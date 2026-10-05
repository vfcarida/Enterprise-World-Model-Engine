"""OpenTelemetry observability adapter for EWM Engine.

Exposes simulation lifecycle events and RunMetrics as OpenTelemetry spans and metrics.
Adheres strictly to the OpenTelemetry Library Guidelines:
- Depends exclusively on `opentelemetry-api` (never `opentelemetry-sdk`).
- Zero network calls and zero telemetry overhead when unattached or unconfigured.
- Emits hierarchical span tree: Simulation -> Rollout -> Step.
- Safe default attributes: never emits secrets, memory dictionaries, or raw state payloads.
- Optional granular state attributes available via `include_state_attributes=True`.
"""

from __future__ import annotations

import logging
from typing import Any

from ewm_engine.hooks.protocol import (
    ConstraintEvaluated,
    HookEvent,
    RolloutCompleted,
    SimulationFinished,
    SimulationStarted,
    StepCompleted,
)
from ewm_engine.simulation.trajectory import TrajectoryStatus

logger = logging.getLogger("ewm_engine.integrations.otel")

# Graceful optional dependency handling
try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Counter, Histogram, Meter
    from opentelemetry.trace import Span, StatusCode, Tracer

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]


class OpenTelemetryHook:
    """Simulation lifecycle hook mapping engine events to OpenTelemetry spans and metrics.

    Span Hierarchy:
        ewm.simulation:{scenario_id}
            ├── ewm.rollout:{rollout_idx}
            │     ├── ewm.step:{step_idx}
            │     └── ...
            └── ...

    Metric Instruments:
        - `ewm.simulation.duration` (Histogram, seconds)
        - `ewm.rollouts.total` (Counter)
        - `ewm.steps.total` (Counter)
        - `ewm.constraints.evaluations` (Counter)
        - `ewm.constraints.hard_violations` (Counter)
        - `ewm.constraints.soft_violations` (Counter)
        - `ewm.dynamics.transitions` (Counter)
        - `ewm.trace.edges` (Counter)
    """

    def __init__(
        self,
        tracer_provider: Any | None = None,
        meter_provider: Any | None = None,
        tracer_name: str = "ewm-engine",
        meter_name: str = "ewm-engine",
        include_state_attributes: bool = False,
        record_step_spans: bool = True,
    ) -> None:
        """Initialize OpenTelemetryHook.

        Args:
            tracer_provider: Optional TracerProvider. If None, uses global tracer provider.
            meter_provider: Optional MeterProvider. If None, uses global meter provider.
            tracer_name: Instrumentation scope name for traces.
            meter_name: Instrumentation scope name for metrics.
            include_state_attributes: When True, records step metrics and non-sensitive state metadata.
                Defaults to False to prevent unintentional payload or PII leakage.
            record_step_spans: When True, records child spans for individual simulation steps.
                Defaults to True.

        Raises:
            ImportError: If `opentelemetry-api` is not installed.
        """
        if not _HAS_OTEL:
            raise ImportError(
                "OpenTelemetry API is required to use OpenTelemetryHook. "
                "Install it via 'pip install ewm-engine[otel]' or 'pip install opentelemetry-api>=1.25.0'."
            )

        self.include_state_attributes = include_state_attributes
        self.record_step_spans = record_step_spans

        # Resolve tracer and meter via the OpenTelemetry API
        self._tracer: Tracer = trace.get_tracer(
            tracer_name,
            tracer_provider=tracer_provider,
        )
        self._meter: Meter = metrics.get_meter(
            meter_name,
            meter_provider=meter_provider,
        )

        # Initialize metric instruments
        self._duration_hist: Histogram = self._meter.create_histogram(
            name="ewm.simulation.duration",
            unit="s",
            description="Total wall-clock duration of simulation execution in seconds",
        )
        self._rollouts_counter: Counter = self._meter.create_counter(
            name="ewm.rollouts.total",
            unit="{rollout}",
            description="Total count of Monte Carlo rollouts executed",
        )
        self._steps_counter: Counter = self._meter.create_counter(
            name="ewm.steps.total",
            unit="{step}",
            description="Total discrete simulation steps executed across all rollouts",
        )
        self._evaluations_counter: Counter = self._meter.create_counter(
            name="ewm.constraints.evaluations",
            unit="{evaluation}",
            description="Total individual constraint evaluations performed",
        )
        self._hard_violations_counter: Counter = self._meter.create_counter(
            name="ewm.constraints.hard_violations",
            unit="{violation}",
            description="Total hard constraint violations encountered",
        )
        self._soft_violations_counter: Counter = self._meter.create_counter(
            name="ewm.constraints.soft_violations",
            unit="{violation}",
            description="Total soft constraint violations encountered",
        )
        self._transitions_counter: Counter = self._meter.create_counter(
            name="ewm.dynamics.transitions",
            unit="{transition}",
            description="Total state dynamics transitions performed",
        )
        self._trace_edges_counter: Counter = self._meter.create_counter(
            name="ewm.trace.edges",
            unit="{edge}",
            description="Total systemic trace causal edges recorded",
        )

        # Active in-flight spans
        self._active_sim_span: Span | None = None
        self._active_rollout_spans: dict[int, Span] = {}

    def on_event(self, event: HookEvent) -> None:
        """Process simulation lifecycle events and dispatch to OpenTelemetry."""
        if not _HAS_OTEL:
            return

        try:
            if isinstance(event, SimulationStarted):
                self._handle_simulation_started(event)
            elif isinstance(event, StepCompleted):
                self._handle_step_completed(event)
            elif isinstance(event, ConstraintEvaluated):
                self._handle_constraint_evaluated(event)
            elif isinstance(event, RolloutCompleted):
                self._handle_rollout_completed(event)
            elif isinstance(event, SimulationFinished):
                self._handle_simulation_finished(event)
        except Exception as exc:
            logger.warning("Error in OpenTelemetryHook handling %s: %s", type(event).__name__, exc)

    def _handle_simulation_started(self, event: SimulationStarted) -> None:
        """Start the root simulation span."""
        attrs: dict[str, Any] = {
            "ewm.scenario_id": event.scenario_id,
            "ewm.horizon": event.horizon,
            "ewm.samples": event.samples,
            "ewm.seed": event.seed,
            "ewm.initial_state_fingerprint": event.initial_state_fingerprint,
        }
        if self.include_state_attributes and event.scenario and event.scenario.metadata:
            for k, v in event.scenario.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    attrs[f"ewm.scenario.metadata.{k}"] = v

        self._active_sim_span = self._tracer.start_span(
            name=f"ewm.simulation:{event.scenario_id}",
            attributes=attrs,
        )

    def _get_or_create_rollout_span(self, rollout_idx: int) -> Span:
        """Get an existing rollout span or create a child of the simulation span."""
        if rollout_idx in self._active_rollout_spans:
            return self._active_rollout_spans[rollout_idx]

        ctx = trace.set_span_in_context(self._active_sim_span) if self._active_sim_span else None
        rollout_span = self._tracer.start_span(
            name=f"ewm.rollout:{rollout_idx}",
            context=ctx,
            attributes={"ewm.rollout_idx": rollout_idx},
        )
        self._active_rollout_spans[rollout_idx] = rollout_span
        return rollout_span

    def _handle_step_completed(self, event: StepCompleted) -> None:
        """Record a completed simulation step as a child span of the active rollout."""
        rollout_span = self._get_or_create_rollout_span(event.rollout_idx)

        if not self.record_step_spans:
            return

        step_ctx = trace.set_span_in_context(rollout_span)
        attrs: dict[str, Any] = {
            "ewm.rollout_idx": event.rollout_idx,
            "ewm.step": event.step,
            "ewm.state_hash": event.state_hash,
            "ewm.actions_accepted_count": len(event.record.actions_accepted),
            "ewm.actions_proposed_count": len(event.record.actions_proposed),
            "ewm.violations_count": len(event.record.constraint_violations),
            "ewm.is_invalid": event.is_invalid,
            "ewm.transition_model": event.record.transition_result.model_name,
        }

        if self.include_state_attributes:
            for k, v in event.record.step_metrics.items():
                attrs[f"ewm.metric.{k}"] = v

        step_span = self._tracer.start_span(
            name=f"ewm.step:{event.step}",
            context=step_ctx,
            attributes=attrs,
        )

        if event.is_invalid:
            step_span.set_status(StatusCode.ERROR, "Fatal constraint violation terminating rollout")
        else:
            step_span.set_status(StatusCode.OK)

        step_span.end()

    def _handle_constraint_evaluated(self, event: ConstraintEvaluated) -> None:
        """Record constraint evaluation as a span event on the active rollout span."""
        if event.rollout_idx in self._active_rollout_spans:
            rollout_span = self._active_rollout_spans[event.rollout_idx]
            rollout_span.add_event(
                name="constraint_evaluated",
                attributes={
                    "ewm.constraint_id": event.constraint_id,
                    "ewm.phase": event.phase.value,
                    "ewm.severity": event.severity.value,
                    "ewm.satisfied": event.satisfied,
                    "ewm.step": event.step,
                },
            )

    def _handle_rollout_completed(self, event: RolloutCompleted) -> None:
        """Complete a rollout span and emit rollout-level metrics."""
        rollout_span = self._active_rollout_spans.pop(event.rollout_idx, None)

        # If rollout span was not created yet (e.g., distributed execution where step events ran out-of-process)
        if rollout_span is None:
            ctx = (
                trace.set_span_in_context(self._active_sim_span) if self._active_sim_span else None
            )
            rollout_span = self._tracer.start_span(
                name=f"ewm.rollout:{event.rollout_idx}",
                context=ctx,
                attributes={"ewm.rollout_idx": event.rollout_idx},
            )

            # Record step spans from trajectory
            if self.record_step_spans and event.trajectory:
                rollout_ctx = trace.set_span_in_context(rollout_span)
                for step_record in event.trajectory.steps:
                    attrs: dict[str, Any] = {
                        "ewm.rollout_idx": event.rollout_idx,
                        "ewm.step": step_record.step,
                        "ewm.state_hash": step_record.state_hash,
                        "ewm.actions_accepted_count": len(step_record.actions_accepted),
                        "ewm.actions_proposed_count": len(step_record.actions_proposed),
                        "ewm.violations_count": len(step_record.constraint_violations),
                        "ewm.transition_model": step_record.transition_result.model_name,
                    }
                    if self.include_state_attributes:
                        for k, v in step_record.step_metrics.items():
                            attrs[f"ewm.metric.{k}"] = v
                    st_span = self._tracer.start_span(
                        name=f"ewm.step:{step_record.step}",
                        context=rollout_ctx,
                        attributes=attrs,
                    )
                    st_span.end()

        # Update rollout span attributes
        rollout_span.set_attribute("ewm.sample_id", event.sample_id)
        rollout_span.set_attribute("ewm.seed", event.trajectory.seed)
        rollout_span.set_attribute("ewm.status", event.status.value)
        rollout_span.set_attribute("ewm.step_count", event.step_count)
        rollout_span.set_attribute(
            "ewm.final_state_fingerprint", event.trajectory.final_state.fingerprint
        )

        if event.status == TrajectoryStatus.INVALID:
            rollout_span.set_status(StatusCode.ERROR, "Rollout invalidated by hard constraint")
        elif event.status == TrajectoryStatus.COMPLETED:
            rollout_span.set_status(StatusCode.OK)

        rollout_span.end()

        # Record rollout metrics
        metric_attrs = {
            "status": event.status.value,
        }
        self._rollouts_counter.add(1, metric_attrs)
        self._steps_counter.add(event.step_count, metric_attrs)

    def _handle_simulation_finished(self, event: SimulationFinished) -> None:
        """Complete the root simulation span and record summary metrics."""
        if self._active_sim_span is not None:
            self._active_sim_span.set_attribute("ewm.duration_seconds", event.duration_seconds)
            self._active_sim_span.set_attribute(
                "ewm.rollout_count", event.run_metrics.rollout_count
            )
            self._active_sim_span.set_attribute(
                "ewm.completed_rollout_count", event.run_metrics.completed_rollout_count
            )
            self._active_sim_span.set_attribute(
                "ewm.invalid_rollout_count", event.run_metrics.invalid_rollout_count
            )
            self._active_sim_span.set_status(StatusCode.OK)
            self._active_sim_span.end()
            self._active_sim_span = None

        # Clean up any remaining rollout spans
        for r_span in self._active_rollout_spans.values():
            r_span.end()
        self._active_rollout_spans.clear()

        # Record RunMetrics via OpenTelemetry meters
        scenario_attrs = {"scenario_id": event.result.scenario.scenario_id}
        self._duration_hist.record(event.run_metrics.simulation_duration_seconds, scenario_attrs)
        self._evaluations_counter.add(event.run_metrics.constraint_evaluation_count, scenario_attrs)
        self._hard_violations_counter.add(event.run_metrics.hard_violation_count, scenario_attrs)
        self._soft_violations_counter.add(event.run_metrics.soft_violation_count, scenario_attrs)
        self._transitions_counter.add(event.run_metrics.dynamics_transition_count, scenario_attrs)
        self._trace_edges_counter.add(event.run_metrics.trace_edge_count, scenario_attrs)
