"""Simulation lifecycle hook protocol and event definitions.

This module provides a lightweight, observational lifecycle event system for EWM Engine.
It allows host applications to monitor simulation progress, log metrics, record traces,
and integrate external observability platforms.

Fault Isolation Policy:
    Hooks are strictly observational and must never alter simulation state, actions,
    RNG streams, or trajectory results. If an exception is raised by a registered hook
    during event dispatch, it is caught, logged with full traceback, and isolated
    so that simulation execution continues unaffected.

Zero-Overhead Contract:
    When no hooks are registered, event creation and dispatch incur zero measurable overhead.

OpenTelemetry Integration Architecture (FUTURE Optional Adapter):
    An OpenTelemetry adapter is planned as a future optional extra package (e.g. `ewm-engine-otel`).
    The core engine does NOT depend on OpenTelemetry SDK or API. Host applications or future
    adapters can implement the `Hook` protocol to export spans and metrics to OpenTelemetry
    collectors without any telemetry dependencies polluting the core engine kernel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
    TrajectoryStatus,
)

logger = logging.getLogger("ewm_engine.hooks")


class HookEvent(BaseModel):
    """Base class for all immutable simulation lifecycle hook events."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class SimulationStarted(HookEvent):
    """Emitted when a simulation experiment run begins, before rollouts execute."""

    scenario_id: str = Field(description="Unique scenario identifier.")
    horizon: int = Field(ge=1, description="Number of simulation time steps per rollout.")
    samples: int = Field(ge=1, description="Number of Monte Carlo rollouts requested.")
    seed: int = Field(description="Master pseudo-random seed.")
    initial_state_fingerprint: str = Field(
        default="",
        description="Deterministic canonical fingerprint of the baseline world state.",
    )
    scenario: Scenario | None = Field(
        default=None,
        description="Optional full scenario specification reference.",
    )


class ConstraintEvaluated(HookEvent):
    """Emitted each time an individual constraint rule is evaluated in a step."""

    rollout_idx: int = Field(ge=0, description="Index of the currently executing rollout.")
    step: int = Field(ge=0, description="Discrete simulation time step index.")
    phase: ConstraintPhase = Field(description="Constraint lifecycle evaluation phase.")
    constraint_id: str = Field(description="Unique identifier of the evaluated constraint.")
    severity: ConstraintSeverity = Field(description="Enforced constraint severity.")
    satisfied: bool = Field(description="Whether the constraint condition was satisfied.")
    result: ConstraintResult = Field(description="Complete constraint evaluation result payload.")


class StepCompleted(HookEvent):
    """Emitted at the completion of each discrete simulation step in a rollout."""

    rollout_idx: int = Field(ge=0, description="Index of the active rollout.")
    step: int = Field(ge=0, description="Discrete simulation time step index.")
    state_hash: str = Field(description="Cryptographic hash of the state prior to transition.")
    record: StepRecord = Field(
        description="Step record containing actions, transitions, and metrics."
    )
    is_invalid: bool = Field(
        default=False,
        description="Whether this step triggered a fatal hard violation terminating the rollout.",
    )


class RolloutCompleted(HookEvent):
    """Emitted upon the completion or invalidation of an individual rollout trajectory."""

    rollout_idx: int = Field(ge=0, description="Rollout index in the batch.")
    sample_id: int = Field(ge=0, description="Sample seed index.")
    status: TrajectoryStatus = Field(description="Terminal status (COMPLETED, INVALID, or FAILED).")
    step_count: int = Field(ge=0, description="Number of steps executed in this rollout.")
    trajectory: Trajectory = Field(description="Final trajectory object containing step records.")


class SimulationFinished(HookEvent):
    """Emitted when the entire simulation experiment run concludes."""

    result: SimulationResult = Field(description="Final assembled simulation result.")
    run_metrics: RunMetrics = Field(description="Summary execution and performance metrics.")
    duration_seconds: float = Field(
        ge=0.0, description="Total wall-clock execution duration in seconds."
    )


@runtime_checkable
class Hook(Protocol):
    """Protocol for simulation lifecycle observation hooks."""

    def on_event(self, event: HookEvent) -> None:
        """Handle an emitted simulation lifecycle event.

        Must not modify simulation state, trajectory outputs, or execution flow.
        """
        ...


class _CallableHookAdapter:
    """Internal adapter allowing arbitrary callables to conform to the Hook protocol."""

    def __init__(self, fn: Callable[[HookEvent], None]) -> None:
        self._fn = fn

    def on_event(self, event: HookEvent) -> None:
        self._fn(event)

    def __repr__(self) -> str:
        return f"CallableHookAdapter({self._fn!r})"


class HookRegistry:
    """Registry managing simulation lifecycle hooks with robust fault isolation.

    Fault isolation policy:
        Hooks are purely observational. If an exception occurs in any hook during `emit()`,
        the error is logged via `ewm_engine.hooks` with full traceback and swallowed so that
        the simulation execution continues without interruption.
    """

    def __init__(
        self,
        hooks: Sequence[Hook | Callable[[HookEvent], None]] | None = None,
    ) -> None:
        self._hooks: list[Hook] = []
        if hooks:
            for h in hooks:
                self.register(h)

    def register(self, hook: Hook | Callable[[HookEvent], None]) -> None:
        """Register a new lifecycle hook or callback into the registry."""
        if hasattr(hook, "on_event") and callable(hook.on_event):
            self._hooks.append(hook)
        elif callable(hook):
            self._hooks.append(_CallableHookAdapter(hook))
        else:
            raise TypeError(
                f"Expected Hook protocol instance (with on_event) or callable, got {type(hook).__name__}"
            )

    def unregister(self, hook: Hook | Callable[[HookEvent], None]) -> None:
        """Unregister an existing hook or callback."""
        if hook in self._hooks:
            self._hooks.remove(hook)
            return
        # If wrapped callable, find and remove adapter
        for h in list(self._hooks):
            if isinstance(h, _CallableHookAdapter) and h._fn is hook:
                self._hooks.remove(h)
                return

    def emit(self, event: HookEvent) -> None:
        """Dispatch a lifecycle event to all registered hooks.

        Isolates any exceptions thrown by hooks to protect simulation integrity.
        """
        if not self._hooks:
            return

        for hook in self._hooks:
            try:
                hook.on_event(event)
            except Exception as exc:
                logger.error(
                    "Simulation lifecycle hook %r raised an exception handling %s: %s",
                    hook,
                    type(event).__name__,
                    exc,
                    exc_info=True,
                )

    def __len__(self) -> int:
        return len(self._hooks)

    def __iter__(self) -> Iterator[Hook]:
        return iter(self._hooks)


__all__ = [
    "ConstraintEvaluated",
    "Hook",
    "HookEvent",
    "HookRegistry",
    "RolloutCompleted",
    "SimulationFinished",
    "SimulationStarted",
    "StepCompleted",
]
