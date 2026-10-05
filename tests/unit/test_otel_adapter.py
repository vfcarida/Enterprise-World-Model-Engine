"""Unit tests for OpenTelemetry observability adapter (Prompt P04).

Verifies:
- Hierarchical span tree generation (simulation -> rollout -> step).
- Metric instruments recording RunMetrics.
- Safe attribute defaults (no raw state or memory payloads leaked).
- Distributed execution compatibility.
- Zero impact on simulation determinism and execution when unattached.
- Clear ImportError when opentelemetry-api is absent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.integrations.otel import OpenTelemetryHook
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.executors import MultiprocessingExecutor
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


@pytest.fixture
def otel_test_setup() -> tuple[
    TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
]:
    """Create isolated TracerProvider and MeterProvider instances with in-memory exporters."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    return tracer_provider, span_exporter, meter_provider, metric_reader


def _create_test_world_and_scenario(
    samples: int = 2,
    horizon: int = 3,
    seed: int = 12345,
    hard_limit: float = 200.0,
) -> tuple[World, Scenario]:
    """Build a deterministic test world and scenario."""
    state = WorldState(
        entities=[Entity(id="warehouse_1", type="warehouse")],
        resources=[Resource(id="inventory", current=100.0, min_value=0.0, max_value=hard_limit)],
        memory={"secret_key": "do_not_leak_this", "counter": 42},
    )
    world = World(
        state=state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory", mean_demand=10.0, std_demand=2.0
        ),
        constraints=[
            ResourceCapacityConstraint(resource_id="inventory", severity=ConstraintSeverity.HARD)
        ],
    )
    scenario = Scenario(
        scenario_id="otel_test_scenario",
        name="OTelTestScenario",
        horizon=horizon,
        samples=samples,
        seed=seed,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(id="restock", type="restock_inventory", parameters={"qty": 20.0}),
            ),
        ),
    )
    return world, scenario


def test_span_tree_hierarchy_and_attributes(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify that OpenTelemetryHook builds a clean 3-level span hierarchy: simulation -> rollout -> step."""
    tracer_provider, span_exporter, meter_provider, _ = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    world, scenario = _create_test_world_and_scenario(samples=2, horizon=3, seed=999)
    engine = SimulationEngine(hooks=[hook])
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 2

    spans = span_exporter.get_finished_spans()
    # 1 simulation span + 2 rollout spans + (2 * 3) step spans = 9 spans total
    assert len(spans) == 9

    # Find simulation span
    sim_spans = [s for s in spans if s.name.startswith("ewm.simulation:")]
    assert len(sim_spans) == 1
    sim_span = sim_spans[0]
    assert sim_span.parent is None
    sim_attrs = sim_span.attributes or {}
    assert sim_attrs["ewm.scenario_id"] == "otel_test_scenario"
    assert sim_attrs["ewm.horizon"] == 3
    assert sim_attrs["ewm.samples"] == 2
    assert sim_attrs["ewm.seed"] == 999
    assert sim_attrs["ewm.rollout_count"] == 2
    assert sim_attrs["ewm.completed_rollout_count"] == 2
    assert "ewm.duration_seconds" in sim_attrs

    # Find rollout spans
    rollout_spans = [s for s in spans if s.name.startswith("ewm.rollout:")]
    assert len(rollout_spans) == 2
    for r_span in rollout_spans:
        assert r_span.parent is not None
        assert r_span.parent.span_id == sim_span.context.span_id
        r_attrs = r_span.attributes or {}
        assert r_attrs["ewm.status"] == "completed"
        assert r_attrs["ewm.step_count"] == 3
        assert "ewm.final_state_fingerprint" in r_attrs

    # Find step spans
    step_spans = [s for s in spans if s.name.startswith("ewm.step:")]
    assert len(step_spans) == 6
    rollout_span_ids = {r.context.span_id for r in rollout_spans}
    for st_span in step_spans:
        assert st_span.parent is not None
        assert st_span.parent.span_id in rollout_span_ids
        st_attrs = st_span.attributes or {}
        assert "ewm.step" in st_attrs
        assert "ewm.state_hash" in st_attrs
        assert "ewm.transition_model" in st_attrs


def test_safe_default_attributes_no_pii_leak(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify that memory dictionaries, secrets, and raw state are NOT in span attributes by default."""
    tracer_provider, span_exporter, meter_provider, _ = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_state_attributes=False,
    )

    world, scenario = _create_test_world_and_scenario(samples=1, horizon=2)
    engine = SimulationEngine(hooks=[hook])
    engine.run(world=world, scenario=scenario)

    spans = span_exporter.get_finished_spans()
    for span in spans:
        span_attrs = span.attributes or {}
        for attr_key, attr_val in span_attrs.items():
            assert "secret_key" not in attr_key
            assert "secret_key" not in str(attr_val)
            assert "do_not_leak_this" not in str(attr_val)
            assert not attr_key.startswith("ewm.metric.")


def test_opt_in_state_attributes(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify that when include_state_attributes=True, step metrics are safely attached."""
    tracer_provider, span_exporter, meter_provider, _ = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        include_state_attributes=True,
    )

    world, scenario = _create_test_world_and_scenario(samples=1, horizon=2)
    engine = SimulationEngine(hooks=[hook])
    engine.run(world=world, scenario=scenario)

    step_spans = [s for s in span_exporter.get_finished_spans() if s.name.startswith("ewm.step:")]
    assert len(step_spans) == 2
    for st_span in step_spans:
        step_attrs = st_span.attributes or {}
        assert any(k.startswith("ewm.metric.") for k in step_attrs)


def test_metric_emission_on_simulation_finished(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify that OpenTelemetryHook records all RunMetrics counters and histograms."""
    tracer_provider, _, meter_provider, metric_reader = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    world, scenario = _create_test_world_and_scenario(samples=3, horizon=4)
    engine = SimulationEngine(hooks=[hook])
    engine.run(world=world, scenario=scenario)

    metric_data = metric_reader.get_metrics_data()
    assert metric_data is not None

    metric_names = set()
    for rm in metric_data.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                metric_names.add(m.name)

    expected_metrics = {
        "ewm.simulation.duration",
        "ewm.rollouts.total",
        "ewm.steps.total",
        "ewm.constraints.evaluations",
        "ewm.constraints.hard_violations",
        "ewm.constraints.soft_violations",
        "ewm.dynamics.transitions",
        "ewm.trace.edges",
    }
    assert expected_metrics.issubset(metric_names), (
        f"Missing metrics: {expected_metrics - metric_names}"
    )


def test_distributed_mode_span_reconstruction(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify that OpenTelemetryHook reconstructs spans when executed with MultiprocessingExecutor."""
    tracer_provider, span_exporter, meter_provider, _ = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
    )

    world, scenario = _create_test_world_and_scenario(samples=2, horizon=2)
    engine = SimulationEngine(hooks=[hook], executor=MultiprocessingExecutor(max_workers=2))
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 2

    spans = span_exporter.get_finished_spans()
    # 1 sim span + 2 rollout spans + (2 * 2) step spans = 7 spans
    assert len(spans) == 7

    sim_spans = [s for s in spans if s.name.startswith("ewm.simulation:")]
    assert len(sim_spans) == 1
    rollout_spans = [s for s in spans if s.name.startswith("ewm.rollout:")]
    assert len(rollout_spans) == 2
    step_spans = [s for s in spans if s.name.startswith("ewm.step:")]
    assert len(step_spans) == 4


def test_disable_step_spans_option(
    otel_test_setup: tuple[
        TracerProvider, InMemorySpanExporter, MeterProvider, InMemoryMetricReader
    ],
) -> None:
    """Verify record_step_spans=False omits granular step spans for high-volume simulations."""
    tracer_provider, span_exporter, meter_provider, _ = otel_test_setup

    hook = OpenTelemetryHook(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        record_step_spans=False,
    )

    world, scenario = _create_test_world_and_scenario(samples=2, horizon=5)
    engine = SimulationEngine(hooks=[hook])
    engine.run(world=world, scenario=scenario)

    spans = span_exporter.get_finished_spans()
    # 1 sim span + 2 rollout spans (0 step spans)
    assert len(spans) == 3
    assert not any(s.name.startswith("ewm.step:") for s in spans)


def test_unattached_engine_behavior_identical() -> None:
    """Verify that running without OpenTelemetryHook produces identical results with zero overhead."""
    world, scenario = _create_test_world_and_scenario(samples=3, horizon=3, seed=42)
    engine = SimulationEngine()

    res1 = engine.run(world=world, scenario=scenario)
    res2 = engine.run(world=world, scenario=scenario)

    assert engine.verify_determinism(world, scenario) is True
    for t1, t2 in zip(res1.trajectories, res2.trajectories, strict=True):
        assert t1.final_state.fingerprint == t2.final_state.fingerprint


def test_missing_opentelemetry_import_error() -> None:
    """Verify descriptive ImportError when opentelemetry is not installed."""
    with patch("ewm_engine.integrations.otel._HAS_OTEL", False):
        with pytest.raises(ImportError, match="pip install ewm-engine\\[otel\\]"):
            OpenTelemetryHook()
