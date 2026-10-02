"""Unit tests for lifecycle hooks protocol, event dispatch, and fault isolation."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pytest

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.hooks.protocol import (
    ConstraintEvaluated,
    HookEvent,
    HookRegistry,
    RolloutCompleted,
    SimulationFinished,
    SimulationStarted,
    StepCompleted,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from ewm_engine.simulation.trajectory import TrajectoryStatus


class DummyDynamics:
    """Simple dynamics model incrementing inventory resource."""

    name: str = "DummyDynamics"
    version: str = "1.0.0"

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: np.random.Generator,
    ) -> TransitionResult:
        delta = 0.0
        for act in actions:
            if act.type == "add_stock":
                delta += float(act.parameters.get("amount", 5.0))
        next_s = state.update_resource("stock", delta=delta)
        return TransitionResult(
            next_state=next_s,
            applied_changes={"stock_delta": delta},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class DummyConstraint:
    """Simple constraint checking stock upper bound."""

    def __init__(self, max_stock: float = 100.0) -> None:
        self.constraint_id = "max_stock_limit"
        self.version = "1.0.0"
        self.severity = ConstraintSeverity.HARD
        self.description = "Limits stock to maximum allowable threshold."
        self.max_stock = max_stock

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        stock = state.resources.get("stock")
        val = stock.current if stock else 0.0
        if val > self.max_stock:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
                satisfied=False,
                message=f"Stock {val} exceeds maximum {self.max_stock}",
                violating_resources=("stock",),
            )
        return ConstraintResult(
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
            satisfied=True,
        )


def _build_test_world_and_scenario(horizon: int = 3, samples: int = 2) -> tuple[World, Scenario]:
    world = World(
        state=WorldState(resources={"stock": Resource(id="stock", current=20.0, max_value=150.0)}),
        dynamics=DummyDynamics(),
        constraints=[DummyConstraint(max_stock=100.0)],
    )
    scenario = Scenario(
        scenario_id="hook_test_scen",
        horizon=horizon,
        samples=samples,
        seed=123,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(id="act_add", type="add_stock", parameters={"amount": 10.0}),
            ),
        ),
    )
    return world, scenario


class EventRecorderHook:
    """Observer collecting all received lifecycle events."""

    def __init__(self) -> None:
        self.events: list[HookEvent] = []

    def on_event(self, event: HookEvent) -> None:
        self.events.append(event)


class CrashingHook:
    """Hook designed to deliberately raise exceptions on every event."""

    def on_event(self, event: HookEvent) -> None:
        raise RuntimeError(f"Deliberate hook failure on {type(event).__name__}")


def test_hook_events_fire_in_correct_order_and_payloads() -> None:
    """AC: Events fire in strictly ordered sequence with correct payloads."""
    world, scenario = _build_test_world_and_scenario(horizon=2, samples=2)
    recorder = EventRecorderHook()
    registry = HookRegistry([recorder])

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario, hooks=registry)

    events = recorder.events
    assert len(events) > 0

    # 1. First event must be SimulationStarted
    assert isinstance(events[0], SimulationStarted)
    assert events[0].scenario_id == scenario.scenario_id
    assert events[0].horizon == 2
    assert events[0].samples == 2
    assert events[0].seed == 123

    # 2. Last event must be SimulationFinished
    assert isinstance(events[-1], SimulationFinished)
    assert events[-1].result is result
    assert events[-1].run_metrics.rollout_count == 2
    assert events[-1].duration_seconds >= 0.0

    # 3. Check sequence of intermediate events
    event_types = [type(e) for e in events]
    assert event_types[0] == SimulationStarted
    assert event_types[-1] == SimulationFinished

    # Verify RolloutCompleted is fired exactly `samples` times
    rollout_events = [e for e in events if isinstance(e, RolloutCompleted)]
    assert len(rollout_events) == 2
    for idx, re in enumerate(rollout_events):
        assert re.rollout_idx == idx
        assert re.status == TrajectoryStatus.COMPLETED
        assert re.step_count == 2

    # Verify StepCompleted is fired `samples * horizon` times
    step_events = [e for e in events if isinstance(e, StepCompleted)]
    assert len(step_events) == 4

    # Verify ConstraintEvaluated events are present and typed
    eval_events = [e for e in events if isinstance(e, ConstraintEvaluated)]
    assert len(eval_events) > 0
    for ee in eval_events:
        assert ee.constraint_id == "max_stock_limit"
        assert ee.severity == ConstraintSeverity.HARD
        assert ee.satisfied is True


def test_crashing_hook_is_isolated_and_never_fails_simulation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC: A raising hook doesn't break the sim; failure is logged and execution proceeds."""
    world, scenario = _build_test_world_and_scenario(horizon=2, samples=1)
    crashing_hook = CrashingHook()
    recorder = EventRecorderHook()

    registry = HookRegistry([crashing_hook, recorder])
    engine = SimulationEngine(hooks=registry)

    with caplog.at_level(logging.ERROR, logger="ewm_engine.hooks"):
        result = engine.run(world=world, scenario=scenario)

    # Simulation must finish completely and return valid result
    assert result.completed_count == 1
    assert len(result.trajectories) == 1

    # Fault isolation check: the second hook (recorder) still received all events
    assert len(recorder.events) > 0
    assert isinstance(recorder.events[0], SimulationStarted)
    assert isinstance(recorder.events[-1], SimulationFinished)

    # Logger check: errors were logged
    assert any("Deliberate hook failure" in record.message for record in caplog.records)


def test_result_equality_with_and_without_hooks() -> None:
    """AC (Guardrail): Registering hooks does not alter state, RNG, or results."""
    world, scenario = _build_test_world_and_scenario(horizon=3, samples=2)

    # Run 1: without hooks
    engine_no_hooks = SimulationEngine()
    result_plain = engine_no_hooks.run(world=world, scenario=scenario)

    # Run 2: with hooks
    recorder = EventRecorderHook()
    engine_with_hooks = SimulationEngine(hooks=HookRegistry([recorder]))
    result_hooked = engine_with_hooks.run(world=world, scenario=scenario)

    # Provenance fingerprints must be identical
    assert (
        result_plain.provenance.scenario_fingerprint
        == result_hooked.provenance.scenario_fingerprint
    )
    assert (
        result_plain.provenance.initial_state_fingerprint
        == result_hooked.provenance.initial_state_fingerprint
    )

    # Trajectories must be bit-for-bit identical
    assert len(result_plain.trajectories) == len(result_hooked.trajectories)
    for t_plain, t_hooked in zip(
        result_plain.trajectories, result_hooked.trajectories, strict=True
    ):
        assert t_plain.seed == t_hooked.seed
        assert t_plain.status == t_hooked.status
        assert t_plain.final_state.fingerprint == t_hooked.final_state.fingerprint
        assert len(t_plain.steps) == len(t_hooked.steps)
        for s_plain, s_hooked in zip(t_plain.steps, t_hooked.steps, strict=True):
            assert s_plain.state_hash == s_hooked.state_hash
            assert s_plain.step_metrics == s_hooked.step_metrics


def test_hook_registry_registration_and_unregistration() -> None:
    """Verify HookRegistry container operations with protocols and callables."""
    registry = HookRegistry()
    assert len(registry) == 0

    h1 = EventRecorderHook()
    registry.register(h1)
    assert len(registry) == 1
    assert h1 in list(registry)

    # Registering callable
    captured: list[HookEvent] = []

    def cb(event: HookEvent) -> None:
        captured.append(event)

    registry.register(cb)
    assert len(registry) == 2

    # Emit event
    dummy_event = SimulationStarted(
        scenario_id="s1",
        horizon=1,
        samples=1,
        seed=42,
    )
    registry.emit(dummy_event)

    assert len(h1.events) == 1
    assert len(captured) == 1

    # Unregister callable
    registry.unregister(cb)
    assert len(registry) == 1

    # Unregister protocol hook
    registry.unregister(h1)
    assert len(registry) == 0

    # Type error on invalid object
    with pytest.raises(TypeError, match="Expected Hook protocol instance"):
        registry.register(12345)  # type: ignore[arg-type]
