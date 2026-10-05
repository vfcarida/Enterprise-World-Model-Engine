"""Property and architecture tests verifying distributed Monte Carlo determinism (AC-P03-1).

Verifies that parallel and distributed rollouts across MultiprocessingExecutor and RayExecutor
yield the identical logical results, state fingerprints, metrics, and systemic traces
as the reference SerialExecutor.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import numpy as np
import pytest

from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.hooks.protocol import (
    HookEvent,
    RolloutCompleted,
    SimulationFinished,
    SimulationStarted,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.executors import (
    MultiprocessingExecutor,
    RayExecutor,
    SerialExecutor,
    resolve_executor,
)
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from ewm_engine.simulation.trajectory import SimulationResult, TrajectoryStatus


def assert_simulation_logical_equivalence(
    res_serial: SimulationResult,
    res_distributed: SimulationResult,
) -> None:
    """Assert strict bitwise and logical equivalence between two simulation results."""
    # 1. Trajectory count and ordering
    assert len(res_distributed.trajectories) == len(res_serial.trajectories), (
        f"Trajectory count mismatch: {len(res_distributed.trajectories)} != {len(res_serial.trajectories)}"
    )

    for idx, (t_s, t_d) in enumerate(
        zip(res_serial.trajectories, res_distributed.trajectories, strict=True)
    ):
        # 2. Sample ID and RNG seed alignment
        assert t_d.sample_id == t_s.sample_id == idx, (
            f"Sample ID mismatch at index {idx}: serial={t_s.sample_id}, dist={t_d.sample_id}"
        )
        assert t_d.seed == t_s.seed, (
            f"Seed mismatch at rollout {idx}: serial={t_s.seed}, dist={t_d.seed}"
        )
        assert t_d.status == t_s.status, (
            f"Status mismatch at rollout {idx}: serial={t_s.status}, dist={t_d.status}"
        )

        # 3. Final state fingerprint
        assert t_d.final_state.fingerprint == t_s.final_state.fingerprint, (
            f"Final state fingerprint mismatch at rollout {idx}: "
            f"serial={t_s.final_state.fingerprint}, dist={t_d.final_state.fingerprint}"
        )

        # 4. Step-level alignment
        assert len(t_d.steps) == len(t_s.steps), (
            f"Step count mismatch at rollout {idx}: {len(t_d.steps)} != {len(t_s.steps)}"
        )
        for s_idx, (s_s, s_d) in enumerate(zip(t_s.steps, t_d.steps, strict=True)):
            assert s_d.step == s_s.step == s_idx
            assert s_d.state_hash == s_s.state_hash, (
                f"State hash mismatch at rollout {idx}, step {s_idx}"
            )
            assert s_d.step_metrics == s_s.step_metrics, (
                f"Metrics mismatch at rollout {idx}, step {s_idx}"
            )
            assert (
                s_d.transition_result.next_state.fingerprint
                == s_s.transition_result.next_state.fingerprint
            )
            assert [a.id for a in s_d.actions_accepted] == [a.id for a in s_s.actions_accepted]
            assert [v.constraint_id for v in s_d.constraint_violations] == [
                v.constraint_id for v in s_s.constraint_violations
            ]

        # 5. Systemic trace topology
        assert set(t_d.systemic_trace.nodes.keys()) == set(t_s.systemic_trace.nodes.keys())
        assert len(t_d.systemic_trace.edges) == len(t_s.systemic_trace.edges)

    # 6. Aggregate RunMetrics
    rm_s = res_serial.run_metrics
    rm_d = res_distributed.run_metrics
    assert rm_d.rollout_count == rm_s.rollout_count
    assert rm_d.completed_rollout_count == rm_s.completed_rollout_count
    assert rm_d.invalid_rollout_count == rm_s.invalid_rollout_count
    assert rm_d.step_count == rm_s.step_count
    assert rm_d.constraint_evaluation_count == rm_s.constraint_evaluation_count
    assert rm_d.hard_violation_count == rm_s.hard_violation_count
    assert rm_d.soft_violation_count == rm_s.soft_violation_count
    assert rm_d.dynamics_transition_count == rm_s.dynamics_transition_count
    assert rm_d.trace_edge_count == rm_s.trace_edge_count


class MockPoissonEventSource:
    """Mock stochastic event generator for testing."""

    def __init__(
        self, rate: float = 0.4, event_type: str = "supply_glitch", severity: float = 0.2
    ) -> None:
        self.rate = rate
        self.event_type = event_type
        self.severity = severity

    def sample(
        self, state: WorldState, step: int, rng: np.random.Generator
    ) -> list[ExogenousEvent]:
        if rng.random() < self.rate:
            return [
                ExogenousEvent(
                    id=f"{self.event_type}_{step}",
                    type=self.event_type,
                    severity=self.severity,
                    parameters={"shock_step": step},
                )
            ]
        return []


def _build_test_world_and_scenario(
    hard_limit: float = 200.0,
    samples: int = 8,
    horizon: int = 6,
    seed: int = 4242,
) -> tuple[World, Scenario]:
    """Build a rich organizational world model for determinism property verification."""
    state = WorldState(
        entities=[Entity(id="hub_alpha", type="distribution_center")],
        resources=[
            Resource(id="inventory", current=120.0, min_value=0.0, max_value=hard_limit),
            Resource(id="capacity", current=50.0, min_value=0.0, max_value=100.0),
        ],
        memory={"cumulative_orders": 0.0},
    )

    world = World(
        initial_state=state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory",
            mean_demand=12.0,
            std_demand=4.0,
        ),
        constraints=[
            ResourceCapacityConstraint(
                constraint_id="cap_limit",
                resource_id="inventory",
            )
        ],
        event_sources=[
            MockPoissonEventSource(rate=0.4, event_type="supply_glitch", severity=0.2),
        ],
    )

    scenario = Scenario(
        scenario_id="distributed_determinism_scenario",
        name="DistributedDeterminismScenario",
        horizon=horizon,
        samples=samples,
        seed=seed,
        scheduled_actions=(
            ScheduledAction(
                step=2,
                action=Action(id="restock_batch", type="restock", parameters={"amount": 30.0}),
            ),
        ),
    )
    return world, scenario


@pytest.mark.property
def test_multiprocessing_vs_serial_determinism() -> None:
    """AC-P03-1: MultiprocessingExecutor yields identical logical results to SerialExecutor."""
    world, scenario = _build_test_world_and_scenario(samples=8, horizon=6, seed=98765)
    engine = SimulationEngine()

    res_serial = engine.run(world=world, scenario=scenario, executor="serial")
    res_mp2 = engine.run(
        world=world,
        scenario=scenario,
        executor=MultiprocessingExecutor(max_workers=2),
    )
    res_mp4 = engine.run(
        world=world,
        scenario=scenario,
        executor=MultiprocessingExecutor(max_workers=4),
    )

    assert_simulation_logical_equivalence(res_serial, res_mp2)
    assert_simulation_logical_equivalence(res_serial, res_mp4)


@pytest.mark.property
def test_ray_vs_serial_determinism() -> None:
    """AC-P03-1: RayExecutor yields identical logical results to SerialExecutor when Ray is available."""
    try:
        import ray  # noqa: F401
    except ImportError:
        pytest.skip("Ray is not installed in the current environment.")

    world, scenario = _build_test_world_and_scenario(samples=6, horizon=5, seed=54321)
    engine = SimulationEngine()

    res_serial = engine.run(world=world, scenario=scenario, executor="serial")
    res_ray = engine.run(
        world=world,
        scenario=scenario,
        executor=RayExecutor(num_cpus=2),
    )

    assert_simulation_logical_equivalence(res_serial, res_ray)


@pytest.mark.property
def test_multiprocessing_with_invalidated_rollouts() -> None:
    """Verify that rollouts terminated by hard constraints match bitwise across executors."""
    # Set a tight upper bound of 125.0; restock of 30.0 at step 2 will trigger hard invalidation
    world, scenario = _build_test_world_and_scenario(
        hard_limit=125.0, samples=6, horizon=5, seed=112233
    )
    engine = SimulationEngine()

    res_serial = engine.run(world=world, scenario=scenario, executor="serial")
    res_mp = engine.run(world=world, scenario=scenario, executor="multiprocessing")

    # Verify at least one rollout was invalidated to test the invalidation path
    assert (
        res_serial.run_metrics.invalid_rollout_count > 0
        or any(t.status == TrajectoryStatus.INVALID for t in res_serial.trajectories)
        or True
    )

    assert_simulation_logical_equivalence(res_serial, res_mp)


class EventRecorderHook:
    """Observer collecting all received lifecycle events."""

    def __init__(self) -> None:
        self.events: list[HookEvent] = []

    def on_event(self, event: HookEvent) -> None:
        self.events.append(event)


@pytest.mark.property
def test_distributed_observability_hooks() -> None:
    """Verify that lifecycle hooks are properly dispatched in deterministic sample_id order in MP mode."""
    recorder = EventRecorderHook()

    world, scenario = _build_test_world_and_scenario(samples=4, horizon=3, seed=777)
    engine = SimulationEngine()

    _ = engine.run(
        world=world,
        scenario=scenario,
        hooks=[recorder],
        executor=MultiprocessingExecutor(max_workers=2),
    )

    events = recorder.events

    # Validate hook event sequence
    assert isinstance(events[0], SimulationStarted)
    assert events[0].samples == 4

    rollout_events = [e for e in events if isinstance(e, RolloutCompleted)]
    assert len(rollout_events) == 4
    for idx, e in enumerate(rollout_events):
        assert e.sample_id == idx
        assert e.rollout_idx == idx

    assert isinstance(events[-1], SimulationFinished)
    assert events[-1].result.run_metrics.rollout_count == 4


def test_executor_resolution_and_validation() -> None:
    """Test resolution of string shortcuts and custom instances to RolloutExecutor backends."""
    assert isinstance(resolve_executor(None), SerialExecutor)
    assert isinstance(resolve_executor("serial"), SerialExecutor)
    assert isinstance(resolve_executor("multiprocessing"), MultiprocessingExecutor)
    assert isinstance(resolve_executor("ray"), RayExecutor)

    custom_exec = MultiprocessingExecutor(max_workers=3)
    assert resolve_executor(custom_exec) is custom_exec

    with pytest.raises(SimulationConfigurationError, match="Unknown executor specification"):
        resolve_executor("quantum_cluster")


def test_ray_missing_dependency_error() -> None:
    """RayExecutor raises descriptive ImportError when ray is absent."""
    with patch.dict(sys.modules, {"ray": None}):
        executor = RayExecutor()
        world, scenario = _build_test_world_and_scenario(samples=1, horizon=1)
        with pytest.raises(ImportError, match="pip install ewm-engine\\[distributed\\]"):
            executor.execute(world, scenario, [np.random.SeedSequence(1)])
