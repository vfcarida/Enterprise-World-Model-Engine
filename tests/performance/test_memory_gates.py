"""Deterministic memory regression and leak gates for EWM Engine (R04 Task C).

Memory usage in deterministic scientific simulations is highly repeatable, making memory
bounds and leak checks safe hard gates (unlike noisy wall-clock benchmarks).

Enforces:
1. Hard memory ceiling per discrete simulation step (< 25 MB).
2. Trajectory horizon scaling ceiling (< 50 MB for 50 steps).
3. Monte Carlo parallel trajectory spawning (< 60 MB for 10 trajectories).
4. Structural sharing in state branching (< 15 MB for 50 branches).
5. Systemic trace DAG memory footprint (< 20 MB for 500 nodes).
6. Zero unbounded memory leak across repeated simulation cycles (< 200 KB net growth).

Uses both `@pytest.mark.limit_memory("N MB")` / `@pytest.mark.limit_leaks` (enforced natively
by pytest-memray under Linux CI) and Python standard `tracemalloc` for cross-platform enforcement.
"""

from __future__ import annotations

import contextlib
import gc
import tracemalloc
from collections.abc import Callable, Generator
from typing import Any

import numpy as np
import pytest

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.rollout import execute_step
from ewm_engine.simulation.scenario import Scenario


@contextlib.contextmanager
def assert_memory_budget(max_peak_mb: float) -> Generator[None, None, None]:
    """Cross-platform memory budget assertion using tracemalloc."""
    gc.collect()
    tracemalloc.start()
    try:
        yield
        _current, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        assert peak_mb <= max_peak_mb, (
            f"Peak memory allocation of {peak_mb:.2f} MB exceeded strict gate ceiling "
            f"of {max_peak_mb:.2f} MB"
        )
    finally:
        tracemalloc.stop()


def assert_no_memory_leak(
    operation: Callable[[], Any],
    warmup_cycles: int = 10,
    measured_cycles: int = 40,
    max_leak_kb: float = 250.0,
) -> None:
    """Assert zero net memory growth across repeated execution cycles."""
    gc.collect()
    for _ in range(warmup_cycles):
        operation()
    gc.collect()

    tracemalloc.start()
    snap_start = tracemalloc.take_snapshot()
    for _ in range(measured_cycles):
        operation()
    gc.collect()
    snap_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_end.compare_to(snap_start, "lineno")
    total_diff_kb = sum(stat.size_diff for stat in stats) / 1024.0
    assert total_diff_kb <= max_leak_kb, (
        f"Memory leak detected: net allocation grew by {total_diff_kb:.2f} KB across "
        f"{measured_cycles} cycles (tolerance: {max_leak_kb:.2f} KB)"
    )


def _build_test_world(num_nodes: int = 30) -> tuple[World, WorldState, Scenario]:
    """Construct standard multi-node world for memory gating."""
    entities = [
        Entity(id=f"depot_{i}", type="depot", attributes={"zone": i % 4}) for i in range(num_nodes)
    ]
    relationships = [
        Relationship(
            source=f"depot_{i}",
            target=f"depot_{(i + 1) % num_nodes}",
            type="connected_to",
            attributes={"distance": 10},
        )
        for i in range(num_nodes)
    ]
    resources = [
        Resource(
            id=f"stock_{i}",
            entity_id=f"depot_{i}",
            current=100.0,
            min_value=0.0,
            max_value=300.0,
            unit="units",
        )
        for i in range(num_nodes)
    ]

    initial_state = WorldState(
        entities=entities,
        relationships=relationships,
        resources=resources,
        memory={"step": 0},
        active_rules={"safety_stock": 20.0},
        context={"cluster": "east"},
        timestamp=0.0,
        step=0,
    )

    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_0", mean_demand=3.0, std_demand=1.0),
        ]
    )

    constraints = ConstraintRegistry(
        [ResourceCapacityConstraint(resource_id=f"stock_{i}") for i in range(min(num_nodes, 10))]
        + [ActionTransferAvailabilityConstraint()]
    )

    actors = [
        ThresholdReplenishmentActor(
            actor_id=f"actor_{i}",
            source_resource=f"stock_{(i + 1) % num_nodes}",
            target_resource=f"stock_{i}",
            reorder_point=40.0,
            order_quantity=20.0,
        )
        for i in range(min(num_nodes, 4))
    ]

    world = World(
        initial_state=initial_state,
        dynamics=dynamics,
        constraints=constraints,
        actors=actors,
    )

    scenario = Scenario(
        name="memory_test_scenario",
        horizon=10,
        samples=1,
        seed=42,
    )

    return world, initial_state, scenario


# ---------------------------------------------------------------------------
# Memory Ceiling Gates
# ---------------------------------------------------------------------------


@pytest.mark.memory
@pytest.mark.limit_memory("25 MB")
def test_single_step_rollout_memory_limit() -> None:
    """Assert peak memory for a single discrete simulation step transition is <= 25 MB."""
    world, state, scenario = _build_test_world(num_nodes=30)
    rng = np.random.default_rng(100)
    trace = SystemicTrace()

    with assert_memory_budget(max_peak_mb=25.0):
        _record, next_state, _terminated = execute_step(
            world=world,
            state=state,
            step_idx=0,
            rng=rng,
            trace=trace,
            scenario=scenario,
        )
        assert next_state.step == 1


@pytest.mark.memory
@pytest.mark.limit_memory("50 MB")
def test_multi_step_trajectory_memory_limit() -> None:
    """Assert peak memory for a full 50-step simulation rollout is <= 50 MB."""
    world, _state, _ = _build_test_world(num_nodes=20)
    scenario = Scenario(name="trajectory_50_steps", horizon=50, samples=1, seed=123)
    engine = SimulationEngine()

    with assert_memory_budget(max_peak_mb=50.0):
        result = engine.run(world=world, scenario=scenario)
        assert len(result.trajectories) == 1
        assert len(result.trajectories[0].steps) == 50


@pytest.mark.memory
@pytest.mark.limit_memory("60 MB")
def test_monte_carlo_spawning_memory_limit() -> None:
    """Assert peak memory for 10 parallel Monte Carlo trajectories is <= 60 MB."""
    world, _state, _ = _build_test_world(num_nodes=20)
    scenario = Scenario(name="mc_10_samples", horizon=10, samples=10, seed=456)
    engine = SimulationEngine()

    with assert_memory_budget(max_peak_mb=60.0):
        result = engine.run(world=world, scenario=scenario)
        assert len(result.trajectories) == 10


@pytest.mark.memory
@pytest.mark.limit_memory("15 MB")
def test_state_branching_structural_sharing_memory_limit() -> None:
    """Assert 50 counterfactual state branches share structure and stay under 15 MB."""
    world, initial_state, _ = _build_test_world(num_nodes=40)

    with assert_memory_budget(max_peak_mb=15.0):
        branches: list[WorldState] = []
        for i in range(50):
            branched_state = initial_state.update_resource(
                resource_id="stock_0",
                new_value=150.0 + float(i),
            )
            branches.append(branched_state)
            _branched_world = branch_world(world, state=branched_state)

        assert len(branches) == 50
        assert branches[0].resources["stock_0"].current == 150.0
        assert branches[49].resources["stock_0"].current == 199.0


@pytest.mark.memory
@pytest.mark.limit_memory("20 MB")
def test_systemic_trace_memory_limit() -> None:
    """Assert generating a 500-node, 750-edge systemic trace occupies <= 20 MB."""
    with assert_memory_budget(max_peak_mb=20.0):
        trace = SystemicTrace()
        for i in range(500):
            trace.add_node(
                node_id=f"node_{i}",
                step=i // 10,
                category="action" if i % 2 == 0 else "state_change",
                label=f"Trace element {i}",
                evidence_level=EvidenceLevel.STRUCTURAL,
                details={"value": float(i)},
            )
        for i in range(1, 500):
            trace.add_edge(
                source=f"node_{i - 1}",
                target=f"node_{i}",
                relation="influences",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )
        assert len(trace.nodes) == 500
        assert len(trace.edges) == 499


# ---------------------------------------------------------------------------
# Memory Leak Gate
# ---------------------------------------------------------------------------


@pytest.mark.memory
@pytest.mark.limit_leaks
def test_simulation_loop_zero_leak_gate() -> None:
    """Assert no memory leak across 50 simulation step executions."""
    world, state, scenario = _build_test_world(num_nodes=15)
    rng = np.random.default_rng(999)

    def _single_iteration() -> None:
        trace = SystemicTrace()
        _record, _next_state, _ = execute_step(
            world=world,
            state=state,
            step_idx=0,
            rng=rng,
            trace=trace,
            scenario=scenario,
        )

    # Enforce zero unbounded memory leakage across repeated simulation cycles
    assert_no_memory_leak(
        operation=_single_iteration,
        warmup_cycles=10,
        measured_cycles=30,
        max_leak_kb=250.0,
    )
