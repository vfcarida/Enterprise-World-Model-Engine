"""Curated performance benchmarks for EWM Engine.

Measures CPU-instruction cost (via CodSpeed instrumentation mode on CI)
and wall-clock execution characteristics (via pytest-benchmark).

Curated Performance-Critical Subset (R04 Task A):
1. rollout step (discrete state transition + constraints + dynamics)
2. Monte Carlo spawn (parallel seeded stream splitting + state replication)
3. fingerprint/canonical hash (cryptographic SHA-256 state hashing)
4. constraint evaluation (action validation + resource boundary predicates)
5. serialization round-trip (JSON dict export and Pydantic model reconstruction)
6. trace build (causal/structural DAG node & edge recording)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.rollout import execute_step
from ewm_engine.simulation.scenario import Scenario


def _create_benchmark_environment(num_nodes: int = 30) -> tuple[World, WorldState, Scenario]:
    """Build a deterministic multi-node supply-chain enterprise world."""
    entities = [
        Entity(id=f"depot_{i}", type="warehouse", attributes={"tier": i % 3})
        for i in range(num_nodes)
    ]
    relationships = [
        Relationship(
            source=f"depot_{i}",
            target=f"depot_{(i + 1) % num_nodes}",
            type="connected_to",
            attributes={"latency_days": 1},
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
        memory={"cycle_count": 0},
        active_rules={"safety_stock": 25.0},
        context={"operating_mode": "nominal"},
        timestamp=0.0,
        step=0,
    )

    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_0", mean_demand=5.0, std_demand=1.0),
        ]
    )

    constraints = ConstraintRegistry(
        [ResourceCapacityConstraint(resource_id=f"stock_{i}") for i in range(min(num_nodes, 10))]
        + [ActionTransferAvailabilityConstraint()]
    )

    actors = [
        ThresholdReplenishmentActor(
            actor_id=f"agent_{i}",
            source_resource=f"stock_{(i + 1) % num_nodes}",
            target_resource=f"stock_{i}",
            reorder_point=40.0,
            order_quantity=20.0,
        )
        for i in range(min(num_nodes, 5))
    ]

    world = World(
        initial_state=initial_state,
        dynamics=dynamics,
        constraints=constraints,
        actors=actors,
    )

    scenario = Scenario(
        name="curated_benchmark_scenario",
        horizon=10,
        samples=5,
        seed=42,
    )

    return world, initial_state, scenario


# ---------------------------------------------------------------------------
# 1. Rollout Step Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_rollout_step(benchmark: Any) -> None:
    """Benchmark a single discrete simulation step transition (dynamics + constraints)."""
    world, state, scenario = _create_benchmark_environment(num_nodes=20)
    rng = np.random.default_rng(12345)
    trace = SystemicTrace()

    def _step_operation() -> WorldState:
        # Step index 0
        _, next_state, _ = execute_step(
            world=world,
            state=state,
            step_idx=0,
            rng=rng,
            trace=trace,
            scenario=scenario,
        )
        return next_state

    result_state = benchmark(_step_operation)
    assert result_state.step == 1
    assert result_state.timestamp > 0.0


# ---------------------------------------------------------------------------
# 2. Monte Carlo Parallel Spawn Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_monte_carlo_spawn(benchmark: Any) -> None:
    """Benchmark seeded parallel stream splitting and worker state replication."""
    _, initial_state, _ = _create_benchmark_environment(num_nodes=30)
    master_seed = 987654321
    num_replicas = 20

    def _spawn_operation() -> list[tuple[np.random.Generator, WorldState]]:
        # High-discipline SeedSequence spawning (AC-004 / R04 policy: never seed + i)
        seed_seq = np.random.SeedSequence(master_seed)
        child_sequences = seed_seq.spawn(num_replicas)
        streams: list[tuple[np.random.Generator, WorldState]] = []
        for child in child_sequences:
            worker_rng = np.random.default_rng(child)
            cloned_state = initial_state.model_copy(deep=True)
            streams.append((worker_rng, cloned_state))
        return streams

    spawned = benchmark(_spawn_operation)
    assert len(spawned) == num_replicas
    assert spawned[0][1].fingerprint == initial_state.fingerprint


# ---------------------------------------------------------------------------
# 3. Fingerprint / Canonical SHA-256 Hash Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_canonical_fingerprint(benchmark: Any) -> None:
    """Benchmark canonical normalization and cryptographic SHA-256 state fingerprinting."""
    _, state, _ = _create_benchmark_environment(num_nodes=40)

    def _hash_operation() -> str:
        return state.fingerprint

    fp = benchmark(_hash_operation)
    assert isinstance(fp, str)
    assert len(fp) == 64  # SHA-256 hex string


# ---------------------------------------------------------------------------
# 4. Constraint Evaluation Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_constraint_evaluation(benchmark: Any) -> None:
    """Benchmark action validation and invariant checks across registered constraints."""
    world, state, _ = _create_benchmark_environment(num_nodes=30)
    actions = [
        Action(
            id=f"act_{i}",
            type="transfer_resource",
            parameters={
                "from_resource_id": f"stock_{i}",
                "to_resource_id": f"stock_{(i + 1) % 30}",
                "amount": 10.0,
            },
        )
        for i in range(15)
    ]

    def _constraint_eval_operation() -> int:
        accepted, _results = world.constraints.validate_actions(state=state, actions=actions)
        return len(accepted)

    accepted_count = benchmark(_constraint_eval_operation)
    assert accepted_count >= 0


# ---------------------------------------------------------------------------
# 5. Serialization Round-Trip Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_serialization_roundtrip(benchmark: Any) -> None:
    """Benchmark state dictionary serialization and Pydantic model validation round-trip."""
    _, state, _ = _create_benchmark_environment(num_nodes=30)

    def _roundtrip_operation() -> WorldState:
        data = state.to_dict()
        reconstructed = WorldState.from_dict(data)
        return reconstructed

    restored = benchmark(_roundtrip_operation)
    assert restored.fingerprint == state.fingerprint
    assert len(restored.resources) == len(state.resources)


# ---------------------------------------------------------------------------
# 6. Systemic Trace Build Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.codspeed
def test_benchmark_systemic_trace_build(benchmark: Any) -> None:
    """Benchmark systemic causal/structural DAG construction and edge linking."""

    def _build_trace_operation() -> SystemicTrace:
        trace = SystemicTrace()
        num_nodes = 50
        # Add nodes
        for i in range(num_nodes):
            trace.add_node(
                node_id=f"node_{i}",
                step=i // 5,
                category="action" if i % 2 == 0 else "state_change",
                label=f"Trace element {i}",
                evidence_level=EvidenceLevel.STRUCTURAL,
                details={"metric_val": float(i)},
            )
        # Add dependency edges
        for i in range(1, num_nodes):
            trace.add_edge(
                source=f"node_{i - 1}",
                target=f"node_{i}",
                relation="influences",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )
        return trace

    trace_built = benchmark(_build_trace_operation)
    assert len(trace_built.nodes) == 50
    assert len(trace_built.edges) == 49
