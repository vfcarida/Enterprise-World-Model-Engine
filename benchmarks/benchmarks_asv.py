"""Airspeed Velocity (ASV) benchmark suite for EWM Engine.

Tracks long-term performance history, change-point regressions, and bisection trends across
commit ranges. Serves as system-of-record for simulation throughput and memory footprint.
"""

from __future__ import annotations

import numpy as np

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


def _build_env(num_nodes: int = 25) -> tuple[World, WorldState, Scenario]:
    """Helper to build a deterministic multi-node simulation world."""
    entities = [
        Entity(id=f"depot_{i}", type="depot", attributes={"tier": i % 3}) for i in range(num_nodes)
    ]
    relationships = [
        Relationship(
            source=f"depot_{i}",
            target=f"depot_{(i + 1) % num_nodes}",
            type="connected_to",
            attributes={"latency": 1},
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
        memory={"step_counter": 0},
        active_rules={"safety_stock": 25.0},
        context={"mode": "standard"},
        timestamp=0.0,
        step=0,
    )

    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_0", mean_demand=4.0, std_demand=1.0),
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
        name="asv_benchmark_scenario",
        horizon=10,
        samples=1,
        seed=42,
    )

    return world, initial_state, scenario


class TimeRolloutStep:
    """Benchmark discrete simulation step transitions."""

    def setup(self) -> None:
        self.world, self.state, self.scenario = _build_env(num_nodes=25)
        self.rng = np.random.default_rng(42)
        self.trace = SystemicTrace()

    def time_single_step(self) -> None:
        execute_step(
            world=self.world,
            state=self.state,
            step_idx=0,
            rng=self.rng,
            trace=self.trace,
            scenario=self.scenario,
        )


class TimeMonteCarloSpawn:
    """Benchmark parallel stream splitting and state cloning for Monte Carlo rollouts."""

    def setup(self) -> None:
        _, self.initial_state, _ = _build_env(num_nodes=25)
        self.master_seed = 999999
        self.n_workers = 25

    def time_seed_sequence_spawn(self) -> None:
        seed_seq = np.random.SeedSequence(self.master_seed)
        child_sequences = seed_seq.spawn(self.n_workers)
        streams = [
            (np.random.default_rng(child), self.initial_state.model_copy(deep=True))
            for child in child_sequences
        ]
        assert len(streams) == self.n_workers


class TimeCanonicalFingerprint:
    """Benchmark canonical state serialization and SHA-256 fingerprinting."""

    def setup(self) -> None:
        _, self.state, _ = _build_env(num_nodes=35)

    def time_state_fingerprint(self) -> None:
        fp = self.state.fingerprint
        assert len(fp) == 64


class TimeConstraintValidation:
    """Benchmark action validation and constraint predicate checks."""

    def setup(self) -> None:
        self.world, self.state, _ = _build_env(num_nodes=25)
        self.actions = [
            Action(
                id=f"act_{i}",
                type="transfer_resource",
                parameters={
                    "from_resource_id": f"stock_{i}",
                    "to_resource_id": f"stock_{(i + 1) % 25}",
                    "amount": 10.0,
                },
            )
            for i in range(15)
        ]

    def time_validate_actions(self) -> None:
        accepted, _ = self.world.constraints.validate_actions(
            state=self.state, actions=self.actions
        )
        assert len(accepted) >= 0


class TimeStateSerialization:
    """Benchmark state dictionary serialization and round-trip deserialization."""

    def setup(self) -> None:
        _, self.state, _ = _build_env(num_nodes=25)
        self.state_dict = self.state.to_dict()

    def time_to_dict(self) -> None:
        _ = self.state.to_dict()

    def time_from_dict(self) -> None:
        _ = WorldState.from_dict(self.state_dict)


class TimeSystemicTrace:
    """Benchmark building systemic causal/structural DAG traces."""

    def time_build_trace_50_nodes(self) -> None:
        trace = SystemicTrace()
        for i in range(50):
            trace.add_node(
                node_id=f"node_{i}",
                step=i // 5,
                category="action" if i % 2 == 0 else "state_change",
                label=f"Trace element {i}",
                evidence_level=EvidenceLevel.STRUCTURAL,
                details={"val": float(i)},
            )
        for i in range(1, 50):
            trace.add_edge(
                source=f"node_{i - 1}",
                target=f"node_{i}",
                relation="influences",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )
        assert len(trace.nodes) == 50


class PeakMemoryWorld:
    """Measure peak memory footprint of simulation rollouts."""

    def setup(self) -> None:
        self.world, self.state, self.scenario = _build_env(num_nodes=30)
        self.rng = np.random.default_rng(42)
        self.trace = SystemicTrace()

    def peakmem_rollout_single_step(self) -> None:
        execute_step(
            world=self.world,
            state=self.state,
            step_idx=0,
            rng=self.rng,
            trace=self.trace,
            scenario=self.scenario,
        )
