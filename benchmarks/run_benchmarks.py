"""Simulation throughput and scale benchmarks for EWM Engine."""

from __future__ import annotations

import time

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
from ewm_engine.simulation.scenario import Scenario


def benchmark_simulation_throughput(
    num_nodes: int = 50, horizon: int = 20, samples: int = 20
) -> dict[str, float]:
    """Measures simulation rollout throughput (steps/sec and samples/sec)."""
    entities = [Entity(id=f"node_{i}", type="depot") for i in range(num_nodes)]
    relationships = [
        Relationship(source=f"node_{i}", target=f"node_{(i + 1) % num_nodes}", type="connected")
        for i in range(num_nodes)
    ]
    resources = [
        Resource(
            id=f"stock_{i}", entity_id=f"node_{i}", current=100.0, min_value=0.0, max_value=250.0
        )
        for i in range(num_nodes)
    ]

    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_0", mean_demand=5.0, std_demand=1.0),
        ]
    )
    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="stock_0"),
            ActionTransferAvailabilityConstraint(),
        ]
    )
    actor = ThresholdReplenishmentActor(
        actor_id="agent_0",
        source_resource="stock_1",
        target_resource="stock_0",
        reorder_point=40.0,
        order_quantity=20.0,
    )

    world = World(state=state, dynamics=dynamics, constraints=constraints, actors=[actor])
    scenario = Scenario(name="ScaleBenchmark", horizon=horizon, samples=samples, seed=42)

    start_t = time.perf_counter()
    result = world.simulate(scenario=scenario)
    elapsed = time.perf_counter() - start_t

    total_steps = horizon * samples
    steps_per_sec = total_steps / elapsed if elapsed > 0 else float("inf")

    return {
        "elapsed_seconds": elapsed,
        "total_rollout_steps": float(total_steps),
        "steps_per_second": steps_per_sec,
        "samples_per_second": samples / elapsed if elapsed > 0 else float("inf"),
        "total_trajectories": float(len(result.trajectories)),
    }


def main() -> None:
    print("=" * 60)
    print("EWM Engine: Scalability & Throughput Benchmark")
    print("=" * 60)
    results = benchmark_simulation_throughput(num_nodes=30, horizon=25, samples=20)
    print(f"Total Rollout Steps: {results['total_rollout_steps']:.0f}")
    print(f"Elapsed Time:        {results['elapsed_seconds']:.3f} s")
    print(
        f"Throughput:          {results['steps_per_second']:.1f} steps/s ({results['samples_per_second']:.1f} rollouts/s)"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
