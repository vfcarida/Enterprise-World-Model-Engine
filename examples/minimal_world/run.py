"""Minimal World Example for Enterprise World Model Engine (EWM Engine).

Demonstrates:
  - WorldState with two warehouses and finite inventory
  - Capacity and stock availability constraints
  - Composite dynamics (transfer flow + stochastic customer demand)
  - Counterfactual scenario branching (Status Quo vs Proactive Transfer Policy)
  - Comparative delta evaluation and distribution quantiles
"""

from __future__ import annotations

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
)
from ewm_engine.core.actions import Intervention
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.evaluation.comparison import ScenarioComparison, compare_scenarios
from ewm_engine.simulation.scenario import Scenario


def run_minimal_world() -> ScenarioComparison:
    # 1. Define initial state S_0
    state = WorldState(
        entities=[
            Entity(id="wh_north", type="warehouse", attributes={"zone": "North"}),
            Entity(id="wh_south", type="warehouse", attributes={"zone": "South"}),
        ],
        relationships=[
            Relationship(
                source="wh_north",
                target="wh_south",
                type="connected_to",
                attributes={"distance_km": 40.0},
            ),
        ],
        resources=[
            Resource(
                id="stock_north",
                entity_id="wh_north",
                current=150.0,
                min_value=0.0,
                max_value=250.0,
                unit="units",
            ),
            Resource(
                id="stock_south",
                entity_id="wh_south",
                current=35.0,
                min_value=0.0,
                max_value=200.0,
                unit="units",
            ),
        ],
    )

    # 2. Register operational constraints and system dynamics
    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="stock_north"),
            ResourceCapacityConstraint(resource_id="stock_south"),
            ResourceNonNegativeConstraint(resource_id="stock_south"),
            ActionTransferAvailabilityConstraint(),
        ]
    )

    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_south", mean_demand=18.0, std_demand=3.0),
        ]
    )

    base_world = World(state=state, dynamics=dynamics, constraints=constraints)

    # 3. Simulate Baseline (Status Quo: No transfer policy)
    baseline_scenario = Scenario(name="Status Quo", horizon=10, samples=20, seed=42)
    result_baseline = base_world.simulate(scenario=baseline_scenario)

    # 4. Branch and Simulate Counterfactual Intervention (Automated Threshold Replenishment)
    proactive_intervention = Intervention(
        id="proactive_replenishment",
        description="Trigger 40-unit transfers when South stock drops <= 40",
        parameters={"reorder_threshold": 40.0, "transfer_qty": 40.0},
    )
    actor = ThresholdReplenishmentActor(
        actor_id="inventory_controller",
        source_resource="stock_north",
        target_resource="stock_south",
        reorder_point=40.0,
        order_quantity=40.0,
    )

    proactive_world = base_world.branch()
    proactive_world.add_actor(actor)

    proactive_scenario = Scenario(
        name="Proactive Policy",
        horizon=10,
        samples=20,
        seed=42,
        intervention=proactive_intervention,
    )
    result_proactive = proactive_world.simulate(scenario=proactive_scenario)

    # 5. Evaluate and Compare Counterfactual Outcomes
    comparison = compare_scenarios(
        baseline=result_baseline,
        candidates=[result_proactive],
        metrics=["resource_stock_south", "resource_stock_north", "violations_count"],
    )

    print("\n" + comparison.summary_table() + "\n")

    # 6. Inspect Sample Systemic Trace
    sample_trace = result_proactive.trajectories[0].systemic_trace
    print("=== Systemic Trace (Sample Trajectory 0) ===")
    print(sample_trace.to_mermaid())
    return comparison


if __name__ == "__main__":
    run_minimal_world()
