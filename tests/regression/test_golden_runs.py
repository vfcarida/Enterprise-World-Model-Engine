"""Regression tests ensuring deterministic stability against golden calibration baselines."""

from __future__ import annotations

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.scenario import Scenario


def test_golden_calibration_run() -> None:
    """Regression test: Fixed golden baseline produces exact expected numerical metrics."""
    state = WorldState(
        entities=[
            Entity(id="hub_a", type="hub"),
            Entity(id="hub_b", type="hub"),
        ],
        resources=[
            Resource(id="inv_a", entity_id="hub_a", current=200.0, min_value=0.0, max_value=300.0),
            Resource(id="inv_b", entity_id="hub_b", current=50.0, min_value=0.0, max_value=300.0),
        ],
    )

    world = World(
        state=state,
        dynamics=CompositeDynamics(
            [
                DeterministicTransferDynamics(),
                StochasticDemandDynamics(resource_id="inv_b", mean_demand=20.0, std_demand=0.0),
            ]
        ),
        constraints=ConstraintRegistry(
            [
                ResourceCapacityConstraint(resource_id="inv_a"),
                ResourceCapacityConstraint(resource_id="inv_b"),
            ]
        ),
        actors=[
            ThresholdReplenishmentActor(
                actor_id="hub_manager",
                source_resource="inv_a",
                target_resource="inv_b",
                reorder_point=40.0,
                order_quantity=50.0,
            )
        ],
    )

    scenario = Scenario(name="GoldenBaseline", horizon=5, samples=1, seed=42)
    result = world.simulate(scenario=scenario)

    traj = result.trajectories[0]
    final_a = traj.final_state.get_resource("inv_a").current
    final_b = traj.final_state.get_resource("inv_b").current

    # Step 0: inv_a=200, inv_b=50. Demand on b = 20 -> inv_b = 30.
    # Step 1: inv_b=30 <= 40 -> Transfer 50: a=150, b=80. Demand on b = 20 -> inv_b = 60.
    # Step 2: inv_b=60 > 40 -> Demand on b = 20 -> inv_b = 40.
    # Step 3: inv_b=40 <= 40 -> Transfer 50: a=100, b=90. Demand on b = 20 -> inv_b = 70.
    # Step 4: inv_b=70 > 40 -> Demand on b = 20 -> inv_b = 50.
    assert final_a == 100.0
    assert final_b == 50.0
    assert traj.total_violations == 0
