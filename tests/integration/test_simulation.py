"""Integration tests for end-to-end Monte Carlo simulation rollouts."""

from __future__ import annotations

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.scenario import Scenario


def test_end_to_end_simulation_rollout(sample_world_state: WorldState) -> None:
    """Test full multi-sample rollout across composite dynamics, actors, and constraints."""
    # Build World
    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="stock_wh2", mean_demand=5.0, std_demand=1.0),
        ]
    )

    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="stock_wh1"),
            ResourceCapacityConstraint(resource_id="stock_wh2"),
            ActionTransferAvailabilityConstraint(),
        ]
    )

    actor = ThresholdReplenishmentActor(
        actor_id="warehouse_manager",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=40.0,
        order_quantity=30.0,
    )

    world = World(
        state=sample_world_state,
        dynamics=dynamics,
        constraints=constraints,
        actors=[actor],
    )

    scenario = Scenario(
        name="ProductionRun",
        horizon=10,
        samples=5,
        seed=123,
    )

    result = world.simulate(scenario=scenario)

    assert len(result.trajectories) == 5
    for traj in result.trajectories:
        assert len(traj.steps) == 10
        assert traj.final_state.step == 10
        assert traj.final_state.timestamp == 10.0
        # Trace nodes must be present
        assert len(traj.systemic_trace.nodes) > 0

    dist = result.metric_distribution("resource_stock_wh2")
    assert dist["p50"] > 0.0
    assert dist["min_val"] <= dist["max_val"]
