"""Unit tests for RecedingHorizonSimulator (MPC) and online re-grounding."""

from __future__ import annotations

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.mpc import RecedingHorizonSimulator


def test_receding_horizon_controller_intervention_selection(
    sample_world_state: WorldState,
) -> None:
    """Verify that MPC evaluates interventions and re-grounds state step-by-step."""
    world = World(
        state=sample_world_state,
        dynamics=DeterministicTransferDynamics(),
        constraints=ConstraintRegistry([ActionTransferAvailabilityConstraint()]),
    )

    # Candidate A: Small replenishment (20 units)
    policy_a = Intervention(id="policy_small", description="Small transfer", parameters={"qty": 20.0})
    # Candidate B: Large replenishment (50 units)
    policy_b = Intervention(id="policy_large", description="Large transfer", parameters={"qty": 50.0})

    sim = RecedingHorizonSimulator(
        lookahead_horizon=2,
        samples_per_candidate=2,
        objective_metric="violations_count",
        minimize=True,
        seed=100,
    )

    result = sim.run(
        world=world,
        total_steps=3,
        candidate_interventions=[policy_a, policy_b],
    )

    assert len(result.decision_history) == 3
    assert len(result.realized_trajectory.steps) == 3
    assert result.final_state.step == 3

    # Inspect decision audit trail
    for decision in result.decision_history:
        assert decision.selected_candidate in ("policy_small", "policy_large")
        assert "policy_small" in decision.candidate_scores
        assert "policy_large" in decision.candidate_scores
        assert len(decision.state_hash_before) == 64
        assert len(decision.state_hash_after) == 64


def test_receding_horizon_controller_actor_selection(sample_world_state: WorldState) -> None:
    """Verify MPC with competing candidate actors."""
    world = World(
        state=sample_world_state,
        dynamics=DeterministicTransferDynamics(),
        constraints=ConstraintRegistry([ActionTransferAvailabilityConstraint()]),
    )

    actor_conservative = ThresholdReplenishmentActor(
        actor_id="actor_cons",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=40.0,
        order_quantity=10.0,
    )
    actor_aggressive = ThresholdReplenishmentActor(
        actor_id="actor_aggr",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=80.0,
        order_quantity=30.0,
    )

    sim = RecedingHorizonSimulator(
        lookahead_horizon=2,
        samples_per_candidate=2,
        objective_metric="violations_count",
        minimize=True,
        seed=101,
    )

    result = sim.run(
        world=world,
        total_steps=2,
        candidate_actors=[actor_conservative, actor_aggressive],
    )

    assert len(result.decision_history) == 2
    summary = result.to_dict()
    assert summary["total_steps"] == 2
    assert len(summary["decisions"]) == 2
