"""Integration tests for AC-006: Hard constraint pre-action rejection."""

from __future__ import annotations

import pytest

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import ConstraintPhase, ConstraintSeverity
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import TrajectoryStatus


class ExcessiveTransferActor(Actor):
    """Actor that proposes an action breaching available inventory."""

    def __init__(self, requested_qty: float) -> None:
        self.requested_qty = requested_qty

    @property
    def actor_id(self) -> str:
        return "excessive_transfer_actor"

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        # Propose a transfer on step 0
        if context.step == 0:
            return [
                Action(
                    id="act_excessive_001",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_wh1",
                        "target_resource": "stock_wh2",
                        "quantity": self.requested_qty,
                    },
                )
            ]
        return []


@pytest.mark.integration
def test_hard_pre_action_constraint_rejects_invalid_action(
    sample_world_state: WorldState,
) -> None:
    """AC-006: HARD + PRE_ACTION violation must reject the action before transition.

    The action must NOT be passed to dynamics, inventory must remain intact,
    and the rejection must be recorded in the systemic trace.
    """
    initial_stock_wh1 = sample_world_state.get_resource("stock_wh1").current  # 100.0
    initial_stock_wh2 = sample_world_state.get_resource("stock_wh2").current  # 50.0

    # Actor attempts to transfer 250 units, exceeding available stock of 100.0
    actor = ExcessiveTransferActor(requested_qty=250.0)

    world = World(
        state=sample_world_state,
        dynamics=DeterministicTransferDynamics(),
        constraints=ConstraintRegistry(
            [
                ActionTransferAvailabilityConstraint(
                    severity=ConstraintSeverity.HARD,
                )
            ]
        ),
        actors=[actor],
    )

    scenario = Scenario(
        scenario_id="scen_pre_action_rejection",
        name="Pre-action Hard Constraint Rejection",
        horizon=2,
        samples=1,
        seed=42,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 1
    traj = result.trajectories[0]

    # Trajectory status must remain COMPLETED because pre-action hard violations filter actions
    # rather than invalidating post-transition state
    assert traj.status == TrajectoryStatus.COMPLETED
    assert len(traj.steps) == 2

    # Step 0 inspection
    step_0 = traj.steps[0]
    assert len(step_0.actions_proposed) == 1
    assert step_0.actions_proposed[0].id == "act_excessive_001"
    # Action was rejected, so actions_accepted must be empty
    assert len(step_0.actions_accepted) == 0

    # Invariant: stock was NOT deducted or credited
    assert traj.final_state.get_resource("stock_wh1").current == initial_stock_wh1
    assert traj.final_state.get_resource("stock_wh2").current == initial_stock_wh2

    # Verify constraint violation was captured with PRE_ACTION phase
    assert len(step_0.constraint_violations) == 1
    viol = step_0.constraint_violations[0]
    assert viol.severity == ConstraintSeverity.HARD
    assert viol.phase == ConstraintPhase.PRE_ACTION
    assert not viol.satisfied
    assert viol.preceding_action_id == "act_excessive_001"

    # Verify systemic trace contains the rejected action node and rejection link
    trace_nodes = traj.systemic_trace.nodes
    assert "action_act_excessive_001_step_0" in trace_nodes
    assert "viol_transfer_stock_available_step_0" in trace_nodes
    rejected_edge = any(
        e.source == "viol_transfer_stock_available_step_0"
        and e.target == "action_act_excessive_001_step_0"
        and e.relation == "rejects"
        for e in traj.systemic_trace.edges
    )
    assert rejected_edge
