"""Integration tests for AC-008: Hard post-transition constraint violation and rollout invalidation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ewm_engine import DynamicsModel, EvidenceLevel, TransitionResult
from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import ConstraintPhase, ConstraintSeverity
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import TrajectoryStatus


class UnconstrainedTransferDynamics(DynamicsModel):
    """Dynamics model that applies full transfer quantity directly without artificial clamping.

    Represents realistic systems where overflowing shipments actually arrive at a facility
    and cause operational breaches unless checked by pre-action gates.
    """

    @property
    def name(self) -> str:
        return "UnconstrainedTransferDynamics"

    @property
    def component_id(self) -> str:
        return "unconstrained_transfer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: np.random.Generator,
    ) -> TransitionResult:
        current_state = state
        for act in actions:
            if act.type == "transfer_resource":
                src = str(act.get("source_resource"))
                tgt = str(act.get("target_resource"))
                qty = float(act.get("quantity", 0.0))
                current_state = current_state.update_resource(src, delta=-qty, clamp=False)
                current_state = current_state.update_resource(tgt, delta=qty, clamp=False)

        return TransitionResult(
            next_state=current_state,
            applied_changes={"transfers_executed": len(actions)},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class OverflowTransferActor(Actor):
    """Proposes a transfer that fits in source stock but causes destination warehouse overflow."""

    def __init__(self, transfer_qty: float) -> None:
        self.transfer_qty = transfer_qty

    @property
    def actor_id(self) -> str:
        return "overflow_transfer_actor"

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        if context.step == 0:
            return [
                Action(
                    id="act_overflow_transfer",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_wh1",
                        "target_resource": "stock_wh2",
                        "quantity": self.transfer_qty,
                    },
                )
            ]
        return []


@pytest.mark.integration
def test_hard_post_transition_violation_invalidates_rollout_without_silent_repair(
    sample_world_state: WorldState,
) -> None:
    """AC-008: A HARD constraint violation evaluated POST_TRANSITION must:

    1. Set the rollout trajectory status to TrajectoryStatus.INVALID.
    2. Abort the rollout immediately (stop horizon loop at that step).
    3. NOT silently project, clamp, or fabricate a valid state: the observed final_state
       must accurately reflect the violating unprojected state.
    """
    # Configure stock_wh2 to have capacity 100.0 (initial is 50.0)
    # stock_wh1 has 100.0 (capacity 200.0)
    # Transferring 70.0 units is valid for source (100 - 70 = 30 >= 0),
    # but causes destination stock_wh2 to reach 50 + 70 = 120.0 > max 100.0!
    wh2_capped = sample_world_state.get_resource("stock_wh2").model_copy(
        update={"max_value": 100.0}
    )
    initial_state = sample_world_state.model_copy(
        update={"resources": {**sample_world_state.resources, "stock_wh2": wh2_capped}}
    )

    actor = OverflowTransferActor(transfer_qty=70.0)

    hard_capacity_constraint = ResourceCapacityConstraint(
        resource_id="stock_wh2",
        severity=ConstraintSeverity.HARD,
        constraint_id="hard_capacity_wh2",
    )

    world = World(
        state=initial_state,
        dynamics=UnconstrainedTransferDynamics(),
        constraints=ConstraintRegistry([hard_capacity_constraint]),
        actors=[actor],
    )

    requested_horizon = 5
    scenario = Scenario(
        scenario_id="scen_post_hard_invalidation",
        name="Post-Transition Hard Invalidation",
        horizon=requested_horizon,
        samples=1,
        seed=777,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 1
    traj = result.trajectories[0]

    # 1. Trajectory must be marked INVALID
    assert traj.status == TrajectoryStatus.INVALID
    assert traj.is_finalized

    # 2. Rollout must have stopped at step 0 (terminated early, did NOT run 5 steps)
    assert len(traj.steps) == 1
    assert len(traj.steps) < requested_horizon

    # 3. No silent state repair: destination must be 120.0, not clamped to 100.0
    violating_wh2 = traj.final_state.get_resource("stock_wh2")
    assert violating_wh2.current == 120.0
    assert violating_wh2.current > violating_wh2.max_value

    # Source stock was transferred
    assert traj.final_state.get_resource("stock_wh1").current == 30.0

    # 4. Check step record and violations
    step_0 = traj.steps[0]
    hard_violations = [
        v for v in step_0.constraint_violations if v.severity == ConstraintSeverity.HARD
    ]
    assert len(hard_violations) == 1
    post_viol = hard_violations[0]
    assert post_viol.constraint_id == "hard_capacity_wh2"
    assert post_viol.phase == ConstraintPhase.POST_TRANSITION
    assert not post_viol.satisfied
    assert post_viol.violating_resources == ("stock_wh2",)

    # 5. SimulationResult metadata accounting
    assert result.invalid_count == 1
    assert result.completed_count == 0
    assert result.violation_rate() == 1.0


@pytest.mark.integration
def test_mixed_completed_and_invalid_trajectories() -> None:
    """AC-008 / AC-009: Multi-sample simulation correctly isolates and aggregates

    mixed COMPLETED and INVALID trajectories without corrupting metrics.
    """

    class StochasticOverflowDynamics(DynamicsModel):
        """Dynamics that overflows stock_wh1 only if random draw exceeds threshold."""

        @property
        def name(self) -> str:
            return "StochasticOverflowDynamics"

        @property
        def component_id(self) -> str:
            return "stochastic_overflow"

        @property
        def version(self) -> str:
            return "1.0.0"

        def transition(
            self,
            state: WorldState,
            actions: Sequence[Action],
            exogenous_events: Sequence[ExogenousEvent],
            rng: np.random.Generator,
        ) -> TransitionResult:
            # 50% chance of overflow
            overflow = rng.random() > 0.5
            new_val = 250.0 if overflow else 150.0
            next_state = state.update_resource("stock_wh1", new_value=new_val, clamp=False)
            return TransitionResult(
                next_state=next_state,
                applied_changes={"stock_wh1": new_val},
                evidence_level=EvidenceLevel.STRUCTURAL,
                model_name=self.name,
            )

    # stock_wh1 max is 200.0. If new_val is 250.0, it breaches HARD capacity.
    from ewm_engine.core.resources import Resource

    res = Resource(
        id="stock_wh1",
        current=100.0,
        min_value=0.0,
        max_value=200.0,
        unit="units",
    )
    initial_state = WorldState(resources=[res])

    world = World(
        state=initial_state,
        dynamics=StochasticOverflowDynamics(),
        constraints=ConstraintRegistry(
            [
                ResourceCapacityConstraint(
                    resource_id="stock_wh1",
                    severity=ConstraintSeverity.HARD,
                )
            ]
        ),
    )

    scenario = Scenario(
        scenario_id="scen_stochastic_invalidation",
        name="Stochastic Invalidation Mix",
        horizon=4,
        samples=10,
        seed=42,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 10
    total_rollouts = result.completed_count + result.invalid_count
    assert total_rollouts == 10
    # Both completed and invalid trajectories must exist under this seed
    assert result.completed_count > 0
    assert result.invalid_count > 0

    for traj in result.trajectories:
        if traj.status == TrajectoryStatus.INVALID:
            assert len(traj.steps) < 4
            assert traj.final_state.get_resource("stock_wh1").current == 250.0
        else:
            assert len(traj.steps) == 4
            assert traj.final_state.get_resource("stock_wh1").current == 150.0
