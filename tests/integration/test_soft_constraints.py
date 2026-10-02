"""Integration tests for AC-007: Soft constraint recording and rollout continuation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import TrajectoryStatus


class SoftBatchSizeConstraint(Constraint):
    """Soft pre-action guideline: warns if single transfer exceeds recommended batch size of 20."""

    @property
    def constraint_id(self) -> ConstraintId:
        return "soft_batch_size_limit"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.SOFT

    @property
    def description(self) -> str:
        return "Soft guideline preferring transfer batches <= 20 units."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.PRE_ACTION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        for act in actions:
            if act.type == "transfer_resource":
                qty = float(act.get("quantity", 0.0))
                if qty > 20.0:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        severity=self.severity,
                        phase=phase,
                        message=f"Transfer batch {qty} exceeds recommended batch size 20.0.",
                        penalty=qty - 20.0,
                        preceding_action_id=act.id,
                    )

        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


class SoftThresholdCapacityConstraint(Constraint):
    """Soft post-transition guideline: warns if stock_wh2 exceeds operational threshold of 70."""

    @property
    def constraint_id(self) -> ConstraintId:
        return "soft_wh2_threshold"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.SOFT

    @property
    def description(self) -> str:
        return "Soft operational guideline: preferred stock_wh2 <= 70 units."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.POST_TRANSITION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        res = state.get_resource("stock_wh2")
        threshold = 70.0
        satisfied = res.current <= threshold
        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
            message="Within operational threshold."
            if satisfied
            else f"stock_wh2 level {res.current} exceeds preferred threshold {threshold}.",
            penalty=max(0.0, res.current - threshold),
            violating_resources=("stock_wh2",) if not satisfied else (),
        )


class LargeBatchActor(Actor):
    """Proposes a 30-unit transfer (breaches soft batch size 20, but within stock 100)."""

    @property
    def actor_id(self) -> str:
        return "large_batch_actor"

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        if context.step == 0:
            return [
                Action(
                    id="act_large_batch",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_wh1",
                        "target_resource": "stock_wh2",
                        "quantity": 30.0,
                    },
                )
            ]
        return []


@pytest.mark.integration
def test_soft_constraints_record_violations_and_continue(
    sample_world_state: WorldState,
) -> None:
    """AC-007: Soft constraints (pre-action and post-transition) must record violations

    without rejecting actions or invalidating trajectories. Rollout must complete
    all horizon steps.
    """
    actor = LargeBatchActor()

    # Configure a soft pre-action constraint (batch size) and a soft post-transition guideline (threshold 70)
    # stock_wh1 has 100, stock_wh2 has 50 (capacity 100).
    # Transfer 30 makes wh1=70, wh2=80 (fits within max_value 100, but exceeds soft threshold 70).
    registry = ConstraintRegistry(
        [
            SoftBatchSizeConstraint(),
            SoftThresholdCapacityConstraint(),
        ]
    )

    world = World(
        state=sample_world_state,
        dynamics=DeterministicTransferDynamics(),
        constraints=registry,
        actors=[actor],
    )

    horizon_steps = 3
    scenario = Scenario(
        scenario_id="scen_soft_constraints",
        name="Soft Constraints Continuation",
        horizon=horizon_steps,
        samples=1,
        seed=101,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    assert len(result.trajectories) == 1
    traj = result.trajectories[0]

    # Rollout must complete the entire horizon without early termination
    assert traj.status == TrajectoryStatus.COMPLETED
    assert len(traj.steps) == horizon_steps
    assert traj.final_state.step == horizon_steps

    step_0 = traj.steps[0]
    # Soft pre-action violation did NOT reject the action
    assert len(step_0.actions_accepted) == 1
    assert step_0.actions_accepted[0].id == "act_large_batch"

    # Action was applied by dynamics
    assert traj.final_state.get_resource("stock_wh1").current == 70.0
    assert traj.final_state.get_resource("stock_wh2").current == 80.0

    # Both soft violations (pre-action batch size and post-transition capacity) were recorded
    violations = step_0.constraint_violations
    assert len(violations) >= 2
    assert all(v.severity == ConstraintSeverity.SOFT for v in violations)

    # Penalties must be tracked in step metrics
    assert step_0.step_metrics["violations_penalty"] > 0.0
    assert step_0.step_metrics["violations_count"] >= 2.0
    assert step_0.step_metrics["hard_violations_count"] == 0.0
