"""Receding-horizon online simulation and Model Predictive Control (MPC) engine.

[EXPERIMENTAL] This module is under active research and subject to breaking changes.
For production workflows, see ewm_engine.experimental.RecedingHorizonSimulator.

Implements the continuous re-grounding feedback loop:
    Observe -> Simulate Short Horizon -> Select Best Policy -> Apply Action -> Re-ground -> Repeat
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.actors.base import Actor
from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory

if TYPE_CHECKING:
    from ewm_engine.experimental.planning import (
        Planner,
        PlanningCandidate,
        PlanningDecision,
        RolloutScorer,
    )


class MPCDecisionRecord(BaseModel):
    """Audit record of a single receding-horizon decision evaluation step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(description="Simulation decision step.")
    selected_candidate: str = Field(description="Identifier or name of chosen policy.")
    candidate_scores: dict[str, float] = Field(description="Evaluated metric scores per candidate.")
    state_hash_before: str = Field(description="Observed ground-truth state hash prior to action.")
    state_hash_after: str = Field(description="Re-grounded state hash after 1-step execution.")


class RecedingHorizonResult:
    """Outcome of an online receding-horizon simulation execution."""

    def __init__(
        self,
        realized_trajectory: Trajectory,
        decision_history: list[MPCDecisionRecord],
        planning_decisions: list[PlanningDecision] | None = None,
    ) -> None:
        self.realized_trajectory = realized_trajectory
        self.decision_history = decision_history
        self.planning_decisions = planning_decisions or []

    @property
    def final_state(self) -> WorldState:
        """The final ground-truth state reached after all decision steps."""
        return self.realized_trajectory.final_state

    def to_dict(self) -> dict[str, Any]:
        """Convert decision history and realized trajectory summary to dictionary."""
        return {
            "total_steps": len(self.decision_history),
            "final_state_hash": self.final_state.state_hash,
            "decisions": [d.model_dump() for d in self.decision_history],
        }


class RecedingHorizonSimulator:
    """Model Predictive Control simulator with continuous state re-grounding.

    Rather than trusting open-loop prophecies over long horizons, this engine simulates
    short lookahead horizons for competing policy candidates, applies the best decision,
    observes the resulting world state, and repeats.
    """

    def __init__(
        self,
        lookahead_horizon: int = 3,
        samples_per_candidate: int = 5,
        objective_metric: str = "violations_count",
        minimize: bool = True,
        seed: int = 42,
        scorer: RolloutScorer | None = None,
        planner: Planner | None = None,
    ) -> None:
        self.lookahead_horizon = lookahead_horizon
        self.samples_per_candidate = samples_per_candidate
        self.objective_metric = objective_metric
        self.minimize = minimize
        self.seed = seed
        if scorer is None:
            from ewm_engine.experimental.planning import ExpectedObjectiveScorer

            self.scorer: RolloutScorer = ExpectedObjectiveScorer(
                metric=objective_metric, minimize=minimize
            )
        else:
            self.scorer = scorer

        if planner is None:
            from ewm_engine.experimental.planning import RolloutPlanner

            self.planner: Planner = RolloutPlanner()
        else:
            self.planner = planner

        self._engine = SimulationEngine()

    def run(
        self,
        world: World,
        total_steps: int,
        candidate_interventions: Sequence[Intervention] | None = None,
        candidate_actors: Sequence[Actor] | None = None,
        candidates: Sequence[PlanningCandidate] | None = None,
    ) -> RecedingHorizonResult:
        """Execute receding-horizon closed-loop simulation over total_steps."""
        from ewm_engine.experimental.planning import (
            PlanningCandidate,
            _StaticActionInjector,
        )

        active_world = world.branch()
        current_state = active_world.initial_state

        decision_history: list[MPCDecisionRecord] = []
        planning_decisions: list[PlanningDecision] = []
        realized_trajectory = Trajectory(
            sample_id=0,
            seed=self.seed,
            initial_state=current_state,
        )

        for step in range(total_steps):
            state_hash_before = current_state.state_hash

            # Assemble candidates
            planning_cands: list[PlanningCandidate] = []
            if candidates is not None:
                planning_cands = list(candidates)
            elif candidate_interventions is not None:
                planning_cands = [
                    PlanningCandidate.from_intervention(i) for i in candidate_interventions
                ]
            elif candidate_actors is not None:
                planning_cands = [PlanningCandidate.from_actor(a) for a in candidate_actors]
            else:
                planning_cands = [PlanningCandidate(id="default", name="default")]

            # Execute planning over candidate lookaheads
            decision = self.planner.plan(
                world=active_world,
                state=current_state,
                candidates=planning_cands,
                horizon=self.lookahead_horizon,
                samples=self.samples_per_candidate,
                scorer=self.scorer,
                step=step,
                seed=self.seed + step,
            )

            chosen = decision.chosen_candidate

            # Execute exactly 1 discrete step in the ground-truth world under the chosen policy
            step_world = branch_world(active_world, state=current_state)
            if chosen.actor is not None:
                step_world.add_actor(chosen.actor)
            elif chosen.actions:
                step_world.add_actor(
                    _StaticActionInjector(chosen.actions, actor_id=f"exec_{chosen.id}")
                )

            exec_scenario = Scenario(
                name="ExecuteSingleStep",
                horizon=1,
                samples=1,
                seed=self.seed + (step * 100),
                intervention=chosen.intervention,
            )
            step_res = self._engine.run(world=step_world, scenario=exec_scenario)
            step_traj = step_res.trajectories[0]

            # Extract executed step record and resulting state
            step_record = step_traj.steps[0]
            current_state = step_traj.final_state

            realized_trajectory.append_step(step_record, current_state)

            decision_record = MPCDecisionRecord(
                step=step,
                selected_candidate=chosen.id,
                candidate_scores=decision.candidate_scores,
                state_hash_before=state_hash_before,
                state_hash_after=current_state.state_hash,
            )
            decision_history.append(decision_record)
            planning_decisions.append(decision)

        return RecedingHorizonResult(
            realized_trajectory=realized_trajectory,
            decision_history=decision_history,
            planning_decisions=planning_decisions,
        )
