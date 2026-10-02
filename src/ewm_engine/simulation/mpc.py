"""Receding-horizon online simulation and Model Predictive Control (MPC) engine.

Implements the continuous re-grounding feedback loop:
    Observe -> Simulate Short Horizon -> Select Best Policy -> Apply Action -> Re-ground -> Repeat
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.actors.base import Actor
from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory


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
    ) -> None:
        self.realized_trajectory = realized_trajectory
        self.decision_history = decision_history

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
    ) -> None:
        self.lookahead_horizon = lookahead_horizon
        self.samples_per_candidate = samples_per_candidate
        self.objective_metric = objective_metric
        self.minimize = minimize
        self.seed = seed
        self._engine = SimulationEngine()

    def run(
        self,
        world: World,
        total_steps: int,
        candidate_interventions: Sequence[Intervention] | None = None,
        candidate_actors: Sequence[Actor] | None = None,
    ) -> RecedingHorizonResult:
        """Execute receding-horizon closed-loop simulation over total_steps."""
        active_world = world.branch()
        current_state = active_world.initial_state

        decision_history: list[MPCDecisionRecord] = []
        realized_trajectory = Trajectory(
            sample_id=0,
            seed=self.seed,
            initial_state=current_state,
        )

        for step in range(total_steps):
            state_hash_before = current_state.state_hash
            candidate_scores: dict[str, float] = {}

            # Evaluate Candidate Interventions
            best_candidate_name = "default"
            best_score = float("inf") if self.minimize else float("-inf")
            chosen_intervention: Intervention | None = None
            chosen_actor: Actor | None = None

            if candidate_interventions:
                for interv in candidate_interventions:
                    sim_world = branch_world(active_world, state=current_state)
                    sim_scenario = Scenario(
                        name=f"Lookahead_{interv.id}",
                        horizon=self.lookahead_horizon,
                        samples=self.samples_per_candidate,
                        seed=self.seed + step,
                        intervention=interv,
                    )
                    sim_res = self._engine.run(world=sim_world, scenario=sim_scenario)
                    score = sim_res.metric_distribution(self.objective_metric)["mean"]
                    candidate_scores[interv.id] = score

                    is_better = (score < best_score) if self.minimize else (score > best_score)
                    if is_better:
                        best_score = score
                        best_candidate_name = interv.id
                        chosen_intervention = interv

            elif candidate_actors:
                for actor in candidate_actors:
                    sim_world = branch_world(active_world, state=current_state)
                    sim_world.add_actor(actor)
                    sim_scenario = Scenario(
                        name=f"Lookahead_{actor.actor_id}",
                        horizon=self.lookahead_horizon,
                        samples=self.samples_per_candidate,
                        seed=self.seed + step,
                    )
                    sim_res = self._engine.run(world=sim_world, scenario=sim_scenario)
                    score = sim_res.metric_distribution(self.objective_metric)["mean"]
                    candidate_scores[actor.actor_id] = score

                    is_better = (score < best_score) if self.minimize else (score > best_score)
                    if is_better:
                        best_score = score
                        best_candidate_name = actor.actor_id
                        chosen_actor = actor
            else:
                candidate_scores["default"] = 0.0

            # Execute exactly 1 discrete step in the ground-truth world under the chosen policy
            step_world = branch_world(active_world, state=current_state)
            if chosen_actor is not None:
                step_world.add_actor(chosen_actor)

            exec_scenario = Scenario(
                name="ExecuteSingleStep",
                horizon=1,
                samples=1,
                seed=self.seed + (step * 100),
                intervention=chosen_intervention,
            )
            step_res = self._engine.run(world=step_world, scenario=exec_scenario)
            step_traj = step_res.trajectories[0]

            # Extract executed step record and resulting state
            step_record = step_traj.steps[0]
            current_state = step_traj.final_state

            realized_trajectory.append_step(step_record, current_state)

            decision_history.append(
                MPCDecisionRecord(
                    step=step,
                    selected_candidate=best_candidate_name,
                    candidate_scores=candidate_scores,
                    state_hash_before=state_hash_before,
                    state_hash_after=current_state.state_hash,
                )
            )

        return RecedingHorizonResult(
            realized_trajectory=realized_trajectory,
            decision_history=decision_history,
        )
