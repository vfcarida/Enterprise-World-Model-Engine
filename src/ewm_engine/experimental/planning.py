"""Planning and controller layer for EWM Engine (Model Predictive Control / Re-Grounding).

[EXPERIMENTAL] This module provides a pluggable planning and rollout scoring layer:
    Observe -> Simulate Short Horizon -> Score under Uncertainty -> Act -> Re-ground -> Repeat

Epistemic Principle:
    Long open-loop rollouts are not forecasts. Real socio-technical systems experience
    unmodeled shocks, structural rule shifts, and autoregressive drift. Continuous
    re-grounding from observed world states is required to bound error compounding.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.branching import branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory, TrajectoryStatus


def extract_trajectory_metric(traj: Trajectory, metric_name: str) -> float:
    """Extract a scalar metric value from a rollout trajectory."""
    # 1. Total or hard constraint violations
    if metric_name in ("violations_count", "total_violations"):
        return float(traj.total_violations)
    if metric_name == "hard_violations":
        return float(traj.hard_violations)
    if metric_name == "invalid_status":
        return 1.0 if traj.status == TrajectoryStatus.INVALID else 0.0

    # 2. Resource check on final state
    if metric_name in traj.final_state.resources:
        return float(traj.final_state.get_resource(metric_name).current)
    if metric_name.startswith("resource:"):
        r_id = metric_name[len("resource:") :]
        if r_id in traj.final_state.resources:
            return float(traj.final_state.get_resource(r_id).current)

    # 3. Step metrics dictionary from Trajectory
    return float(traj.final_metric(metric_name))


class ScorerResult(BaseModel):
    """Result of scoring a candidate decision over a set of rollout trajectories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(
        description="Unified scalar utility score (higher is always better).",
    )
    raw_objective: float = Field(
        description="Unadjusted empirical expectation or summary of the primary metric.",
    )
    details: dict[str, float] = Field(
        default_factory=dict,
        description="Component breakdown (e.g. mean, std, cvar, penalties).",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata from the scoring evaluation.",
    )


class PlanningCandidate(BaseModel):
    """A candidate policy option evaluated by a world-model planner."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    id: str = Field(description="Unique candidate identifier.")
    name: str = Field(default="", description="Descriptive human-readable name.")
    actions: tuple[Action, ...] = Field(
        default_factory=tuple,
        description="Immediate actions proposed by this candidate.",
    )
    intervention: Intervention | None = Field(
        default=None,
        description="Simulation intervention applied for this candidate.",
    )
    actor: Actor | None = Field(
        default=None,
        description="Candidate policy actor plugged into the simulation lookahead.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual candidate metadata.",
    )

    @classmethod
    def from_action(cls, action: Action, id: str | None = None) -> PlanningCandidate:
        """Create a candidate proposing a single immediate action."""
        cand_id = id or action.id or f"action_{action.type}"
        return cls(id=cand_id, name=action.type, actions=(action,))

    @classmethod
    def from_actions(cls, actions: Sequence[Action], id: str, name: str = "") -> PlanningCandidate:
        """Create a candidate proposing a sequence of immediate actions."""
        return cls(id=id, name=name or id, actions=tuple(actions))

    @classmethod
    def from_intervention(cls, intervention: Intervention) -> PlanningCandidate:
        """Create a candidate defined by a policy intervention."""
        return cls(
            id=intervention.id,
            name=intervention.description or intervention.id,
            intervention=intervention,
        )

    @classmethod
    def from_actor(cls, actor: Actor, id: str | None = None) -> PlanningCandidate:
        """Create a candidate defined by an autonomous policy actor."""
        cand_id = id or actor.actor_id
        return cls(id=cand_id, name=cand_id, actor=actor)


@runtime_checkable
class RolloutScorer(Protocol):
    """Protocol for scoring candidate rollout distributions."""

    @property
    def name(self) -> str:
        """Human-readable scorer name."""
        ...

    def score_rollouts(
        self,
        trajectories: Sequence[Trajectory],
        candidate: PlanningCandidate,
    ) -> ScorerResult:
        """Evaluate rollout trajectories and return a unified scalar score (higher is better)."""
        ...


class ExpectedObjectiveScorer:
    """Scores rollouts by the sample mean of a designated metric across trajectories."""

    def __init__(
        self,
        metric: str,
        minimize: bool = False,
        name: str = "ExpectedObjectiveScorer",
    ) -> None:
        self.metric = metric
        self.minimize = minimize
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score_rollouts(
        self,
        trajectories: Sequence[Trajectory],
        candidate: PlanningCandidate,
    ) -> ScorerResult:
        if not trajectories:
            return ScorerResult(score=float("-inf"), raw_objective=0.0)

        vals = [extract_trajectory_metric(t, self.metric) for t in trajectories]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        score = -mu if self.minimize else mu

        return ScorerResult(
            score=score,
            raw_objective=mu,
            details={
                "mean": mu,
                "std": sigma,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            },
            metadata={"metric": self.metric, "minimize": self.minimize},
        )


class CVaRScorer:
    """Risk-averse scorer evaluating Conditional Value-at-Risk (tail risk) across rollouts."""

    def __init__(
        self,
        metric: str,
        alpha: float = 0.10,
        minimize: bool = False,
        name: str = "CVaRScorer",
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.metric = metric
        self.alpha = alpha
        self.minimize = minimize
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score_rollouts(
        self,
        trajectories: Sequence[Trajectory],
        candidate: PlanningCandidate,
    ) -> ScorerResult:
        if not trajectories:
            return ScorerResult(score=float("-inf"), raw_objective=0.0)

        vals = np.array(
            [extract_trajectory_metric(t, self.metric) for t in trajectories], dtype=float
        )
        mu = float(np.mean(vals))

        if self.minimize:
            # When minimizing cost/violations, worst cases are in the upper tail (high values)
            quantile_val = float(np.percentile(vals, 100.0 * (1.0 - self.alpha)))
            tail_vals = vals[vals >= quantile_val]
            cvar = float(np.mean(tail_vals)) if len(tail_vals) > 0 else quantile_val
            # Score is negated so lower tail cost yields higher score
            score = -cvar
        else:
            # When maximizing profit/utility, worst cases are in the lower tail (low values)
            quantile_val = float(np.percentile(vals, 100.0 * self.alpha))
            tail_vals = vals[vals <= quantile_val]
            cvar = float(np.mean(tail_vals)) if len(tail_vals) > 0 else quantile_val
            score = cvar

        return ScorerResult(
            score=score,
            raw_objective=mu,
            details={
                "cvar": cvar,
                "quantile": quantile_val,
                "mean": mu,
                "std": float(np.std(vals)),
                "alpha": self.alpha,
            },
            metadata={"metric": self.metric, "minimize": self.minimize},
        )


class ConstraintPenalizedScorer:
    """Scores rollouts with explicit linear penalties for hard and soft constraint violations."""

    def __init__(
        self,
        base_metric: str,
        violation_penalty_weight: float = 100.0,
        invalid_trajectory_penalty: float = 500.0,
        minimize: bool = False,
        name: str = "ConstraintPenalizedScorer",
    ) -> None:
        self.base_metric = base_metric
        self.violation_penalty_weight = violation_penalty_weight
        self.invalid_trajectory_penalty = invalid_trajectory_penalty
        self.minimize = minimize
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score_rollouts(
        self,
        trajectories: Sequence[Trajectory],
        candidate: PlanningCandidate,
    ) -> ScorerResult:
        if not trajectories:
            return ScorerResult(score=float("-inf"), raw_objective=0.0)

        base_vals = [extract_trajectory_metric(t, self.base_metric) for t in trajectories]
        mu_base = float(np.mean(base_vals))
        oriented_base = -mu_base if self.minimize else mu_base

        violations = [t.total_violations for t in trajectories]
        mean_violations = float(np.mean(violations))

        invalid_count = sum(1 for t in trajectories if t.status == TrajectoryStatus.INVALID)
        invalid_rate = invalid_count / len(trajectories)

        penalty = (self.violation_penalty_weight * mean_violations) + (
            self.invalid_trajectory_penalty * invalid_rate
        )
        score = oriented_base - penalty

        return ScorerResult(
            score=score,
            raw_objective=mu_base,
            details={
                "oriented_base": oriented_base,
                "mean_violations": mean_violations,
                "invalid_rate": invalid_rate,
                "total_penalty": penalty,
            },
            metadata={"base_metric": self.base_metric, "minimize": self.minimize},
        )


class UncertaintyPenalizedScorer:
    """Penalizes outcome variance across rollouts to favor predictable, robust decisions."""

    def __init__(
        self,
        metric: str,
        uncertainty_weight: float = 1.0,
        minimize: bool = False,
        measure: str = "std",
        name: str = "UncertaintyPenalizedScorer",
    ) -> None:
        self.metric = metric
        self.uncertainty_weight = uncertainty_weight
        self.minimize = minimize
        self.measure = measure.lower()
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score_rollouts(
        self,
        trajectories: Sequence[Trajectory],
        candidate: PlanningCandidate,
    ) -> ScorerResult:
        if not trajectories:
            return ScorerResult(score=float("-inf"), raw_objective=0.0)

        vals = [extract_trajectory_metric(t, self.metric) for t in trajectories]
        mu = float(np.mean(vals))
        sigma = float(np.std(vals))
        spread = sigma**2 if self.measure == "var" else sigma

        penalty = self.uncertainty_weight * spread

        if self.minimize:
            # Minimizing cost: worst case is higher cost + higher uncertainty
            score = -(mu + penalty)
        else:
            # Maximizing utility: reward mean minus uncertainty penalty
            score = mu - penalty

        return ScorerResult(
            score=score,
            raw_objective=mu,
            details={
                "mean": mu,
                "std": sigma,
                "spread": spread,
                "penalty": penalty,
            },
            metadata={"metric": self.metric, "minimize": self.minimize},
        )


class PlanningDecision(BaseModel):
    """Audit record and decision provenance for an evaluated planning step."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    decision_id: str = Field(description="Unique identifier for this decision step.")
    step: int = Field(description="Simulation decision step index.")
    chosen_candidate_id: str = Field(description="ID of the candidate selected by the planner.")
    chosen_candidate: PlanningCandidate = Field(description="Full chosen candidate object.")
    candidate_scores: dict[str, float] = Field(description="Evaluated utility score per candidate.")
    score_details: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Component breakdowns per candidate from the RolloutScorer.",
    )
    lookahead_horizon: int = Field(description="Lookahead horizon used for simulation.")
    samples_per_candidate: int = Field(description="Monte Carlo sample count per candidate.")
    seed: int = Field(description="Deterministic random seed used for this planning step.")
    scorer_name: str = Field(description="Name of RolloutScorer utilized.")
    timestamp: datetime = Field(description="Timestamp when decision was planned.")
    state_hash_before: str = Field(description="SHA-256 state hash before action execution.")
    state_hash_after: str | None = Field(
        default=None,
        description="Re-grounded SHA-256 state hash after 1-step realization.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class _StaticActionInjector:
    """Internal actor that injects a candidate's proposed actions on step 0."""

    def __init__(self, actions: Sequence[Action], actor_id: str = "action_injector") -> None:
        self._actor_id = actor_id
        self._actions = tuple(actions)

    @property
    def actor_id(self) -> str:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        if context.step == 0:
            return self._actions
        return ()


@runtime_checkable
class Planner(Protocol):
    """Protocol for world-model-based planners."""

    def plan(
        self,
        *,
        world: World,
        state: WorldState,
        candidates: Sequence[PlanningCandidate],
        horizon: int,
        samples: int,
        scorer: RolloutScorer,
        step: int = 0,
        seed: int = 42,
    ) -> PlanningDecision:
        """Simulate candidate rollouts, evaluate scores, and return a PlanningDecision."""
        ...


class RolloutPlanner:
    """Standard Monte Carlo lookahead planner evaluating candidates in a branched world."""

    def __init__(self, engine: SimulationEngine | None = None) -> None:
        self.engine = engine or SimulationEngine()

    def plan(
        self,
        *,
        world: World,
        state: WorldState,
        candidates: Sequence[PlanningCandidate],
        horizon: int,
        samples: int,
        scorer: RolloutScorer,
        step: int = 0,
        seed: int = 42,
    ) -> PlanningDecision:
        if not candidates:
            raise ValueError("Planner.plan requires at least one PlanningCandidate.")

        state_hash_before = state.state_hash
        candidate_scores: dict[str, float] = {}
        score_details: dict[str, dict[str, float]] = {}

        for cand_idx, cand in enumerate(candidates):
            sim_world = branch_world(world, state=state)

            if cand.actor is not None:
                sim_world.actors = [
                    a for a in sim_world.actors if a.actor_id != cand.actor.actor_id
                ]
                sim_world.add_actor(cand.actor)
            elif cand.actions:
                sim_world.add_actor(
                    _StaticActionInjector(cand.actions, actor_id=f"injector_{cand.id}")
                )

            sim_scenario = Scenario(
                name=f"Lookahead_{cand.id}",
                horizon=horizon,
                samples=samples,
                seed=seed + (cand_idx * 1000) + step,
                intervention=cand.intervention,
            )

            sim_res = self.engine.run(world=sim_world, scenario=sim_scenario)
            scorer_res = scorer.score_rollouts(sim_res.trajectories, cand)

            candidate_scores[cand.id] = scorer_res.score
            score_details[cand.id] = scorer_res.details

        # Deterministic selection: highest score, tie-break by candidate id
        best_cand = max(
            candidates,
            key=lambda c: (candidate_scores[c.id], -hash(c.id)),
        )

        return PlanningDecision(
            decision_id=str(uuid.uuid4()),
            step=step,
            chosen_candidate_id=best_cand.id,
            chosen_candidate=best_cand,
            candidate_scores=candidate_scores,
            score_details=score_details,
            lookahead_horizon=horizon,
            samples_per_candidate=samples,
            seed=seed,
            scorer_name=scorer.name,
            timestamp=datetime.now(UTC),
            state_hash_before=state_hash_before,
        )


class PlanningActor:
    """An Actor that executes receding-horizon planning at each decision step."""

    def __init__(
        self,
        actor_id: str,
        world_model: World,
        candidate_generator: Callable[[WorldState, ActorContext], Sequence[PlanningCandidate]],
        scorer: RolloutScorer,
        lookahead_horizon: int = 3,
        samples_per_candidate: int = 5,
        planner: Planner | None = None,
    ) -> None:
        self._actor_id = actor_id
        self.world_model = world_model
        self.candidate_generator = candidate_generator
        self.scorer = scorer
        self.lookahead_horizon = lookahead_horizon
        self.samples_per_candidate = samples_per_candidate
        self.planner: Planner = planner or RolloutPlanner()
        self.decision_history: list[PlanningDecision] = []

    @property
    def actor_id(self) -> str:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        candidates = self.candidate_generator(state, context)
        if not candidates:
            return ()

        # Ensure lookahead world does not contain this PlanningActor to avoid recursive planning
        cleaned_world = branch_world(self.world_model, state=state)
        cleaned_world.actors = [a for a in cleaned_world.actors if a.actor_id != self._actor_id]

        # Derive a deterministic sub-seed for this step
        step_seed = (
            int(context.rng.integers(0, 2**31 - 1))
            if hasattr(context.rng, "integers")
            else 42 + context.step
        )

        decision = self.planner.plan(
            world=cleaned_world,
            state=state,
            candidates=candidates,
            horizon=self.lookahead_horizon,
            samples=self.samples_per_candidate,
            scorer=self.scorer,
            step=context.step,
            seed=step_seed,
        )
        self.decision_history.append(decision)
        return decision.chosen_candidate.actions
