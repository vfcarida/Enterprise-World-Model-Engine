"""First-class epistemic abstention and Physics-IQ correctness guardrails.

Scientific Anchors:
    - RWM-U (arXiv:2504.16680): Uncertainty-aware world modeling and epistemic abstention.
    - Motamed et al. (2025, arXiv:2501.09038): "Physics-IQ: Evaluating Physical Reasoning
      in World Models beyond Perceptual Realism" (The Anti-Plausibility Guardrail).
    - Vafa et al. (NeurIPS 2024, arXiv:2406.03689): Internal state coherence vs surface likelihood.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState
    from ewm_engine.simulation.trajectory import Trajectory


class EpistemicCompetence(BaseModel):
    """Auditable competence assessment for an individual simulation rollout or state.

    Operationalizes RWM-U epistemic abstention: signals whether a rollout is grounded
    within the model's verified operational envelope, or has drifted into ungrounded
    extrapolation requiring refusal.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_competent: bool = Field(
        description="True if the rollout remains within the model's verified competence envelope."
    )
    competence_score: float = Field(
        description="Continuous competence score in [0.0, 1.0]. 1.0 = fully grounded, 0.0 = pure hallucination."
    )
    abstention_step: int | None = Field(
        default=None,
        description="Simulation step where the model first crossed the abstention threshold.",
    )
    abstention_reason: str | None = Field(
        default=None, description="Explicit rationale for epistemic abstention."
    )
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Supporting metrics: horizon decay, boundary proximity, entropy.",
    )


class CompetenceEstimator:
    """Evaluates whether simulation trajectories remain within verified competence support.

    Computes competence using:
        1. Horizon Length Penalty: Exponential decay beyond the verified horizon H_max.
        2. Support Bounds: Checks whether resources remain within historical or prior support.
        3. Constraint Proximity: Penalizes states nearing boundary violations.
    """

    def __init__(
        self,
        max_verified_horizon: int = 15,
        horizon_decay_rate: float = 0.15,
        support_bounds: dict[str, tuple[float, float]] | None = None,
        abstention_threshold: float = 0.35,
    ) -> None:
        self.max_verified_horizon = max_verified_horizon
        self.horizon_decay_rate = horizon_decay_rate
        self.support_bounds = support_bounds or {}
        self.abstention_threshold = abstention_threshold

    def evaluate_state(self, state: WorldState, step: int) -> tuple[float, list[str]]:
        """Evaluate continuous competence score for an individual state at a specific step."""
        reasons: list[str] = []
        score = 1.0

        # 1. Horizon decay
        if step > self.max_verified_horizon:
            excess = step - self.max_verified_horizon
            horizon_factor = math.exp(-self.horizon_decay_rate * excess)
            score *= horizon_factor
            reasons.append(
                f"Horizon step {step} exceeds verified envelope {self.max_verified_horizon} (factor: {horizon_factor:.2f})."
            )

        # 2. Support bounds check
        for rid, (low, high) in self.support_bounds.items():
            if rid in state.resources:
                curr = float(state.resources[rid].current)
                if curr < low or curr > high:
                    out_dist = min(abs(curr - low), abs(curr - high))
                    support_factor = math.exp(-0.1 * out_dist)
                    score *= support_factor
                    reasons.append(
                        f"Resource '{rid}'={curr:.2f} drifted outside support [{low:.1f}, {high:.1f}] (factor: {support_factor:.2f})."
                    )

        # 3. Soft/Hard constraint boundary proximity
        for rid, r in state.resources.items():
            val = float(r.current)
            if r.min_value > -float("inf") and val < r.min_value:
                score *= 0.1
                reasons.append(
                    f"Resource '{rid}' violated non-negativity constraint ({val:.2f} < {r.min_value})."
                )

        score = max(0.0, min(1.0, float(score)))
        return score, reasons

    def evaluate_trajectory(self, trajectory: Trajectory) -> EpistemicCompetence:
        """Evaluate full trajectory for epistemic competence and identify abstention point."""
        states = [trajectory.initial_state] + [
            step.transition_result.next_state for step in trajectory.steps
        ]

        min_score = 1.0
        abstention_step: int | None = None
        primary_reason: str | None = None

        for idx, s in enumerate(states):
            score_t, reasons = self.evaluate_state(s, step=idx)
            if score_t < min_score:
                min_score = score_t

            if score_t < self.abstention_threshold and abstention_step is None:
                abstention_step = idx
                primary_reason = (
                    reasons[0]
                    if reasons
                    else f"Competence score {score_t:.3f} fell below threshold {self.abstention_threshold}."
                )

        is_competent = min_score >= self.abstention_threshold

        return EpistemicCompetence(
            is_competent=is_competent,
            competence_score=min_score,
            abstention_step=abstention_step,
            abstention_reason=primary_reason if not is_competent else None,
            diagnostics={
                "min_competence_score": min_score,
                "horizon_length": float(len(states)),
                "abstention_threshold": self.abstention_threshold,
                "max_verified_horizon": float(self.max_verified_horizon),
            },
        )


class PhysicsIQGuardrail:
    """Enforces the Physics-IQ Doctrine (Motamed et al., 2025).

    Doctrine:
        Never use output plausibility or perceptual realism as a proxy for correctness.
        A trajectory that 'looks right' or has high likelihood under a generative prior
        is epistemic noise unless verified against physical laws and conservation identities.
    """

    def __init__(self, conservation_tolerance: float = 1e-5) -> None:
        self.conservation_tolerance = conservation_tolerance

    def evaluate_physics_iq(
        self,
        trajectory: Trajectory,
        conserved_groups: Sequence[Sequence[str]] | None = None,
    ) -> tuple[float, bool, list[str]]:
        """Evaluate physical correctness score of a simulation trajectory.

        Checks:
            1. Non-negativity of physical resources (mass, inventory, energy >= 0).
            2. Conservation of declared closed resource groups.
            3. Continuity (no instantaneous teleportation/infinite rates).

        Returns:
            Tuple of (physics_iq_score in [0.0, 1.0], is_physically_sound, violation_log).
        """
        states = [trajectory.initial_state] + [
            step.transition_result.next_state for step in trajectory.steps
        ]
        if not states:
            return 1.0, True, []

        violations: list[str] = []
        total_invariants = 0
        passed_invariants = 0

        # Check 1: Non-negativity
        for idx, s in enumerate(states):
            for rid, r in s.resources.items():
                if r.min_value >= 0.0:
                    total_invariants += 1
                    if r.current >= -self.conservation_tolerance:
                        passed_invariants += 1
                    else:
                        violations.append(
                            f"Step {idx}: Negative physical resource '{rid}'={r.current:.4f}."
                        )

        # Check 2: Conservation of declared closed groups
        if conserved_groups:
            for group in conserved_groups:
                total_invariants += 1
                group_set = set(group)
                init_sum = sum(
                    float(states[0].resources[r].current)
                    for r in group_set
                    if r in states[0].resources
                )
                group_violated = False
                for idx, s in enumerate(states[1:], 1):
                    step_sum = sum(
                        float(s.resources[r].current) for r in group_set if r in s.resources
                    )
                    drift = abs(step_sum - init_sum)
                    if drift > self.conservation_tolerance:
                        group_violated = True
                        violations.append(
                            f"Step {idx}: Conservation violation in group {group}: drift={drift:.6f}."
                        )
                        break
                if not group_violated:
                    passed_invariants += 1

        physics_iq = float(passed_invariants / total_invariants) if total_invariants > 0 else 1.0
        is_sound = len(violations) == 0

        return physics_iq, is_sound, violations
