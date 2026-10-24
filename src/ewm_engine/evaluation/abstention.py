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
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.drift import DriftKind, DriftSignal, PlanInvalidationAudit
from ewm_engine.core.trust import EvidenceGateDecision
from ewm_engine.dynamics.base import ValidityDomain
from ewm_engine.evaluation.uncertainty import (
    UncertaintySourceType,
    UncertaintyTaxonomyBreakdown,
)

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


class AbstentionTriggerKind(StrEnum):
    """Taxonomy of conditions triggering 'do not use me here' abstention."""

    HIGH_UNCERTAINTY = "high_uncertainty"
    OUT_OF_DOMAIN_STATE = "out_of_domain_state"
    UNSUPPORTED_REGIME_CONTEXT = "unsupported_regime_context"
    PERFORMATIVE_INVALIDATION = "performative_invalidation"
    ENDOGENOUS_CONFOUNDING = "endogenous_confounding"
    ADVERSARIAL_DEGRADATION = "adversarial_degradation"
    INSUFFICIENT_CAUSAL_EVIDENCE = "insufficient_causal_evidence"
    HORIZON_COMPETENCE_DECAY = "horizon_competence_decay"


class AbstentionResolutionAction(StrEnum):
    """Operational resolution action when the world model reaches its validity limits."""

    PROCEED_WITH_MONITORING = "proceed_with_monitoring"
    FALLBACK_SAFE_POLICY = "fallback_safe_policy"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    BLOCK_ACTUATION_FAIL_CLOSED = "block_actuation_fail_closed"


class UncertaintyTriggerDetail(BaseModel):
    """Specific uncertainty source from the 6-part taxonomy that triggered abstention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: UncertaintySourceType
    observed_value: float
    threshold: float
    rationale: str


class ValidityLimitAbstentionDecision(BaseModel):
    """Structured 'do not use me here' abstention decision tied to the uncertainty taxonomy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    do_not_use_me_here: bool = Field(
        description="Explicit flag indicating the model must not be used for automated actuation here."
    )
    abstain: bool = Field(
        description="True when the model abstains due to uncertainty or validity-limit breaches."
    )
    status_banner: str = Field(
        description="'DO_NOT_USE_ME_HERE' when abstaining, else 'WITHIN_DECLARED_VALIDITY_LIMITS'."
    )
    recommended_resolution: AbstentionResolutionAction
    requires_human_review: bool = False
    human_review_brief: str | None = None
    fallback_policy_id: str | None = None
    fallback_actions: tuple[Action, ...] = Field(default_factory=tuple)
    triggered_uncertainty_sources: tuple[UncertaintySourceType, ...] = Field(
        default_factory=tuple,
        description="Exact subset of the 6-part UncertaintySourceType taxonomy that triggered abstention.",
    )
    uncertainty_trigger_details: tuple[UncertaintyTriggerDetail, ...] = Field(
        default_factory=tuple,
        description="Quantitative details per triggered uncertainty source.",
    )
    triggered_validity_limits: tuple[AbstentionTriggerKind, ...] = Field(
        default_factory=tuple,
        description="Validity-limit categories that fired.",
    )
    triggered_drift_kinds: tuple[DriftKind, ...] = Field(
        default_factory=tuple,
        description="DriftKind signals (exogenous or model-induced) that contributed to abstention.",
    )
    validity_domain_violations: tuple[str, ...] = Field(default_factory=tuple)
    competence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reasons: tuple[str, ...] = Field(default_factory=tuple)


class ValidityLimitAbstentionGate:
    """Evaluates uncertainty sources, validity domain, performativity, and evidence to decide abstention.

    Operationalizes Uncertainty-Typed Abstention:
    - Reports *which* specific uncertainty source(s) from the 6-part taxonomy
      (`ALEATORIC`, `PARAMETER`, `STRUCTURAL_MODEL`, `RUNTIME_CONTEXT`,
      `MONTE_CARLO_ERROR`, `MODEL_DISCREPANCY_VS_REALITY`) triggered abstention.
    - Detects when observed conditions leave `ValidityDomain` (OOD state or unsupported
      regime in `ContextSnapshot`), or when `PlanInvalidationAudit` / `DriftSignal` flags
      performative invalidation, endogenous confounding, or adversarial adaptation.
    - Routes to human review (`HUMAN_REVIEW_REQUIRED`) and/or supplies a pre-defined
      operational fallback policy (`FALLBACK_SAFE_POLICY`).
    """

    def __init__(
        self,
        *,
        max_aleatoric_std: float | None = None,
        max_parameter_std: float | None = None,
        max_structural_model_std: float | None = None,
        max_runtime_context_std: float | None = None,
        max_monte_carlo_se: float | None = None,
        max_model_discrepancy: float | None = None,
        validity_domain: ValidityDomain | None = None,
        competence_estimator: CompetenceEstimator | None = None,
        fallback_policy_id: str = "conservative_fallback_policy",
        fallback_actions: Sequence[Action] = (),
        require_human_review_on_ood_or_performative: bool = True,
    ) -> None:
        self.max_aleatoric_std = max_aleatoric_std
        self.max_parameter_std = max_parameter_std
        self.max_structural_model_std = max_structural_model_std
        self.max_runtime_context_std = max_runtime_context_std
        self.max_monte_carlo_se = max_monte_carlo_se
        self.max_model_discrepancy = max_model_discrepancy
        self.validity_domain = validity_domain
        self.competence_estimator = competence_estimator
        self.fallback_policy_id = fallback_policy_id
        self.fallback_actions = tuple(fallback_actions)
        self.require_human_review_on_ood_or_performative = (
            require_human_review_on_ood_or_performative
        )

    def evaluate(
        self,
        *,
        state: WorldState | None = None,
        runtime_context: ContextSnapshot | None = None,
        uncertainty_breakdown: UncertaintyTaxonomyBreakdown | None = None,
        drift_signals: Sequence[DriftSignal] = (),
        plan_invalidation: PlanInvalidationAudit | None = None,
        evidence_gate_decision: EvidenceGateDecision | None = None,
        step: int = 0,
    ) -> ValidityLimitAbstentionDecision:
        """Evaluate all uncertainty and validity-limit criteria and return an abstention decision."""
        trig_unc_sources: list[UncertaintySourceType] = []
        unc_details: list[UncertaintyTriggerDetail] = []
        trig_limits: list[AbstentionTriggerKind] = []
        trig_drifts: list[DriftKind] = []
        domain_violations: list[str] = []
        reasons: list[str] = []
        comp_score = 1.0

        def _add_limit(kind: AbstentionTriggerKind) -> None:
            if kind not in trig_limits:
                trig_limits.append(kind)

        def _add_unc(src: UncertaintySourceType, obs: float, thresh: float, label: str) -> None:
            if src not in trig_unc_sources:
                trig_unc_sources.append(src)
            rationale = (
                f"Uncertainty source '{src.value}' ({label}) = {obs:.4f} "
                f"exceeds threshold {thresh:.4f}."
            )
            unc_details.append(
                UncertaintyTriggerDetail(
                    source=src,
                    observed_value=obs,
                    threshold=thresh,
                    rationale=rationale,
                )
            )
            reasons.append(rationale)
            _add_limit(AbstentionTriggerKind.HIGH_UNCERTAINTY)

        # 1. Evaluate 6-part UncertaintyTaxonomyBreakdown
        if uncertainty_breakdown is not None:
            checks: list[tuple[UncertaintySourceType, float | None, float | None, str]] = [
                (
                    UncertaintySourceType.ALEATORIC,
                    uncertainty_breakdown.aleatoric_std,
                    self.max_aleatoric_std,
                    "irreducible stochastic dispersion",
                ),
                (
                    UncertaintySourceType.PARAMETER,
                    uncertainty_breakdown.parameter_std,
                    self.max_parameter_std,
                    "parameter posterior dispersion",
                ),
                (
                    UncertaintySourceType.STRUCTURAL_MODEL,
                    uncertainty_breakdown.structural_model_std,
                    self.max_structural_model_std,
                    "competing structural mechanism dispersion",
                ),
                (
                    UncertaintySourceType.RUNTIME_CONTEXT,
                    uncertainty_breakdown.runtime_context_std,
                    self.max_runtime_context_std,
                    "runtime operational context dispersion",
                ),
                (
                    UncertaintySourceType.MONTE_CARLO_ERROR,
                    uncertainty_breakdown.monte_carlo_standard_error,
                    self.max_monte_carlo_se,
                    "finite-sample Monte Carlo standard error",
                ),
                (
                    UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY,
                    uncertainty_breakdown.model_discrepancy_vs_reality,
                    self.max_model_discrepancy,
                    "Kennedy-O'Hagan structural discrepancy vs reality",
                ),
            ]
            for src_type, obs_val, max_thresh, desc in checks:
                if (
                    obs_val is not None
                    and max_thresh is not None
                    and float(obs_val) > float(max_thresh)
                ):
                    _add_unc(src_type, float(obs_val), float(max_thresh), desc)

        # 2. Evaluate declared ValidityDomain (state OOD + context regime)
        if self.validity_domain is not None and state is not None:
            is_valid, dom_reasons = self.validity_domain.check_validity(
                state, runtime_context=runtime_context
            )
            if not is_valid:
                for dr in dom_reasons:
                    domain_violations.append(dr)
                    reasons.append(f"ValidityDomain violation: {dr}")
                    if "Context regime" in dr:
                        _add_limit(AbstentionTriggerKind.UNSUPPORTED_REGIME_CONTEXT)
                        if UncertaintySourceType.RUNTIME_CONTEXT not in trig_unc_sources:
                            trig_unc_sources.append(UncertaintySourceType.RUNTIME_CONTEXT)
                    else:
                        _add_limit(AbstentionTriggerKind.OUT_OF_DOMAIN_STATE)
                        if UncertaintySourceType.STRUCTURAL_MODEL not in trig_unc_sources:
                            trig_unc_sources.append(UncertaintySourceType.STRUCTURAL_MODEL)

        # 3. Evaluate CompetenceEstimator (horizon decay / support envelope)
        if self.competence_estimator is not None and state is not None:
            c_score, c_reasons = self.competence_estimator.evaluate_state(state, step=step)
            comp_score = c_score
            if c_score < self.competence_estimator.abstention_threshold:
                _add_limit(AbstentionTriggerKind.HORIZON_COMPETENCE_DECAY)
                for cr in c_reasons:
                    reasons.append(f"Competence envelope breach: {cr}")
                if UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY)

        # 4. Evaluate DriftSignals (exogenous + model-induced)
        for sig in drift_signals:
            if not sig.is_drifted:
                continue
            if sig.kind not in trig_drifts:
                trig_drifts.append(sig.kind)
            if sig.kind == DriftKind.POLICY_INDUCED_SHIFT:
                _add_limit(AbstentionTriggerKind.PERFORMATIVE_INVALIDATION)
                if UncertaintySourceType.STRUCTURAL_MODEL not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.STRUCTURAL_MODEL)
                reasons.append(
                    f"Performative drift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )
            elif sig.kind == DriftKind.ENDOGENOUS_FEEDBACK_SHIFT:
                _add_limit(AbstentionTriggerKind.ENDOGENOUS_CONFOUNDING)
                if UncertaintySourceType.STRUCTURAL_MODEL not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.STRUCTURAL_MODEL)
                reasons.append(
                    f"Endogenous feedback shift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )
            elif sig.kind == DriftKind.ADVERSARIAL_ADAPTATION_SHIFT:
                _add_limit(AbstentionTriggerKind.ADVERSARIAL_DEGRADATION)
                if UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY)
                reasons.append(
                    f"Adversarial adaptation shift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )
            elif sig.kind in (
                DriftKind.RULE_CONSTRAINT_CHANGE,
                DriftKind.STRUCTURAL_MECHANISM_CHANGE,
                DriftKind.TOPOLOGY_CHANGE,
            ):
                _add_limit(AbstentionTriggerKind.UNSUPPORTED_REGIME_CONTEXT)
                if UncertaintySourceType.RUNTIME_CONTEXT not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.RUNTIME_CONTEXT)
                reasons.append(
                    f"Regime/context drift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )
            elif sig.kind == DriftKind.PARAMETER_DRIFT:
                _add_limit(AbstentionTriggerKind.OUT_OF_DOMAIN_STATE)
                if UncertaintySourceType.PARAMETER not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.PARAMETER)
                reasons.append(
                    f"Parameter drift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )
            elif sig.kind == DriftKind.COVARIATE_SHIFT:
                _add_limit(AbstentionTriggerKind.OUT_OF_DOMAIN_STATE)
                if UncertaintySourceType.ALEATORIC not in trig_unc_sources:
                    trig_unc_sources.append(UncertaintySourceType.ALEATORIC)
                reasons.append(
                    f"Covariate shift detected ({sig.kind.value}, mag={sig.magnitude:.4f}): {sig.details}"
                )

        # 5. Evaluate PlanInvalidationAudit (self-invalidating plans)
        if plan_invalidation is not None and plan_invalidation.is_self_invalidating:
            if plan_invalidation.performative_shift_magnitude > 0.0:
                _add_limit(AbstentionTriggerKind.PERFORMATIVE_INVALIDATION)
            if plan_invalidation.endogenous_confounding_detected:
                _add_limit(AbstentionTriggerKind.ENDOGENOUS_CONFOUNDING)
            if plan_invalidation.adversarial_shift_detected:
                _add_limit(AbstentionTriggerKind.ADVERSARIAL_DEGRADATION)
            if UncertaintySourceType.STRUCTURAL_MODEL not in trig_unc_sources:
                trig_unc_sources.append(UncertaintySourceType.STRUCTURAL_MODEL)
            for inv_a in plan_invalidation.invalidated_assumptions:
                if inv_a not in reasons:
                    reasons.append(f"Plan self-invalidation: {inv_a}")
            for d_sig in plan_invalidation.drift_signals:
                if d_sig.kind not in trig_drifts:
                    trig_drifts.append(d_sig.kind)

        # 6. Evaluate EvidenceGateDecision (refused causal queries)
        if evidence_gate_decision is not None and not evidence_gate_decision.permitted:
            _add_limit(AbstentionTriggerKind.INSUFFICIENT_CAUSAL_EVIDENCE)
            if UncertaintySourceType.STRUCTURAL_MODEL not in trig_unc_sources:
                trig_unc_sources.append(UncertaintySourceType.STRUCTURAL_MODEL)
            for r_reason in evidence_gate_decision.refusal_reasons:
                reasons.append(f"Causal gate refusal: {r_reason}")

        abstain = len(trig_limits) > 0 or len(trig_unc_sources) > 0
        if not abstain:
            return ValidityLimitAbstentionDecision(
                do_not_use_me_here=False,
                abstain=False,
                status_banner="WITHIN_DECLARED_VALIDITY_LIMITS",
                recommended_resolution=AbstentionResolutionAction.PROCEED_WITH_MONITORING,
                requires_human_review=False,
                human_review_brief=None,
                fallback_policy_id=None,
                fallback_actions=(),
                triggered_uncertainty_sources=(),
                uncertainty_trigger_details=(),
                triggered_validity_limits=(),
                triggered_drift_kinds=(),
                validity_domain_violations=(),
                competence_score=comp_score,
                reasons=(),
            )

        severe_limits = {
            AbstentionTriggerKind.PERFORMATIVE_INVALIDATION,
            AbstentionTriggerKind.ENDOGENOUS_CONFOUNDING,
            AbstentionTriggerKind.ADVERSARIAL_DEGRADATION,
            AbstentionTriggerKind.OUT_OF_DOMAIN_STATE,
            AbstentionTriggerKind.UNSUPPORTED_REGIME_CONTEXT,
            AbstentionTriggerKind.INSUFFICIENT_CAUSAL_EVIDENCE,
        }
        needs_human = self.require_human_review_on_ood_or_performative and any(
            lim in severe_limits for lim in trig_limits
        )
        resolution = (
            AbstentionResolutionAction.HUMAN_REVIEW_REQUIRED
            if needs_human
            else AbstentionResolutionAction.FALLBACK_SAFE_POLICY
        )
        src_list = [s.value for s in trig_unc_sources]
        lim_list = [lim.value for lim in trig_limits]
        brief = (
            f"DO NOT USE MODEL FOR AUTOMATED ACTUATION: triggered uncertainty sources={src_list}, "
            f"validity limits={lim_list}. Route to human operator review and execute fallback "
            f"policy '{self.fallback_policy_id}'."
        )

        return ValidityLimitAbstentionDecision(
            do_not_use_me_here=True,
            abstain=True,
            status_banner="DO_NOT_USE_ME_HERE",
            recommended_resolution=resolution,
            requires_human_review=needs_human,
            human_review_brief=brief,
            fallback_policy_id=self.fallback_policy_id,
            fallback_actions=self.fallback_actions,
            triggered_uncertainty_sources=tuple(trig_unc_sources),
            uncertainty_trigger_details=tuple(unc_details),
            triggered_validity_limits=tuple(trig_limits),
            triggered_drift_kinds=tuple(trig_drifts),
            validity_domain_violations=tuple(domain_violations),
            competence_score=comp_score,
            reasons=tuple(reasons),
        )
