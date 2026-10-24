"""Responsible AI trust guarantees, anti-overclaiming enforcement, and gating hooks.

This module enforces epistemic humility at the API boundary:
1. Causal claims carry an EvidenceLevel that CANNOT be auto-promoted without formal identification.
2. Forecast and scenario outputs always ship with uncertainty intervals and extrapolation flags.
3. Structured MANAGE warnings alert users to wide intervals, ungrounded regimes, and confounding.
4. Configurable ResponsibleAIGate hooks can block downstream decision pipelines on high-risk projections.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.provenance.evidence import (
    EVIDENCE_LEVEL_TO_NATURE,
    EVIDENCE_NATURE_TO_DEFAULT_LEVEL,
    PERMISSIBLE_QUERIES_BY_NATURE,
    DependencyEvidenceNature,
    DependencyQueryKind,
    EvidenceLevel,
    EvidenceNature,
    evidence_level_to_nature,
    evidence_nature_to_level,
)

__all__ = [
    "EVIDENCE_LEVEL_TO_NATURE",
    "EVIDENCE_NATURE_TO_DEFAULT_LEVEL",
    "EVIDENCE_QUERY_MATRIX_CONTRACT",
    "INDUCTION_EVIDENCE_LADDER",
    "PERMISSIBLE_QUERIES_BY_NATURE",
    "BarePointEstimateWarning",
    "DecisionBlockedError",
    "DependencyEvidenceNature",
    "DependencyQueryKind",
    "EvidenceGateDecision",
    "EvidenceLevel",
    "EvidenceNature",
    "ExtrapolationWarning",
    "GateValidationReport",
    "InductionRefusalError",
    "InsufficientCausalEvidenceWarning",
    "ResponsibleAIGate",
    "ResponsibleAIWarning",
    "ScenarioForecastResult",
    "UncertaintyRequiredError",
    "UnqualifiedCausalClaimError",
    "WideUncertaintyWarning",
    "evaluate_dependency_chain_for_query",
    "evaluate_evidence_nature_gate",
    "evidence_level_to_nature",
    "evidence_nature_to_level",
    "is_evidence_at_least",
    "validate_causal_claim",
    "validate_dependency_for_query",
]

if TYPE_CHECKING:
    pass


# ==============================================================================
# 1. Custom Exceptions & Structured Warnings
# ==============================================================================


class ResponsibleAIWarning(UserWarning):
    """Base class for all Responsible AI epistemic and trust warnings."""


class WideUncertaintyWarning(ResponsibleAIWarning):
    """Emitted when confidence intervals or dispersion are excessively wide for decision-making."""


class InsufficientCausalEvidenceWarning(ResponsibleAIWarning):
    """Emitted when an interventional or causal claim lacks sufficient structural backing."""


class ExtrapolationWarning(ResponsibleAIWarning):
    """Emitted when a simulation trajectory leaves the grounded support regime into OOD space."""


class BarePointEstimateWarning(ResponsibleAIWarning):
    """Emitted when a user accesses a bare point estimate without acknowledging uncertainty."""


class UnqualifiedCausalClaimError(ValueError):
    """Raised when code or configuration attempts to emit an unqualified causal claim."""


class UncertaintyRequiredError(RuntimeError):
    """Raised when an operation attempts to output a point estimate without mandatory uncertainty."""


class DecisionBlockedError(RuntimeError):
    """Raised when a ResponsibleAIGate blocks a downstream decision pipeline on safety criteria."""

    def __init__(self, message: str, reasons: list[str]) -> None:
        super().__init__(message)
        self.reasons = reasons


class InductionRefusalError(DecisionBlockedError):
    """Raised when unasserted/unidentifiable elements attempt to enter a WorldSpec."""


# ==============================================================================
# 2. Epistemic Ladder & Evidence Ranking for Induction
# ==============================================================================

INDUCTION_EVIDENCE_LADDER: tuple[EvidenceLevel, ...] = (
    EvidenceLevel.ASSERTED,
    EvidenceLevel.PROVISIONAL,
    EvidenceLevel.DIRECTIONAL_UNCERTAIN,
    EvidenceLevel.UNIDENTIFIED,
    EvidenceLevel.CONFOUNDED,
)

_INDUCTION_LADDER_RANKS: dict[EvidenceLevel, int] = {
    EvidenceLevel.ASSERTED: 5,
    EvidenceLevel.PROVISIONAL: 4,
    EvidenceLevel.DIRECTIONAL_UNCERTAIN: 3,
    EvidenceLevel.UNIDENTIFIED: 2,
    EvidenceLevel.CONFOUNDED: 1,
}


def is_evidence_at_least(level: EvidenceLevel, min_level: EvidenceLevel) -> bool:
    """Check if an evidence level meets or exceeds the minimum required rank."""
    return _INDUCTION_LADDER_RANKS.get(level, 0) >= _INDUCTION_LADDER_RANKS.get(min_level, 0)


MatrixCellCondition = Literal[
    "always_permitted",
    "requires_overlap",
    "requires_full_identification",
    "requires_overlap_and_functional_identification",
    "forbidden",
]

# Full 5 x 4 (DependencyEvidenceNature x DependencyQueryKind) contract matrix
EVIDENCE_QUERY_MATRIX_CONTRACT: dict[
    tuple[DependencyEvidenceNature, DependencyQueryKind], MatrixCellCondition
] = {
    # 1. STATISTICAL_ASSOCIATION: observational prediction/evolution only; causal queries FORBIDDEN
    (
        DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
        DependencyQueryKind.STATE_EVOLUTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
        DependencyQueryKind.INTERVENTIONAL_DO,
    ): "forbidden",
    (
        DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ): "forbidden",
    # 2. CAUSAL_HYPOTHESIS: requires full identification for do(), forbidden for twin counterfactual
    (
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        DependencyQueryKind.STATE_EVOLUTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        DependencyQueryKind.INTERVENTIONAL_DO,
    ): "requires_full_identification",
    (
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ): "forbidden",
    # 3. EXPERIMENTALLY_SUPPORTED: requires overlap for do(), plus functional id + provenance for twin
    (
        DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        DependencyQueryKind.STATE_EVOLUTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        DependencyQueryKind.INTERVENTIONAL_DO,
    ): "requires_overlap",
    (
        DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ): "requires_overlap_and_functional_identification",
    # 4. BUSINESS_RULE_MECHANISM: permitted across all query types within domain overlap
    (
        DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
        DependencyQueryKind.STATE_EVOLUTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
        DependencyQueryKind.INTERVENTIONAL_DO,
    ): "requires_overlap",
    (
        DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ): "requires_overlap",
    # 5. KNOWN_FUNCTIONAL: permitted across all query types within domain overlap
    (
        DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        DependencyQueryKind.STATE_EVOLUTION,
    ): "always_permitted",
    (
        DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        DependencyQueryKind.INTERVENTIONAL_DO,
    ): "requires_overlap",
    (
        DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ): "requires_overlap",
}


class EvidenceGateDecision(BaseModel):
    """Structured fail-closed decision record for an (evidence_nature x query_kind) evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dependency_id: str
    evidence_nature: DependencyEvidenceNature
    query_kind: DependencyQueryKind
    matrix_condition: MatrixCellCondition
    permitted: bool
    effective_evidence_level: EvidenceLevel
    downgraded: bool = False
    is_identifiable: bool = False
    overlap_satisfied: bool = True
    identification_provenance: str | None = None
    refusal_reasons: tuple[str, ...] = Field(default_factory=tuple)


def evaluate_evidence_nature_gate(
    evidence_nature: DependencyEvidenceNature,
    query_kind: DependencyQueryKind,
    *,
    is_identifiable: bool = False,
    overlap_satisfied: bool = True,
    identification_provenance: str | None = None,
    dependency_id: str = "<unnamed>",
    strict: bool = False,
) -> EvidenceGateDecision:
    """Evaluate the fail-closed (evidence_nature x query_kind) refusal gate and return a structured decision."""
    cell_cond = EVIDENCE_QUERY_MATRIX_CONTRACT[(evidence_nature, query_kind)]
    has_provenance = identification_provenance is not None and bool(
        identification_provenance.strip()
    )
    has_id_evidence = is_identifiable and overlap_satisfied and has_provenance

    if query_kind in (
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
        DependencyQueryKind.STATE_EVOLUTION,
    ):
        lvl = evidence_nature_to_level(evidence_nature, has_identification_evidence=has_id_evidence)
        return EvidenceGateDecision(
            dependency_id=dependency_id,
            evidence_nature=evidence_nature,
            query_kind=query_kind,
            matrix_condition=cell_cond,
            permitted=True,
            effective_evidence_level=lvl,
            downgraded=False,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=identification_provenance,
            refusal_reasons=(),
        )

    try:
        lvl = validate_dependency_for_query(
            evidence_nature=evidence_nature,
            query_kind=query_kind,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=identification_provenance,
            dependency_id=dependency_id,
            strict=True,
        )
        return EvidenceGateDecision(
            dependency_id=dependency_id,
            evidence_nature=evidence_nature,
            query_kind=query_kind,
            matrix_condition=cell_cond,
            permitted=True,
            effective_evidence_level=lvl,
            downgraded=False,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=identification_provenance,
            refusal_reasons=(),
        )
    except UnqualifiedCausalClaimError as exc:
        if strict:
            raise
        warnings.warn(
            f"{exc} Automatically downgrading to PREDICTIVE.",
            category=InsufficientCausalEvidenceWarning,
            stacklevel=2,
        )
        return EvidenceGateDecision(
            dependency_id=dependency_id,
            evidence_nature=evidence_nature,
            query_kind=query_kind,
            matrix_condition=cell_cond,
            permitted=False,
            effective_evidence_level=EvidenceLevel.PREDICTIVE,
            downgraded=True,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=identification_provenance,
            refusal_reasons=(str(exc),),
        )


def evaluate_dependency_chain_for_query(
    chain: Sequence[Any],
    query_kind: DependencyQueryKind,
    *,
    strict: bool = True,
) -> EvidenceGateDecision:
    """Evaluate a multi-link dependency chain fail-closed (bounded by the weakest link).

    Accepts a sequence of ``Relationship``, ``StructuralAssignment``, or ``EvidenceGateDecision``
    objects. If any link in the chain violates the query requirement for ``query_kind``, the
    entire chain is refused (when ``strict=True``) or downgraded to ``EvidenceLevel.PREDICTIVE``
    (when ``strict=False``).
    """
    from ewm_engine.core.epistemic import resolve_fail_closed_evidence_level

    if not chain:
        raise ValueError("Cannot evaluate an empty dependency chain.")

    link_decisions: list[EvidenceGateDecision] = []
    all_reasons: list[str] = []
    levels: list[EvidenceLevel] = []

    for idx, item in enumerate(chain):
        if isinstance(item, EvidenceGateDecision):
            dec = item
            if not dec.permitted and strict:
                raise UnqualifiedCausalClaimError(
                    " | ".join(dec.refusal_reasons)
                    or f"Chain link {idx} ({dec.dependency_id}) refused {query_kind.value}."
                )
        else:
            ev_nat = getattr(
                item, "evidence_nature", DependencyEvidenceNature.STATISTICAL_ASSOCIATION
            )
            raw_attrs = getattr(item, "attributes", {}) or {}
            is_id = bool(getattr(item, "is_identifiable", raw_attrs.get("is_identifiable", False)))
            ovl = bool(getattr(item, "overlap_satisfied", raw_attrs.get("overlap_satisfied", True)))
            prov = getattr(
                item,
                "identification_provenance",
                raw_attrs.get("identification_provenance", None),
            )
            dep_id = getattr(item, "mechanism_id", None) or getattr(item, "type", f"link_{idx}")
            dec = evaluate_evidence_nature_gate(
                evidence_nature=ev_nat,
                query_kind=query_kind,
                is_identifiable=is_id,
                overlap_satisfied=ovl,
                identification_provenance=prov,
                dependency_id=str(dep_id),
                strict=strict,
            )
        link_decisions.append(dec)
        levels.append(dec.effective_evidence_level)
        if not dec.permitted:
            all_reasons.extend(dec.refusal_reasons)

    all_permitted = all(d.permitted for d in link_decisions)
    eff_level = (
        resolve_fail_closed_evidence_level(levels) if all_permitted else EvidenceLevel.PREDICTIVE
    )
    weakest_nature = evidence_level_to_nature(eff_level)
    chain_id = "->".join(d.dependency_id for d in link_decisions)

    return EvidenceGateDecision(
        dependency_id=chain_id,
        evidence_nature=weakest_nature,
        query_kind=query_kind,
        matrix_condition=EVIDENCE_QUERY_MATRIX_CONTRACT[(weakest_nature, query_kind)],
        permitted=all_permitted,
        effective_evidence_level=eff_level,
        downgraded=not all_permitted,
        is_identifiable=all(d.is_identifiable for d in link_decisions),
        overlap_satisfied=all(d.overlap_satisfied for d in link_decisions),
        identification_provenance=(
            "; ".join(
                d.identification_provenance for d in link_decisions if d.identification_provenance
            )
            or None
        ),
        refusal_reasons=tuple(all_reasons),
    )


def validate_dependency_for_query(
    evidence_nature: DependencyEvidenceNature,
    query_kind: DependencyQueryKind,
    *,
    is_identifiable: bool = False,
    overlap_satisfied: bool = True,
    identification_provenance: str | None = None,
    dependency_id: str = "<unnamed>",
    strict: bool = True,
) -> EvidenceLevel:
    """Validate whether a dependency's ``DependencyEvidenceNature`` permits ``query_kind``.

    Epistemic Invariants:
    1. ``STATISTICAL_ASSOCIATION`` is strictly observational ($P(Y \\mid X)$). It may
       be used for ``OBSERVATIONAL_PREDICTION`` and ``STATE_EVOLUTION``, but is
       **never** permitted for ``INTERVENTIONAL_DO`` or ``COUNTERFACTUAL_TWIN``,
       regardless of flags.
    2. ``CAUSAL_HYPOTHESIS`` cannot be used for ``INTERVENTIONAL_DO`` unless
       explicitly backed by identification evidence (``is_identifiable=True``,
       ``overlap_satisfied=True``, and non-empty ``identification_provenance``),
       and cannot be used for ``COUNTERFACTUAL_TWIN`` without a verified functional
       mechanism.
    3. ``EXPERIMENTALLY_SUPPORTED`` permits ``INTERVENTIONAL_DO`` when overlap holds;
       ``COUNTERFACTUAL_TWIN`` additionally requires structural functional
       identifiability and provenance.
    4. ``BUSINESS_RULE_MECHANISM`` and ``KNOWN_FUNCTIONAL`` permit all query classes
       within their active policy/physical regime provided overlap holds.
    """
    has_provenance = identification_provenance is not None and bool(
        identification_provenance.strip()
    )
    has_id_evidence = is_identifiable and overlap_satisfied and has_provenance

    if query_kind in (
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
        DependencyQueryKind.STATE_EVOLUTION,
    ):
        return evidence_nature_to_level(
            evidence_nature, has_identification_evidence=has_id_evidence
        )

    reasons: list[str] = []

    if evidence_nature == DependencyEvidenceNature.STATISTICAL_ASSOCIATION:
        reasons.append(
            f"Dependency '{dependency_id}' has evidence_nature=STATISTICAL_ASSOCIATION "
            f"(observational P(Y|X)) and cannot be treated as causal or used for "
            f"'{query_kind.value}' queries."
        )
    elif evidence_nature == DependencyEvidenceNature.CAUSAL_HYPOTHESIS:
        if query_kind == DependencyQueryKind.COUNTERFACTUAL_TWIN:
            reasons.append(
                f"Dependency '{dependency_id}' has evidence_nature=CAUSAL_HYPOTHESIS "
                "and lacks a verified structural functional assignment required for "
                "counterfactual twin queries."
            )
        else:
            if not is_identifiable:
                reasons.append(
                    f"Dependency '{dependency_id}' (CAUSAL_HYPOTHESIS) is not mathematically "
                    "identifiable (open backdoor paths or unobserved confounding)."
                )
            if not overlap_satisfied:
                reasons.append(
                    f"Dependency '{dependency_id}' (CAUSAL_HYPOTHESIS) violates positivity / "
                    "common support overlap."
                )
            if not has_provenance:
                reasons.append(
                    f"Dependency '{dependency_id}' (CAUSAL_HYPOTHESIS) lacks explicit "
                    "identification_provenance (adjustment set, IV, or refuter pass)."
                )
    elif evidence_nature == DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED:
        if not overlap_satisfied:
            reasons.append(
                f"Dependency '{dependency_id}' (EXPERIMENTALLY_SUPPORTED) violates "
                "positivity / common support overlap in target regime."
            )
        if query_kind == DependencyQueryKind.COUNTERFACTUAL_TWIN and not (
            is_identifiable and has_provenance
        ):
            reasons.append(
                f"Dependency '{dependency_id}' (EXPERIMENTALLY_SUPPORTED) requires "
                "structural functional identifiability and provenance for unit-level "
                "counterfactual twin queries."
            )
    elif evidence_nature in (
        DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
        DependencyEvidenceNature.KNOWN_FUNCTIONAL,
    ):
        if not overlap_satisfied:
            reasons.append(
                f"Dependency '{dependency_id}' ({evidence_nature.value}) violates "
                "domain support / overlap constraints."
            )

    if reasons:
        msg = "Unqualified causal query on dependency: " + " ".join(reasons)
        if strict:
            raise UnqualifiedCausalClaimError(msg)
        warnings.warn(
            f"{msg} Automatically downgrading to PREDICTIVE.",
            category=InsufficientCausalEvidenceWarning,
            stacklevel=2,
        )
        return EvidenceLevel.PREDICTIVE

    return evidence_nature_to_level(evidence_nature, has_identification_evidence=has_id_evidence)


def validate_causal_claim(
    claimed_level: EvidenceLevel,
    is_identifiable: bool,
    overlap_satisfied: bool = True,
    *,
    evidence_nature: DependencyEvidenceNature | None = None,
    query_kind: DependencyQueryKind = DependencyQueryKind.INTERVENTIONAL_DO,
    identification_provenance: str | None = None,
    require_provenance: bool = False,
    strict: bool = True,
) -> EvidenceLevel:
    """Validate that a causal claim does not exceed structural mathematical guarantees.

    Epistemic Guarantee:
        Under NO circumstances can an observational association P(Y | X) be promoted
        to INTERVENTIONAL or STRUCTURAL without passing mathematical identifiability
        and positivity overlap criteria. When ``evidence_nature`` is supplied:
        - ``STATISTICAL_ASSOCIATION`` is refused for interventional/counterfactual claims.
        - ``CAUSAL_HYPOTHESIS`` is refused for interventional claims unless backed by
          identification evidence (``is_identifiable=True``, ``overlap_satisfied=True``,
          and ``identification_provenance``).

    Args:
        claimed_level: The target EvidenceLevel being asserted.
        is_identifiable: Whether Backdoor Criterion or do-calculus identifiability passed.
        overlap_satisfied: Whether common support / positivity overlap is satisfied.
        evidence_nature: Optional explicit DependencyEvidenceNature classification.
        query_kind: Query class being validated (defaults to INTERVENTIONAL_DO).
        identification_provenance: Optional provenance string documenting the source of
            causal identification assumptions (e.g., declared DAG + backdoor adjustment set).
        require_provenance: If True, refuses causal standing when identification_provenance
            is absent or empty.
        strict: If True, raises UnqualifiedCausalClaimError on violation; if False,
            automatically downgrades to EvidenceLevel.PREDICTIVE with a structured warning.

    Returns:
        The validated or downgraded EvidenceLevel.
    """
    if evidence_nature is not None:
        nature_level = validate_dependency_for_query(
            evidence_nature=evidence_nature,
            query_kind=query_kind,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=identification_provenance,
            strict=strict,
        )
        if nature_level == EvidenceLevel.PREDICTIVE and query_kind in (
            DependencyQueryKind.INTERVENTIONAL_DO,
            DependencyQueryKind.COUNTERFACTUAL_TWIN,
        ):
            return EvidenceLevel.PREDICTIVE

    if claimed_level in (
        EvidenceLevel.COUNTERFACTUAL,
        EvidenceLevel.INTERVENTIONAL,
        EvidenceLevel.QUASI_CAUSAL,
    ):
        reasons: list[str] = []
        if not is_identifiable:
            reasons.append(
                "Effect is NOT mathematically identifiable (open backdoor paths or unobserved confounding)."
            )
        if not overlap_satisfied:
            reasons.append(
                "Positivity / common support breached (treatment and control lack overlapping covariate support)."
            )
        if require_provenance and (
            identification_provenance is None or not identification_provenance.strip()
        ):
            reasons.append(
                "Causal identification provenance is missing (must cite explicit structural DAG, adjustment set, or RCT design)."
            )

        if reasons:
            msg = (
                f"Unqualified causal claim: cannot emit {claimed_level.value.upper()} standing. "
                + " ".join(reasons)
            )
            if strict:
                raise UnqualifiedCausalClaimError(msg)
            else:
                warnings.warn(
                    f"{msg} Automatically downgrading to PREDICTIVE.",
                    category=InsufficientCausalEvidenceWarning,
                    stacklevel=2,
                )
                return EvidenceLevel.PREDICTIVE

    if (
        evidence_nature == DependencyEvidenceNature.CAUSAL_HYPOTHESIS
        and claimed_level == EvidenceLevel.ASSUMED
        and query_kind == DependencyQueryKind.INTERVENTIONAL_DO
    ):
        return EvidenceLevel.QUASI_CAUSAL

    return claimed_level


# ==============================================================================
# 3. Mandatory Uncertainty Forecast Output
# ==============================================================================


class ScenarioForecastResult(BaseModel):
    """A forecast or scenario evaluation output that structurally couples point estimates with uncertainty.

    Epistemic Invariant:
        Prevents downstream systems from ignoring uncertainty. Point estimates cannot be
        extracted without explicit awareness of confidence intervals and extrapolation status.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Name of the evaluated metric or resource.")
    estimand: str = Field(
        default="E_{P_model}[Y]",
        description="Formal statistical estimand under simulator measure P_model (not P_real unless externally validated).",
    )
    point_estimate: float = Field(description="Central tendency estimate (mean or median).")
    ci_lower: float = Field(description="Lower bound of the confidence/credible interval.")
    ci_upper: float = Field(description="Upper bound of the confidence/credible interval.")
    confidence_level: float = Field(
        default=0.90,
        ge=0.50,
        lt=1.0,
        description="Nominal coverage probability of the interval.",
    )
    distribution: Any = Field(
        description="Full distributional summary across Monte Carlo rollouts.",
    )
    extrapolation_flag: bool = Field(
        default=False,
        description="True if any step in the generating rollout left the grounded training/support regime.",
    )
    sample_count: int = Field(
        default=1,
        ge=1,
        description="Number of stochastic rollout trajectories evaluated.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.PREDICTIVE,
        description="Epistemic evidence level backing this projection.",
    )
    identification_provenance: str | None = Field(
        default=None,
        description="Explicit provenance for causal identification assumptions when applicable.",
    )
    warnings_issued: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Structured warnings issued regarding interval width or extrapolation.",
    )

    @property
    def relative_ci_width(self) -> float:
        """Width of the confidence interval relative to the absolute point estimate."""
        denom = abs(self.point_estimate)
        if denom < 1e-9:
            return float("inf") if (self.ci_upper - self.ci_lower) > 0 else 0.0
        return float((self.ci_upper - self.ci_lower) / denom)

    @property
    def is_grounded(self) -> bool:
        """True if the projection remains within the grounded empirical support regime."""
        return not self.extrapolation_flag

    def get_point_estimate(self, *, acknowledge_no_uncertainty: bool = False) -> float:
        """Retrieve bare point estimate with explicit user opt-in.

        Args:
            acknowledge_no_uncertainty: Must be explicitly set to True to retrieve
                the bare point estimate. If False, raises UncertaintyRequiredError.
        """
        if not acknowledge_no_uncertainty:
            raise UncertaintyRequiredError(
                f"Responsible AI Policy Violation: Accessing bare point estimate ({self.point_estimate}) "
                f"without uncertainty interval [{self.ci_lower:.3f}, {self.ci_upper:.3f}] is prohibited. "
                f"Pass acknowledge_no_uncertainty=True to confirm you have logged the uncertainty."
            )
        warnings.warn(
            f"Bare point estimate for '{self.metric_name}' extracted without uncertainty.",
            category=BarePointEstimateWarning,
            stacklevel=2,
        )
        return self.point_estimate

    def __str__(self) -> str:
        pct = int(self.confidence_level * 100)
        ood_status = "EXTRAPOLATING (OOD)" if self.extrapolation_flag else "Grounded"
        return (
            f"Forecast[{self.metric_name}]: point={self.point_estimate:.3f} | "
            f"{pct}% CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}] | "
            f"IQR={self.distribution.iqr:.3f} | Regime={ood_status} | "
            f"Evidence={self.evidence_level.value}"
        )

    def __repr__(self) -> str:
        return (
            f"ScenarioForecastResult(metric='{self.metric_name}', point={self.point_estimate:.3f}, "
            f"ci=[{self.ci_lower:.3f}, {self.ci_upper:.3f}], level={self.confidence_level}, "
            f"extrapolation={self.extrapolation_flag}, evidence={self.evidence_level.value})"
        )


# ==============================================================================
# 4. MANAGE Gating Hook & Pipeline Guard
# ==============================================================================


class GateValidationReport(BaseModel):
    """Audit report produced by a ResponsibleAIGate evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool = Field(description="True if all configured trust gates passed.")
    reasons: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Specific safety violations if passed is False.",
    )
    warnings: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Advisory warnings emitted during validation.",
    )
    relative_ci_width: float | None = None
    is_extrapolating: bool = False
    evidence_level: EvidenceLevel | None = None


class ResponsibleAIGate:
    """Configurable pipeline guard aligned with NIST AI RMF 'MANAGE' function.

    Evaluates simulation and scenario forecasts before automated actuation or policy commit.
    """

    def __init__(
        self,
        *,
        max_relative_ci_width: float | None = 2.0,
        require_causal_identifiability: bool = True,
        block_on_extrapolation: bool = True,
        strict: bool = True,
    ) -> None:
        self.max_relative_ci_width = max_relative_ci_width
        self.require_causal_identifiability = require_causal_identifiability
        self.block_on_extrapolation = block_on_extrapolation
        self.strict = strict

    def validate_forecast(self, forecast: ScenarioForecastResult) -> GateValidationReport:
        """Validate a scenario forecast against configured safety thresholds."""
        failures: list[str] = []
        advisories: list[str] = []

        # 1. CI width check
        rel_width = forecast.relative_ci_width
        if self.max_relative_ci_width is not None and rel_width > self.max_relative_ci_width:
            msg = (
                f"Uncertainty interval for '{forecast.metric_name}' is too wide for confident actuation: "
                f"relative width {rel_width:.2f} exceeds threshold {self.max_relative_ci_width:.2f}."
            )
            failures.append(msg)
            if not self.strict:
                warnings.warn(msg, category=WideUncertaintyWarning, stacklevel=2)

        # 2. Extrapolation check
        if self.block_on_extrapolation and forecast.extrapolation_flag:
            msg = (
                f"Forecast for '{forecast.metric_name}' is EXTRAPOLATING beyond grounded support. "
                f"Rollout trajectory passed into ungrounded out-of-distribution regime."
            )
            failures.append(msg)
            if not self.strict:
                warnings.warn(msg, category=ExtrapolationWarning, stacklevel=2)

        # 3. Causal standing check
        if self.require_causal_identifiability:
            if forecast.evidence_level not in (
                EvidenceLevel.COUNTERFACTUAL,
                EvidenceLevel.STRUCTURAL,
                EvidenceLevel.INTERVENTIONAL,
                EvidenceLevel.QUASI_CAUSAL,
            ):
                advisories.append(
                    f"Forecast carries non-causal standing ({forecast.evidence_level.value}). "
                    f"Treat as observational prediction rather than interventional proof."
                )

        passed = len(failures) == 0

        report = GateValidationReport(
            passed=passed,
            reasons=tuple(failures),
            warnings=tuple(advisories),
            relative_ci_width=rel_width,
            is_extrapolating=forecast.extrapolation_flag,
            evidence_level=forecast.evidence_level,
        )

        if not passed and self.strict:
            raise DecisionBlockedError(
                f"Responsible AI Gate BLOCKED decision pipeline with {len(failures)} violations: "
                + " | ".join(failures),
                reasons=failures,
            )

        return report
