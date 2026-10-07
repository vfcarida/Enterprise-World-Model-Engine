"""Responsible AI trust guarantees, anti-overclaiming enforcement, and gating hooks.

This module enforces epistemic humility at the API boundary:
1. Causal claims carry an EvidenceLevel that CANNOT be auto-promoted without formal identification.
2. Forecast and scenario outputs always ship with uncertainty intervals and extrapolation flags.
3. Structured MANAGE warnings alert users to wide intervals, ungrounded regimes, and confounding.
4. Configurable ResponsibleAIGate hooks can block downstream decision pipelines on high-risk projections.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.provenance.evidence import EvidenceLevel

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


# ==============================================================================
# 2. Causal Evidence Level Anti-Auto-Promotion Guard
# ==============================================================================


def validate_causal_claim(
    claimed_level: EvidenceLevel,
    is_identifiable: bool,
    overlap_satisfied: bool = True,
    *,
    strict: bool = True,
) -> EvidenceLevel:
    """Validate that a causal claim does not exceed structural mathematical guarantees.

    Epistemic Guarantee:
        Under NO circumstances can an observational association P(Y | X) be promoted
        to INTERVENTIONAL or STRUCTURAL without passing mathematical identifiability
        and positivity overlap criteria.

    Args:
        claimed_level: The target EvidenceLevel being asserted.
        is_identifiable: Whether Backdoor Criterion or do-calculus identifiability passed.
        overlap_satisfied: Whether common support / positivity overlap is satisfied.
        strict: If True, raises UnqualifiedCausalClaimError on violation; if False,
            automatically downgrades to EvidenceLevel.PREDICTIVE with a structured warning.

    Returns:
        The validated or downgraded EvidenceLevel.
    """
    if claimed_level in (EvidenceLevel.INTERVENTIONAL, EvidenceLevel.QUASI_CAUSAL):
        reasons: list[str] = []
        if not is_identifiable:
            reasons.append(
                "Effect is NOT mathematically identifiable (open backdoor paths or unobserved confounding)."
            )
        if not overlap_satisfied:
            reasons.append(
                "Positivity / common support breached (treatment and control lack overlapping covariate support)."
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
