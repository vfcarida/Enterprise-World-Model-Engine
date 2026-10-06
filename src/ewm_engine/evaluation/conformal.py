"""Distribution-free conformal prediction, Adaptive Conformal Inference (ACI), and UQ Pareto frontier.

Epistemic Foundations:
    - Standard conformal prediction provides finite-sample distribution-free coverage
      guarantees under exchangeability (Vovk et al., 2005; Romano et al., 2019).
    - Under non-stationary simulation drift or real-world distribution shift, exchangeability fails.
      Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021) and Conformal-PID (Angelopoulos
      et al., 2023) dynamically update coverage levels, serving as an online regime-shift / OOD detector.
    - Uncertainty Quantification (UQ) Pareto analysis models the fundamental trade-off between
      sharpness (width), coverage, calibration, and compute cost, reflecting the impossibility
      of empirical aleatoric/epistemic disentanglement without untestable structural assumptions
      (Mucsányi et al., 2024).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    ParetoFrontier,
    compute_pareto_frontier,
)


class ConformalInterval(BaseModel):
    """Calibrated prediction interval for a single evaluation step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point_prediction: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    nominal_coverage: float
    is_covered: bool | None = Field(
        default=None,
        description="True if ground truth outcome fell within [lower_bound, upper_bound].",
    )


class ConformalEvaluationSummary(BaseModel):
    """Aggregate audit of conformal interval predictor performance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal_coverage: float
    empirical_coverage: float
    coverage_gap: float
    mean_interval_width: float
    median_interval_width: float
    guarantee_satisfied: bool = Field(
        description="True if empirical coverage >= nominal_coverage - tolerance.",
    )
    total_evaluations: int
    epistemic_note: str = Field(
        default=(
            "Conformal prediction provides finite-sample distribution-free validity under exchangeability. "
            "Under non-stationary dynamics or regime shifts, use Adaptive Conformal Inference (ACI)."
        ),
    )


class ConformalIntervalPredictor:
    """Split Conformal / CQR-style interval predictor providing distribution-free coverage.

    Calibrates non-conformity scores on a hold-out calibration set:
        q_hat = Quantile_{ceil((n+1)(1-alpha)) / n}(scores)
    yielding finite-sample coverage P(Y in C(X)) >= 1 - alpha under exchangeability.
    """

    def __init__(
        self,
        nominal_coverage: float = 0.90,
        normalized: bool = False,
    ) -> None:
        """Initialize conformal interval predictor.

        Args:
            nominal_coverage: Target confidence level 1 - alpha (e.g. 0.90 for 90% coverage).
            normalized: If True, uses normalized non-conformity scores |y - mu| / sigma
                for locally adaptive interval widths (CQR style).
        """
        if not 0.0 < nominal_coverage < 1.0:
            raise ValueError(f"nominal_coverage must be in (0, 1), got {nominal_coverage}")
        self.nominal_coverage = float(nominal_coverage)
        self.alpha = float(1.0 - nominal_coverage)
        self.normalized = normalized
        self.conformal_quantile: float | None = None
        self._n_calibrated: int = 0

    def calibrate(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
        uncertainties: Sequence[float] | None = None,
    ) -> float:
        """Calibrate non-conformity scores on a calibration dataset.

        Args:
            actuals: Observed reference targets on calibration set.
            predictions: Model point predictions mu_hat.
            uncertainties: Model predictive standard deviations sigma_hat (required if normalized=True).

        Returns:
            The calibrated non-conformity quantile threshold q_hat.
        """
        n = min(len(actuals), len(predictions))
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        y = np.asarray(actuals[:n], dtype=float)
        mu = np.asarray(predictions[:n], dtype=float)
        residuals = np.abs(y - mu)

        if self.normalized:
            if uncertainties is None:
                raise ValueError("uncertainties must be provided when normalized=True.")
            sigma = np.clip(np.asarray(uncertainties[:n], dtype=float), 1e-6, None)
            scores = residuals / sigma
        else:
            scores = residuals

        # Finite-sample conformal quantile with conservative ceil((n + 1)(1 - alpha)) / n rule
        p_quantile = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        q_hat = float(np.percentile(scores, p_quantile * 100.0))

        self.conformal_quantile = max(0.0, q_hat)
        self._n_calibrated = n
        return self.conformal_quantile

    def predict_interval(
        self,
        prediction: float,
        uncertainty: float | None = None,
        actual: float | None = None,
    ) -> ConformalInterval:
        """Construct conformal prediction interval for a new test point.

        Args:
            prediction: Point prediction mu_hat.
            uncertainty: Model uncertainty sigma_hat (used if normalized=True).
            actual: Optional observed ground truth value to evaluate coverage.

        Returns:
            ConformalInterval with bounds, width, and coverage status.
        """
        if self.conformal_quantile is None:
            raise RuntimeError("ConformalIntervalPredictor must be calibrated before predicting.")

        if self.normalized:
            scale = max(1e-6, float(uncertainty if uncertainty is not None else 1.0))
            half_width = self.conformal_quantile * scale
        else:
            half_width = self.conformal_quantile

        lower = float(prediction - half_width)
        upper = float(prediction + half_width)
        is_cov = (lower <= actual <= upper) if actual is not None else None

        return ConformalInterval(
            point_prediction=float(prediction),
            lower_bound=lower,
            upper_bound=upper,
            interval_width=float(upper - lower),
            nominal_coverage=self.nominal_coverage,
            is_covered=is_cov,
        )

    def evaluate(
        self,
        actuals: Sequence[float],
        predictions: Sequence[float],
        uncertainties: Sequence[float] | None = None,
        tolerance: float = 0.05,
    ) -> ConformalEvaluationSummary:
        """Audit conformal predictor empirical coverage and sharpness across test points."""
        n = min(len(actuals), len(predictions))
        if n == 0:
            return ConformalEvaluationSummary(
                nominal_coverage=self.nominal_coverage,
                empirical_coverage=0.0,
                coverage_gap=-self.nominal_coverage,
                mean_interval_width=0.0,
                median_interval_width=0.0,
                guarantee_satisfied=False,
                total_evaluations=0,
            )

        covered_count = 0
        widths: list[float] = []

        for i in range(n):
            unc = uncertainties[i] if uncertainties is not None else None
            interval = self.predict_interval(
                prediction=predictions[i],
                uncertainty=unc,
                actual=actuals[i],
            )
            if interval.is_covered:
                covered_count += 1
            widths.append(interval.interval_width)

        emp_cov = float(covered_count / n)
        cov_gap = float(emp_cov - self.nominal_coverage)
        mean_w = float(np.mean(widths))
        median_w = float(np.median(widths))

        return ConformalEvaluationSummary(
            nominal_coverage=self.nominal_coverage,
            empirical_coverage=emp_cov,
            coverage_gap=cov_gap,
            mean_interval_width=mean_w,
            median_interval_width=median_w,
            guarantee_satisfied=bool(emp_cov >= (self.nominal_coverage - tolerance)),
            total_evaluations=n,
        )


class ACIState(BaseModel):
    """Snapshot of online Adaptive Conformal Inference state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int
    alpha_t: float
    current_width: float
    recent_miscoverage_rate: float
    regime_shift_detected: bool
    diagnostic_message: str


class AdaptiveConformalInference:
    """Adaptive Conformal Inference (ACI / Conformal-PID) for time-series dynamics.

    Dynamically adjusts interval coverage under non-exchangeable distribution shifts
    (Gibbs & Candès, 2021; Angelopoulos et al., 2023). Consecutive miscoverages serve
    as an early-warning signal for out-of-distribution (OOD) regime shifts.
    """

    def __init__(
        self,
        nominal_coverage: float = 0.90,
        gamma: float = 0.05,
        base_half_width: float = 1.0,
        window_size: int = 20,
        shift_miscoverage_threshold: float = 0.40,
    ) -> None:
        """Initialize Adaptive Conformal Inference tracker.

        Args:
            nominal_coverage: Target marginal coverage (e.g. 0.90).
            gamma: Learning rate for online error updates alpha_{t+1} = alpha_t + gamma * (alpha - err_t).
            base_half_width: Base interval width scaling.
            window_size: Moving window size for monitoring empirical miscoverage rate.
            shift_miscoverage_threshold: Empirical miscoverage rate above which regime shift is alerted.
        """
        self.nominal_coverage = nominal_coverage
        self.target_alpha = 1.0 - nominal_coverage
        self.alpha_t = self.target_alpha
        self.gamma = gamma
        self.base_half_width = base_half_width
        self.window_size = window_size
        self.shift_threshold = shift_miscoverage_threshold

        self.step = 0
        self._history: deque[int] = deque(maxlen=window_size)
        self._consecutive_misses = 0

    def predict_and_update(
        self,
        point_prediction: float,
        actual: float,
        scale: float = 1.0,
    ) -> tuple[ConformalInterval, ACIState]:
        """Predict time-series interval at step t and update ACI state with observed outcome.

        Args:
            point_prediction: Forecast mean mu_t.
            actual: True observed outcome y_t.
            scale: Local variance scaling factor.

        Returns:
            Tuple of (ConformalInterval, ACIState).
        """
        self.step += 1

        # Current interval half-width is inversely proportional to alpha_t
        # Higher alpha_t -> narrower interval; lower alpha_t -> wider interval
        multiplier = max(0.1, float(1.0 / (self.alpha_t + 1e-6)))
        half_width = self.base_half_width * multiplier * scale

        lower = point_prediction - half_width
        upper = point_prediction + half_width
        is_cov = bool(lower <= actual <= upper)

        # Miscoverage indicator: err_t = 1 if missed, 0 if covered
        err_t = 0 if is_cov else 1
        self._history.append(err_t)

        if err_t == 1:
            self._consecutive_misses += 1
        else:
            self._consecutive_misses = 0

        # ACI gradient update:
        # If missed (err=1), alpha_{t+1} decreases -> wider intervals next time
        # If covered (err=0), alpha_{t+1} increases -> narrower intervals
        self.alpha_t = float(
            np.clip(self.alpha_t + self.gamma * (self.target_alpha - err_t), 0.005, 0.99)
        )

        recent_err_rate = float(np.mean(list(self._history))) if self._history else 0.0
        regime_shift = bool(
            (
                len(self._history) >= min(5, self.window_size)
                and recent_err_rate >= self.shift_threshold
            )
            or self._consecutive_misses >= 3
        )

        msg = f"Step {self.step}: Miscoverage rate = {recent_err_rate:.2f} (target alpha = {self.target_alpha:.2f})."
        if regime_shift:
            msg += (
                f" REGIME SHIFT DETECTED: Consecutive misses = {self._consecutive_misses}, "
                f"recent error rate = {recent_err_rate:.2f} exceeds threshold {self.shift_threshold:.2f}."
            )

        interval = ConformalInterval(
            point_prediction=point_prediction,
            lower_bound=lower,
            upper_bound=upper,
            interval_width=upper - lower,
            nominal_coverage=self.nominal_coverage,
            is_covered=is_cov,
        )

        state = ACIState(
            step=self.step,
            alpha_t=self.alpha_t,
            current_width=upper - lower,
            recent_miscoverage_rate=recent_err_rate,
            regime_shift_detected=regime_shift,
            diagnostic_message=msg,
        )

        return interval, state


class UQParetoCandidate(BaseModel):
    """Specification of an evaluated model's Uncertainty Quantification performance profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Identifier of model / UQ configuration.")
    sharpness: float = Field(
        description="Average interval width across evaluations (lower is better)."
    )
    empirical_coverage: float = Field(
        description="Observed empirical interval coverage rate in [0.0, 1.0] (higher is better)."
    )
    calibration_error: float = Field(
        description="ACE or Continuous Ranked Probability Score (lower is better)."
    )
    latency_ms: float = Field(
        description="Compute latency or sampling runtime overhead in milliseconds (lower is better)."
    )
    extra_metrics: dict[str, float] = Field(default_factory=dict)


def compute_uq_pareto_front(
    candidates: Sequence[UQParetoCandidate],
) -> ParetoFrontier:
    """Evaluate UQ trade-offs across candidates along the multi-objective Pareto frontier.

    Trade-off Dimensions:
        1. sharpness (minimize interval width)
        2. empirical_coverage (maximize coverage)
        3. calibration_error (minimize ACE / CRPS)
        4. latency_ms (minimize computational overhead)

    Epistemic Grounding:
        As demonstrated by Mucsányi et al. (2024), pure disentanglement of aleatoric and epistemic
        uncertainty is fundamentally underdetermined without restrictive structural priors.
        Engineers must treat UQ selection as a multi-criteria decision over Pareto-optimal candidates.
    """
    scores: dict[str, dict[str, float]] = {}
    for cand in candidates:
        scores[cand.name] = {
            "sharpness": cand.sharpness,
            "empirical_coverage": cand.empirical_coverage,
            "calibration_error": cand.calibration_error,
            "latency_ms": cand.latency_ms,
            **cand.extra_metrics,
        }

    objectives = [
        ObjectiveSpec(metric_name="sharpness", direction=ObjectiveDirection.MINIMIZE),
        ObjectiveSpec(metric_name="empirical_coverage", direction=ObjectiveDirection.MAXIMIZE),
        ObjectiveSpec(metric_name="calibration_error", direction=ObjectiveDirection.MINIMIZE),
        ObjectiveSpec(metric_name="latency_ms", direction=ObjectiveDirection.MINIMIZE),
    ]

    return compute_pareto_frontier(scores=scores, objectives=objectives)
