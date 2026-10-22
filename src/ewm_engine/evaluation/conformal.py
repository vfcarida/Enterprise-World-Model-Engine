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
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    ParetoFrontier,
    compute_pareto_frontier,
)

FloatSequence = Sequence[float] | np.ndarray[Any, np.dtype[Any]]


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
    """Aggregate audit of conformal interval predictor performance.

    Epistemic Guardrail (Marginal vs. Conditional Coverage):
        Split conformal prediction guarantees finite-sample **marginal** coverage
        $\\mathbb{P}(Y_{n+1} \\in \\hat{C}(X_{n+1})) \\ge 1 - \\alpha$ over the joint draw of
        calibration and test points under exchangeability. It NEVER guarantees exact
        pointwise conditional coverage $\\mathbb{P}(Y_{n+1} \\in \\hat{C}(x) \\mid X_{n+1}=x) \\ge 1 - \\alpha$
        without distributional assumptions (Foygel Barber et al., 2021). Always inspect
        ``slice_coverages`` and ``worst_slice_coverage`` to detect conditional failure regions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal_coverage: float
    empirical_coverage: float
    coverage_gap: float
    mean_interval_width: float
    median_interval_width: float
    guarantee_satisfied: bool = Field(
        description="True if empirical MARGINAL coverage >= nominal_coverage - tolerance.",
    )
    total_evaluations: int
    coverage_scope: str = Field(
        default="marginal",
        description="Explicit scope of coverage guarantee ('marginal', NEVER 'conditional').",
    )
    calibration_population: str = Field(
        default="exchangeable_calibration_holdout",
        description="Description of the calibration population defining the reference distribution.",
    )
    calibration_sample_size: int = Field(
        default=0,
        description="Number of calibration samples n_cal used to estimate conformal quantile q_hat.",
    )
    finite_sample_marginal_bounds: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Theoretical finite-sample marginal coverage bounds [1 - alpha, 1 - alpha + 1/(n_cal + 1)] "
            "under exchangeability and continuous non-conformity scores (Lei et al., 2018)."
        ),
    )
    slice_coverages: dict[str, float] = Field(
        default_factory=dict,
        description="Empirical coverage stratified by subgroup/regime slice to expose conditional under-coverage.",
    )
    worst_slice_coverage: float | None = Field(
        default=None,
        description="Minimum empirical coverage observed across evaluated slices.",
    )
    slice_coverage_gap_warning: bool = Field(
        default=False,
        description="True if any subgroup slice under-covers nominal_coverage - tolerance (conditional miscoverage).",
    )
    assumptions: tuple[str, ...] = Field(
        default_factory=lambda: (
            "Exchangeability of calibration and test pairs (X_i, Y_i) under the reference distribution (Vovk et al., 2005).",
            "Guarantee is strictly MARGINAL over (X, Y); conditional coverage P(Y in C(X) | X=x) is not guaranteed distribution-free (Barber et al., 2021).",
        ),
    )
    epistemic_note: str = Field(
        default=(
            "Conformal prediction provides finite-sample distribution-free MARGINAL validity under exchangeability "
            "(never conditional validity). Under non-stationary dynamics or regime shifts, exchangeability fails; "
            "use Adaptive Conformal Inference (ACI) and audit slice_coverages."
        ),
    )


class ConformalShiftExperimentResult(BaseModel):
    """Explicit evaluation of conformal coverage under distribution shift vs. exchangeability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal_coverage: float
    calibration_sample_size: int
    in_distribution_empirical_coverage: float
    shifted_split_conformal_coverage: float
    shifted_aci_empirical_coverage: float
    coverage_drop_under_shift: float = Field(
        description="Drop in split-conformal coverage from in-distribution to shifted test set."
    )
    exchangeability_violated: bool
    aci_regime_shift_detected: bool
    slice_coverages: dict[str, float]
    epistemic_conclusion: str


class ConformalIntervalPredictor:
    """Split Conformal / CQR-style interval predictor providing distribution-free marginal coverage.

    Calibrates non-conformity scores on a hold-out calibration set:
        q_hat = Quantile_{ceil((n+1)(1-alpha)) / n}(scores)
    yielding finite-sample marginal coverage P(Y in C(X)) >= 1 - alpha under exchangeability.
    """

    def __init__(
        self,
        nominal_coverage: float = 0.90,
        normalized: bool = False,
        calibration_population: str = "exchangeable_calibration_holdout",
    ) -> None:
        """Initialize conformal interval predictor.

        Args:
            nominal_coverage: Target confidence level 1 - alpha (e.g. 0.90 for 90% coverage).
            normalized: If True, uses normalized non-conformity scores |y - mu| / sigma
                for locally adaptive interval widths (CQR style).
            calibration_population: Identifier of the calibration population.
        """
        if not 0.0 < nominal_coverage < 1.0:
            raise ValueError(f"nominal_coverage must be in (0, 1), got {nominal_coverage}")
        self.nominal_coverage = float(nominal_coverage)
        self.alpha = float(1.0 - nominal_coverage)
        self.normalized = normalized
        self.calibration_population = calibration_population
        self.conformal_quantile: float | None = None
        self._n_calibrated: int = 0

    def calibrate(
        self,
        actuals: FloatSequence,
        predictions: FloatSequence,
        uncertainties: FloatSequence | None = None,
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

        # Exact finite-sample order statistic index k = ceil((n + 1)(1 - alpha))
        # Using 'higher' / order-statistic indexing ensures exact finite-sample validity
        # P(Y_{n+1} in C(X_{n+1})) >= 1 - alpha under exchangeability (Vovk et al., 2005; Lei et al., 2018).
        sorted_scores = np.sort(scores)
        k_idx = min(n - 1, max(0, int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1))
        q_hat = float(sorted_scores[k_idx])

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
        actuals: FloatSequence,
        predictions: FloatSequence,
        uncertainties: FloatSequence | None = None,
        tolerance: float = 0.05,
        slice_labels: Sequence[str] | None = None,
    ) -> ConformalEvaluationSummary:
        """Audit conformal predictor empirical marginal and slice coverage and sharpness."""
        n = min(len(actuals), len(predictions))
        fs_bounds = (
            (
                self.nominal_coverage,
                min(1.0, self.nominal_coverage + 1.0 / (self._n_calibrated + 1)),
            )
            if self._n_calibrated > 0
            else None
        )
        if n == 0:
            return ConformalEvaluationSummary(
                nominal_coverage=self.nominal_coverage,
                empirical_coverage=0.0,
                coverage_gap=-self.nominal_coverage,
                mean_interval_width=0.0,
                median_interval_width=0.0,
                guarantee_satisfied=False,
                total_evaluations=0,
                calibration_population=self.calibration_population,
                calibration_sample_size=self._n_calibrated,
                finite_sample_marginal_bounds=fs_bounds,
            )

        covered_count = 0
        widths: list[float] = []
        slice_hits: dict[str, list[int]] = {}

        for i in range(n):
            unc = uncertainties[i] if uncertainties is not None else None
            interval = self.predict_interval(
                prediction=predictions[i],
                uncertainty=unc,
                actual=actuals[i],
            )
            hit = 1 if interval.is_covered else 0
            covered_count += hit
            widths.append(interval.interval_width)
            if slice_labels is not None and i < len(slice_labels):
                slice_hits.setdefault(str(slice_labels[i]), []).append(hit)

        emp_cov = float(covered_count / n)
        cov_gap = float(emp_cov - self.nominal_coverage)
        mean_w = float(np.mean(widths))
        median_w = float(np.median(widths))

        slice_cov_map = {
            s_name: float(np.mean(hits)) for s_name, hits in sorted(slice_hits.items()) if hits
        }
        worst_slice = min(slice_cov_map.values()) if slice_cov_map else None
        slice_warn = bool(
            worst_slice is not None and worst_slice < (self.nominal_coverage - tolerance)
        )

        return ConformalEvaluationSummary(
            nominal_coverage=self.nominal_coverage,
            empirical_coverage=emp_cov,
            coverage_gap=cov_gap,
            mean_interval_width=mean_w,
            median_interval_width=median_w,
            guarantee_satisfied=bool(emp_cov >= (self.nominal_coverage - tolerance)),
            total_evaluations=n,
            calibration_population=self.calibration_population,
            calibration_sample_size=self._n_calibrated,
            finite_sample_marginal_bounds=fs_bounds,
            slice_coverages=slice_cov_map,
            worst_slice_coverage=worst_slice,
            slice_coverage_gap_warning=slice_warn,
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


def evaluate_conformal_under_shift(
    cal_actuals: FloatSequence,
    cal_predictions: FloatSequence,
    id_test_actuals: FloatSequence,
    id_test_predictions: FloatSequence,
    shifted_test_actuals: FloatSequence,
    shifted_test_predictions: FloatSequence,
    *,
    nominal_coverage: float = 0.90,
    aci_gamma: float = 0.08,
) -> ConformalShiftExperimentResult:
    """Run an explicit distribution-shift experiment comparing static split conformal vs. online ACI.

    Demonstrates three statistical facts required for epistemic honesty:
        1. Under in-distribution exchangeability, split conformal meets its finite-sample marginal guarantee.
        2. Under covariate/regime/mechanism shift (exchangeability violated), static split conformal
           loses coverage guarantee and can severely under-cover.
        3. Adaptive Conformal Inference (ACI; Gibbs & Candès, 2021) detects the miscoverage burst
           and dynamically widens intervals to recover long-run coverage.
    """
    predictor = ConformalIntervalPredictor(
        nominal_coverage=nominal_coverage,
        calibration_population="in_distribution_calibration_holdout",
    )
    q_hat = predictor.calibrate(actuals=cal_actuals, predictions=cal_predictions)

    id_summary = predictor.evaluate(actuals=id_test_actuals, predictions=id_test_predictions)
    shifted_summary = predictor.evaluate(
        actuals=shifted_test_actuals, predictions=shifted_test_predictions
    )

    # Evaluate combined stream with slice labels ("in_distribution" vs "shifted_regime")
    combined_actuals = [*list(id_test_actuals), *list(shifted_test_actuals)]
    combined_preds = [*list(id_test_predictions), *list(shifted_test_predictions)]
    combined_slices = ["in_distribution"] * len(id_test_actuals) + ["shifted_regime"] * len(
        shifted_test_actuals
    )
    combined_summary = predictor.evaluate(
        actuals=combined_actuals,
        predictions=combined_preds,
        slice_labels=combined_slices,
    )

    # Run online ACI across the shifted regime initialized with base_half_width = q_hat * (1 - nominal_coverage)
    aci = AdaptiveConformalInference(
        nominal_coverage=nominal_coverage,
        gamma=aci_gamma,
        base_half_width=max(1e-6, q_hat * (1.0 - nominal_coverage)),
    )
    aci_hits = 0
    shift_detected_any = False
    n_shift = min(len(shifted_test_actuals), len(shifted_test_predictions))
    for i in range(n_shift):
        interval, state = aci.predict_and_update(
            point_prediction=float(shifted_test_predictions[i]),
            actual=float(shifted_test_actuals[i]),
        )
        if interval.is_covered:
            aci_hits += 1
        if state.regime_shift_detected:
            shift_detected_any = True

    aci_cov = float(aci_hits / n_shift) if n_shift > 0 else 0.0
    cov_drop = float(id_summary.empirical_coverage - shifted_summary.empirical_coverage)
    exch_violated = bool(shifted_summary.empirical_coverage < (nominal_coverage - 0.05))

    return ConformalShiftExperimentResult(
        nominal_coverage=nominal_coverage,
        calibration_sample_size=len(cal_actuals),
        in_distribution_empirical_coverage=id_summary.empirical_coverage,
        shifted_split_conformal_coverage=shifted_summary.empirical_coverage,
        shifted_aci_empirical_coverage=aci_cov,
        coverage_drop_under_shift=cov_drop,
        exchangeability_violated=exch_violated,
        aci_regime_shift_detected=shift_detected_any,
        slice_coverages=combined_summary.slice_coverages,
        epistemic_conclusion=(
            f"Split conformal achieved {id_summary.empirical_coverage:.3f} marginal coverage in-distribution "
            f"(target {nominal_coverage:.2f}), dropped to {shifted_summary.empirical_coverage:.3f} under shift "
            f"(exchangeability violated={exch_violated}), while online ACI recovered {aci_cov:.3f} coverage "
            f"(regime_shift_detected={shift_detected_any})."
        ),
    )
