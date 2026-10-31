"""Learned Dynamics Calibration, Conformal Prediction, OOD Detection, and Abstention.

Wraps neural and empirical learned dynamics models in distribution-free conformal
prediction intervals (Vovk et al., 2005), exact ensemble epistemic variance
($\\text{Var}(\\mu) = \\frac{1}{N} \\sum_{i=1}^N (\\mu_i - \\bar{\\mu})^2$), and multivariate
Mahalanobis out-of-distribution (OOD) detection. Raises `AbstainException` when
predictive uncertainty or OOD distance exceeds configured operational thresholds.

Scientific Grounding:
    - Vovk, V., Gammerman, A., & Shafer, G. (2005): "Algorithmic Learning in a Random World",
      Springer. (Split conformal prediction & finite-sample valid coverage bounds).
    - Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017): "Simple and Scalable
      Predictive Uncertainty Estimation using Deep Ensembles", NeurIPS.
    - Lee, K., Lee, K., Lee, H., & Shin, J. (2018): "A Simple Unified Framework for
      Detecting Out-of-Distribution Samples and Adversarial Attacks", NeurIPS.
    - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017): "On Calibration of
      Modern Neural Networks", ICML.
    - Kuleshov, V., Fenner, N., & Ermon, S. (2018): "Accurate Uncertainties for Deep
      Learning Using Calibrated Regression", ICML.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import DynamicsModel
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.dynamics.router import (
    CalibratedPrediction,
    NumericalInstabilityError,
    validate_learned_prediction_no_nan,
)


def _normal_ppf_two_sided(nominal_coverage: float) -> float:
    """Return the two-sided standard normal critical value $z_{1 - \\alpha/2}$ for coverage $1 - \\alpha$."""
    cov = min(max(float(nominal_coverage), 1e-6), 1.0 - 1e-6)
    # Rational approximation (Abramowitz & Stegun 26.2.23) for Phi^{-1}(p) where p = (1 + cov) / 2
    p = 0.5 * (1.0 + cov)
    q = 1.0 - p
    t = math.sqrt(-2.0 * math.log(q))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - ((c2 * t + c1) * t + c0) / (((d3 * t + d2) * t + d1) * t + 1.0)
    return max(0.0, float(z))


def compute_ensemble_variance(
    member_predictions: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute ensemble mean $\\bar{\\mu}$ and exact population variance $\\text{Var}(\\mu)$.

    Enforces the exact mathematical identity required by EWM-003 Section 8:
    $$\\text{Var}(\\mu) = \\frac{1}{N} \\sum_{i=1}^N (\\mu_i - \\bar{\\mu})^2, \\quad \\bar{\\mu} = \\frac{1}{N} \\sum_{i=1}^N \\mu_i$$

    Args:
        member_predictions: Non-empty sequence of resource-delta mappings from $N$ ensemble members.

    Returns:
        Tuple `(mean_deltas, variance_deltas)` indexed by resource ID.
    """
    n_members = len(member_predictions)
    if n_members == 0:
        return {}, {}

    keys: list[str] = sorted({str(k) for pred in member_predictions for k in pred})
    means: dict[str, float] = {}
    variances: dict[str, float] = {}

    for key in keys:
        vals = np.array(
            [float(pred.get(key, 0.0)) for pred in member_predictions], dtype=np.float64
        )
        if not np.all(np.isfinite(vals)):
            raise NumericalInstabilityError(
                f"Ensemble member emitted NaN or Inf for target {key!r}: {vals.tolist()}"
            )
        mu_bar = float(np.sum(vals) / float(n_members))
        var_mu = float(np.sum((vals - mu_bar) ** 2) / float(n_members))
        means[key] = mu_bar
        variances[key] = var_mu

    return means, variances


class ConfidenceInterval(BaseModel):
    """Statistically calibrated confidence interval and epistemic uncertainty summary.

    Supports direct threshold comparison (`if confidence_interval > threshold:`) as
    specified in EWM-003 Section 6.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: dict[str, float] = Field(
        default_factory=dict,
        description="Lower calibrated interval bound per resource ID.",
    )
    upper: dict[str, float] = Field(
        default_factory=dict,
        description="Upper calibrated interval bound per resource ID.",
    )
    half_width: dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated half-width (radius) per resource ID.",
    )
    width: dict[str, float] = Field(
        default_factory=dict,
        description="Total confidence interval width (upper - lower) per resource ID.",
    )
    max_width: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum interval width across all predicted dimensions.",
    )
    max_half_width: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum half-width across all predicted dimensions.",
    )
    ensemble_variance: dict[str, float] = Field(
        default_factory=dict,
        description="Ensemble variance Var(mu) = (1/N) sum (mu_i - mu_bar)^2 per resource ID.",
    )
    max_ensemble_variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum ensemble variance across all predicted dimensions.",
    )
    ood_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Multivariate Mahalanobis OOD distance D_M(x) for the input state.",
    )
    nominal_coverage: float = Field(
        default=0.90,
        gt=0.0,
        lt=1.0,
        description="Target coverage probability (1 - alpha) in (0, 1).",
    )

    @property
    def effective_uncertainty(self) -> float:
        """Scalar uncertainty statistic used for threshold comparisons (`CI > threshold`)."""
        return max(
            float(self.max_width),
            float(self.max_ensemble_variance),
            float(self.ood_score),
        )

    def exceeds(self, threshold: float) -> bool:
        """Return True if interval width, ensemble variance, or OOD score exceeds `threshold`."""
        thresh = float(threshold)
        if math.isinf(thresh) and thresh > 0:
            return False
        return bool(self.effective_uncertainty > thresh)

    def __gt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            return self.exceeds(float(other))
        if isinstance(other, ConfidenceInterval):
            return self.effective_uncertainty > other.effective_uncertainty
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            thresh = float(other)
            if math.isinf(thresh) and thresh > 0:
                return False
            return bool(self.effective_uncertainty >= thresh)
        if isinstance(other, ConfidenceInterval):
            return self.effective_uncertainty >= other.effective_uncertainty
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            thresh = float(other)
            if math.isinf(thresh) and thresh > 0:
                return True
            return bool(self.effective_uncertainty < thresh)
        if isinstance(other, ConfidenceInterval):
            return self.effective_uncertainty < other.effective_uncertainty
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, (int, float)):
            thresh = float(other)
            if math.isinf(thresh) and thresh > 0:
                return True
            return bool(self.effective_uncertainty <= thresh)
        if isinstance(other, ConfidenceInterval):
            return self.effective_uncertainty <= other.effective_uncertainty
        return NotImplemented


class AbstainException(Exception):
    """Raised when a learned model abstains due to OOD inputs or `ConfidenceInterval > threshold`."""

    def __init__(
        self,
        message: str,
        *,
        confidence_interval: ConfidenceInterval | None = None,
        ood_score: float = 0.0,
        ensemble_variance: float = 0.0,
        threshold: float = float("inf"),
        trigger: str = "confidence_interval_exceeded",
    ) -> None:
        super().__init__(message)
        self.reason = message
        self.confidence_interval = confidence_interval
        self.ood_score = float(ood_score)
        self.ensemble_variance = float(ensemble_variance)
        self.threshold = float(threshold)
        self.trigger = trigger


class DriftStatusReport(BaseModel):
    """Online covariate and epistemic drift diagnostic report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drift_detected: bool = Field(
        description="True if sliding-window Mahalanobis score exceeds drift threshold."
    )
    window_mean_ood_score: float = Field(
        description="Mean Mahalanobis distance across the recent sliding window."
    )
    window_ood_rate: float = Field(
        description="Fraction of recent inputs in the sliding window exceeding ood_threshold."
    )
    sample_count: int = Field(description="Number of recent samples in the drift window.")


class MahalanobisOODAndDriftDetector:
    """Multivariate Mahalanobis distance and ensemble-variance OOD & drift detector.

    Computes the regularized Mahalanobis distance:
    $$D_M(x) = \\sqrt{(x - \\hat{\\mu})^\\top (\\hat{\\Sigma} + \\lambda I)^{-1} (x - \\hat{\\mu})}$$
    over state-action feature vectors, and tracks sliding-window covariate drift.
    All thresholds default to `float("inf")` (disabled) to preserve legacy behavior
    unless explicitly turned on (EWM-003 Section 13).
    """

    def __init__(
        self,
        *,
        feature_keys: Sequence[str] | None = None,
        ood_threshold: float = float("inf"),
        variance_threshold: float = float("inf"),
        regularization: float = 1e-5,
        drift_window_size: int = 50,
    ) -> None:
        self.feature_keys: tuple[str, ...] | None = (
            tuple(feature_keys) if feature_keys is not None else None
        )
        self.ood_threshold = float(ood_threshold)
        self.variance_threshold = float(variance_threshold)
        self.regularization = float(regularization)
        self.drift_window_size = max(1, int(drift_window_size))

        self._mean: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._precision: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._std: np.ndarray[Any, np.dtype[np.float64]] | None = None
        self._in_dist_q99: float = 1.0
        self._is_fitted: bool = False
        self._recent_scores: deque[float] = deque(maxlen=self.drift_window_size)

    @property
    def is_fitted(self) -> bool:
        """Return True if the detector has been fitted on in-distribution reference data."""
        return self._is_fitted

    def extract_features(
        self,
        state: WorldState | Mapping[str, float] | Sequence[float] | np.ndarray[Any, Any],
        actions: Sequence[Action] = (),
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Extract a canonical 1D float64 feature vector from a `WorldState` or numeric array."""
        if isinstance(state, np.ndarray):
            vec = np.asarray(state, dtype=np.float64).ravel()
        elif isinstance(state, Sequence) and not isinstance(state, (str, bytes)):
            vec = np.asarray(list(state), dtype=np.float64).ravel()
        elif isinstance(state, Mapping):
            keys = (
                list(self.feature_keys)
                if self.feature_keys is not None
                else sorted(str(k) for k in state)
            )
            vec = np.array([float(state.get(k, 0.0)) for k in keys], dtype=np.float64)
        elif isinstance(state, WorldState):
            keys = (
                list(self.feature_keys)
                if self.feature_keys is not None
                else sorted(state.resources.keys())
            )
            feats: list[float] = []
            for k in keys:
                res = state.resources.get(k)
                feats.append(float(res.current) if res is not None else 0.0)
            if actions:
                act_sum = 0.0
                for act in actions:
                    for pv in act.parameters.values():
                        if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                            act_sum += float(pv)
                if self.feature_keys is not None and "__action_sum__" in self.feature_keys:
                    feats.append(act_sum)
            vec = np.array(feats, dtype=np.float64)
        else:
            raise TypeError(
                f"Unsupported state input for OOD feature extraction: {type(state).__name__}"
            )

        if not np.all(np.isfinite(vec)):
            raise NumericalInstabilityError(
                f"Non-finite state features encountered during OOD evaluation: {vec.tolist()}"
            )
        return vec

    def fit(
        self,
        reference_data: (
            TransitionDataset
            | Sequence[WorldState]
            | Sequence[Sequence[float]]
            | np.ndarray[Any, Any]
        ),
        *,
        tune_quantile: float | None = None,
        safety_margin: float = 1.25,
    ) -> MahalanobisOODAndDriftDetector:
        """Fit empirical mean $\\hat{\\mu}$ and regularized precision matrix $(\\hat{\\Sigma} + \\lambda I)^{-1}$.

        Args:
            reference_data: In-distribution `TransitionDataset`, sequence of `WorldState`s,
                or 2D feature matrix of shape `(n_samples, n_features)`.
            tune_quantile: Optional quantile in `(0, 1)` (e.g., `0.99`) to automatically tune
                `self.ood_threshold` from the empirical in-distribution Mahalanobis distribution.
            safety_margin: Multiplicative safety factor applied when `tune_quantile` is specified.
        """
        rows: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        if isinstance(reference_data, TransitionDataset):
            if self.feature_keys is None and len(reference_data) > 0:
                self.feature_keys = tuple(sorted(reference_data[0].state.resources.keys()))
            for sample in reference_data:
                rows.append(self.extract_features(sample.state, sample.actions))
        elif isinstance(reference_data, np.ndarray):
            arr = np.asarray(reference_data, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            for i in range(arr.shape[0]):
                rows.append(arr[i])
        else:
            seq: list[Any] = list(reference_data)
            if len(seq) > 0 and isinstance(seq[0], WorldState) and self.feature_keys is None:
                self.feature_keys = tuple(sorted(seq[0].resources.keys()))
            for item in seq:
                rows.append(self.extract_features(item))

        if not rows:
            raise ValueError("Cannot fit MahalanobisOODAndDriftDetector on empty reference data.")

        x_mat = np.vstack(rows)
        n_samples, n_dims = x_mat.shape
        self._mean = np.mean(x_mat, axis=0)
        self._std = np.maximum(np.std(x_mat, axis=0), 1e-6)

        if n_samples > 1:
            centered = x_mat - self._mean
            cov = (centered.T @ centered) / float(n_samples)
        else:
            cov = np.zeros((n_dims, n_dims), dtype=np.float64)

        reg_cov = cov + np.eye(n_dims, dtype=np.float64) * self.regularization
        self._precision = np.linalg.pinv(reg_cov)
        self._is_fitted = True

        # Compute empirical in-distribution Mahalanobis distances
        diffs = x_mat - self._mean
        quad = np.einsum("ij,jk,ik->i", diffs, self._precision, diffs)
        dists = np.sqrt(np.maximum(0.0, quad))
        self._in_dist_q99 = max(1e-6, float(np.quantile(dists, 0.99)))

        if tune_quantile is not None:
            q_val = min(max(float(tune_quantile), 0.50), 0.9999)
            emp_q = float(np.quantile(dists, q_val))
            # Theoretical chi-square tail fallback for small sample sizes
            chi2_approx = math.sqrt(float(n_dims)) + 2.5758 * math.sqrt(2.0 * float(n_dims))
            self.ood_threshold = max(emp_q * float(safety_margin), chi2_approx * 0.85, 3.0)

        return self

    def mahalanobis_distance(
        self,
        state: WorldState | Mapping[str, float] | Sequence[float] | np.ndarray[Any, Any],
        actions: Sequence[Action] = (),
    ) -> float:
        """Compute the Mahalanobis distance $D_M(x)$ for a state or feature vector."""
        vec = self.extract_features(state, actions)
        if not self._is_fitted or self._mean is None or self._precision is None:
            return 0.0
        if vec.shape[0] != self._mean.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch in OOD detector: expected {self._mean.shape[0]}, "
                f"got {vec.shape[0]}"
            )
        diff = vec - self._mean
        max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
        if max_abs == 0.0:
            dist = 0.0
        else:
            u = diff / max_abs
            quad_u = float(u.T @ self._precision @ u)
            if not math.isfinite(quad_u) or quad_u <= 0.0:
                unit_dist = float(np.linalg.norm(u))
            else:
                unit_dist = math.sqrt(quad_u)
            dist = max_abs * unit_dist
            if not math.isfinite(dist):
                dist = float("inf")
        self._recent_scores.append(dist)
        return dist

    def is_ood(
        self,
        state: WorldState | Mapping[str, float] | Sequence[float] | np.ndarray[Any, Any],
        actions: Sequence[Action] = (),
        *,
        ensemble_variance: float = 0.0,
    ) -> tuple[bool, float, str]:
        """Evaluate whether `state` or `ensemble_variance` exceeds configured OOD thresholds.

        Returns:
            Tuple `(is_ood_flag, mahalanobis_distance, trigger_reason)`.
        """
        dist = self.mahalanobis_distance(state, actions)
        if not math.isinf(self.ood_threshold) and dist > self.ood_threshold:
            return (
                True,
                dist,
                f"mahalanobis_distance ({dist:.4f}) > ood_threshold ({self.ood_threshold:.4f})",
            )
        if not math.isinf(self.variance_threshold) and ensemble_variance > self.variance_threshold:
            return (
                True,
                dist,
                f"ensemble_variance ({ensemble_variance:.4f}) > variance_threshold ({self.variance_threshold:.4f})",
            )
        return False, dist, "in_distribution"

    def drift_report(self, drift_score_threshold: float | None = None) -> DriftStatusReport:
        """Evaluate online covariate drift across the sliding window of recent states."""
        if not self._recent_scores:
            return DriftStatusReport(
                drift_detected=False,
                window_mean_ood_score=0.0,
                window_ood_rate=0.0,
                sample_count=0,
            )
        scores = list(self._recent_scores)
        mean_score = float(np.mean(scores))
        eff_thresh = (
            float(drift_score_threshold)
            if drift_score_threshold is not None
            else self.ood_threshold
        )
        if math.isinf(eff_thresh):
            ood_rate = 0.0
            drifted = False
        else:
            ood_rate = float(sum(1 for s in scores if s > eff_thresh)) / float(len(scores))
            drifted = bool(mean_score > eff_thresh or ood_rate >= 0.25)
        return DriftStatusReport(
            drift_detected=drifted,
            window_mean_ood_score=mean_score,
            window_ood_rate=ood_rate,
            sample_count=len(scores),
        )


class CalibrationEvaluationReport(BaseModel):
    """Synthetic and held-out validation calibration report logging Expected Calibration Error (ECE)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expected_calibration_error: float = Field(
        ge=0.0,
        description="Expected Calibration Error (ECE) of the calibrated predictor.",
    )
    uncalibrated_ece: float = Field(
        ge=0.0,
        description="Baseline Expected Calibration Error prior to conformal/temperature calibration.",
    )
    empirical_coverage_at_nominal: float = Field(
        ge=0.0,
        le=1.0,
        description="Empirical coverage achieved at the predictor's nominal coverage (1 - alpha).",
    )
    nominal_coverage: float = Field(
        gt=0.0,
        lt=1.0,
        description="Target nominal coverage probability (1 - alpha).",
    )
    num_samples: int = Field(
        ge=0,
        description="Number of validation samples evaluated.",
    )
    num_bins: int = Field(
        ge=1,
        description="Number of confidence levels / bins used to compute ECE.",
    )
    bin_nominal_confidences: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Evaluated confidence levels p_m in (0, 1).",
    )
    bin_empirical_accuracies: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Empirical coverage / accuracy observed in each confidence bin.",
    )
    bin_counts: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Sample count per calibration bin.",
    )


def compute_expected_calibration_error(
    confidences: Sequence[float] | np.ndarray[Any, Any],
    accuracies: Sequence[float | bool] | np.ndarray[Any, Any],
    *,
    num_bins: int = 10,
) -> tuple[float, dict[str, Any]]:
    """Compute Expected Calibration Error (ECE; Naeini et al., 2015; Guo et al., 2017).

    $$\\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N} \\left| \\text{acc}(B_m) - \\text{conf}(B_m) \\right|$$

    Args:
        confidences: Predicted confidence / probability scores in `[0, 1]`.
        accuracies: Binary indicators (`0`/`1` or `bool`) of whether the event/interval covered truth.
        num_bins: Number of uniform partition bins across `[0, 1]`.

    Returns:
        Tuple `(ece_value, diagnostics_dict)`.
    """
    conf_arr = np.asarray(confidences, dtype=np.float64).ravel()
    acc_arr = np.asarray(accuracies, dtype=np.float64).ravel()
    if conf_arr.shape[0] != acc_arr.shape[0]:
        raise ValueError(
            f"Length mismatch between confidences ({conf_arr.shape[0]}) and accuracies ({acc_arr.shape[0]})."
        )
    n_total = conf_arr.shape[0]
    if n_total == 0:
        return 0.0, {"bin_confidences": [], "bin_accuracies": [], "bin_counts": []}

    bins = np.linspace(0.0, 1.0, max(2, int(num_bins) + 1))
    ece = 0.0
    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []

    for i in range(len(bins) - 1):
        lo, hi = float(bins[i]), float(bins[i + 1])
        if i == 0:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr > lo) & (conf_arr <= hi)
        count = int(np.sum(mask))
        bin_counts.append(count)
        if count > 0:
            avg_conf = float(np.mean(conf_arr[mask]))
            avg_acc = float(np.mean(acc_arr[mask]))
            ece += (float(count) / float(n_total)) * abs(avg_acc - avg_conf)
            bin_confs.append(avg_conf)
            bin_accs.append(avg_acc)
        else:
            bin_confs.append(0.5 * (lo + hi))
            bin_accs.append(0.0)

    return float(ece), {
        "bin_confidences": bin_confs,
        "bin_accuracies": bin_accs,
        "bin_counts": bin_counts,
    }


PredictorModelType = (
    DynamicsModel
    | Callable[[WorldState, Sequence[Action]], Mapping[str, float] | CalibratedPrediction]
    | Callable[[WorldState], Mapping[str, float] | CalibratedPrediction]
)


class ConformalPredictorWrapper:
    """Conformal prediction & ensemble uncertainty wrapper for learned world-model dynamics.

    Implements the EWM-003 Section 6 contract:
        `Predictor.predict(state)` -> `(CalibratedPrediction, ConfidenceInterval)`
        If `ConfidenceInterval > threshold`, raises `AbstainException`.

    Supports:
    1. Single learned models (`LinearResidualDynamics`, `DynamicsModel`, or callables) and
       deep/bootstrap ensembles of $N$ models with exact variance
       $\\text{Var}(\\mu) = \\frac{1}{N} \\sum_{i=1}^N (\\mu_i - \\bar{\\mu})^2$.
    2. Distribution-free split conformal calibration (Vovk et al., 2005) on held-out
       calibration transitions with finite-sample quantile $\\lceil (n+1)(1-\\alpha) \\rceil / n$.
    3. Multivariate Mahalanobis OOD distance scaling and abstention gating.
    4. Thresholds defaulting to `float("inf")` (disabled) for legacy backwards compatibility.
    """

    def __init__(
        self,
        models: PredictorModelType | Sequence[PredictorModelType],
        *,
        ood_detector: MahalanobisOODAndDriftDetector | None = None,
        threshold: float = float("inf"),
        ood_threshold: float = float("inf"),
        variance_threshold: float = float("inf"),
        nominal_coverage: float = 0.90,
        default_half_width: float = 0.10,
        ood_uncertainty_scale: float = 0.25,
        seed: int = 42,
    ) -> None:
        if isinstance(models, Sequence):
            self.models: tuple[PredictorModelType, ...] = tuple(models)
        else:
            self.models = (models,)
        if len(self.models) == 0:
            raise ValueError("ConformalPredictorWrapper requires at least one learned model.")

        self.ood_detector = ood_detector
        self.threshold = float(threshold)
        self.ood_threshold = float(ood_threshold)
        self.variance_threshold = float(variance_threshold)
        self.nominal_coverage = float(nominal_coverage)
        self.default_half_width = float(default_half_width)
        self.ood_uncertainty_scale = float(ood_uncertainty_scale)
        self.seed = int(seed)

        self._conformal_quantiles: dict[str, float] = {}
        self._normalized_quantiles: dict[str, float] = {}
        self._residual_stds: dict[str, float] = {}
        self._calibration_temperature: float = 1.0
        self._is_calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        """Return True if conformal calibration has been executed on a calibration dataset."""
        return self._is_calibrated

    @property
    def conformal_quantiles(self) -> dict[str, float]:
        """Return a copy of the fitted split-conformal half-width quantiles per resource."""
        return dict(self._conformal_quantiles)

    def _invoke_single_member(
        self,
        member: PredictorModelType,
        state: WorldState,
        actions: Sequence[Action],
        structural_state: WorldState | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Evaluate a single ensemble member and extract `(mean_deltas, std_deltas)`."""
        base_state = structural_state if structural_state is not None else state
        if isinstance(member, DynamicsModel):
            rng = np.random.default_rng(self.seed)
            res = member.transition(
                state=base_state,
                actions=actions,
                exogenous_events=(),
                rng=rng,
            )
            target_res = getattr(member, "target_resource", None)
            deltas: dict[str, float] = {}
            if isinstance(target_res, str) and target_res in res.next_state.resources:
                r_curr = base_state.resources.get(target_res)
                curr_val = r_curr.current if r_curr is not None else 0.0
                deltas[target_res] = float(
                    res.next_state.get_resource(target_res).current - curr_val
                )
            else:
                for rid, r_next in res.next_state.resources.items():
                    r_curr = base_state.resources.get(rid)
                    curr_val = r_curr.current if r_curr is not None else 0.0
                    deltas[rid] = float(r_next.current - curr_val)
            std_val = float(getattr(member, "residual_std", 0.0))
            stds = {rid: max(0.0, std_val) for rid in deltas}
            return deltas, stds

        if callable(member):
            raw: Any
            try:
                raw = member(state, actions)  # type: ignore[call-arg]
            except TypeError:
                raw = member(state)  # type: ignore[call-arg]
            validated = validate_learned_prediction_no_nan(raw)
            return dict(validated.mean_deltas), dict(validated.std_deltas)

        raise TypeError(f"Unsupported learned model member type: {type(member).__name__}")

    def _predict_raw_ensemble(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        structural_state: WorldState | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Compute ensemble mean $\\bar{\\mu}$, ensemble variance $\\text{Var}(\\mu)$, and member aleatoric std."""
        member_means: list[dict[str, float]] = []
        member_stds: list[dict[str, float]] = []
        for member in self.models:
            m_deltas, s_deltas = self._invoke_single_member(
                member, state, actions, structural_state
            )
            member_means.append(m_deltas)
            member_stds.append(s_deltas)

        means, variances = compute_ensemble_variance(member_means)
        avg_stds: dict[str, float] = {}
        for k in means:
            s_vals = [m_s.get(k, 0.0) for m_s in member_stds]
            avg_stds[k] = float(np.mean(s_vals)) if s_vals else 0.0
        return means, variances, avg_stds

    def calibrate(
        self,
        calibration_dataset: TransitionDataset | Sequence[TransitionSample],
        *,
        alpha: float | None = None,
        fit_ood_detector: bool = True,
        tune_ood_quantile: float | None = None,
    ) -> dict[str, float]:
        """Calibrate split conformal prediction intervals (Vovk et al., 2005) on held-out data.

        Computes the finite-sample order statistic at index
        $k = \\lceil (n + 1)(1 - \\alpha) \\rceil$ over nonconformity scores
        $R_j = |\\Delta y_j - \\hat{\\mu}(x_j)|$ so marginal coverage is guaranteed $\\ge 1 - \\alpha$.

        Args:
            calibration_dataset: Held-out calibration transitions $(S_t, A_t, E_t, S_{t+1})$.
            alpha: Miscoverage rate $\\alpha \\in (0, 1)$. Defaults to `1.0 - self.nominal_coverage`.
            fit_ood_detector: If True and an OOD detector is attached (or created), fits it on
                calibration states.
            tune_ood_quantile: Optional quantile (e.g., `0.99`) to auto-tune the OOD threshold.

        Returns:
            Dictionary of calibrated conformal half-widths $\\hat{q}_{1-\\alpha}$ per resource ID.
        """
        samples = list(calibration_dataset)
        if not samples:
            raise ValueError("Cannot calibrate ConformalPredictorWrapper on an empty dataset.")

        target_cov = 1.0 - float(alpha) if alpha is not None else float(self.nominal_coverage)
        if not (0.0 < target_cov < 1.0):
            raise ValueError(f"Nominal coverage (1 - alpha) must be in (0, 1), got {target_cov}")
        self.nominal_coverage = target_cov

        if fit_ood_detector:
            if self.ood_detector is None:
                self.ood_detector = MahalanobisOODAndDriftDetector(
                    ood_threshold=self.ood_threshold,
                    variance_threshold=self.variance_threshold,
                )
            ds = (
                calibration_dataset
                if isinstance(calibration_dataset, TransitionDataset)
                else TransitionDataset(samples)
            )
            self.ood_detector.fit(ds, tune_quantile=tune_ood_quantile)
            if tune_ood_quantile is not None and math.isinf(self.ood_threshold):
                self.ood_threshold = self.ood_detector.ood_threshold

        abs_residuals_by_res: dict[str, list[float]] = {}
        norm_residuals_by_res: dict[str, list[float]] = {}
        raw_signed_by_res: dict[str, list[tuple[float, float]]] = {}

        for sample in samples:
            means, variances, avg_stds = self._predict_raw_ensemble(sample.state, sample.actions)
            for rid, pred_delta in means.items():
                if rid not in sample.state.resources or rid not in sample.next_state.resources:
                    continue
                true_delta = (
                    sample.next_state.get_resource(rid).current
                    - sample.state.get_resource(rid).current
                )
                abs_err = abs(true_delta - pred_delta)
                scale = math.sqrt(
                    max(variances.get(rid, 0.0), 0.0) + max(avg_stds.get(rid, 0.0), 1e-6) ** 2
                )
                abs_residuals_by_res.setdefault(rid, []).append(abs_err)
                norm_residuals_by_res.setdefault(rid, []).append(abs_err / max(scale, 1e-6))
                raw_signed_by_res.setdefault(rid, []).append((true_delta - pred_delta, scale))

        for rid, res_list in abs_residuals_by_res.items():
            n_cal = len(res_list)
            # Vovk et al. (2005) finite-sample adjusted rank: ceil((n + 1) * (1 - alpha))
            rank_idx = min(n_cal - 1, max(0, math.ceil((n_cal + 1) * target_cov) - 1))
            sorted_abs = sorted(res_list)
            sorted_norm = sorted(norm_residuals_by_res[rid])
            self._conformal_quantiles[rid] = float(sorted_abs[rank_idx])
            self._normalized_quantiles[rid] = float(sorted_norm[rank_idx])
            signed_errs = [pair[0] for pair in raw_signed_by_res[rid]]
            self._residual_stds[rid] = max(1e-6, float(np.std(signed_errs)))

        # Fit global temperature scaling factor for Gaussian predictive std alignment
        all_ratios: list[float] = []
        z_crit = max(1e-6, _normal_ppf_two_sided(target_cov))
        for rid, q_val in self._conformal_quantiles.items():
            base_std = self._residual_stds.get(rid, self.default_half_width)
            all_ratios.append(q_val / max(1e-6, z_crit * base_std))
        if all_ratios:
            self._calibration_temperature = max(1e-4, float(np.mean(all_ratios)))

        self._is_calibrated = True
        return dict(self._conformal_quantiles)

    def predict(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        structural_state: WorldState | None = None,
    ) -> tuple[CalibratedPrediction, ConfidenceInterval]:
        """Predict next-step resource deltas and calibrated `ConfidenceInterval`.

        Contract (EWM-003 Section 6):
            `Predictor.predict(state)` -> `(Prediction, ConfidenceInterval)`
            If `ConfidenceInterval > threshold` (or OOD/variance threshold exceeded),
            raises `AbstainException`.
        """
        # 1. Compute OOD Mahalanobis score and effective thresholds
        ood_score = 0.0
        if self.ood_detector is not None and self.ood_detector.is_fitted:
            ood_score = self.ood_detector.mahalanobis_distance(state, actions)

        eff_ood_thresh = (
            min(self.ood_threshold, self.ood_detector.ood_threshold)
            if self.ood_detector is not None
            else self.ood_threshold
        )
        eff_var_thresh = (
            min(self.variance_threshold, self.ood_detector.variance_threshold)
            if self.ood_detector is not None
            else self.variance_threshold
        )

        # 2. Evaluate ensemble mean and exact variance Var(mu) = (1/N) sum (mu_i - mu_bar)^2
        try:
            means, variances, avg_stds = self._predict_raw_ensemble(
                state=state,
                actions=actions,
                structural_state=structural_state,
            )
        except (NumericalInstabilityError, OverflowError) as exc:
            if not math.isinf(eff_ood_thresh) and ood_score > eff_ood_thresh:
                raise AbstainException(
                    f"OOD abstention triggered prior to numerical overflow: "
                    f"Mahalanobis distance {ood_score:.4f} > ood_threshold {eff_ood_thresh:.4f}",
                    ood_score=ood_score,
                    threshold=eff_ood_thresh,
                    trigger="ood_mahalanobis_threshold",
                ) from exc
            raise

        z_crit = _normal_ppf_two_sided(self.nominal_coverage)
        ood_inflation = (
            math.hypot(1.0, math.sqrt(max(0.0, self.ood_uncertainty_scale)) * ood_score)
            if ood_score > 0.0
            else 1.0
        )

        lower_map: dict[str, float] = {}
        upper_map: dict[str, float] = {}
        half_width_map: dict[str, float] = {}
        width_map: dict[str, float] = {}
        std_deltas: dict[str, float] = {}

        for rid, mu_val in means.items():
            var_val = variances.get(rid, 0.0)
            epistemic_std = math.sqrt(max(0.0, var_val))
            aleatoric_std = max(avg_stds.get(rid, 0.0), self._residual_stds.get(rid, 0.0))

            if self._is_calibrated and rid in self._conformal_quantiles:
                base_hw = self._conformal_quantiles[rid]
                inflate_active = not math.isinf(self.ood_threshold) or not math.isinf(
                    self.threshold
                )
                eff_scale = ood_inflation if inflate_active else 1.0
                base_radius = base_hw + z_crit * epistemic_std
                if eff_scale > 1.0 and base_radius == 0.0:
                    base_radius = max(self.default_half_width * 1e-6, 1e-6)
                hw = base_radius * eff_scale
            else:
                combined_std = math.hypot(
                    epistemic_std, max(aleatoric_std, self.default_half_width, 1e-6)
                )
                hw = z_crit * combined_std * ood_inflation

            if math.isnan(hw) or hw < 0.0:
                hw = float("inf")

            half_width_map[rid] = float(hw)
            width_map[rid] = float(2.0 * hw)
            lower_map[rid] = float(mu_val - hw)
            upper_map[rid] = float(mu_val + hw)
            std_deltas[rid] = float(hw / max(z_crit, 1e-6))

        max_w = max(width_map.values(), default=0.0)
        max_hw = max(half_width_map.values(), default=0.0)
        max_var = max(variances.values(), default=0.0)

        ci = ConfidenceInterval(
            lower=lower_map,
            upper=upper_map,
            half_width=half_width_map,
            width=width_map,
            max_width=float(max_w),
            max_half_width=float(max_hw),
            ensemble_variance=variances,
            max_ensemble_variance=float(max_var),
            ood_score=float(ood_score),
            nominal_coverage=self.nominal_coverage,
        )

        prediction = CalibratedPrediction(
            mean_deltas=means,
            std_deltas=std_deltas,
            proposed_actions=(),
            calibration_temperature=self._calibration_temperature,
        )

        # 3. Check abstention gates (thresholds default to +inf so legacy behavior is preserved)

        if not math.isinf(eff_ood_thresh) and ood_score > eff_ood_thresh:
            raise AbstainException(
                f"OOD abstention triggered: Mahalanobis distance {ood_score:.4f} > "
                f"ood_threshold {eff_ood_thresh:.4f}",
                confidence_interval=ci,
                ood_score=ood_score,
                ensemble_variance=max_var,
                threshold=eff_ood_thresh,
                trigger="ood_mahalanobis_threshold",
            )

        if not math.isinf(eff_var_thresh) and max_var > eff_var_thresh:
            raise AbstainException(
                f"Ensemble variance abstention triggered: Var(mu)={max_var:.4f} > "
                f"variance_threshold {eff_var_thresh:.4f}",
                confidence_interval=ci,
                ood_score=ood_score,
                ensemble_variance=max_var,
                threshold=eff_var_thresh,
                trigger="ensemble_variance_threshold",
            )

        if ci > self.threshold:
            raise AbstainException(
                f"ConfidenceInterval abstention triggered: effective_uncertainty "
                f"{ci.effective_uncertainty:.4f} (max_width={ci.max_width:.4f}) > "
                f"threshold {self.threshold:.4f}",
                confidence_interval=ci,
                ood_score=ood_score,
                ensemble_variance=max_var,
                threshold=self.threshold,
                trigger="confidence_interval_threshold",
            )

        return prediction, ci

    def evaluate_calibration(
        self,
        validation_dataset: TransitionDataset | Sequence[TransitionSample],
        *,
        num_bins: int = 10,
        uncalibrated_scale: float = 0.25,
    ) -> CalibrationEvaluationReport:
        """Evaluate predictive interval calibration (ECE) across nominal levels $p_m \\in (0, 1)$.

        Follows Kuleshov et al. (2018) & Vovk et al. (2005): for $M$ nominal confidence
        levels $p_1, \\dots, p_M \\in (0, 1)$, computes empirical coverage $\\hat{p}(p_m)$
        of the conformal quantile intervals and reports:
        $$\\text{ECE} = \\sum_{m=1}^M \\frac{|B_m|}{N_{\\text{total}}} |\\hat{p}(p_m) - p_m|$$
        alongside the uncalibrated baseline ECE.
        """
        val_samples = list(validation_dataset)
        if not val_samples:
            raise ValueError("Cannot evaluate calibration on an empty validation dataset.")

        m_bins = max(2, int(num_bins))
        nominal_levels = np.linspace(1.0 / (m_bins + 1), m_bins / (m_bins + 1), m_bins)

        # Collect residuals on validation set
        abs_errors: list[float] = []
        base_stds: list[float] = []
        covered_at_target: list[bool] = []

        # Temporarily evaluate without abstaining so all validation samples are scored
        saved_thresh = (self.threshold, self.ood_threshold, self.variance_threshold)
        try:
            self.threshold = float("inf")
            self.ood_threshold = float("inf")
            self.variance_threshold = float("inf")

            for sample in val_samples:
                pred, ci = self.predict(sample.state, sample.actions)
                for rid, pred_delta in pred.mean_deltas.items():
                    if rid not in sample.state.resources or rid not in sample.next_state.resources:
                        continue
                    true_delta = (
                        sample.next_state.get_resource(rid).current
                        - sample.state.get_resource(rid).current
                    )
                    err = abs(true_delta - pred_delta)
                    abs_errors.append(err)
                    base_stds.append(
                        max(1e-6, self._residual_stds.get(rid, self.default_half_width))
                    )
                    lo = ci.lower.get(rid, pred_delta)
                    hi = ci.upper.get(rid, pred_delta)
                    covered_at_target.append(bool(lo <= true_delta <= hi))
        finally:
            self.threshold, self.ood_threshold, self.variance_threshold = saved_thresh

        n_eval = len(abs_errors)
        err_arr = np.array(abs_errors, dtype=np.float64)
        std_arr = np.array(base_stds, dtype=np.float64)

        cal_emp_accs: list[float] = []
        uncal_emp_accs: list[float] = []
        counts: list[int] = []

        # Reference conformal scale at self.nominal_coverage
        z_ref = max(1e-6, _normal_ppf_two_sided(self.nominal_coverage))
        q_ref = (
            float(np.mean(list(self._conformal_quantiles.values())))
            if self._conformal_quantiles
            else float(np.std(err_arr)) * z_ref
        )
        calibrated_sigma = q_ref / z_ref

        for p_level in nominal_levels:
            z_p = _normal_ppf_two_sided(float(p_level))
            # Calibrated half-width at level p_level
            cal_hw = z_p * calibrated_sigma
            # Uncalibrated (overconfident) half-width at level p_level
            uncal_hw = z_p * std_arr * float(uncalibrated_scale)

            cal_cov = float(np.mean(err_arr <= cal_hw))
            uncal_cov = float(np.mean(err_arr <= uncal_hw))
            cal_emp_accs.append(cal_cov)
            uncal_emp_accs.append(uncal_cov)
            counts.append(n_eval)

        cal_ece = float(np.mean(np.abs(np.array(cal_emp_accs) - nominal_levels)))
        uncal_ece = float(np.mean(np.abs(np.array(uncal_emp_accs) - nominal_levels)))
        emp_cov_nominal = float(np.mean(covered_at_target)) if covered_at_target else 0.0

        return CalibrationEvaluationReport(
            expected_calibration_error=cal_ece,
            uncalibrated_ece=uncal_ece,
            empirical_coverage_at_nominal=emp_cov_nominal,
            nominal_coverage=self.nominal_coverage,
            num_samples=n_eval,
            num_bins=m_bins,
            bin_nominal_confidences=tuple(float(x) for x in nominal_levels),
            bin_empirical_accuracies=tuple(cal_emp_accs),
            bin_counts=tuple(counts),
        )


# Alias matching Section 6 contract (`Predictor.predict(state) -> (Prediction, ConfidenceInterval)`)
Predictor = ConformalPredictorWrapper


class LearnedModelRegistry:
    """Registry of calibrated learned predictors with versioned provenance fingerprints."""

    def __init__(self) -> None:
        self._predictors: dict[str, ConformalPredictorWrapper] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        model_id: str,
        predictor: ConformalPredictorWrapper,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Register a calibrated predictor under `model_id` and return its registry fingerprint."""
        if not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        self._predictors[model_id] = predictor
        meta_dict = dict(metadata or {})
        meta_dict["is_calibrated"] = predictor.is_calibrated
        meta_dict["nominal_coverage"] = predictor.nominal_coverage
        meta_dict["threshold"] = (
            "infinity" if math.isinf(predictor.threshold) else predictor.threshold
        )
        meta_dict["ood_threshold"] = (
            "infinity" if math.isinf(predictor.ood_threshold) else predictor.ood_threshold
        )
        fp = canonical_sha256({"model_id": model_id, **meta_dict})
        meta_dict["fingerprint"] = fp
        self._metadata[model_id] = meta_dict
        return fp

    def get(self, model_id: str) -> ConformalPredictorWrapper:
        """Retrieve a registered `ConformalPredictorWrapper` by `model_id`."""
        if model_id not in self._predictors:
            raise KeyError(f"Unknown model_id {model_id!r} in LearnedModelRegistry.")
        return self._predictors[model_id]

    def metadata(self, model_id: str) -> dict[str, Any]:
        """Return a copy of the registered metadata for `model_id`."""
        if model_id not in self._metadata:
            raise KeyError(f"Unknown model_id {model_id!r} in LearnedModelRegistry.")
        return dict(self._metadata[model_id])

    def list_models(self) -> tuple[str, ...]:
        """Return sorted tuple of registered model identifiers."""
        return tuple(sorted(self._predictors.keys()))


__all__ = [
    "AbstainException",
    "CalibrationEvaluationReport",
    "ConfidenceInterval",
    "ConformalPredictorWrapper",
    "DriftStatusReport",
    "LearnedModelRegistry",
    "MahalanobisOODAndDriftDetector",
    "Predictor",
    "compute_ensemble_variance",
    "compute_expected_calibration_error",
]
