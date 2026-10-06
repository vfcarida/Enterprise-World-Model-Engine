"""Uncertainty quantification and distribution statistics for Monte Carlo rollouts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class UncertaintyDistribution(BaseModel):
    """Statistical summary of metric outcomes across Monte Carlo rollouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float = Field(description="Sample mean across trajectories.")
    std: float = Field(description="Sample standard deviation.")
    median: float = Field(description="Sample median (50th percentile).")
    p05: float = Field(description="5th percentile (lower tail risk).")
    p25: float = Field(description="25th percentile (first quartile).")
    p75: float = Field(description="75th percentile (third quartile).")
    p95: float = Field(description="95th percentile (upper tail risk).")
    min_val: float = Field(description="Observed minimum.")
    max_val: float = Field(description="Observed maximum.")
    iqr: float = Field(description="Interquartile range (p75 - p25).")
    cvar_05: float = Field(description="Conditional Value-at-Risk below 5th percentile.")


def summarize_distribution(values: Sequence[float]) -> UncertaintyDistribution:
    """Compute comprehensive uncertainty metrics from a sequence of rollout values."""
    if values is None or len(values) == 0:
        return UncertaintyDistribution(
            mean=0.0,
            std=0.0,
            median=0.0,
            p05=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            min_val=0.0,
            max_val=0.0,
            iqr=0.0,
            cvar_05=0.0,
        )

    arr = np.array(values, dtype=float)
    p05 = float(np.percentile(arr, 5))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))

    tail_values = arr[arr <= p05]
    cvar_05 = float(np.mean(tail_values)) if len(tail_values) > 0 else p05

    return UncertaintyDistribution(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        median=p50,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        iqr=p75 - p25,
        cvar_05=cvar_05,
    )


class BootstrapConfidenceInterval(BaseModel):
    """Bootstrap confidence interval for a metric estimate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float = Field(description="Sample mean of the observed distribution.")
    ci_lower: float = Field(description="Lower bound of confidence interval.")
    ci_upper: float = Field(description="Upper bound of confidence interval.")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g. 0.95).")
    standard_error: float = Field(description="Standard error of the bootstrap estimate.")


class BootstrapDelta(BaseModel):
    """Statistical delta between candidate and baseline with bootstrap uncertainty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    absolute_delta: float = Field(
        description="Point estimate of candidate mean minus baseline mean."
    )
    percent_delta: float = Field(description="Percentage change relative to baseline mean.")
    ci_lower: float = Field(description="Lower bound of bootstrap CI for absolute delta.")
    ci_upper: float = Field(description="Upper bound of bootstrap CI for absolute delta.")
    confidence_level: float = Field(default=0.95, description="Confidence level for delta CI.")
    p_value: float = Field(description="Empirical two-tailed bootstrap p-value for H0: delta = 0.")
    is_significant: bool = Field(description="True if delta CI strictly excludes zero.")
    n_resamples: int = Field(default=1000, description="Number of bootstrap resamples.")


def compute_bootstrap_ci(
    values: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> BootstrapConfidenceInterval:
    """Compute non-parametric bootstrap confidence interval for the sample mean.

    Epistemic Note:
        The resulting confidence interval quantifies stochastic simulation sampling
        variance across Monte Carlo rollouts (P_model). It does not reflect empirical
        real-world ground truth confidence unless the model is empirically identified.
    """
    if values is None or len(values) == 0:
        return BootstrapConfidenceInterval(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            standard_error=0.0,
        )

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n < 2 or n_resamples <= 0:
        mean_val = float(np.mean(arr))
        return BootstrapConfidenceInterval(
            mean=mean_val,
            ci_lower=mean_val,
            ci_upper=mean_val,
            confidence_level=confidence_level,
            standard_error=0.0,
        )

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.mean(arr[boot_indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))
    std_err = float(np.std(resampled_means))

    return BootstrapConfidenceInterval(
        mean=float(np.mean(arr)),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        standard_error=std_err,
    )


def compute_bootstrap_delta(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
) -> BootstrapDelta:
    """Compute non-parametric bootstrap confidence interval and significance for a counterfactual delta.

    Epistemic Guardrail:
        Statistical significance flags in this engine measure whether a simulated policy
        difference is distinguishable from zero given simulation stochasticity.
        It MUST NOT be cited as empirical causal proof (P(Y|do(X))) without real-world
        observational identification and unconfoundedness verification.
    """
    if (
        baseline_values is None
        or len(baseline_values) == 0
        or candidate_values is None
        or len(candidate_values) == 0
    ):
        return BootstrapDelta(
            absolute_delta=0.0,
            percent_delta=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            p_value=1.0,
            is_significant=False,
            n_resamples=n_resamples,
        )

    base_arr = np.asarray(baseline_values, dtype=float)
    cand_arr = np.asarray(candidate_values, dtype=float)

    base_mean = float(np.mean(base_arr))
    cand_mean = float(np.mean(cand_arr))
    abs_delta = cand_mean - base_mean
    pct_delta = (abs_delta / base_mean * 100.0) if base_mean != 0.0 else 0.0

    n_base = len(base_arr)
    n_cand = len(cand_arr)

    if (n_base < 2 and n_cand < 2) or n_resamples <= 0:
        return BootstrapDelta(
            absolute_delta=abs_delta,
            percent_delta=pct_delta,
            ci_lower=abs_delta,
            ci_upper=abs_delta,
            confidence_level=confidence_level,
            p_value=0.0 if abs_delta != 0.0 else 1.0,
            is_significant=abs_delta != 0.0,
            n_resamples=n_resamples,
        )

    rng = np.random.default_rng(seed)
    base_idx = rng.integers(0, n_base, size=(n_resamples, n_base))
    cand_idx = rng.integers(0, n_cand, size=(n_resamples, n_cand))

    boot_base_means = np.mean(base_arr[base_idx], axis=1)
    boot_cand_means = np.mean(cand_arr[cand_idx], axis=1)
    boot_deltas = boot_cand_means - boot_base_means

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_deltas, lower_pct))
    ci_upper = float(np.percentile(boot_deltas, upper_pct))

    is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)

    p_neg = float(np.mean(boot_deltas <= 0.0))
    p_pos = float(np.mean(boot_deltas >= 0.0))
    p_val = float(min(1.0, 2.0 * min(p_neg, p_pos)))

    return BootstrapDelta(
        absolute_delta=abs_delta,
        percent_delta=pct_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        p_value=p_val,
        is_significant=is_significant,
        n_resamples=n_resamples,
    )


class CalibrationDiagnostic(BaseModel):
    """Calibration diagnostic evaluating predictive distribution validity against test rollouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nominal_coverage: float = Field(
        description="Nominal confidence level of the interval (e.g. 0.90)."
    )
    empirical_coverage: float = Field(description="Observed empirical coverage rate in [0.0, 1.0].")
    coverage_error: float = Field(description="empirical_coverage minus nominal_coverage.")
    mean_crps: float | None = Field(
        default=None, description="Average Continuous Ranked Probability Score."
    )
    epistemic_note: str = Field(
        default=(
            "Rollout calibration measures internal consistency between Monte Carlo branches "
            "under P_model, NOT empirical physical-world calibration."
        ),
        description="Epistemic humility guardrail.",
    )


def compute_interval_coverage(
    actuals: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> float:
    """Compute empirical coverage rate of predicted intervals over evaluated targets.

    Args:
        actuals: Sequence of observed or reference outcome values.
        lower_bounds: Sequence of predicted lower interval bounds.
        upper_bounds: Sequence of predicted upper interval bounds.

    Returns:
        Empirical coverage fraction in [0.0, 1.0].
    """
    if (
        actuals is None
        or len(actuals) == 0
        or lower_bounds is None
        or len(lower_bounds) == 0
        or upper_bounds is None
        or len(upper_bounds) == 0
    ):
        return 0.0
    n = min(len(actuals), len(lower_bounds), len(upper_bounds))
    y = np.asarray(actuals[:n], dtype=float)
    lb = np.asarray(lower_bounds[:n], dtype=float)
    ub = np.asarray(upper_bounds[:n], dtype=float)
    covered = (y >= lb) & (y <= ub)
    return float(np.mean(covered))


def compute_crps(samples: Sequence[float], actual: float) -> float:
    """Compute sample Continuous Ranked Probability Score (CRPS) for a probabilistic forecast.

    Uses the non-parametric energy kernel representation (Gneiting & Raftery, 2007):
        CRPS(F_N, y) = (1/N) * sum(|x_i - y|) - (1 / (2*N^2)) * sum_i sum_j |x_i - x_j|

    Epistemic Note:
        CRPS evaluates the calibration and sharpness of Monte Carlo rollout distributions
        against reference targets. It assesses simulation model calibration under P_model.

    Args:
        samples: Monte Carlo rollout sample values representing the predictive distribution.
        actual: The observed or reference evaluation target value.

    Returns:
        Non-negative CRPS score (lower is better; 0 indicates perfect deterministic prediction).
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    if n == 1:
        return float(abs(arr[0] - actual))

    term1 = float(np.mean(np.abs(arr - actual)))
    diff_matrix = np.abs(arr[:, None] - arr[None, :])
    term2 = float(0.5 * np.mean(diff_matrix))

    return float(max(0.0, term1 - term2))


def compute_tail_metrics(
    values: Sequence[float],
    alpha: float = 0.05,
    worst_k: int = 3,
) -> dict[str, Any]:
    """Compute tail risk diagnostics including CVaR and worst-k observations.

    Args:
        values: Sequence of rollout outcome values.
        alpha: Lower tail risk threshold percentile (e.g. 0.05 for worst 5%).
        worst_k: Number of worst-case individual values to extract.

    Returns:
        Dictionary containing 'cvar', 'worst_k_values', and 'tail_threshold'.
    """
    if values is None or len(values) == 0:
        return {"cvar": 0.0, "worst_k_values": [], "tail_threshold": 0.0}

    arr = np.asarray(values, dtype=float)
    sorted_arr = np.sort(arr)
    tail_idx = max(1, int(np.ceil(alpha * len(arr))))
    tail_vals = sorted_arr[:tail_idx]
    cvar_val = float(np.mean(tail_vals))
    k_actual = min(worst_k, len(arr))
    worst_k_vals = [float(v) for v in sorted_arr[:k_actual]]

    return {
        "cvar": cvar_val,
        "worst_k_values": worst_k_vals,
        "tail_threshold": float(sorted_arr[tail_idx - 1]),
    }


# ---------------------------------------------------------------------------
# Proper Scoring Rules & Calibration Diagnostics
# ---------------------------------------------------------------------------


def compute_mean_crps(
    forecast_samples: Sequence[Sequence[float]],
    actuals: Sequence[float],
) -> float:
    """Compute sample-average Continuous Ranked Probability Score across multiple targets.

    Args:
        forecast_samples: Sequence of Monte Carlo sample sets for each evaluation target.
        actuals: Sequence of observed target outcome values.

    Returns:
        Average non-negative CRPS (lower is better; 0 indicates perfect deterministic prediction).
    """
    if (
        forecast_samples is None
        or len(forecast_samples) == 0
        or actuals is None
        or len(actuals) == 0
    ):
        return 0.0
    n = min(len(forecast_samples), len(actuals))
    if n == 0:
        return 0.0
    scores = [compute_crps(forecast_samples[i], actuals[i]) for i in range(n)]
    return float(np.mean(scores))


def compute_log_score(
    means: Sequence[float],
    stds: Sequence[float],
    actuals: Sequence[float],
    eps: float = 1e-6,
) -> float:
    """Compute mean negative Gaussian log-likelihood (proper log score).

    For Gaussian predictive distributions N(mu_i, sigma_i^2), the negative log-likelihood is:
        NLL(y_i) = 0.5 * log(2 * pi * sigma_i^2) + (y_i - mu_i)^2 / (2 * sigma_i^2)

    Lower is better.

    Args:
        means: Predicted predictive distribution means.
        stds: Predicted predictive distribution standard deviations.
        actuals: Observed target outcomes.
        eps: Minimum standard deviation floor for numerical stability.

    Returns:
        Average negative log-likelihood score.
    """
    if (
        means is None
        or len(means) == 0
        or stds is None
        or len(stds) == 0
        or actuals is None
        or len(actuals) == 0
    ):
        return 0.0
    n = min(len(means), len(stds), len(actuals))
    if n == 0:
        return 0.0

    mu = np.asarray(means[:n], dtype=float)
    sigma = np.clip(np.asarray(stds[:n], dtype=float), eps, None)
    y = np.asarray(actuals[:n], dtype=float)

    nll = 0.5 * np.log(2.0 * np.pi * (sigma**2)) + ((y - mu) ** 2) / (2.0 * (sigma**2))
    return float(np.mean(nll))


def compute_brier_score(
    probabilities: Sequence[Sequence[float]] | Sequence[float],
    labels: Sequence[int] | Sequence[float],
    n_classes: int | None = None,
) -> float:
    """Compute the multi-class or binary Brier score (strictly proper scoring rule).

    For binary predictions p in [0, 1] and labels y in {0, 1}:
        Brier = (1/N) * sum((p_i - y_i)^2)

    For K-class categorical distributions p_ik and one-hot labels:
        Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    Lower is better (0.0 is perfect; random binary guessing gives 0.25).

    Args:
        probabilities: Predicted class probabilities (1D for binary positive class, 2D for multiclass).
        labels: True integer class indices or binary indicators.
        n_classes: Number of categorical classes (inferred if None).

    Returns:
        Brier score in [0.0, 2.0].
    """
    if probabilities is None or len(probabilities) == 0 or labels is None or len(labels) == 0:
        return 0.0

    n = min(len(probabilities), len(labels))
    if n == 0:
        return 0.0

    first_elem = probabilities[0]
    if isinstance(first_elem, (list, tuple, np.ndarray)):
        # Multiclass probability matrix (N, K)
        p_matrix = np.asarray(probabilities[:n], dtype=float)
        k = p_matrix.shape[1] if n_classes is None else n_classes
        y_int = np.asarray(labels[:n], dtype=int)
        y_one_hot = np.zeros_like(p_matrix)
        for i, idx in enumerate(y_int):
            if 0 <= idx < k:
                y_one_hot[i, idx] = 1.0
        return float(np.mean(np.sum((p_matrix - y_one_hot) ** 2, axis=1)))
    else:
        # Binary probabilities (N,)
        p_vec = np.asarray(probabilities[:n], dtype=float)
        y_vec = np.asarray(labels[:n], dtype=float)
        return float(np.mean((p_vec - y_vec) ** 2))


class ReliabilityBin(BaseModel):
    """Single bin diagnostic in a reliability diagram."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bin_index: int
    lower_edge: float
    upper_edge: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_gap: float


class AdaptiveCalibrationResult(BaseModel):
    """Adaptive calibration evaluation with quantile-spaced equal-mass binning.

    Epistemic Note:
        Fixed-width Expected Calibration Error (ECE) suffers from empty bin distortion and
        masks severe overconfidence in sparse tail intervals. Adaptive Calibration Error (ACE)
        uses quantile partitioning to ensure equal empirical mass per bin, providing a
        debiased, proper measure of classification/regime calibration.
        (Mucsányi et al., 2024; Roelofs et al., 2022).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ace: float = Field(description="Adaptive Calibration Error (weighted absolute gap).")
    debiased_ace: float = Field(
        description="Debiased ACE correcting for finite-sample bin variance bias."
    )
    n_bins: int
    bins: tuple[ReliabilityBin, ...]
    epistemic_note: str = Field(
        default=(
            "Adaptive binning prevents empty-bin artifacts. Aleatoric vs epistemic "
            "uncertainty disentanglement remains fundamentally unidentifiable from observational data "
            "(Mucsányi et al., 2024)."
        )
    )


def compute_adaptive_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> AdaptiveCalibrationResult:
    """Compute Adaptive Calibration Error (ACE) using equal-mass quantile binning.

    Args:
        probabilities: Predicted probabilities for the positive outcome in [0, 1].
        labels: True binary outcomes in {0, 1}.
        n_bins: Number of equal-mass quantile bins.

    Returns:
        AdaptiveCalibrationResult containing ACE, debiased ACE, and per-bin diagnostics.
    """
    if (
        probabilities is None
        or len(probabilities) == 0
        or labels is None
        or len(labels) == 0
        or n_bins <= 0
    ):
        return AdaptiveCalibrationResult(
            ace=0.0,
            debiased_ace=0.0,
            n_bins=n_bins,
            bins=(),
        )

    n = min(len(probabilities), len(labels))
    p = np.clip(np.asarray(probabilities[:n], dtype=float), 0.0, 1.0)
    y = np.asarray(labels[:n], dtype=float)

    if n < n_bins:
        # Fall back to single bin if dataset is tiny
        mean_p = float(np.mean(p))
        mean_y = float(np.mean(y))
        gap = abs(mean_p - mean_y)
        bin_diag = ReliabilityBin(
            bin_index=0,
            lower_edge=0.0,
            upper_edge=1.0,
            count=n,
            mean_confidence=mean_p,
            empirical_accuracy=mean_y,
            calibration_gap=gap,
        )
        return AdaptiveCalibrationResult(
            ace=gap,
            debiased_ace=gap,
            n_bins=1,
            bins=(bin_diag,),
        )

    # Sort by confidence to establish equal-mass quantile bins
    sort_idx = np.argsort(p)
    p_sorted = p[sort_idx]
    y_sorted = y[sort_idx]

    bin_splits = np.array_split(np.arange(n), n_bins)
    reliability_bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    debias_correction = 0.0

    for idx, split in enumerate(bin_splits):
        if len(split) == 0:
            continue
        p_bin = p_sorted[split]
        y_bin = y_sorted[split]
        count = len(split)

        conf = float(np.mean(p_bin))
        acc = float(np.mean(y_bin))
        gap = abs(conf - acc)
        weight = count / n
        weighted_gap += weight * gap

        # Broecker / Roelofs finite-sample variance debiasing term: Var(accuracy) / count
        bin_var = float(np.var(y_bin)) if count > 1 else 0.0
        debias_correction += weight * (bin_var / count)

        reliability_bins.append(
            ReliabilityBin(
                bin_index=idx,
                lower_edge=float(p_bin[0]),
                upper_edge=float(p_bin[-1]),
                count=count,
                mean_confidence=conf,
                empirical_accuracy=acc,
                calibration_gap=gap,
            )
        )

    debiased_ace = float(max(0.0, weighted_gap - np.sqrt(debias_correction)))

    return AdaptiveCalibrationResult(
        ace=float(weighted_gap),
        debiased_ace=debiased_ace,
        n_bins=len(reliability_bins),
        bins=tuple(reliability_bins),
    )


class PITBin(BaseModel):
    """Histogram bin for Probability Integral Transform (PIT) evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bin_index: int
    lower_edge: float
    upper_edge: float
    count: int
    empirical_frequency: float
    expected_frequency: float


class PITCalibrationResult(BaseModel):
    """Continuous predictive distribution calibration evaluated via Probability Integral Transform.

    Under an ideally calibrated probabilistic forecast F_i, the transform u_i = F_i(y_i)
    is uniformly distributed on [0, 1]. Significant deviation from uniformity indicates
    overconfidence (U-shaped), underconfidence (hump-shaped), or biased drift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ks_statistic: float = Field(
        description="Kolmogorov-Smirnov distance from standard uniform distribution U(0, 1)."
    )
    is_well_calibrated: bool = Field(
        description="True if empirical PIT distribution does not severely diverge from uniform (KS < 0.15)."
    )
    quantile_coverages: dict[str, float] = Field(
        description="Empirical coverage rates for central intervals (e.g. 50%, 80%, 90%, 95%)."
    )
    pit_bins: tuple[PITBin, ...]
    epistemic_note: str = Field(
        default=(
            "PIT assesses predictive distribution calibration against evaluation targets. "
            "Passing PIT calibration does not prove correctness of underlying causal mechanisms."
        )
    )


def compute_pit_calibration(
    forecast_samples: Sequence[Sequence[float]],
    actuals: Sequence[float],
    n_bins: int = 10,
) -> PITCalibrationResult:
    """Evaluate continuous distribution calibration via Probability Integral Transform (PIT).

    Args:
        forecast_samples: Sequence of sample sets from predictive distribution for each step.
        actuals: Sequence of observed target outcome values.
        n_bins: Number of equal-width bins for the PIT histogram.

    Returns:
        PITCalibrationResult with KS distance, coverage rates, and histogram.
    """
    if (
        forecast_samples is None
        or len(forecast_samples) == 0
        or actuals is None
        or len(actuals) == 0
        or n_bins <= 0
    ):
        return PITCalibrationResult(
            ks_statistic=1.0,
            is_well_calibrated=False,
            quantile_coverages={},
            pit_bins=(),
        )

    n = min(len(forecast_samples), len(actuals))
    if n == 0:
        return PITCalibrationResult(
            ks_statistic=1.0,
            is_well_calibrated=False,
            quantile_coverages={},
            pit_bins=(),
        )

    # Compute empirical PIT values u_i = (1/M) sum_j 1(x_ij <= y_i)
    pit_values: list[float] = []
    nominal_levels = [0.50, 0.80, 0.90, 0.95]
    interval_counts: dict[float, int] = dict.fromkeys(nominal_levels, 0)

    for i in range(n):
        samples = np.asarray(forecast_samples[i], dtype=float)
        y = float(actuals[i])
        m = len(samples)
        if m == 0:
            continue
        # Non-parametric empirical CDF with mid-P continuity correction
        count_less = np.sum(samples < y)
        count_equal = np.sum(samples == y)
        u_val = float((count_less + 0.5 * count_equal) / m)
        pit_values.append(np.clip(u_val, 0.0, 1.0))

        # Check coverage of central intervals
        for alpha in nominal_levels:
            lower_q = (1.0 - alpha) / 2.0
            upper_q = 1.0 - lower_q
            q_low = float(np.percentile(samples, lower_q * 100.0))
            q_high = float(np.percentile(samples, upper_q * 100.0))
            if q_low <= y <= q_high:
                interval_counts[alpha] += 1

    if not pit_values:
        return PITCalibrationResult(
            ks_statistic=1.0,
            is_well_calibrated=False,
            quantile_coverages={},
            pit_bins=(),
        )

    u_arr = np.sort(np.asarray(pit_values, dtype=float))
    total_u = len(u_arr)

    # Kolmogorov-Smirnov test against standard uniform CDF F(u) = u
    ecdf = np.arange(1, total_u + 1) / total_u
    d_plus = np.max(ecdf - u_arr)
    d_minus = np.max(u_arr - (np.arange(0, total_u) / total_u))
    ks_stat = float(max(d_plus, d_minus))

    # Binning for PIT histogram
    bins_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(u_arr, bins=bins_edges)
    expected_freq = 1.0 / n_bins

    pit_bins: list[PITBin] = []
    for b_idx in range(n_bins):
        cnt = int(counts[b_idx])
        pit_bins.append(
            PITBin(
                bin_index=b_idx,
                lower_edge=float(bins_edges[b_idx]),
                upper_edge=float(bins_edges[b_idx + 1]),
                count=cnt,
                empirical_frequency=float(cnt / total_u),
                expected_frequency=expected_freq,
            )
        )

    coverages = {
        f"central_{int(alpha * 100)}": float(interval_counts[alpha] / total_u)
        for alpha in nominal_levels
    }

    return PITCalibrationResult(
        ks_statistic=ks_stat,
        is_well_calibrated=bool(ks_stat < 0.15),
        quantile_coverages=coverages,
        pit_bins=tuple(pit_bins),
    )
