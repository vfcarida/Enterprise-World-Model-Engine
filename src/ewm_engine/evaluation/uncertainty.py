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
    if not values:
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
    if not values:
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
    if not baseline_values or not candidate_values:
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
    if not actuals or not lower_bounds or not upper_bounds:
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
    if not values:
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
