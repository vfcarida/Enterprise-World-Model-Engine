"""Uncertainty quantification and distribution statistics for Monte Carlo rollouts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

BootstrapIntervalMethod = Literal["percentile", "basic", "normal", "studentized", "bca"]
FloatSequence = Sequence[float] | np.ndarray[Any, np.dtype[Any]]


class UncertaintySourceType(StrEnum):
    """Principled 6-part taxonomy of uncertainty sources in enterprise world modeling."""

    ALEATORIC = "aleatoric"
    PARAMETER = "parameter"
    STRUCTURAL_MODEL = "structural_model"
    RUNTIME_CONTEXT = "runtime_context"
    MONTE_CARLO_ERROR = "monte_carlo_error"
    MODEL_DISCREPANCY_VS_REALITY = "model_discrepancy_vs_reality"


class UncertaintyTaxonomyBreakdown(BaseModel):
    """Explicit decomposition of uncertainty by epistemic/ontic source.

    Core Principle:
        Never report a generic 'uncertainty' scalar when the source is knowable,
        and never pretend a source is estimated when it is unidentifiable from the
        experiment (e.g., model discrepancy vs. physical reality in purely
        in-silico simulations).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    aleatoric_std: float | None = Field(
        default=None,
        description="Irreducible stochastic dispersion within the declared transition/observation model (sigma_aleatoric).",
    )
    parameter_std: float | None = Field(
        default=None,
        description="Epistemic dispersion induced by posterior uncertainty over continuous parameters theta.",
    )
    structural_model_std: float | None = Field(
        default=None,
        description="Epistemic dispersion across competing mechanism graphs or functional forms.",
    )
    runtime_context_std: float | None = Field(
        default=None,
        description="Dispersion attributable to shifting or partially observed runtime operational context C_t.",
    )
    monte_carlo_standard_error: float | None = Field(
        default=None,
        description="Finite-sample Monte Carlo estimation standard error (SE_MC = sigma / sqrt(N)), vanishing as N -> inf.",
    )
    model_discrepancy_vs_reality: float | None = Field(
        default=None,
        description=(
            "Kennedy & O'Hagan (2001) structural discrepancy |Y_real - Y_model|. "
            "Must be None unless external real-world validation data is provided."
        ),
    )
    unestimable_sources: tuple[str, ...] = Field(
        default_factory=lambda: (
            UncertaintySourceType.PARAMETER.value,
            UncertaintySourceType.STRUCTURAL_MODEL.value,
            UncertaintySourceType.RUNTIME_CONTEXT.value,
            UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value,
        ),
        description="Uncertainty sources that cannot be estimated from the current experiment design.",
    )
    source_status: dict[str, str] = Field(
        default_factory=dict,
        description="Explicit audit status per uncertainty source ('estimated', 'zero_deterministic', or reason unestimable).",
    )

    @classmethod
    def from_simulation_rollouts(
        cls,
        values: FloatSequence,
        *,
        parameter_std: float | None = None,
        structural_model_std: float | None = None,
        runtime_context_std: float | None = None,
        model_discrepancy_vs_reality: float | None = None,
    ) -> UncertaintyTaxonomyBreakdown:
        """Construct an honest uncertainty breakdown from simulated rollout outcomes."""
        arr = np.asarray(values, dtype=float) if values is not None else np.array([], dtype=float)
        n = len(arr)
        if n == 0:
            aleatoric = 0.0
            mc_se = 0.0
            aleatoric_status = "empty_sample"
            mc_status = "empty_sample"
        elif n == 1:
            aleatoric = 0.0
            mc_se = 0.0
            aleatoric_status = "unestimable_single_trajectory"
            mc_status = "unestimable_single_trajectory"
        else:
            aleatoric = float(np.std(arr, ddof=1))
            mc_se = float(aleatoric / math.sqrt(n))
            aleatoric_status = (
                "zero_deterministic" if aleatoric == 0.0 else "estimated_from_rollouts"
            )
            mc_status = "zero_deterministic" if mc_se == 0.0 else "estimated_from_rollouts"

        unestimable: list[str] = []
        status: dict[str, str] = {
            UncertaintySourceType.ALEATORIC.value: aleatoric_status,
            UncertaintySourceType.MONTE_CARLO_ERROR.value: mc_status,
        }
        if n == 1:
            unestimable.append(UncertaintySourceType.ALEATORIC.value)
            unestimable.append(UncertaintySourceType.MONTE_CARLO_ERROR.value)

        if parameter_std is None:
            unestimable.append(UncertaintySourceType.PARAMETER.value)
            status[UncertaintySourceType.PARAMETER.value] = (
                "unestimable_fixed_parameters_no_posterior_draws"
            )
        else:
            status[UncertaintySourceType.PARAMETER.value] = "estimated_from_parameter_ensemble"

        if structural_model_std is None:
            unestimable.append(UncertaintySourceType.STRUCTURAL_MODEL.value)
            status[UncertaintySourceType.STRUCTURAL_MODEL.value] = (
                "unestimable_single_mechanism_specification"
            )
        else:
            status[UncertaintySourceType.STRUCTURAL_MODEL.value] = "estimated_from_model_ensemble"

        if runtime_context_std is None:
            unestimable.append(UncertaintySourceType.RUNTIME_CONTEXT.value)
            status[UncertaintySourceType.RUNTIME_CONTEXT.value] = (
                "unestimable_fixed_context_specification"
            )
        else:
            status[UncertaintySourceType.RUNTIME_CONTEXT.value] = "estimated_from_context_sweeps"

        if model_discrepancy_vs_reality is None:
            unestimable.append(UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value)
            status[UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value] = (
                "unestimable_without_external_real_world_validation"
            )
        else:
            status[UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value] = (
                "estimated_from_external_telemetry_residuals"
            )

        return cls(
            aleatoric_std=aleatoric if n >= 2 else None,
            parameter_std=parameter_std,
            structural_model_std=structural_model_std,
            runtime_context_std=runtime_context_std,
            monte_carlo_standard_error=mc_se if n >= 2 else None,
            model_discrepancy_vs_reality=model_discrepancy_vs_reality,
            unestimable_sources=tuple(unestimable),
            source_status=status,
        )


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF Phi^{-1}(p) via bisection to 1e-12 accuracy."""
    p_clipped = min(max(float(p), 1e-12), 1.0 - 1e-12)
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _norm_cdf(mid) < p_clipped:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
    estimand: str = Field(
        default="E_{P_model}[Y]",
        description="Formal statistical estimand under the simulation distribution P_model.",
    )
    uncertainty_taxonomy: UncertaintyTaxonomyBreakdown = Field(
        default_factory=lambda: UncertaintyTaxonomyBreakdown.from_simulation_rollouts([]),
        description="Explicit decomposition of uncertainty into aleatoric, Monte Carlo, and unestimable sources.",
    )

    @property
    def mc_standard_error(self) -> float:
        """Monte Carlo standard error SE_MC = sample_std_ddof1 / sqrt(N)."""
        if (
            self.uncertainty_taxonomy is not None
            and self.uncertainty_taxonomy.monte_carlo_standard_error is not None
        ):
            return self.uncertainty_taxonomy.monte_carlo_standard_error
        return 0.0


def summarize_distribution(
    values: FloatSequence,
    *,
    estimand: str = "E_{P_model}[Y]",
    parameter_std: float | None = None,
    structural_model_std: float | None = None,
    runtime_context_std: float | None = None,
    model_discrepancy_vs_reality: float | None = None,
) -> UncertaintyDistribution:
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
            estimand=estimand,
            uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts(
                [],
                parameter_std=parameter_std,
                structural_model_std=structural_model_std,
                runtime_context_std=runtime_context_std,
                model_discrepancy_vs_reality=model_discrepancy_vs_reality,
            ),
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
        estimand=estimand,
        uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts(
            values,
            parameter_std=parameter_std,
            structural_model_std=structural_model_std,
            runtime_context_std=runtime_context_std,
            model_discrepancy_vs_reality=model_discrepancy_vs_reality,
        ),
    )


class BootstrapConfidenceInterval(BaseModel):
    """Bootstrap confidence interval for a metric estimate with explicit estimand and assumptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float = Field(description="Sample mean of the observed distribution.")
    ci_lower: float = Field(description="Lower bound of confidence interval.")
    ci_upper: float = Field(description="Upper bound of confidence interval.")
    confidence_level: float = Field(default=0.95, description="Confidence level (e.g. 0.95).")
    standard_error: float = Field(description="Standard error of the bootstrap estimate.")
    estimand: str = Field(
        default="E_{P_model}[Y]",
        description="Formal target estimand (under P_model unless external calibration/identification is supplied).",
    )
    interval_method: BootstrapIntervalMethod = Field(
        default="percentile",
        description="Bootstrap interval construction strategy ('percentile', 'basic', 'normal', 'studentized', 'bca').",
    )
    resampling_unit: str = Field(
        default="independent_trajectory",
        description="Unit of resampling (e.g., 'independent_trajectory').",
    )
    n_samples: int = Field(default=0, description="Number of observed trajectories/samples.")
    n_resamples: int = Field(default=1000, description="Number of bootstrap replicates.")
    seed: int | None = Field(default=None, description="PRNG seed used for bootstrap resampling.")
    is_degenerate: bool = Field(
        default=False,
        description="True when sample size < 2 or sample variance is identically zero.",
    )
    small_sample_warning: bool = Field(
        default=False,
        description="True when n_samples < 30, where asymptotic bootstrap coverage may under-cover.",
    )
    assumptions: tuple[str, ...] = Field(
        default_factory=lambda: (
            "Independent and identically distributed (i.i.d.) trajectory draws under P_model.",
            "Quantifies Monte Carlo simulation error under P_model, NOT real-world structural discrepancy.",
        ),
        description="Statistical assumptions required for interval validity.",
    )
    uncertainty_taxonomy: UncertaintyTaxonomyBreakdown | None = Field(
        default=None,
        description="Explicit uncertainty taxonomy decomposition for this estimate.",
    )

    @property
    def point_estimate(self) -> float:
        """Alias for mean point estimate."""
        return self.mean

    @property
    def zero_variance(self) -> bool:
        """Alias for is_degenerate."""
        return self.is_degenerate


class BootstrapDelta(BaseModel):
    """Statistical delta between candidate and baseline with bootstrap uncertainty and estimand."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    absolute_delta: float = Field(
        description="Point estimate of candidate mean minus baseline mean."
    )
    percent_delta: float = Field(description="Percentage change relative to baseline mean.")
    ci_lower: float = Field(description="Lower bound of bootstrap CI for absolute delta.")
    ci_upper: float = Field(description="Upper bound of bootstrap CI for absolute delta.")
    confidence_level: float = Field(default=0.95, description="Confidence level for delta CI.")
    p_value: float = Field(description="Empirical two-tailed bootstrap p-value for H0: delta = 0.")
    is_significant: bool = Field(
        description=(
            "True if and only if the comparison is stochastic (is_deterministic=False) "
            "and the bootstrap CI strictly excludes zero under P_model."
        )
    )
    is_deterministic: bool = Field(
        default=False,
        description="True when both baseline and candidate distributions are degenerate point masses (zero variance or single sample).",
    )
    status_label: str = Field(
        default="non_significant",
        description=(
            "Semantic comparison status: 'statistically_significant' (stochastic bootstrap CI excludes 0 under P_model), "
            "'deterministic_difference' (zero-variance deterministic delta != 0), "
            "'deterministic_identical' (zero-variance deterministic delta == 0), or "
            "'non_significant' (stochastic bootstrap CI spans 0)."
        ),
    )
    n_resamples: int = Field(default=1000, description="Number of bootstrap resamples.")
    estimand: str = Field(
        default="E_{P_model}[Y_candidate] - E_{P_model}[Y_baseline]",
        description="Formal target estimand of the comparison (under P_model, not real-world P(Y|do(X)) unless identified).",
    )
    design: Literal["independent", "paired"] = Field(
        default="independent",
        description="Experimental resampling design ('independent' two-sample vs 'paired' common-random-numbers).",
    )
    interval_method: BootstrapIntervalMethod = Field(
        default="percentile",
        description="Bootstrap interval construction strategy ('percentile', 'basic', 'normal', 'studentized', 'bca').",
    )
    resampling_unit: str = Field(
        default="independent_trajectory",
        description="Resampling unit ('independent_trajectory' or 'paired_seed_trajectory').",
    )
    n_baseline: int = Field(default=0, description="Number of baseline trajectories.")
    n_candidate: int = Field(default=0, description="Number of candidate trajectories.")
    seed: int | None = Field(default=None, description="PRNG seed used for bootstrap resampling.")
    effect_size_cohens_d: float | None = Field(
        default=None,
        description="Standardized mean difference (Cohen's d) when pooled sample variance > 0; None if deterministic.",
    )
    small_sample_warning: bool = Field(
        default=False,
        description="True when min(n_baseline, n_candidate) < 30, where bootstrap intervals may under-cover.",
    )
    assumptions: tuple[str, ...] = Field(
        default_factory=lambda: (
            "Estimand is evaluated under the simulator distribution P_model.",
            "Does NOT establish real-world causal effect P(Y|do(X)) without external causal identification and calibration.",
        ),
        description="Statistical and epistemic assumptions governing this delta.",
    )
    limitations: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Explicit methodological limitations (e.g. zero-variance degeneracy, small sample size, unmodeled reality gap).",
    )
    uncertainty_taxonomy: UncertaintyTaxonomyBreakdown | None = Field(
        default=None,
        description="Explicit uncertainty taxonomy decomposition for the delta.",
    )

    @property
    def zero_variance(self) -> bool:
        """True if the comparison has zero sample variance (deterministic point mass)."""
        return self.is_deterministic


def _compute_bootstrap_bounds_from_replicates(
    point_estimate: float,
    replicates: np.ndarray,
    confidence_level: float,
    method: BootstrapIntervalMethod,
    *,
    raw_sample: np.ndarray | None = None,
    boot_se_replicates: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval bounds under the specified strategy."""
    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    std_err = float(np.std(replicates, ddof=1)) if len(replicates) > 1 else 0.0

    if std_err <= 1e-15:
        return point_estimate, point_estimate

    if method == "percentile":
        return (
            float(np.percentile(replicates, lower_pct)),
            float(np.percentile(replicates, upper_pct)),
        )
    if method == "basic":
        q_low = float(np.percentile(replicates, lower_pct))
        q_high = float(np.percentile(replicates, upper_pct))
        return (2.0 * point_estimate - q_high, 2.0 * point_estimate - q_low)
    if method == "normal":
        z = _norm_ppf(1.0 - alpha / 2.0)
        return (point_estimate - z * std_err, point_estimate + z * std_err)
    if method == "studentized":
        if boot_se_replicates is not None and raw_sample is not None and len(raw_sample) >= 2:
            sample_se = float(np.std(raw_sample, ddof=1) / math.sqrt(len(raw_sample)))
            safe_boot_se = np.where(boot_se_replicates > 1e-12, boot_se_replicates, sample_se)
            if sample_se > 1e-12:
                t_stats = (replicates - point_estimate) / safe_boot_se
                t_low = float(np.percentile(t_stats, lower_pct))
                t_high = float(np.percentile(t_stats, upper_pct))
                return (
                    point_estimate - t_high * sample_se,
                    point_estimate - t_low * sample_se,
                )
        return (
            float(np.percentile(replicates, lower_pct)),
            float(np.percentile(replicates, upper_pct)),
        )
    if method == "bca":
        # Bias-Corrected and Accelerated (BCa; Efron, 1987)
        prop_less = float(np.mean(replicates < point_estimate))
        if prop_less <= 0.0 or prop_less >= 1.0 or raw_sample is None or len(raw_sample) < 3:
            return (
                float(np.percentile(replicates, lower_pct)),
                float(np.percentile(replicates, upper_pct)),
            )
        z0 = _norm_ppf(prop_less)
        n = len(raw_sample)
        total_sum = float(np.sum(raw_sample))
        jack_means = (total_sum - raw_sample) / float(n - 1)
        jack_mean_avg = float(np.mean(jack_means))
        diffs = jack_mean_avg - jack_means
        denom = float(np.sum(diffs**2)) ** 1.5
        a_hat = float(np.sum(diffs**3)) / (6.0 * denom) if denom > 1e-18 else 0.0

        z_low = _norm_ppf(alpha / 2.0)
        z_high = _norm_ppf(1.0 - alpha / 2.0)

        denom_low = 1.0 - a_hat * (z0 + z_low)
        denom_high = 1.0 - a_hat * (z0 + z_high)
        if abs(denom_low) < 1e-9 or abs(denom_high) < 1e-9:
            return (
                float(np.percentile(replicates, lower_pct)),
                float(np.percentile(replicates, upper_pct)),
            )
        a1 = _norm_cdf(z0 + (z0 + z_low) / denom_low) * 100.0
        a2 = _norm_cdf(z0 + (z0 + z_high) / denom_high) * 100.0
        a1_c = min(max(a1, 0.0), 100.0)
        a2_c = min(max(a2, 0.0), 100.0)
        lo_b = float(np.percentile(replicates, min(a1_c, a2_c)))
        hi_b = float(np.percentile(replicates, max(a1_c, a2_c)))
        return lo_b, hi_b

    raise ValueError(f"Unsupported bootstrap interval method: {method!r}")


def compute_bootstrap_ci(
    values: FloatSequence,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
    *,
    method: BootstrapIntervalMethod | None = None,
    interval_method: BootstrapIntervalMethod = "percentile",
    estimand: str = "E_{P_model}[Y]",
    resampling_unit: str = "independent_trajectory",
) -> BootstrapConfidenceInterval:
    """Compute non-parametric bootstrap confidence interval for the sample mean.

    Supports multiple explicitly named interval strategies:
        - ``"percentile"``: Standard percentile bootstrap $[\\hat{\\theta}^*_{(\\alpha/2)}, \\hat{\\theta}^*_{(1-\\alpha/2)}]$.
        - ``"basic"``: Basic pivotal bootstrap $[2\\hat{\\theta} - \\hat{\\theta}^*_{(1-\\alpha/2)}, 2\\hat{\\theta} - \\hat{\\theta}^*_{(\\alpha/2)}]$.
        - ``"normal"``: Normal approximation $\\hat{\\theta} \\pm z_{1-\\alpha/2} \\widehat{\\text{SE}}_{\\text{boot}}$.
        - ``"studentized"``: Studentized bootstrap-t interval using per-resample plug-in SE.
        - ``"bca"``: Bias-Corrected and Accelerated bootstrap (Efron, 1987) with jackknife skewness acceleration.

    Epistemic Note:
        The resulting confidence interval quantifies stochastic simulation sampling
        variance across Monte Carlo rollouts ($P_{\\text{model}}$). It does not reflect empirical
        real-world ground truth confidence unless the model is empirically identified.
    """
    resolved_method: BootstrapIntervalMethod = method if method is not None else interval_method
    if values is None or len(values) == 0:
        return BootstrapConfidenceInterval(
            mean=0.0,
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=confidence_level,
            standard_error=0.0,
            estimand=estimand,
            interval_method=resolved_method,
            resampling_unit=resampling_unit,
            n_samples=0,
            n_resamples=n_resamples,
            seed=seed,
            is_degenerate=True,
            small_sample_warning=True,
            uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts([]),
        )

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean_val = float(np.mean(arr))
    sample_std = float(np.std(arr))
    is_degen = n < 2 or sample_std == 0.0 or n_resamples <= 0

    if is_degen:
        return BootstrapConfidenceInterval(
            mean=mean_val,
            ci_lower=mean_val,
            ci_upper=mean_val,
            confidence_level=confidence_level,
            standard_error=0.0,
            estimand=estimand,
            interval_method=resolved_method,
            resampling_unit=resampling_unit,
            n_samples=n,
            n_resamples=n_resamples,
            seed=seed,
            is_degenerate=True,
            small_sample_warning=n < 30,
            uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts(values),
        )

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_samples = arr[boot_indices]
    resampled_means = np.mean(boot_samples, axis=1)
    boot_ses = (
        np.std(boot_samples, axis=1, ddof=1) / math.sqrt(n)
        if resolved_method == "studentized"
        else None
    )

    ci_lower, ci_upper = _compute_bootstrap_bounds_from_replicates(
        point_estimate=mean_val,
        replicates=resampled_means,
        confidence_level=confidence_level,
        method=resolved_method,
        raw_sample=arr,
        boot_se_replicates=boot_ses,
    )
    std_err = float(np.std(resampled_means))

    return BootstrapConfidenceInterval(
        mean=mean_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        standard_error=std_err,
        estimand=estimand,
        interval_method=resolved_method,
        resampling_unit=resampling_unit,
        n_samples=n,
        n_resamples=n_resamples,
        seed=seed,
        is_degenerate=False,
        small_sample_warning=n < 30,
        uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts(values),
    )


def compute_bootstrap_delta(
    baseline_values: FloatSequence,
    candidate_values: FloatSequence,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int | None = None,
    *,
    paired: bool = False,
    method: BootstrapIntervalMethod | None = None,
    interval_method: BootstrapIntervalMethod = "percentile",
    estimand: str | None = None,
) -> BootstrapDelta:
    """Compute non-parametric bootstrap confidence interval and significance for a counterfactual delta.

    Epistemic Guardrail:
        Statistical significance flags in this engine measure whether a simulated policy
        difference is distinguishable from zero given simulation stochasticity under $P_{\\text{model}}$.
        When both baseline and candidate rollouts have zero sample variance (or $n < 2$),
        the comparison is tagged with ``is_deterministic=True``, ``is_significant=False``,
        and ``status_label='deterministic_difference'`` (or ``'deterministic_identical'``)
        rather than claiming stochastic statistical significance.
        It MUST NOT be cited as empirical real-world causal proof ($P(Y \\mid \\text{do}(X))$)
        without external causal identification and empirical validation.
    """
    resolved_method: BootstrapIntervalMethod = method if method is not None else interval_method
    resolved_design: Literal["independent", "paired"] = "paired" if paired else "independent"
    resolved_unit = "paired_seed_trajectory" if paired else "independent_trajectory"
    resolved_estimand = estimand or (
        "E_{P_model}[Y_candidate(U) - Y_baseline(U)]"
        if paired
        else "E_{P_model}[Y_candidate] - E_{P_model}[Y_baseline]"
    )

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
            is_deterministic=False,
            status_label="non_significant",
            n_resamples=n_resamples,
            estimand=resolved_estimand,
            design=resolved_design,
            interval_method=resolved_method,
            resampling_unit=resolved_unit,
            n_baseline=0,
            n_candidate=0,
            seed=seed,
            effect_size_cohens_d=None,
            small_sample_warning=True,
            limitations=("Empty baseline or candidate sample.",),
            uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts([]),
        )

    base_arr = np.asarray(baseline_values, dtype=float)
    cand_arr = np.asarray(candidate_values, dtype=float)

    n_base = len(base_arr)
    n_cand = len(cand_arr)
    if paired and n_base != n_cand:
        raise ValueError(
            f"Paired bootstrap design requires equal sample sizes, got n_baseline={n_base}, n_candidate={n_cand}."
        )

    base_mean = float(np.mean(base_arr))
    cand_mean = float(np.mean(cand_arr))
    abs_delta = cand_mean - base_mean
    pct_delta = (abs_delta / base_mean * 100.0) if base_mean != 0.0 else 0.0

    if paired:
        diff_arr = cand_arr - base_arr
        zero_variance = float(np.std(diff_arr)) == 0.0
    else:
        diff_arr = None
        zero_variance = float(np.std(base_arr)) == 0.0 and float(np.std(cand_arr)) == 0.0

    small_sample = min(n_base, n_cand) < 30
    limitations_list: list[str] = [
        "Evaluated under simulator measure P_model; real-world model discrepancy is unestimated."
    ]
    if small_sample:
        limitations_list.append(
            f"Small trajectory sample size (n_baseline={n_base}, n_candidate={n_cand} < 30); bootstrap intervals may under-cover tails."
        )

    if (n_base < 2 and n_cand < 2) or zero_variance or n_resamples <= 0:
        limitations_list.append(
            "Degenerate zero-variance deterministic sample: frequentist p-values and stochastic significance do not apply."
        )
        return BootstrapDelta(
            absolute_delta=abs_delta,
            percent_delta=pct_delta,
            ci_lower=abs_delta,
            ci_upper=abs_delta,
            confidence_level=confidence_level,
            p_value=1.0,
            is_significant=False,
            is_deterministic=True,
            status_label="deterministic_difference"
            if abs_delta != 0.0
            else "deterministic_identical",
            n_resamples=n_resamples,
            estimand=resolved_estimand,
            design=resolved_design,
            interval_method=resolved_method,
            resampling_unit=resolved_unit,
            n_baseline=n_base,
            n_candidate=n_cand,
            seed=seed,
            effect_size_cohens_d=None,
            small_sample_warning=small_sample,
            limitations=tuple(limitations_list),
            uncertainty_taxonomy=UncertaintyTaxonomyBreakdown.from_simulation_rollouts([abs_delta]),
        )

    rng = np.random.default_rng(seed)
    if paired and diff_arr is not None:
        idx = rng.integers(0, n_base, size=(n_resamples, n_base))
        boot_samples = diff_arr[idx]
        boot_deltas = np.mean(boot_samples, axis=1)
        boot_ses = (
            np.std(boot_samples, axis=1, ddof=1) / math.sqrt(n_base)
            if resolved_method == "studentized"
            else None
        )
        raw_for_method = diff_arr
        sd_diff = float(np.std(diff_arr, ddof=1))
        cohens_d = float(abs_delta / sd_diff) if sd_diff > 1e-12 else None
    else:
        base_idx = rng.integers(0, n_base, size=(n_resamples, n_base))
        cand_idx = rng.integers(0, n_cand, size=(n_resamples, n_cand))

        boot_base_samples = base_arr[base_idx]
        boot_cand_samples = cand_arr[cand_idx]
        boot_base_means = np.mean(boot_base_samples, axis=1)
        boot_cand_means = np.mean(boot_cand_samples, axis=1)
        boot_deltas = boot_cand_means - boot_base_means
        boot_ses = None
        if resolved_method == "studentized" and n_base >= 2 and n_cand >= 2:
            var_b = np.var(boot_base_samples, axis=1, ddof=1) / n_base
            var_c = np.var(boot_cand_samples, axis=1, ddof=1) / n_cand
            boot_ses = np.sqrt(var_b + var_c)
        raw_for_method = cand_arr - np.mean(base_arr) if n_base == n_cand else None

        var_base = float(np.var(base_arr, ddof=1)) if n_base >= 2 else 0.0
        var_cand = float(np.var(cand_arr, ddof=1)) if n_cand >= 2 else 0.0
        pooled_sd = math.sqrt(0.5 * (var_base + var_cand))
        cohens_d = float(abs_delta / pooled_sd) if pooled_sd > 1e-12 else None

    ci_lower, ci_upper = _compute_bootstrap_bounds_from_replicates(
        point_estimate=abs_delta,
        replicates=boot_deltas,
        confidence_level=confidence_level,
        method=resolved_method,
        raw_sample=raw_for_method,
        boot_se_replicates=boot_ses,
    )

    is_significant = (ci_lower > 0.0) or (ci_upper < 0.0)

    p_neg = float(np.mean(boot_deltas <= 0.0))
    p_pos = float(np.mean(boot_deltas >= 0.0))
    p_val = float(min(1.0, 2.0 * min(p_neg, p_pos)))

    # Construct delta taxonomy breakdown
    delta_se = float(np.std(boot_deltas, ddof=1)) if len(boot_deltas) > 1 else 0.0
    taxonomy = UncertaintyTaxonomyBreakdown(
        aleatoric_std=float(np.std(cand_arr, ddof=1)) if n_cand >= 2 else None,
        monte_carlo_standard_error=delta_se,
        unestimable_sources=(
            UncertaintySourceType.PARAMETER.value,
            UncertaintySourceType.STRUCTURAL_MODEL.value,
            UncertaintySourceType.RUNTIME_CONTEXT.value,
            UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value,
        ),
        source_status={
            UncertaintySourceType.ALEATORIC.value: "estimated_from_rollouts",
            UncertaintySourceType.MONTE_CARLO_ERROR.value: "estimated_from_bootstrap_replicates",
            UncertaintySourceType.PARAMETER.value: "unestimable_fixed_parameters_no_posterior_draws",
            UncertaintySourceType.STRUCTURAL_MODEL.value: "unestimable_single_mechanism_specification",
            UncertaintySourceType.RUNTIME_CONTEXT.value: "unestimable_fixed_context_specification",
            UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY.value: "unestimable_without_external_real_world_validation",
        },
    )

    return BootstrapDelta(
        absolute_delta=abs_delta,
        percent_delta=pct_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        p_value=p_val,
        is_significant=is_significant,
        is_deterministic=False,
        status_label="statistically_significant" if is_significant else "non_significant",
        n_resamples=n_resamples,
        estimand=resolved_estimand,
        design=resolved_design,
        interval_method=resolved_method,
        resampling_unit=resolved_unit,
        n_baseline=n_base,
        n_candidate=n_cand,
        seed=seed,
        effect_size_cohens_d=cohens_d,
        small_sample_warning=small_sample,
        limitations=tuple(limitations_list),
        uncertainty_taxonomy=taxonomy,
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


# ==============================================================================
# Safe-Testing e-Values & Anytime-Valid e-Processes (Grünwald et al., 2024; Ramdas et al., 2023)
# ==============================================================================


class SafeTestingEValueResult(BaseModel):
    """Anytime-valid sequential safe-testing e-value / e-process summary.

    Terminology Disambiguation (CRITICAL):
        This is a **safe-testing e-value / e-process** ($e_t \\ge 0$ with
        $\\mathbb{E}_{H_0}[e_t] \\le 1$ and Ville's inequality guarantee
        $\\mathbb{P}_{H_0}(\\exists t \\ge 1 : E_t \\ge 1/\\alpha) \\le \\alpha$ under optional stopping;
        Grünwald et al., 2024, JRSSB; Ramdas et al., 2023, Statistical Science).
        It MUST NEVER be confused with VanderWeele & Ding's (2017) **Sensitivity E-value**
        ($E_{\\text{sens}} = RR + \\sqrt{RR(RR-1)} \\ge 1$), which measures unmeasured
        confounding sensitivity on the risk-ratio scale in observational causal studies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    concept_type: Literal["safe_testing_e_value"] = "safe_testing_e_value"
    e_value: float = Field(
        ge=0.0,
        description="Final anytime-valid test martingale / e-process value E_T >= 0.",
    )
    running_max_e_value: float = Field(
        ge=0.0,
        description="Supremum of the e-process trajectory sup_{1 <= t <= T} E_t.",
    )
    anytime_valid_p_value: float = Field(
        ge=0.0,
        le=1.0,
        description="Anytime-valid p-value p_t = min(1.0, 1.0 / sup_{s <= t} E_s) via Ville's inequality.",
    )
    alpha: float = Field(default=0.05, description="Target Type-I error threshold alpha.")
    reject_null: bool = Field(
        description="True if running_max_e_value >= 1 / alpha (valid under optional stopping)."
    )
    n_observations: int = Field(description="Number of sequential observations processed.")
    null_mean: float = Field(default=0.0, description="Tested null hypothesis mean mu_0.")
    estimand: str = Field(
        default="E_{P_model}[D_t] - mu_0",
        description="Sequential drift estimand tested against H_0: E[D_t | F_{t-1}] = mu_0.",
    )
    assumptions: tuple[str, ...] = Field(
        default_factory=lambda: (
            "Observations (D_t - mu_0)/sigma are conditionally 1-sub-Gaussian under H_0 given filtration F_{t-1}.",
            "Betting / mixture parameter lambda_t is predictable (chosen prior to observing D_t).",
            "Valid under arbitrary filtration-adapted stopping times tau via Ville's maximal inequality.",
        ),
        description="Mathematical assumptions required for anytime-valid Type-I error control.",
    )
    disambiguation_note: str = Field(
        default=(
            "Safe-testing e-value / e-process (E_{H0}[E] <= 1, Grunwald et al., 2024; Ramdas et al., 2023) "
            "for sequential optional stopping. Distinct from VanderWeele & Ding (2017) confounding Sensitivity E-value."
        ),
    )


class SafeTestingEProcess:
    """Sequential anytime-valid e-process (test supermartingale) for optional-stopping inference.

    Maintains a non-negative wealth process $E_t = \\prod_{i=1}^t e_i$ testing
    $H_0: \\mathbb{E}[X_i \\mid \\mathcal{F}_{i-1}] = \\mu_0$ for conditionally
    $\\sigma$-sub-Gaussian increments using predictable mixture / plug-in betting:

    $$e_i = \\exp\\left( \\lambda_i z_i - \\frac{1}{2} \\lambda_i^2 \\right), \\quad z_i = \\frac{X_i - \\mu_0}{\\sigma}$$

    Because $\\lambda_i$ is $\\mathcal{F}_{i-1}$-measurable (computed strictly from
    past observations $1, \\dots, i-1$), $\\mathbb{E}_{H_0}[e_i \\mid \\mathcal{F}_{i-1}] \\le 1$
    holds at every step, and $(E_t)_{t \\ge 0}$ is a non-negative supermartingale with $E_0 = 1$.
    By Ville's inequality (1939):

    $$\\mathbb{P}_{H_0}\\left( \\exists t \\ge 1 : E_t \\ge \\frac{1}{\\alpha} \\right) \\le \\alpha$$
    """

    def __init__(
        self,
        null_mean: float = 0.0,
        sigma: float = 1.0,
        alpha: float = 0.05,
        max_lambda: float = 0.8,
        estimand: str = "E_{P_model}[D_t] - mu_0",
    ) -> None:
        if sigma <= 0.0:
            raise ValueError(f"sigma must be strictly positive, got {sigma}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.null_mean = float(null_mean)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.max_lambda = float(max_lambda)
        self.estimand = estimand

        self._log_e: float = 0.0
        self._max_log_e: float = 0.0
        self._n: int = 0
        self._sum_z: float = 0.0
        self._history: list[float] = []

    def step(self, observation: float) -> SafeTestingEValueResult:
        """Process one sequential observation X_t and update the test martingale E_t."""
        # Predictable plug-in parameter lambda_t depends ONLY on F_{t-1} (past observations)
        if self._n == 0:
            lam_t = 0.0
        else:
            running_mean_z = self._sum_z / math.sqrt(self._n * (self._n + 1))
            lam_t = float(np.clip(running_mean_z, -self.max_lambda, self.max_lambda))

        z_t = (float(observation) - self.null_mean) / self.sigma
        log_inc = (lam_t * z_t) - (0.5 * (lam_t**2))
        self._log_e = float(np.clip(self._log_e + log_inc, -500.0, 500.0))
        self._max_log_e = max(self._max_log_e, self._log_e)
        self._n += 1
        self._sum_z += z_t
        e_val = float(math.exp(self._log_e))
        self._history.append(e_val)
        return self.summary()

    def update_batch(self, observations: FloatSequence) -> SafeTestingEValueResult:
        """Process a sequence of observations in order."""
        for obs in observations:
            self.step(float(obs))
        return self.summary()

    def summary(self) -> SafeTestingEValueResult:
        """Return the current anytime-valid SafeTestingEValueResult."""
        e_val = float(math.exp(self._log_e))
        max_e = float(math.exp(self._max_log_e))
        anytime_p = float(min(1.0, 1.0 / max(1.0, max_e)))
        threshold = 1.0 / self.alpha
        return SafeTestingEValueResult(
            e_value=e_val,
            running_max_e_value=max_e,
            anytime_valid_p_value=anytime_p,
            alpha=self.alpha,
            reject_null=bool(max_e >= threshold),
            n_observations=self._n,
            null_mean=self.null_mean,
            estimand=self.estimand,
        )


def compute_safe_e_value(
    observations: FloatSequence,
    *,
    null_mean: float = 0.0,
    sigma: float | None = None,
    alpha: float = 0.05,
    estimand: str = "E_{P_model}[D_t] - mu_0",
) -> SafeTestingEValueResult:
    """Compute an anytime-valid safe-testing e-value / e-process over a sequence of observations.

    Note on Sigma:
        For strict finite-sample supermartingale validity under $H_0$, ``sigma`` should be a
        known upper bound on the sub-Gaussian scale of $(X_t - \\mu_0)$. If ``sigma=None``,
        an empirical pre-standardized scale is used with an explicit assumption caveat.
    """
    if observations is None or len(observations) == 0:
        return SafeTestingEValueResult(
            e_value=1.0,
            running_max_e_value=1.0,
            anytime_valid_p_value=1.0,
            alpha=alpha,
            reject_null=False,
            n_observations=0,
            null_mean=null_mean,
            estimand=estimand,
        )

    arr = np.asarray(observations, dtype=float)
    if sigma is None:
        emp_std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
        if emp_std <= 1e-12:
            return SafeTestingEValueResult(
                e_value=1.0,
                running_max_e_value=1.0,
                anytime_valid_p_value=1.0,
                alpha=alpha,
                reject_null=False,
                n_observations=len(arr),
                null_mean=null_mean,
                estimand=estimand,
                assumptions=(
                    "Degenerate zero-variance sample: sequential safe-testing e-process remains at neutral E_0 = 1.0.",
                ),
            )
        scale = emp_std
        custom_assumptions = (
            "Empirical scale sigma_hat plugged in from sample; exact finite-sample supermartingale validity requires known sub-Gaussian bound sigma.",
            "Predictable betting parameter lambda_t chosen strictly from filtration F_{t-1}.",
            "Anytime-valid under optional stopping via Ville's inequality.",
        )
    else:
        scale = float(sigma)
        custom_assumptions = None

    proc = SafeTestingEProcess(
        null_mean=null_mean,
        sigma=scale,
        alpha=alpha,
        estimand=estimand,
    )
    res = proc.update_batch(arr.tolist())
    if custom_assumptions is not None:
        return res.model_copy(update={"assumptions": custom_assumptions})
    return res
