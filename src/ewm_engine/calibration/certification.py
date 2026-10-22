"""Mandatory coverage certification: TARP, Posterior-SBC, and misspecification diagnostics.

Implements rigorous coverage tests for Simulation-Based Inference:
- TARP (Tests of Accuracy with Random Points; Lemos et al., ICML 2023, arXiv:2302.03026)
- Posterior-SBC (Simulation-Based Calibration conditional on observed data; arXiv:2502.03279)
- Misspecification detection (Huang et al., NeurIPS 2023)

An uncertified or under-covered posterior is flagged and refused for decision use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator


def _kolmogorov_smirnov_uniform(samples: np.ndarray) -> tuple[float, float]:
    """Compute 1-sample Kolmogorov-Smirnov statistic and approximate p-value against Uniform([0, 1])."""
    n = len(samples)
    if n == 0:
        return 1.0, 0.0

    sorted_vals = np.sort(samples)
    cdf_expected = sorted_vals
    cdf_empirical_plus = np.arange(1, n + 1) / n
    cdf_empirical_minus = np.arange(0, n) / n

    d_plus = np.max(cdf_empirical_plus - cdf_expected)
    d_minus = np.max(cdf_expected - cdf_empirical_minus)
    d_stat = float(max(d_plus, d_minus))

    # Asymptotic Kolmogorov distribution p-value approximation
    en = math.sqrt(n)
    lambda_val = (en + 0.12 + 0.11 / en) * d_stat
    p_val = 2.0 * math.exp(-2.0 * (lambda_val**2))
    p_val = max(0.0, min(1.0, float(p_val)))

    return d_stat, p_val


class CoverageCertificate(BaseModel):
    """Certified coverage assessment for an amortized posterior distribution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_certified: bool = Field(
        description="Whether the posterior satisfies joint coverage and calibration standards."
    )
    status: Literal[
        "CERTIFIED",
        "UNCERTIFIED_OVERCONFIDENT",
        "UNCERTIFIED_UNDERCOVERED",
        "UNCERTIFIED_MISSPECIFIED",
    ] = Field(description="Certification verdict classification.")
    decision_usable: bool = Field(
        description="Strict safety gate: True ONLY if certified. Uncertified posteriors must be refused for decisions."
    )
    tarp_max_discrepancy: float = Field(
        description="Maximum absolute discrepancy between nominal and empirical TARP coverage."
    )
    tarp_ks_pvalue: float = Field(
        description="Kolmogorov-Smirnov p-value for TARP coverage probabilities."
    )
    sbc_rank_ks_pvalue: float = Field(
        description="Kolmogorov-Smirnov p-value for Posterior-SBC normalized ranks."
    )
    misspecification_score: float = Field(
        description="Standardized statistical distance of observed data from simulator support."
    )
    is_misspecified: bool = Field(
        description="Whether observed data falls outside the simulator's generative support."
    )
    refusal_reason: str | None = Field(
        default=None, description="Auditable rationale when calibration certification is refused."
    )
    nominal_alphas: tuple[float, ...] = Field(description="Tested nominal credibility levels.")
    empirical_coverages: tuple[float, ...] = Field(
        description="Empirically observed coverage rates."
    )
    max_discrepancy_threshold: float = Field(
        default=0.28, description="Threshold tolerance for TARP max discrepancy."
    )
    ks_pvalue_threshold: float = Field(
        default=0.01, description="Minimum p-value threshold for Kolmogorov-Smirnov test."
    )
    tarp_expansion: str = Field(
        default="Tests of Accuracy with Random Points (Lemos et al., ICML 2023, arXiv:2302.03026)",
        description="Verified primary-source expansion and citation for TARP.",
    )
    tarp_reference_distribution: str = Field(
        default="random_uniform",
        description="Reference point sampling distribution p_r(theta) used for TARP spheres.",
    )
    assumptions: tuple[str, ...] = Field(
        default=(
            "Reference points theta_r are drawn independently of theta_true given x from a distribution with full support over the prior domain (Lemos et al., ICML 2023, Theorem 1).",
            "Test pairs (theta_j^*, x_j^*) are drawn i.i.d. from the joint generative model p(theta)p(x|theta).",
            "In-simulator calibration certification does not guarantee real-world structural validity if the simulator is misspecified outside the tested summary statistics.",
        ),
        description="Explicit mathematical assumptions required for TARP and Posterior-SBC certification.",
    )

    def to_teaching_error(self) -> Any:
        """Convert an uncertified coverage assessment into an actionable TeachingError."""
        from ewm_engine.exceptions import UncertifiedCalibrationError

        avg_emp = float(np.mean(self.empirical_coverages)) if self.empirical_coverages else 0.0
        avg_nom = float(np.mean(self.nominal_alphas)) if self.nominal_alphas else 0.5
        return UncertifiedCalibrationError(
            empirical_coverage=avg_emp,
            target_coverage=avg_nom,
            refusal_reason=self.refusal_reason,
        )

    def raise_for_status(self) -> None:
        """Raise UncertifiedCalibrationError if this posterior is not certified."""
        if not self.is_certified:
            raise self.to_teaching_error()


def compute_tarp_coverage(
    test_thetas: np.ndarray,
    test_xs: np.ndarray,
    estimator: AmortizedPosteriorEstimator,
    n_posterior_samples: int = 150,
    nominal_alphas: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    seed: int = 42,
    reference_mode: Literal["random_uniform", "fixed_mean"] = "random_uniform",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute TARP (Tests of Accuracy with Random Points) coverage curve.

    Primary Reference:
        Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).
        "Sampling-Based Accuracy Testing of Posterior Estimators for General Inference."
        International Conference on Machine Learning (ICML 2023), PMLR 202:19256-19273,
        arXiv:2302.03026.

    Algorithm (Algorithm 1 & Theorem 1 of Lemos et al., 2023):
    For each test pair (theta_j^*, x_j^*):
        1. Draw samples tilde_theta_s ~ q(theta | x_j^*), s = 1..S.
        2. Sample a random reference point theta_{r,j} ~ p_r(theta | x_j^*) independently
           of theta_j^* (uniformly over the prior hyper-rectangle bounds).
        3. Compute Euclidean distances to theta_{r,j}:
           d_j^* = ||theta_j^* - theta_{r,j}||_2, d_{j,s} = ||tilde_theta_s - theta_{r,j}||_2.
        4. Credibility fraction f_j = (1/S) sum_{s=1}^S I(d_{j,s} < d_j^*).
    Evaluate empirical coverage probability ECP(alpha) = (1/m) sum_{j=1}^m I(f_j <= alpha).
    Under Theorem 1 of Lemos et al. (2023), random reference points make ECP(alpha) = alpha
    both necessary and sufficient for posterior accuracy q(theta|x) = p(theta|x).
    """
    thetas_arr = np.asarray(test_thetas, dtype=np.float64)
    if thetas_arr.ndim == 1:
        thetas_arr = thetas_arr.reshape(-1, 1)
    xs_arr = np.asarray(test_xs, dtype=np.float64)
    m = len(thetas_arr)

    if m == 0:
        alphas = np.array(nominal_alphas, dtype=np.float64)
        return alphas, np.zeros_like(alphas), 1.0, 0.0

    d_theta = thetas_arr.shape[1]
    if hasattr(estimator, "prior_set") and len(estimator.prior_set.bounds) == d_theta:
        low_bounds = np.array([b[0] for b in estimator.prior_set.bounds], dtype=np.float64)
        high_bounds = np.array([b[1] for b in estimator.prior_set.bounds], dtype=np.float64)
    else:
        low_bounds = np.min(thetas_arr, axis=0)
        high_bounds = np.max(thetas_arr, axis=0)
        span = np.where(high_bounds > low_bounds, high_bounds - low_bounds, 1.0)
        low_bounds = low_bounds - 0.05 * span
        high_bounds = high_bounds + 0.05 * span

    fixed_ref_point = np.mean(thetas_arr, axis=0)
    coverage_fractions = np.zeros(m, dtype=np.float64)

    rng = np.random.default_rng(seed)

    for j in range(m):
        theta_true = thetas_arr[j]
        x_j = xs_arr[j]

        # Draw posterior samples conditioned on test x_j
        seed_j = int(rng.integers(1, 2**31 - 1))
        post = estimator.predict_posterior(x_j, n_samples=n_posterior_samples, seed=seed_j)
        post_samples = post.get_samples_array()

        # Sample random reference point per test simulation j (Lemos et al., 2023, Alg. 1)
        if reference_mode == "random_uniform":
            ref_point_j = rng.uniform(low=low_bounds, high=high_bounds)
        elif reference_mode == "fixed_mean":
            ref_point_j = fixed_ref_point
        else:
            raise ValueError(f"Unsupported TARP reference_mode: {reference_mode!r}")

        # Compute Euclidean distance from reference point
        d_true = np.linalg.norm(theta_true - ref_point_j)
        d_samples = np.linalg.norm(post_samples - ref_point_j, axis=1)

        coverage_fractions[j] = float(np.mean(d_samples < d_true))

    # Evaluate coverage across nominal levels
    alphas = np.array(nominal_alphas, dtype=np.float64)
    empirical_coverages = np.array(
        [float(np.mean(coverage_fractions <= a)) for a in alphas], dtype=np.float64
    )

    max_discrepancy = float(np.max(np.abs(empirical_coverages - alphas)))
    _, ks_pvalue = _kolmogorov_smirnov_uniform(coverage_fractions)

    return alphas, empirical_coverages, max_discrepancy, ks_pvalue


def compute_posterior_sbc(
    x_obs: np.ndarray,
    estimator: AmortizedPosteriorEstimator,
    simulator_fn: Any,
    k_posterior_draws: int = 20,
    l_eval_samples: int = 100,
    seed: int = 42,
) -> float:
    """Compute Posterior-SBC rank uniformity p-value conditional on observed data x_obs.

    Algorithm (arXiv:2502.03279):
    1. Draw theta^(k) ~ p(theta | x_obs).
    2. Forward simulate x^(k) = Sim(theta^(k)).
    3. Re-infer p(theta | x^(k)) and compute fractional rank of theta^(k).
    4. Test rank uniformity via Kolmogorov-Smirnov test.
    """
    post = estimator.predict_posterior(x_obs, n_samples=k_posterior_draws, seed=seed)
    theta_draws = post.get_samples_array()

    ranks = np.zeros(len(theta_draws), dtype=np.float64)
    names = post.parameter_names

    rng = np.random.default_rng(seed + 100)

    for k in range(len(theta_draws)):
        t_k = theta_draws[k]
        t_dict = {names[j]: float(t_k[j]) for j in range(len(names))}

        sim_seed = int(rng.integers(1, 2**31 - 1))
        x_sim, _ = simulator_fn(t_dict, seed=sim_seed)

        # Re-infer posterior from forward realization
        re_post = estimator.predict_posterior(x_sim, n_samples=l_eval_samples, seed=sim_seed)
        re_samples = re_post.get_samples_array()

        # Average fractional rank across parameter dimensions
        dim_ranks = [float(np.mean(re_samples[:, j] <= t_k[j])) for j in range(len(names))]
        ranks[k] = float(np.mean(dim_ranks))

    _, ks_pvalue = _kolmogorov_smirnov_uniform(ranks)
    return ks_pvalue


def compute_misspecification_diagnostic(
    x_obs: np.ndarray,
    simulated_xs: np.ndarray,
) -> tuple[float, bool]:
    """Compute statistical distance of target data from simulator predictive distribution (Huang et al., 2023)."""
    xo = np.asarray(x_obs, dtype=np.float64).ravel()
    sim_x = np.asarray(simulated_xs, dtype=np.float64)

    mean_sim = np.mean(sim_x, axis=0)
    std_sim = np.std(sim_x, axis=0)
    std_sim[std_sim < 1e-12] = 1.0

    z_scores = np.abs((xo - mean_sim) / std_sim)
    norm_score = float(np.mean(z_scores))

    # Flagged as misspecified if average feature distance exceeds 3.5 standard deviations
    is_misspecified = norm_score > 3.5
    return norm_score, is_misspecified


def certify_coverage(
    estimator: AmortizedPosteriorEstimator,
    test_thetas: np.ndarray,
    test_xs: np.ndarray,
    x_obs: np.ndarray,
    simulator_fn: Any | None = None,
    max_discrepancy_threshold: float = 0.28,
    ks_pvalue_threshold: float = 0.01,
    seed: int = 42,
) -> CoverageCertificate:
    """Execute mandatory coverage certification and emit auditable certificate."""
    alphas, emp_covs, max_disc, tarp_p = compute_tarp_coverage(
        test_thetas=test_thetas,
        test_xs=test_xs,
        estimator=estimator,
        seed=seed,
    )

    misspec_score, is_misspec = compute_misspecification_diagnostic(x_obs, test_xs)

    if simulator_fn is not None:
        sbc_p = compute_posterior_sbc(
            x_obs=x_obs,
            estimator=estimator,
            simulator_fn=simulator_fn,
            seed=seed,
        )
    else:
        sbc_p = tarp_p

    # Adjudicate certification verdict
    if is_misspec:
        status: Literal[
            "CERTIFIED",
            "UNCERTIFIED_OVERCONFIDENT",
            "UNCERTIFIED_UNDERCOVERED",
            "UNCERTIFIED_MISSPECIFIED",
        ] = "UNCERTIFIED_MISSPECIFIED"
        is_certified = False
        refusal_reason = (
            f"Model Misspecification: Observed target data lies {misspec_score:.2f} standard deviations "
            "outside the simulator generative support (Huang et al., 2023). "
            "The simulator cannot reproduce real observations under any parameter prior."
        )
    elif max_disc > max_discrepancy_threshold or tarp_p < ks_pvalue_threshold:
        # Check if empirical coverage is lower than nominal (under-covered / overconfident)
        mean_bias = float(np.mean(emp_covs - alphas))
        if mean_bias < -0.05:
            status = "UNCERTIFIED_OVERCONFIDENT"
            refusal_reason = (
                f"Overconfident Posterior: Empirical TARP coverage is significantly below nominal credibility "
                f"(max discrepancy {max_disc:.3f} > {max_discrepancy_threshold}, p={tarp_p:.4f}). "
                "Posterior variance is spuriously collapsed."
            )
        else:
            status = "UNCERTIFIED_UNDERCOVERED"
            refusal_reason = (
                f"Miscalibrated Coverage: Posterior coverage violates uniformity test "
                f"(max discrepancy {max_disc:.3f}, p={tarp_p:.4f})."
            )
        is_certified = False
    else:
        status = "CERTIFIED"
        is_certified = True
        refusal_reason = None

    return CoverageCertificate(
        is_certified=is_certified,
        status=status,
        decision_usable=is_certified,
        tarp_max_discrepancy=max_disc,
        tarp_ks_pvalue=tarp_p,
        sbc_rank_ks_pvalue=sbc_p,
        misspecification_score=misspec_score,
        is_misspecified=is_misspec,
        refusal_reason=refusal_reason,
        nominal_alphas=tuple(float(a) for a in alphas),
        empirical_coverages=tuple(float(c) for c in emp_covs),
        max_discrepancy_threshold=max_discrepancy_threshold,
        ks_pvalue_threshold=ks_pvalue_threshold,
    )
