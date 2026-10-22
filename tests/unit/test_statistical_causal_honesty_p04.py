"""Unit, oracle, property, and shift tests for PROMPT 04 statistical and causal honesty.

Tests analytical oracles, bootstrap interval strategies, paired vs. independent designs,
degenerate zero-variance reporting, Sensitivity E-values vs. safe-testing e-processes under
optional stopping, TARP random reference points (Lemos et al., ICML 2023), 6-part uncertainty
taxonomy, marginal vs. conditional conformal coverage, conformal distribution shift, and
fail-closed causal identification provenance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ewm_engine.calibration.certification import CoverageCertificate, compute_tarp_coverage
from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator
from ewm_engine.core.scm import CausalGraph
from ewm_engine.core.trust import (
    EvidenceLevel,
    InsufficientCausalEvidenceWarning,
    ScenarioForecastResult,
    UnqualifiedCausalClaimError,
    validate_causal_claim,
)
from ewm_engine.evaluation.conformal import (
    ConformalIntervalPredictor,
    evaluate_conformal_under_shift,
)
from ewm_engine.evaluation.uncertainty import (
    SafeTestingEProcess,
    UncertaintyTaxonomyBreakdown,
    compute_bootstrap_ci,
    compute_bootstrap_delta,
    compute_safe_e_value,
    summarize_distribution,
)
from ewm_engine.experimental.causal import (
    SensitivityEValueResult,
    check_backdoor_identifiability,
    compute_e_value,
    compute_sensitivity_e_value,
)

# ---------------------------------------------------------------------------
# 1. Analytical Oracles & Uncertainty Taxonomy Tests
# ---------------------------------------------------------------------------


def test_analytical_oracle_summarize_distribution_and_taxonomy() -> None:
    """Verify exact sample statistics, Bessel-corrected MC SE, and 6-part uncertainty taxonomy."""
    values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    dist = summarize_distribution(
        values,
        estimand="E_{P_model}[inventory_cost]",
        parameter_std=0.75,
        structural_model_std=1.25,
    )

    # Analytical population mean = 5.0, population variance = 4.0 (std = 2.0)
    # Sample variance (ddof=1) = 32 / 7 => sample std = sqrt(32/7), MC SE = sqrt(32/7) / sqrt(8) = sqrt(4/7)
    expected_mc_se = math.sqrt(4.0 / 7.0)
    assert math.isclose(dist.mean, 5.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(dist.std, 2.0, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(dist.mc_standard_error, expected_mc_se, rel_tol=1e-12, abs_tol=1e-12)
    assert dist.estimand == "E_{P_model}[inventory_cost]"

    tax: UncertaintyTaxonomyBreakdown = dist.uncertainty_taxonomy
    assert math.isclose(tax.aleatoric_std or 0.0, math.sqrt(32.0 / 7.0), rel_tol=1e-12)
    assert math.isclose(tax.monte_carlo_standard_error or 0.0, expected_mc_se, rel_tol=1e-12)
    assert math.isclose(tax.parameter_std or 0.0, 0.75, rel_tol=1e-12)
    assert math.isclose(tax.structural_model_std or 0.0, 1.25, rel_tol=1e-12)
    assert tax.runtime_context_std is None
    assert tax.model_discrepancy_vs_reality is None
    assert "runtime_context" in tax.unestimable_sources
    assert "model_discrepancy_vs_reality" in tax.unestimable_sources
    assert tax.source_status["model_discrepancy_vs_reality"].startswith("unestimable")


# ---------------------------------------------------------------------------
# 2. Bootstrap Interval Strategies, Paired Design & Degenerate Zero-Variance
# ---------------------------------------------------------------------------


def test_degenerate_zero_variance_delta_is_not_statistically_significant() -> None:
    """Zero-variance deterministic runs must NEVER be labeled statistically significant."""
    base = [100.0, 100.0, 100.0, 100.0]
    cand = [70.0, 70.0, 70.0, 70.0]

    delta = compute_bootstrap_delta(base, cand, seed=42)
    assert delta.zero_variance is True
    assert delta.is_deterministic is True
    assert delta.is_significant is False
    assert delta.status_label == "deterministic_difference"
    assert delta.p_value == 1.0
    assert delta.small_sample_warning is True


def test_paired_vs_independent_bootstrap_variance_reduction_oracle() -> None:
    """Paired bootstrap under common random numbers must match analytical paired SE << independent SE."""
    rng = np.random.default_rng(12345)
    n = 80
    # Shared exogenous shock Z ~ N(0, 10^2) plus small policy treatment noise eps ~ N(0, 0.5^2)
    shared_shock = rng.normal(0.0, 10.0, size=n)
    true_effect = 2.0
    treatment_noise = rng.normal(0.0, 0.5, size=n)

    baseline = shared_shock
    candidate = shared_shock + true_effect + treatment_noise

    indep_delta = compute_bootstrap_delta(
        baseline, candidate, paired=False, n_resamples=2000, seed=42
    )
    paired_delta = compute_bootstrap_delta(
        baseline, candidate, paired=True, n_resamples=2000, seed=42
    )

    indep_width = indep_delta.ci_upper - indep_delta.ci_lower
    paired_width = paired_delta.ci_upper - paired_delta.ci_lower

    # Analytical SE of independent difference is sqrt(10^2/80 + 10^2/80) ≈ 1.58 -> 95% CI width ≈ 6.2
    # Analytical SE of paired difference is 0.5 / sqrt(80) ≈ 0.056 -> 95% CI width ≈ 0.22
    assert paired_delta.design == "paired"
    assert indep_delta.design == "independent"
    assert paired_width < 0.35
    assert indep_width > 4.0
    assert paired_delta.is_significant is True
    assert indep_delta.is_significant is False  # Unpaired noise masks the +2.0 effect!


@pytest.mark.parametrize(
    "method",
    ["percentile", "basic", "normal", "studentized", "bca"],
)
def test_bootstrap_interval_methods_on_skewed_distribution(method: str) -> None:
    """All 5 bootstrap interval strategies must produce valid intervals enclosing sample mean."""
    rng = np.random.default_rng(2026)
    # Shifted exponential sample with skewness = 2
    samples = rng.exponential(scale=2.0, size=60) + 1.0

    ci = compute_bootstrap_ci(
        samples,
        confidence_level=0.90,
        n_resamples=1000,
        seed=99,
        interval_method=method,  # type: ignore[arg-type]
    )
    assert ci.interval_method == method
    assert ci.ci_lower < ci.point_estimate < ci.ci_upper
    assert not ci.zero_variance
    assert not ci.small_sample_warning


def test_bootstrap_nominal_coverage_calibration_on_gaussian_oracle() -> None:
    """Empirical coverage of 90% bootstrap CIs across 160 Gaussian trials must match 0.90 within binomial bounds."""
    master_rng = np.random.default_rng(777)
    true_mu = 10.0
    true_sigma = 3.0
    n_trials = 160
    n_samples = 45
    covered = 0

    for trial in range(n_trials):
        x = master_rng.normal(true_mu, true_sigma, size=n_samples)
        ci = compute_bootstrap_ci(
            x,
            confidence_level=0.90,
            n_resamples=400,
            seed=1000 + trial,
            interval_method="studentized",
        )
        if ci.ci_lower <= true_mu <= ci.ci_upper:
            covered += 1

    empirical_coverage = covered / n_trials
    # For N=160, p=0.90, SE = sqrt(0.9*0.1/160) ≈ 0.0237; 99% binomial interval is [0.83, 0.96]
    assert 0.82 <= empirical_coverage <= 0.97


# ---------------------------------------------------------------------------
# 3. E-Value Disambiguation & Optional Stopping (Safe-Testing vs. Sensitivity)
# ---------------------------------------------------------------------------


def test_e_value_disambiguation_sensitivity_vs_safe_testing() -> None:
    """Sensitivity E-value (VanderWeele & Ding, 2017) and Safe-testing e-value (Grünwald et al., 2024) are distinct."""
    # 1. VanderWeele & Ding (2017) confounding Sensitivity E-value for RR = 2.0:
    # Analytical formula: E_sens = RR + sqrt(RR * (RR - 1)) = 2 + sqrt(2) = 3.414213562373095
    sens = compute_sensitivity_e_value(
        point_estimate=2.0,
        ci_lower=1.2,
        ci_upper=3.1,
        is_risk_ratio=True,
    )
    assert isinstance(sens, SensitivityEValueResult)
    assert sens.concept_type == "confounding_sensitivity_e_value"
    assert math.isclose(sens.e_value_point, 2.0 + math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(sens.e_value_ci_limit or 0.0, 1.2 + math.sqrt(1.2 * 0.2), rel_tol=1e-12)
    assert "VanderWeele & Ding" in sens.disambiguation_note
    assert compute_e_value(2.0, is_risk_ratio=True).e_value_point == sens.e_value_point

    # 2. Safe-testing e-value / e-process under H0: mu = 0
    rng = np.random.default_rng(42)
    null_obs = rng.normal(0.0, 1.0, size=50)
    safe_res = compute_safe_e_value(null_obs, null_mean=0.0, sigma=1.0, alpha=0.05)
    assert safe_res.e_value >= 0.0
    assert 0.0 <= safe_res.anytime_valid_p_value <= 1.0
    assert "Grunwald" in safe_res.disambiguation_note or "Grünwald" in safe_res.disambiguation_note
    assert "VanderWeele & Ding" in safe_res.disambiguation_note


def test_safe_e_process_optional_stopping_controls_type_i_error() -> None:
    """Verify Ville's inequality P_{H0}(exists t <= T: E_t >= 1/alpha) <= alpha under continuous peeking."""
    rng = np.random.default_rng(314159)
    alpha = 0.05
    n_trajectories = 400
    horizon = 60

    e_process_false_positives = 0
    naive_z_false_positives = 0

    for _ in range(n_trajectories):
        proc = SafeTestingEProcess(null_mean=0.0, sigma=1.0, alpha=alpha)
        draws = rng.normal(0.0, 1.0, size=horizon)
        res = proc.update_batch(draws.tolist())

        if res.reject_null:
            e_process_false_positives += 1

        # Compare against naive repeated z-test peeking at every step t >= 5
        running_sum = 0.0
        naive_rejected = False
        for t_idx, z in enumerate(draws, start=1):
            running_sum += float(z)
            if t_idx >= 5:
                z_stat = abs(running_sum / math.sqrt(t_idx))
                if z_stat > 1.95996:  # 2-sided alpha=0.05 threshold without sequential correction
                    naive_rejected = True
                    break
        if naive_rejected:
            naive_z_false_positives += 1

    e_process_fpr = e_process_false_positives / n_trajectories
    naive_z_fpr = naive_z_false_positives / n_trajectories

    # Ville's inequality guarantees e-process FPR <= alpha (allowing tiny Monte Carlo fluctuation)
    assert e_process_fpr <= alpha + 0.015
    # Naive optional stopping severely inflates Type-I error (> 20%)
    assert naive_z_fpr > 0.20

    # Under H1 (true mean = +0.8), e-process must reject H0 with high power
    alt_proc = SafeTestingEProcess(null_mean=0.0, sigma=1.0, alpha=alpha)
    alt_res = alt_proc.update_batch(rng.normal(0.8, 1.0, size=80).tolist())
    assert alt_res.reject_null is True
    assert alt_res.running_max_e_value >= 1.0 / alpha


# ---------------------------------------------------------------------------
# 4. TARP Random Reference Points Verification (Lemos et al., ICML 2023)
# ---------------------------------------------------------------------------


def test_tarp_random_reference_points_detect_location_collapse() -> None:
    """Random reference points (Lemos et al., ICML 2023) detect location-collapsed posteriors."""
    p1 = CalibrationPrior(name="theta", target="resource", target_id="r1", bounds=(0.0, 10.0))
    prior_set = PriorSet(priors=(p1,))

    rng = np.random.default_rng(101)
    train_thetas = prior_set.sample(n=120, rng=rng)
    train_xs = train_thetas * 2.0 + rng.normal(0.0, 0.15, size=train_thetas.shape)

    estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    estimator.fit(thetas=train_thetas, xs=train_xs)

    test_thetas = prior_set.sample(n=60, rng=np.random.default_rng(202))
    test_xs = test_thetas * 2.0 + rng.normal(0.0, 0.15, size=test_thetas.shape)

    _, _, max_disc_well, p_val_well = compute_tarp_coverage(
        test_thetas=test_thetas,
        test_xs=test_xs,
        estimator=estimator,
        seed=42,
        reference_mode="random_uniform",
    )
    assert max_disc_well < 0.30
    assert p_val_well > 0.01

    # Verify CoverageCertificate carries verified primary-source TARP citation
    cert = CoverageCertificate(
        is_certified=True,
        status="CERTIFIED",
        decision_usable=True,
        tarp_max_discrepancy=max_disc_well,
        tarp_ks_pvalue=p_val_well,
        sbc_rank_ks_pvalue=p_val_well,
        misspecification_score=0.5,
        is_misspecified=False,
        nominal_alphas=(0.1, 0.5, 0.9),
        empirical_coverages=(0.1, 0.5, 0.9),
    )
    assert "Tests of Accuracy with Random Points" in cert.tarp_expansion
    assert "Lemos et al., ICML 2023" in cert.tarp_expansion
    assert "arXiv:2302.03026" in cert.tarp_expansion


# ---------------------------------------------------------------------------
# 5. Conformal Marginal vs. Conditional Coverage & Distribution Shift
# ---------------------------------------------------------------------------


def test_conformal_marginal_vs_slice_conditional_coverage() -> None:
    """Conformal prediction guarantees marginal coverage but exposes conditional slice under-coverage."""
    rng = np.random.default_rng(42)
    n_cal = 200
    n_test = 300

    # Heteroscedastic errors: slice "easy" (80% of data) has sigma=0.5; slice "hard" (20%) has sigma=5.0
    cal_slices = ["easy" if i < int(0.8 * n_cal) else "hard" for i in range(n_cal)]
    cal_preds = np.zeros(n_cal)
    cal_actuals = np.array(
        [rng.normal(0.0, 0.5) if s == "easy" else rng.normal(0.0, 5.0) for s in cal_slices]
    )
    # Shuffle calibration order so it is exchangeable
    perm = rng.permutation(n_cal)
    cal_preds = cal_preds[perm]
    cal_actuals = cal_actuals[perm]

    test_slices = ["easy" if i < int(0.8 * n_test) else "hard" for i in range(n_test)]
    test_preds = np.zeros(n_test)
    test_actuals = np.array(
        [rng.normal(0.0, 0.5) if s == "easy" else rng.normal(0.0, 5.0) for s in test_slices]
    )

    predictor = ConformalIntervalPredictor(
        nominal_coverage=0.90, calibration_population="heteroscedastic_synthetic_population"
    )
    predictor.calibrate(actuals=cal_actuals, predictions=cal_preds)
    summary = predictor.evaluate(
        actuals=test_actuals, predictions=test_preds, slice_labels=test_slices
    )

    assert summary.coverage_scope == "marginal"
    assert summary.calibration_population == "heteroscedastic_synthetic_population"
    assert summary.calibration_sample_size == n_cal
    # Marginal coverage is close to 90%
    assert 0.84 <= summary.empirical_coverage <= 0.96
    # Conditional coverage on the "hard" slice is severely under-covered (< 65%)
    assert summary.slice_coverages["hard"] < 0.65
    assert summary.slice_coverages["easy"] > 0.95
    assert summary.slice_coverage_gap_warning is True


def test_conformal_distribution_shift_and_aci_recovery() -> None:
    """Under post-calibration regime shift, static split conformal under-covers while online ACI recovers."""
    rng = np.random.default_rng(909)
    cal_preds = np.zeros(150)
    cal_actuals = rng.normal(0.0, 1.0, size=150)

    id_test_preds = np.zeros(150)
    id_test_actuals = rng.normal(0.0, 1.0, size=150)

    # Shifted test regime: scale inflates 3x + bias shift
    shifted_preds = np.zeros(200)
    shifted_actuals = rng.normal(1.5, 3.0, size=200)

    shift_res = evaluate_conformal_under_shift(
        cal_actuals=cal_actuals,
        cal_predictions=cal_preds,
        id_test_actuals=id_test_actuals,
        id_test_predictions=id_test_preds,
        shifted_test_actuals=shifted_actuals,
        shifted_test_predictions=shifted_preds,
        nominal_coverage=0.90,
        aci_gamma=0.08,
    )

    assert shift_res.exchangeability_violated is True
    assert shift_res.shifted_split_conformal_coverage < 0.65
    assert (
        shift_res.shifted_aci_empirical_coverage > shift_res.shifted_split_conformal_coverage + 0.15
    )


# ---------------------------------------------------------------------------
# 6. Causal Hierarchy & Provenance Enforcement
# ---------------------------------------------------------------------------


def test_causal_claim_requires_identification_and_provenance() -> None:
    """Causal claims fail closed when unidentifiable, overlap-violated, or lacking provenance."""
    # 1. Missing provenance when require_provenance=True raises UnqualifiedCausalClaimError
    with pytest.raises(UnqualifiedCausalClaimError, match="provenance is missing"):
        validate_causal_claim(
            claimed_level=EvidenceLevel.INTERVENTIONAL,
            is_identifiable=True,
            overlap_satisfied=True,
            identification_provenance="",
            require_provenance=True,
            strict=True,
        )

    # 2. Non-strict mode downgrades to PREDICTIVE with warning
    with pytest.warns(InsufficientCausalEvidenceWarning):
        downgraded = validate_causal_claim(
            claimed_level=EvidenceLevel.INTERVENTIONAL,
            is_identifiable=True,
            overlap_satisfied=True,
            identification_provenance=None,
            require_provenance=True,
            strict=False,
        )
    assert downgraded == EvidenceLevel.PREDICTIVE

    # 3. Backdoor check records explicit identification provenance
    dag = CausalGraph(
        nodes={"Z", "X", "Y"},
        edges={("Z", "X"), ("Z", "Y"), ("X", "Y")},
    )
    id_unadjusted = check_backdoor_identifiability(
        dag, treatment="X", outcome="Y", conditioning_set=()
    )
    assert not id_unadjusted.is_identifiable
    assert id_unadjusted.causal_level == EvidenceLevel.PREDICTIVE
    assert id_unadjusted.identification_provenance == "unidentified_open_backdoor_paths"

    id_adjusted = check_backdoor_identifiability(
        dag, treatment="X", outcome="Y", conditioning_set=("Z",)
    )
    assert id_adjusted.is_identifiable
    assert id_adjusted.causal_level == EvidenceLevel.INTERVENTIONAL
    assert "declared_dag_backdoor_adjustment" in id_adjusted.identification_provenance

    # 4. ScenarioForecastResult carries explicit P_model estimand
    dist = summarize_distribution([10.0, 12.0, 14.0])
    forecast = ScenarioForecastResult(
        metric_name="throughput",
        point_estimate=dist.mean,
        ci_lower=dist.p05,
        ci_upper=dist.p95,
        distribution=dist,
    )
    assert forecast.estimand == "E_{P_model}[Y]"


def test_metamorphic_affine_invariance_of_bootstrap_ci() -> None:
    """Metamorphic test: affine transformation Y' = a*Y + c shifts and scales bootstrap bounds exactly."""
    rng = np.random.default_rng(8080)
    y = rng.normal(loc=5.0, scale=2.0, size=50)
    a, c = 3.5, -12.0
    y_transformed = a * y + c

    ci_orig = compute_bootstrap_ci(y, confidence_level=0.95, n_resamples=800, seed=55)
    ci_trans = compute_bootstrap_ci(y_transformed, confidence_level=0.95, n_resamples=800, seed=55)

    assert math.isclose(ci_trans.mean, a * ci_orig.mean + c, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ci_trans.ci_lower, a * ci_orig.ci_lower + c, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(ci_trans.ci_upper, a * ci_orig.ci_upper + c, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(
        ci_trans.standard_error, a * ci_orig.standard_error, rel_tol=1e-12, abs_tol=1e-12
    )


def test_tarp_random_reference_catches_antipodal_bias_that_fixed_mean_misses() -> None:
    """Prove Lemos et al. (ICML 2023) Theorem 1: random reference points catch antipodal bias invisible to fixed_mean."""
    p1 = CalibrationPrior(name="theta", target="resource", target_id="r1", bounds=(0.0, 10.0))
    prior_set = PriorSet(priors=(p1,))

    rng = np.random.default_rng(505)
    test_thetas = prior_set.sample(n=80, rng=rng)
    mean_theta = float(np.mean(test_thetas))

    # Construct an antipodally biased training set where theta_hat = 2 * mean_theta - theta_true
    # Every posterior sample is reflected across mean_theta, so distance to fixed_mean is identical!
    train_thetas = prior_set.sample(n=150, rng=rng)
    antipodal_thetas = 2.0 * mean_theta - train_thetas
    train_xs = train_thetas * 2.0

    biased_estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    biased_estimator.fit(thetas=antipodal_thetas, xs=train_xs)

    test_xs = test_thetas * 2.0

    _, _, disc_fixed, _ = compute_tarp_coverage(
        test_thetas=test_thetas,
        test_xs=test_xs,
        estimator=biased_estimator,
        seed=42,
        reference_mode="fixed_mean",
    )
    _, _, disc_random, _ = compute_tarp_coverage(
        test_thetas=test_thetas,
        test_xs=test_xs,
        estimator=biased_estimator,
        seed=42,
        reference_mode="random_uniform",
    )

    # Random reference points expose the severe antipodal location bias
    assert disc_random > disc_fixed
