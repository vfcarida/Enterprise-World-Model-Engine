"""Unit tests for mandatory coverage certification (TARP, Posterior-SBC, Misspecification)."""

from __future__ import annotations

import numpy as np

from ewm_engine.calibration.certification import (
    certify_coverage,
    compute_misspecification_diagnostic,
    compute_tarp_coverage,
)
from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator


def test_tarp_coverage_well_calibrated() -> None:
    p1 = CalibrationPrior(name="th1", target="resource", target_id="r1", bounds=(0.0, 10.0))
    prior_set = PriorSet(priors=(p1,))

    rng = np.random.default_rng(42)
    thetas = prior_set.sample(n=100, rng=rng)
    xs = thetas * 2.0 + rng.normal(0, 0.2, size=thetas.shape)

    estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    estimator.fit(thetas=thetas, xs=xs)

    test_thetas = prior_set.sample(n=30, rng=np.random.default_rng(99))
    test_xs = test_thetas * 2.0 + rng.normal(0, 0.2, size=test_thetas.shape)

    alphas, coverages, max_disc, p_val = compute_tarp_coverage(
        test_thetas=test_thetas,
        test_xs=test_xs,
        estimator=estimator,
        seed=42,
    )

    assert len(alphas) == len(coverages)
    assert max_disc < 0.35
    assert p_val > 0.001


def test_misspecification_diagnostic() -> None:
    simulated_xs = np.random.default_rng(42).normal(loc=50.0, scale=5.0, size=(100, 2))

    # Well-specified observation
    x_in_support = np.array([52.0, 48.0])
    score_in, is_misspec_in = compute_misspecification_diagnostic(x_in_support, simulated_xs)
    assert score_in < 2.0
    assert not is_misspec_in

    # Grossly misspecified observation (e.g. 500.0 instead of 50.0)
    x_out = np.array([500.0, 500.0])
    score_out, is_misspec_out = compute_misspecification_diagnostic(x_out, simulated_xs)
    assert score_out > 20.0
    assert is_misspec_out


def test_certify_coverage_refusal_on_misspecification() -> None:
    p1 = CalibrationPrior(name="param", target="resource", target_id="r", bounds=(0.0, 10.0))
    prior_set = PriorSet(priors=(p1,))

    estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    thetas = np.array([[2.0], [4.0], [6.0], [8.0]])
    xs = np.array([[4.0], [8.0], [12.0], [16.0]])
    estimator.fit(thetas=thetas, xs=xs)

    # Simulated target data that is hopelessly misspecified (x_obs = 99999.0)
    x_obs_bad = np.array([99999.0])
    cert = certify_coverage(
        estimator=estimator,
        test_thetas=thetas,
        test_xs=xs,
        x_obs=x_obs_bad,
        seed=42,
    )

    assert not cert.is_certified
    assert not cert.decision_usable
    assert cert.status == "UNCERTIFIED_MISSPECIFIED"
    assert cert.is_misspecified
    assert "Misspecification" in (cert.refusal_reason or "")


def test_certify_coverage_refusal_on_under_coverage() -> None:
    p1 = CalibrationPrior(name="param", target="resource", target_id="r", bounds=(0.0, 10.0))
    prior_set = PriorSet(priors=(p1,))

    estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    thetas = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    estimator.fit(thetas=thetas, xs=xs)

    # Force a very tight discrepancy threshold to trigger refusal
    cert = certify_coverage(
        estimator=estimator,
        test_thetas=thetas,
        test_xs=xs,
        x_obs=np.array([3.0]),
        max_discrepancy_threshold=0.0001,  # Impolitely strict threshold
        seed=42,
    )

    assert not cert.is_certified
    assert not cert.decision_usable
    assert cert.status in ("UNCERTIFIED_OVERCONFIDENT", "UNCERTIFIED_UNDERCOVERED")
    assert cert.refusal_reason is not None
