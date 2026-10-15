"""Unit tests for Amortized SBI Bayesian estimator and posterior distribution representations."""

from __future__ import annotations

import numpy as np

from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator, PosteriorDistribution


def test_posterior_distribution_properties() -> None:
    samples = np.random.default_rng(42).normal(loc=[10.0, 50.0], scale=[1.0, 5.0], size=(1000, 2))
    p1 = CalibrationPrior(name="param_a", target="resource", target_id="ra", bounds=(0.0, 30.0))
    p2 = CalibrationPrior(name="param_b", target="resource", target_id="rb", bounds=(0.0, 100.0))
    prior_set = PriorSet(priors=(p1, p2))

    post = PosteriorDistribution.from_samples(
        samples=samples,
        prior_set=prior_set,
        n_effective_samples=1000.0,
        bandwidth=0.5,
    )

    # Invariant: Never a single point estimate
    assert isinstance(post.mean, dict)
    assert isinstance(post.credible_intervals_90, dict)
    assert "param_a" in post.mean and "param_b" in post.mean

    assert 9.5 < post.mean["param_a"] < 10.5
    assert 48.0 < post.mean["param_b"] < 52.0

    ci_a = post.credible_intervals_90["param_a"]
    assert ci_a[0] < post.mean["param_a"] < ci_a[1]
    assert ci_a[0] >= 0.0 and ci_a[1] <= 30.0

    # Test sampling
    post_draws = post.sample(n=100, seed=123)
    assert post_draws.shape == (100, 2)
    assert np.all(post_draws[:, 0] >= 0.0) and np.all(post_draws[:, 0] <= 30.0)

    # Test log_prob
    logp = post.log_prob(post_draws[:5])
    assert len(logp) == 5
    assert np.all(np.isfinite(logp))


def test_amortized_posterior_estimator_learning() -> None:
    p1 = CalibrationPrior(name="alpha", target="resource", target_id="r1", bounds=(5.0, 25.0))
    p2 = CalibrationPrior(name="beta", target="resource", target_id="r2", bounds=(10.0, 50.0))
    prior_set = PriorSet(priors=(p1, p2))

    # Generate synthetic training pairs: x1 = 2 * alpha + noise, x2 = 3 * beta + noise
    rng = np.random.default_rng(42)
    thetas = prior_set.sample(n=150, rng=rng)
    noise = rng.normal(0, 0.5, size=thetas.shape)
    xs = np.column_stack([2.0 * thetas[:, 0], 3.0 * thetas[:, 1]]) + noise

    estimator = AmortizedPosteriorEstimator(prior_set=prior_set)
    estimator.fit(thetas=thetas, xs=xs)

    # Given observed x_obs = [20.0, 90.0], true theta should be roughly [10.0, 30.0]
    x_obs = np.array([20.0, 90.0])
    posterior = estimator.predict_posterior(x_obs=x_obs, n_samples=300, seed=42)

    assert abs(posterior.mean["alpha"] - 10.0) < 2.5
    assert abs(posterior.mean["beta"] - 30.0) < 3.5

    ci_alpha = posterior.credible_intervals_90["alpha"]
    assert ci_alpha[0] <= 10.0 <= ci_alpha[1]

    ci_beta = posterior.credible_intervals_90["beta"]
    assert ci_beta[0] <= 30.0 <= ci_beta[1]
