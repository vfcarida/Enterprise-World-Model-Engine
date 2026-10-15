"""Unit tests for calibration prior distributions and prior sets."""

from __future__ import annotations

import numpy as np

from ewm_engine.calibration.priors import CalibrationPrior, PriorSet


def test_uniform_prior_sampling_and_bounds() -> None:
    prior = CalibrationPrior(
        name="initial_inventory",
        target="resource",
        target_id="stock_main",
        distribution="uniform",
        bounds=(10.0, 50.0),
    )
    rng = np.random.default_rng(42)
    samples = prior.sample(100, rng)

    assert len(samples) == 100
    assert np.all(samples >= 10.0)
    assert np.all(samples <= 50.0)

    # In bounds vs out of bounds log_prob
    logp = prior.log_prob(np.array([20.0, 5.0, 55.0]))
    assert np.isfinite(logp[0])
    assert logp[1] == -np.inf
    assert logp[2] == -np.inf


def test_gaussian_prior_sampling_and_clipping() -> None:
    prior = CalibrationPrior(
        name="transfer_rate",
        target="parameter",
        target_id="rate_k",
        distribution="gaussian",
        bounds=(0.0, 1.0),
        mean=0.5,
        std=0.1,
    )
    rng = np.random.default_rng(123)
    samples = prior.sample(200, rng)

    assert len(samples) == 200
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)
    assert 0.4 < float(np.mean(samples)) < 0.6

    # Log probability peak at mean
    logp_mean = prior.log_prob(np.array([0.5]))
    logp_tail = prior.log_prob(np.array([0.1]))
    assert logp_mean[0] > logp_tail[0]


def test_log_uniform_prior() -> None:
    prior = CalibrationPrior(
        name="scale_factor",
        target="context",
        target_id="market_volatility",
        distribution="log_uniform",
        bounds=(0.01, 100.0),
    )
    rng = np.random.default_rng(999)
    samples = prior.sample(300, rng)

    assert np.all(samples >= 0.01)
    assert np.all(samples <= 100.0)
    # Median of log-uniform on [0.01, 100] is sqrt(0.01 * 100) = 1.0
    median_val = float(np.median(samples))
    assert 0.5 < median_val < 2.0


def test_prior_set_multivariate() -> None:
    p1 = CalibrationPrior(name="theta_1", target="resource", target_id="r1", distribution="uniform", bounds=(0.0, 10.0))
    p2 = CalibrationPrior(name="theta_2", target="parameter", target_id="p2", distribution="gaussian", bounds=(10.0, 20.0), mean=15.0, std=1.0)

    prior_set = PriorSet(priors=(p1, p2))
    assert prior_set.dim == 2
    assert prior_set.names == ("theta_1", "theta_2")

    rng = np.random.default_rng(42)
    thetas = prior_set.sample(50, rng)
    assert thetas.shape == (50, 2)
    assert np.all(thetas[:, 0] >= 0.0) and np.all(thetas[:, 0] <= 10.0)
    assert np.all(thetas[:, 1] >= 10.0) and np.all(thetas[:, 1] <= 20.0)

    # Conversion to dict list
    dicts = prior_set.to_dict_list(thetas)
    assert len(dicts) == 50
    assert set(dicts[0].keys()) == {"theta_1", "theta_2"}

    # Joint log_prob
    logp = prior_set.log_prob(thetas)
    assert len(logp) == 50
    assert np.all(np.isfinite(logp))
