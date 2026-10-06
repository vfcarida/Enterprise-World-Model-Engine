"""Unit tests for validation scorers and walk-forward rolling-origin backtesting."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.experimentation.backtest import (
    BacktestReport,
    run_walk_forward_backtest,
    score_crps,
    score_empirical_coverage,
    score_moments_distance,
    score_rmse,
    score_spectral_distance,
)


def test_scorers_rmse() -> None:
    """Test RMSE computation under exact match and known error."""
    assert score_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
    # Constant offset of 2.0 -> RMSE = 2.0
    assert score_rmse([3.0, 4.0, 5.0], [1.0, 2.0, 3.0]) == pytest.approx(2.0)
    # Empty
    assert score_rmse([], []) == 0.0
    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        score_rmse([1.0], [1.0, 2.0])


def test_scorers_moments_distance() -> None:
    """Test method-of-moments distance against scalar observations."""
    ensemble = [10.0, 12.0, 14.0]  # mean = 12.0, std = sqrt(8/3) ~ 1.633
    obs = 10.0
    moments = score_moments_distance(ensemble, obs)
    assert moments["mean"] == pytest.approx(12.0)
    assert moments["bias"] == pytest.approx(2.0)
    assert moments["abs_error"] == pytest.approx(2.0)
    assert moments["z_score"] > 0.0


def test_scorers_crps() -> None:
    """Test Continuous Ranked Probability Score (CRPS)."""
    # Deterministic ensemble matching observation -> CRPS = 0.0
    assert score_crps([5.0, 5.0, 5.0], 5.0) == pytest.approx(0.0, abs=1e-6)
    # Dispersion around observation
    ensemble = [8.0, 10.0, 12.0]
    crps_val = score_crps(ensemble, 10.0)
    assert crps_val >= 0.0
    # Further observation increases CRPS
    crps_distant = score_crps(ensemble, 20.0)
    assert crps_distant > crps_val


def test_scorers_empirical_coverage() -> None:
    """Test empirical coverage across credible intervals (SBC/TARP concept)."""
    # Ensemble of 100 samples per step across horizon of 3 steps
    rng = np.random.default_rng(42)
    ensemble = rng.normal(loc=10.0, scale=1.0, size=(100, 3))
    # Observed point exactly at mean
    observed = [10.0, 10.0, 10.0]
    cov = score_empirical_coverage(ensemble, observed, credible_intervals=(0.50, 0.80, 0.95))

    # Center observation should be contained in all credible intervals
    assert cov["ci_50"] == 1.0
    assert cov["ci_80"] == 1.0
    assert cov["ci_95"] == 1.0

    # Outlier observation far beyond distribution
    outlier = [100.0, 100.0, 100.0]
    cov_outlier = score_empirical_coverage(ensemble, outlier)
    assert cov_outlier["ci_95"] == 0.0


def test_scorers_spectral_distance() -> None:
    """Test frequency-domain spectral Fourier distance."""
    # Identical series -> 0.0
    s1 = [1.0, 2.0, 3.0, 2.0, 1.0]
    assert score_spectral_distance(s1, s1) == pytest.approx(0.0, abs=1e-6)
    # Different frequencies
    s2 = [1.0, -1.0, 1.0, -1.0, 1.0]
    dist = score_spectral_distance(s1, s2)
    assert dist > 0.0


def test_run_walk_forward_backtest() -> None:
    """Test walk-forward rolling-origin backtest on a synthetic state sequence."""
    total_steps = 10
    world_history: list[World] = []
    observed_signal: list[float] = []

    for i in range(total_steps):
        stock_val = 100.0 - 5.0 * i
        world_history.append(
            World(
                initial_state=WorldState(
                    step=i,
                    entities=[Entity(id="warehouse", type="facility")],
                    resources=[Resource(id="inventory", current=stock_val)],
                )
            )
        )
        observed_signal.append(stock_val)

    report: BacktestReport = run_walk_forward_backtest(
        world_history=world_history,
        observed_signal_history=observed_signal,
        target_signal_name="inventory",
        horizon=3,
        step_size=2,
        samples=4,
        seed=42,
    )

    assert report.target_signal == "inventory"
    assert report.window_count > 0
    assert len(report.windows) == report.window_count
    assert report.mean_rmse >= 0.0
    assert report.mean_crps >= 0.0
    assert "ci_50" in report.aggregate_coverage
    assert "ci_95" in report.aggregate_coverage
    assert len(report.fingerprint) == 64


def test_backtest_scorers_edge_cases() -> None:
    """Test scorers on empty arrays, zero lengths, and dimension mismatches."""
    # RMSE empty and length mismatch
    assert score_rmse([], []) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        score_rmse([1.0], [1.0, 2.0])

    # Moments empty
    moments_empty = score_moments_distance([], 10.0)
    assert moments_empty["bias"] == 0.0

    # CRPS empty
    assert score_crps([], 5.0) == 0.0

    # Spectral empty and length mismatch
    assert score_spectral_distance([], []) == 0.0
    with pytest.raises(ValueError, match="lengths must match"):
        score_spectral_distance([1.0], [1.0, 2.0])


def test_backtest_default_extractor_signals() -> None:
    """Test default extractor with state memory signal."""
    world = World(initial_state=WorldState(step=0, memory={"latency": 40.0}))
    rep = run_walk_forward_backtest(
        world_history=[world, world, world],
        observed_signal_history=[40.0, 42.0, 44.0],
        target_signal_name="latency",
        horizon=1,
        step_size=1,
        samples=2,
    )
    assert rep.window_count >= 1
