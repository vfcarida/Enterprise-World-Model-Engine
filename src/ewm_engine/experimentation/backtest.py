"""Validation and walk-forward backtest harness.

Conforms to Track T4: Core Substrate (Zero-dep, pure NumPy/stdlib).
Evaluates simulation ensemble predictions against empirical observed time series
using method-of-moments distance, empirical coverage (SBC/TARP concept), RMSE,
CRPS, and spectral Fourier distance under rolling-origin walk-forward cross-validation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.world import World
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory


def score_rmse(
    predictions: Sequence[float] | np.ndarray, observed: Sequence[float] | np.ndarray
) -> float:
    """Compute Root Mean Squared Error (RMSE) between predicted and observed series."""
    pred_arr = np.asarray(predictions, dtype=np.float64)
    obs_arr = np.asarray(observed, dtype=np.float64)
    if len(pred_arr) != len(obs_arr):
        raise ValueError(
            f"Length mismatch: {len(pred_arr)} predictions vs {len(obs_arr)} observed."
        )
    if len(pred_arr) == 0:
        return 0.0
    return float(np.sqrt(np.mean((pred_arr - obs_arr) ** 2)))


def score_moments_distance(
    ensemble_samples: Sequence[float] | np.ndarray,
    observed_value: float,
) -> dict[str, float]:
    """Compute method-of-moments distances (mean bias, standard deviation, standardized z-score)."""
    samples = np.asarray(ensemble_samples, dtype=np.float64)
    if len(samples) == 0:
        return {"bias": 0.0, "std": 0.0, "z_score": 0.0, "abs_error": 0.0}

    mean_val = float(np.mean(samples))
    std_val = float(np.std(samples))
    bias = mean_val - float(observed_value)
    abs_err = abs(bias)
    z_score = float(bias / std_val) if std_val > 1e-12 else 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "bias": bias,
        "abs_error": abs_err,
        "z_score": z_score,
    }


def score_empirical_coverage(
    ensemble_matrix: np.ndarray,
    observed_series: Sequence[float],
    credible_intervals: Sequence[float] = (0.50, 0.80, 0.90, 0.95),
) -> dict[str, float]:
    """Compute empirical coverage fractions (SBC/TARP concept).

    Measures the empirical fraction of time steps where the ground-truth observed value
    falls within the nominal (1 - alpha) quantile confidence intervals of the ensemble.
    """
    obs = np.asarray(observed_series, dtype=np.float64)
    # ensemble_matrix is shape (n_rollouts, horizon)
    if ensemble_matrix.ndim != 2:
        raise ValueError(
            f"ensemble_matrix must be 2D (samples, horizon); got {ensemble_matrix.shape}"
        )
    if ensemble_matrix.shape[1] != len(obs):
        raise ValueError(
            f"Horizon mismatch: ensemble horizon {ensemble_matrix.shape[1]} vs observed {len(obs)}"
        )

    coverage_results: dict[str, float] = {}

    for ci in credible_intervals:
        alpha = 1.0 - ci
        lower_q = alpha / 2.0 * 100.0
        upper_q = (1.0 - alpha / 2.0) * 100.0

        lowers = np.percentile(ensemble_matrix, lower_q, axis=0)
        uppers = np.percentile(ensemble_matrix, upper_q, axis=0)

        in_interval = (obs >= lowers) & (obs <= uppers)
        empirical_cov = float(np.mean(in_interval))
        ci_key = f"ci_{round(ci * 100)}"
        coverage_results[ci_key] = empirical_cov

    return coverage_results


def score_crps(ensemble_samples: Sequence[float] | np.ndarray, observed_value: float) -> float:
    """Compute Continuous Ranked Probability Score (CRPS) for an ensemble against a scalar observation.

    Uses the exact pure-NumPy representation:
    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
    """
    x = np.asarray(ensemble_samples, dtype=np.float64)
    y = float(observed_value)
    m = len(x)
    if m == 0:
        return 0.0
    term1 = float(np.mean(np.abs(x - y)))
    # Pairwise differences
    diffs = np.abs(x[:, None] - x[None, :])
    term2 = float(np.sum(diffs) / (2.0 * m * m))
    return float(term1 - term2)


def score_spectral_distance(
    predicted_series: Sequence[float] | np.ndarray,
    observed_series: Sequence[float] | np.ndarray,
) -> float:
    """Compute spectral Fourier distance between predicted and observed timeseries.

    Measures frequency-domain discrepancy via Euclidean distance between 1D FFT magnitude spectra.
    """
    p = np.asarray(predicted_series, dtype=np.float64)
    o = np.asarray(observed_series, dtype=np.float64)
    if len(p) != len(o):
        raise ValueError(f"Series lengths must match; got {len(p)} vs {len(o)}")
    if len(p) == 0:
        return 0.0

    fft_p = np.abs(np.fft.rfft(p))
    fft_o = np.abs(np.fft.rfft(o))
    return float(np.linalg.norm(fft_p - fft_o) / len(fft_p))


class BacktestWindow(BaseModel):
    """Evaluation outcomes for an individual rolling-origin window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    origin_step: int = Field(ge=0, description="Step index from which rollouts originated.")
    horizon: int = Field(ge=1, description="Forecast horizon steps.")
    rmse: float = Field(ge=0.0, description="Root Mean Squared Error.")
    mean_crps: float = Field(ge=0.0, description="Average CRPS across the forecast horizon.")
    spectral_distance: float = Field(ge=0.0, description="Frequency-domain spectral distance.")
    moments: dict[str, float] = Field(description="Method-of-moments distances at horizon end.")
    coverage: dict[str, float] = Field(description="Empirical coverage across credible intervals.")


class BacktestReport(BaseModel):
    """Comprehensive walk-forward backtest evaluation report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_signal: str = Field(description="Target resource or variable evaluated.")
    window_count: int = Field(ge=0, description="Number of rolling-origin windows evaluated.")
    samples_per_window: int = Field(ge=1, description="Number of Monte Carlo rollouts per origin.")
    mean_rmse: float = Field(ge=0.0, description="Average RMSE across all walk-forward windows.")
    mean_crps: float = Field(ge=0.0, description="Average CRPS across all walk-forward windows.")
    mean_spectral_distance: float = Field(ge=0.0, description="Average spectral Fourier distance.")
    aggregate_coverage: dict[str, float] = Field(
        description="Empirical coverage aggregated over all forecast steps."
    )
    windows: tuple[BacktestWindow, ...] = Field(description="Individual origin window reports.")
    fingerprint: str = Field(description="Cryptographic SHA-256 evaluation fingerprint.")


def run_walk_forward_backtest(
    world_history: Sequence[World],
    observed_signal_history: Sequence[float],
    target_signal_name: str,
    horizon: int = 5,
    step_size: int = 1,
    samples: int = 10,
    seed: int = 42,
    signal_extractor: Callable[[Trajectory], Sequence[float]] | None = None,
    engine: SimulationEngine | None = None,
) -> BacktestReport:
    """Execute rolling-origin walk-forward backtesting comparing rollouts against historical observations."""
    sim_engine = engine or SimulationEngine()
    total_steps = len(observed_signal_history)

    def default_extractor(traj: Trajectory) -> list[float]:
        # Extract signal timeseries across steps
        series: list[float] = []
        for step_rec in traj.steps:
            if step_rec.transition_result is not None:
                st = step_rec.transition_result.next_state
                if target_signal_name in st.resources:
                    series.append(float(st.resources[target_signal_name].current))
                elif target_signal_name in st.memory:
                    series.append(float(st.memory[target_signal_name]))
                elif (
                    step_rec.step_metrics is not None
                    and target_signal_name in step_rec.step_metrics.metrics
                ):
                    series.append(float(step_rec.step_metrics.metrics[target_signal_name]))
                else:
                    series.append(0.0)
        return series

    extractor = signal_extractor or default_extractor

    windows: list[BacktestWindow] = []
    ss = np.random.SeedSequence(seed)

    origin_steps = list(range(0, total_steps - horizon, step_size))
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(origin_steps))]

    for origin, origin_seed in zip(origin_steps, child_seeds, strict=True):
        origin_world = world_history[origin]
        target_observed = observed_signal_history[origin + 1 : origin + 1 + horizon]
        if len(target_observed) < horizon:
            continue

        scenario = Scenario(
            scenario_id=f"backtest_origin_{origin}",
            horizon=horizon,
            samples=samples,
            seed=origin_seed,
        )

        sim_res = sim_engine.run(origin_world, scenario)
        ensemble_runs: list[list[float]] = []
        for traj in sim_res.trajectories:
            series = list(extractor(traj))
            # pad or truncate to horizon
            if len(series) < horizon:
                pad_val = series[-1] if series else 0.0
                series = series + [pad_val] * (horizon - len(series))
            ensemble_runs.append(series[:horizon])

        ensemble_arr = np.array(ensemble_runs, dtype=np.float64)  # (samples, horizon)
        mean_prediction = np.mean(ensemble_arr, axis=0)

        # Compute window scores
        rmse_val = score_rmse(mean_prediction, target_observed)
        crps_vals = [score_crps(ensemble_arr[:, h], target_observed[h]) for h in range(horizon)]
        mean_crps = float(np.mean(crps_vals))
        spectral_dist = score_spectral_distance(mean_prediction, target_observed)
        moments = score_moments_distance(ensemble_arr[:, -1], target_observed[-1])
        coverage = score_empirical_coverage(ensemble_arr, target_observed)

        windows.append(
            BacktestWindow(
                origin_step=origin,
                horizon=horizon,
                rmse=rmse_val,
                mean_crps=mean_crps,
                spectral_distance=spectral_dist,
                moments=moments,
                coverage=coverage,
            )
        )

    if not windows:
        return BacktestReport(
            target_signal=target_signal_name,
            window_count=0,
            samples_per_window=samples,
            mean_rmse=0.0,
            mean_crps=0.0,
            mean_spectral_distance=0.0,
            aggregate_coverage={},
            windows=(),
            fingerprint="0" * 64,
        )

    mean_rmse = float(np.mean([w.rmse for w in windows]))
    mean_crps_all = float(np.mean([w.mean_crps for w in windows]))
    mean_spec_all = float(np.mean([w.spectral_distance for w in windows]))

    # Aggregate coverage
    all_cis = list(windows[0].coverage.keys())
    agg_coverage: dict[str, float] = {}
    for ci_key in all_cis:
        agg_coverage[ci_key] = float(np.mean([w.coverage[ci_key] for w in windows]))

    fingerprint = canonical_sha256(
        {
            "target_signal": target_signal_name,
            "window_count": len(windows),
            "samples_per_window": samples,
            "mean_rmse": mean_rmse,
            "mean_crps": mean_crps_all,
            "aggregate_coverage": agg_coverage,
        }
    )

    return BacktestReport(
        target_signal=target_signal_name,
        window_count=len(windows),
        samples_per_window=samples,
        mean_rmse=mean_rmse,
        mean_crps=mean_crps_all,
        mean_spectral_distance=mean_spec_all,
        aggregate_coverage=agg_coverage,
        windows=tuple(windows),
        fingerprint=fingerprint,
    )
