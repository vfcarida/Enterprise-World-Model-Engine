"""Automated performance and throughput regression guard tests (AUDIT-003, AC-P03-1)."""

from __future__ import annotations

import pytest

from benchmarks.run_benchmarks import (
    benchmark_scale_out_comparison,
    benchmark_simulation_throughput,
)


@pytest.mark.benchmark
def test_simulation_throughput_benchmark_regression_guard() -> None:
    """Assert simulation engine achieves minimum throughput threshold on reference topology."""
    results = benchmark_simulation_throughput(num_nodes=20, horizon=15, samples=10)

    assert results["total_rollout_steps"] == 150.0
    assert results["total_trajectories"] == 10.0
    assert results["elapsed_seconds"] > 0.0

    # Conservative threshold to prevent CI flakiness across varied hardware
    min_steps_per_sec = 50.0
    assert results["steps_per_second"] >= min_steps_per_sec, (
        f"Throughput regression detected: {results['steps_per_second']:.1f} steps/s "
        f"is below threshold of {min_steps_per_sec} steps/s"
    )


@pytest.mark.benchmark
def test_scale_out_parallel_benchmark_determinism() -> None:
    """Assert scale-out benchmark produces verified identical results and positive throughput."""
    results = benchmark_scale_out_comparison(num_nodes=15, horizon=10, samples=12, workers=2)

    assert results["determinism_verified"] is True
    assert results["samples"] == 12
    assert results["horizon"] == 10
    assert results["mp_steps_per_sec"] > 0.0
    assert results["serial_steps_per_sec"] > 0.0
