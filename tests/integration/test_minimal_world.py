"""Integration test executing the minimal world runnable demonstration."""

from __future__ import annotations

from examples.minimal_world.run import run_minimal_world


def test_minimal_world_execution() -> None:
    """Ensure minimal_world example runs to completion and satisfies behavioral assertions."""
    comparison = run_minimal_world()
    assert comparison is not None
    assert comparison.baseline_name == "Status Quo"
    assert "Proactive Policy" in comparison.results_by_scenario

    # Verify North warehouse stock was transferred to South
    stock_north_sq = comparison.results_by_scenario["Status Quo"]["resource_stock_north"].mean
    stock_north_proactive = comparison.results_by_scenario["Proactive Policy"][
        "resource_stock_north"
    ].mean
    assert stock_north_sq == 150.0
    assert stock_north_proactive == 30.0
    assert stock_north_proactive < stock_north_sq

    # Verify deltas reflect inventory transfer
    delta_north = comparison.deltas["Proactive Policy"]["resource_stock_north"]
    assert delta_north["absolute_delta"] == -120.0
    assert delta_north["percent_delta"] == -80.0

    # Verify constraint violation is detected when attempting to transfer without remaining stock
    proactive_violations = comparison.results_by_scenario["Proactive Policy"][
        "violations_count"
    ].mean
    assert proactive_violations >= 1.0
