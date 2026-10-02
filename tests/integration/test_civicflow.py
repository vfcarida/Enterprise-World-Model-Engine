"""Integration test executing the CivicFlow flagship simulation demonstration."""

from __future__ import annotations

from examples.civicflow.run import run_civicflow_simulation


def test_civicflow_simulation_execution() -> None:
    """Ensure CivicFlow flood response simulation runs and achieves expected policy deltas."""
    comparison = run_civicflow_simulation()
    assert comparison is not None
    assert comparison.baseline_name == "Policy A (Myopic Nearest)"
    assert "Policy B (Proactive Regional)" in comparison.results_by_scenario

    # Verify Policy B achieves significant (>20%) reduction in unserved relief supplies
    water_delta = comparison.deltas["Policy B (Proactive Regional)"][
        "mem_cumulative_unserved_water"
    ]
    rations_delta = comparison.deltas["Policy B (Proactive Regional)"][
        "mem_cumulative_unserved_rations"
    ]

    assert water_delta["percent_delta"] < -20.0
    assert rations_delta["percent_delta"] < -20.0

    # Verify zero constraint violations in both policy scenarios
    base_violations = comparison.results_by_scenario["Policy A (Myopic Nearest)"][
        "violations_count"
    ].mean
    cand_violations = comparison.results_by_scenario["Policy B (Proactive Regional)"][
        "violations_count"
    ].mean
    assert base_violations == 0.0
    assert cand_violations == 0.0
