"""Tests for the ARE/Gaia2-inspired agent evaluation example (AC-014/P05)."""

from __future__ import annotations

import pytest

from examples.agent_evaluation.run import run_agent_evaluation


@pytest.mark.example
def test_agent_evaluation_example_executes() -> None:
    """Verify agent evaluation runs end-to-end and produces verifiable state scores and audit certs."""
    results = run_agent_evaluation()

    assert results is not None
    assert "comparison" in results
    assert "oracle_scores" in results
    assert "unsat_core" in results

    comparison = results["comparison"]
    assert len(comparison.results_by_scenario) == 4
    table = comparison.summary_table()
    assert "Status Quo (No Transfers)" in table
    assert "Greedy Myopic Agent" in table
    assert "Proportional Heuristic Agent" in table
    assert "CP-SAT Optimization Agent" in table
    assert "resource_cumulative_unserved" in table

    scores = results["oracle_scores"]

    # Greedy agent triggered corridor capacity violations
    assert scores["Greedy Myopic"]["invariants_preserved"] is False
    assert scores["Greedy Myopic"]["violations"] > 0
    assert scores["Greedy Myopic"]["composite_verifiable_score"] == 0.0

    # CP-SAT optimizer satisfied all physical constraints and outperformed baseline
    assert scores["CP-SAT Optimizer"]["invariants_preserved"] is True
    assert scores["CP-SAT Optimizer"]["violations"] == 0
    assert scores["CP-SAT Optimizer"]["unserved_demand"] < scores["Status Quo"]["unserved_demand"]
    assert (
        scores["CP-SAT Optimizer"]["unserved_demand"] < scores["Greedy Myopic"]["unserved_demand"]
    )
    assert scores["CP-SAT Optimizer"]["service_level"] > scores["Status Quo"]["service_level"]

    # Infeasibility certificate / unsat core surfaced from CP-SAT under crisis shock
    core = results["unsat_core"]
    assert len(core) >= 2
    assert any("supply_capacity" in c for c in core)
    assert any("demand_target" in c for c in core)
