"""Unit tests for the Software Engineering Progressive-Delivery application."""

from __future__ import annotations

from examples.swe import run_swe_deployment_application


def test_swe_deployment_application_captures_shared_pool_starvation_and_negatives() -> None:
    summary = run_swe_deployment_application(seed=42, n_seeds=3)
    assert "payment_worker" not in summary.static_dependents_of_auth_lib
    assert "payment_worker" in summary.dynamic_impacted_under_burst
    assert summary.payment_worker_missed_by_static_graph is True
    assert summary.static_graph_deployment_regret_usd > 10_000.0
    assert summary.ewm_deployment_regret_usd == 0.0
    assert summary.bitemporal_withheld_records_at_t5 > 0
    assert summary.negative_findings_confirmed == 3
    assert len(summary.output_fingerprint) == 64
