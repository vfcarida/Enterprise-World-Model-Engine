"""End-to-end integration and comparative benchmark tests for the AIOps World Model."""

from __future__ import annotations

import pytest

from ewm_engine.aiops.evaluation.evaluator import AIOpsWorldModelEvaluator


@pytest.mark.integration
def test_end_to_end_aiops_world_model_comparative_evaluation() -> None:
    """Validate MTTD advance warning (30-60 min), MTTR (< 5 min), and RCA accuracy vs baselines."""
    evaluator = AIOpsWorldModelEvaluator()
    results = evaluator.run_comparative_evaluation(
        incidents=("memory_leak", "bad_deployment", "traffic_spike"),
        seed=42,
    )

    expected_models = {
        "Threshold Alerts",
        "Pearson Correlation RCA",
        "Full AIOps World Model",
    }
    assert set(results.keys()) == expected_models

    aiops_metrics = results["Full AIOps World Model"]
    threshold_metrics = results["Threshold Alerts"]
    pearson_metrics = results["Pearson Correlation RCA"]

    assert len(aiops_metrics) == 3
    assert len(threshold_metrics) == 3
    assert len(pearson_metrics) == 3

    # 1. Verify Full AIOps World Model isolates the true root cause across all 3 cascading incidents
    aiops_rca_accuracy = sum(1 for m in aiops_metrics if m.root_cause_correct) / len(aiops_metrics)
    threshold_rca_accuracy = sum(1 for m in threshold_metrics if m.root_cause_correct) / len(
        threshold_metrics
    )
    pearson_rca_accuracy = sum(1 for m in pearson_metrics if m.root_cause_correct) / len(
        pearson_metrics
    )

    assert aiops_rca_accuracy == pytest.approx(1.0)
    assert aiops_rca_accuracy > threshold_rca_accuracy
    assert aiops_rca_accuracy > pearson_rca_accuracy

    # 2. Verify MTTD improvement: 30-60 minutes advance warning vs. reactive threshold alerts (< 5 min)
    for m_aiops, m_thresh in zip(aiops_metrics, threshold_metrics, strict=True):
        assert 30.0 <= m_aiops.mttd_advance_warning_minutes <= 60.0
        assert m_thresh.mttd_advance_warning_minutes < 5.0
        assert m_aiops.mttd_advance_warning_minutes > m_thresh.mttd_advance_warning_minutes + 25.0

    # 3. Verify MTTR reduction: AI-remediated < 5 min vs. human-remediated > 20 min
    for m_aiops, m_thresh, m_pearson in zip(
        aiops_metrics, threshold_metrics, pearson_metrics, strict=True
    ):
        assert m_aiops.mttr_minutes < 5.0
        assert m_thresh.mttr_minutes > 20.0
        assert m_pearson.mttr_minutes > 20.0
        assert m_aiops.opa_allowed is True
        assert m_aiops.chaos_validated is True
        assert m_aiops.cascade_f1_score >= 0.80
