"""Self-tests for Decision-First Comparative Evaluation & De-Scripted Suite.

Tests:
1. Pre-registration lock and artifact synchronization.
2. Strict separation of the four capabilities (Prediction, Simulation, Intervention,
   Planning/Decision) across the 5-rung comparative ladder, 4 architectural baselines,
   and 6 single-component ablations.
3. Internal coherence vs. external reality fidelity separation in Simulation.
4. Evidence-nature causal refusal gate in Intervention.
5. Both directions of the Headline Prediction-vs-Decision Gap.
6. Single-component ablations on Rung 5 (including zero-decision-benefit components).
7. First-class negative results (N1-N4) and operational cost / viability metrics.
8. Objective construct validity cautionary cases inside the exact ground-truth EWM.
9. De-scripted Enterprise Benchmark Suite verification (Families A-L).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewm_engine.flagship.comparative_evaluation import (
    PRE_REGISTRATION_ID,
    ComparativeEvaluationReport,
    run_comparative_evaluation,
)
from scripts.run_comparative_evaluation import (
    DEFAULT_JSON_PATH,
    DEFAULT_MD_PATH,
    ENTERPRISE_SUITE_DESCRIPting_AUDIT_ROWS,
    render_comparative_evaluation_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def comparative_report_smoke() -> ComparativeEvaluationReport:
    """Run the 5-seed smoke comparative evaluation once for module-scoped tests."""
    return run_comparative_evaluation(master_seed=42, smoke=True, horizon=20)


def test_pre_registration_locked_and_valid() -> None:
    """Verify that the comparative ladder pre-registration JSON and Markdown exist and match ID."""
    prereg_json = (
        REPO_ROOT / "docs" / "results" / "artifacts" / "pre_registration_comparative_ladder.json"
    )
    prereg_md = REPO_ROOT / "docs" / "scientific" / "pre_registration_comparative_ladder.md"
    assert prereg_json.exists()
    assert prereg_md.exists()

    payload = json.loads(prereg_json.read_text(encoding="utf-8"))
    assert payload["pre_registration_id"] == PRE_REGISTRATION_ID
    assert len(payload["capabilities"]) == 4
    assert len(payload["comparative_ladder"]) == 5
    assert len(payload["negative_result_hypotheses"]) == 4


def test_four_capabilities_separated_and_leak_free(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify all 5 ladder rungs, 4 architectural baselines, and 6 ablations are evaluated."""
    assert comparative_report_smoke.leakage_certificate.passed is True
    assert comparative_report_smoke.leakage_certificate.state_fingerprint_overlap_count == 0
    assert comparative_report_smoke.leakage_certificate.temporal_boundary_verified is True
    assert comparative_report_smoke.leakage_certificate.config_boundary_verified is True

    assert len(comparative_report_smoke.ladder_rungs) == 5
    assert len(comparative_report_smoke.architectural_baselines) == 4
    assert len(comparative_report_smoke.ablations) == 6

    for rec in (
        *comparative_report_smoke.ladder_rungs,
        *comparative_report_smoke.architectural_baselines,
        *comparative_report_smoke.ablations,
    ):
        assert rec.prediction.call_arrival_mae.mean > 0.0
        assert 0.0 <= rec.prediction.empirical_coverage_90.mean <= 1.0
        assert 0.0 <= rec.simulation.internal_coherence_pass_rate.mean <= 1.0
        assert rec.simulation.external_trajectory_nmae.mean >= 0.0
        assert rec.planning_decision.decision_regret_usd.mean >= 0.0
        assert -1.0 <= rec.planning_decision.policy_rank_correlation.mean <= 1.0
        assert rec.operational_cost.mean_latency_ms.mean > 0.0
        assert rec.operational_cost.peak_memory_kb.mean > 0.0
        assert rec.operational_cost.specification_burden_components >= 1


def test_simulation_internal_coherence_vs_external_fidelity(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify Rung 4 (predict-only DES) is 100% internally coherent yet externally biased under shift."""
    by_id = {
        r.rung_id: r
        for r in (
            *comparative_report_smoke.ladder_rungs,
            *comparative_report_smoke.architectural_baselines,
            *comparative_report_smoke.ablations,
        )
    }
    r1 = by_id["rung1_simple_statistical"]
    r4 = by_id["rung4_explicit_dynamics_predict_only"]
    r5 = by_id["rung5_full_ewm"]

    assert r1.simulation.internal_coherence_pass_rate.mean == 0.0
    assert r4.simulation.internal_coherence_pass_rate.mean == pytest.approx(1.0)
    assert r5.simulation.internal_coherence_pass_rate.mean == pytest.approx(1.0)

    # Rung 4 misses the causal effect of the mid-rollout rule shift -> higher external NMAE than Rung 5
    assert r4.simulation.external_trajectory_nmae.mean > r5.simulation.external_trajectory_nmae.mean
    assert (
        "INTERNALLY COHERENT YET EXTERNALLY BIASED" in r4.simulation.coherence_vs_fidelity_gap_note
    )


def test_intervention_causal_refusal_gate_and_ate_recovery(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify Rungs 1-4 are refused interventional standing while Rung 5 recovers ground-truth ATE."""
    by_id = {r.rung_id: r for r in comparative_report_smoke.ladder_rungs}

    for rid in (
        "rung1_simple_statistical",
        "rung2_standard_ml_gbt",
        "rung3_ml_engineered_features",
        "rung4_explicit_dynamics_predict_only",
    ):
        assert by_id[rid].intervention.causal_gate_refused is True
        assert by_id[rid].intervention.refusal_reason is not None
        assert by_id[rid].intervention.diagnostic_ungated_utility_ate_mae.mean > 5000.0

    r5 = by_id["rung5_full_ewm"]
    assert r5.intervention.causal_gate_refused is False
    assert r5.intervention.diagnostic_ungated_utility_ate_mae.mean == pytest.approx(0.0, abs=1e-6)


def test_headline_prediction_vs_decision_gap_both_directions(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify both directions of the Prediction-vs-Decision Gap (Section C)."""
    gap = comparative_report_smoke.prediction_vs_decision_gap

    # Direction 1: Rung 1 -> Rung 3 improves 1-step prediction MAE, but does NOT improve decision regret
    assert gap.rung3_call_mae < gap.rung1_call_mae
    assert gap.rung1_to_rung3_mae_reduction_pct > 15.0
    assert gap.rung3_decision_regret_usd > 15000.0
    assert gap.rung1_to_rung3_regret_reduction_usd == pytest.approx(0.0, abs=1e-6)
    assert gap.rung5_decision_regret_usd == pytest.approx(0.0, abs=1e-6)

    # Direction 2: Arch Rules-Only has higher net-cash state error than Rung 3 yet 0.0 decision regret
    assert gap.rules_only_net_cash_mae > gap.rung3_net_cash_mae
    assert gap.rules_only_decision_regret_usd == pytest.approx(0.0, abs=1e-6)


def test_ablations_and_zero_benefit_components(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify single-component ablations on Rung 5 (Section D)."""
    by_id = {r.rung_id: r for r in comparative_report_smoke.ablations}

    # Ablating cascade dynamics or do() surgery causes high decision regret
    assert by_id["ablate_cascade_dynamics"].planning_decision.decision_regret_usd.mean > 15000.0
    assert (
        by_id["ablate_do_intervention_operator"].planning_decision.decision_regret_usd.mean
        > 15000.0
    )

    # Ablating feasible projection Pi_C causes 100% constraint violation rate
    assert by_id[
        "ablate_feasible_projection"
    ].planning_decision.constraint_violation_rate.mean == pytest.approx(1.0)

    # Ablating UQ collapses 90% coverage to 0%, but leaves risk-neutral decision regret at 0.0
    assert by_id[
        "ablate_uncertainty_quantification"
    ].prediction.empirical_coverage_90.mean == pytest.approx(0.0)
    assert by_id[
        "ablate_uncertainty_quantification"
    ].planning_decision.decision_regret_usd.mean == pytest.approx(0.0)

    # Ablating engineered lag features has 0.0 impact on decision regret and ATE recovery
    assert by_id[
        "ablate_engineered_lag_features"
    ].planning_decision.decision_regret_usd.mean == pytest.approx(0.0)


def test_negative_results_and_construct_validity_cautionary_cases(
    comparative_report_smoke: ComparativeEvaluationReport,
) -> None:
    """Verify negative results (N1-N4) and construct-validity cautionary cases (Section E & F)."""
    neg_by_id = {n.hypothesis_id: n for n in comparative_report_smoke.negative_results}
    assert set(neg_by_id.keys()) == {"N1", "N2", "N3", "N4"}
    for n_rec in neg_by_id.values():
        assert n_rec.confirmed is True

    cv_by_id = {c.objective_id: c for c in comparative_report_smoke.construct_validity_cases}
    assert cv_by_id[
        "obj_balanced_enterprise_utility"
    ].true_enterprise_decision_regret_usd == pytest.approx(0.0)
    # Optimizing narrow fraud-loss proxy inside the exact EWM produces severe true regret
    narrow = cv_by_id["obj_narrow_fraud_loss_minimization"]
    assert narrow.is_misspecified_proxy is True
    assert narrow.realized_sla_breach_steps > 0.0
    assert narrow.true_enterprise_decision_regret_usd > 15000.0


def test_committed_artifacts_and_descripting_audit_sync() -> None:
    """Verify committed JSON and Markdown artifacts match and de-scripting audit table is complete."""
    assert len(ENTERPRISE_SUITE_DESCRIPting_AUDIT_ROWS) >= 35
    assert DEFAULT_JSON_PATH.exists()
    assert DEFAULT_MD_PATH.exists()

    committed = ComparativeEvaluationReport.model_validate_json(
        DEFAULT_JSON_PATH.read_text(encoding="utf-8")
    )
    assert committed.n_eval_seeds == 10
    expected_md = render_comparative_evaluation_markdown(committed)
    actual_md = DEFAULT_MD_PATH.read_text(encoding="utf-8")
    assert actual_md == expected_md
