"""Unit, contract, and property tests for Fenced Experimental Lines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ewm_engine
from ewm_engine.experimental.domain_composition import (
    COMPOSITION_CANDIDATE_POLICIES,
    COMPOSITION_EPISTEMIC_STATUS,
    CallCenterCapacitySubmodel,
    ComposableDomainSubmodel,
    CompositionAssumptionError,
    FraudDeclinesCustomerSubmodel,
    build_default_boundary_contract,
    run_composed_cascade_rollout,
    run_domain_composition_experiment,
)
from ewm_engine.experimental.swe_ecosystem import (
    SWE_EPISTEMIC_STATUS,
    SWEEcosystemEnvironment,
    compute_static_transitive_dependents,
    run_swe_ecosystem_experiment,
)
from ewm_engine.provenance.evidence import DependencyQueryKind

ROOT = Path(__file__).resolve().parent.parent.parent


def test_stable_public_contract_untouched_by_experimental_lines() -> None:
    """Fenced experimental lines must live strictly in `ewm_engine.experimental`."""
    stable_exports = set(ewm_engine.__all__)
    assert "DomainBoundaryContract" not in stable_exports
    assert "SWEEcosystemEnvironment" not in stable_exports
    assert "run_domain_composition_experiment" not in stable_exports
    assert "run_swe_ecosystem_experiment" not in stable_exports


def test_composable_submodel_protocol_and_boundary_leak_prevention() -> None:
    """Verify `ComposableDomainSubmodel` protocol and `ObservationFilter` boundary projection."""
    upstream = FraudDeclinesCustomerSubmodel()
    downstream = CallCenterCapacitySubmodel()
    assert isinstance(upstream, ComposableDomainSubmodel)
    assert isinstance(downstream, ComposableDomainSubmodel)
    assert upstream.specification_component_count == 6
    assert downstream.specification_component_count == 5

    contract = build_default_boundary_contract()
    traj_a = run_composed_cascade_rollout(
        seed=42, policy=COMPOSITION_CANDIDATE_POLICIES[2], sync_cadence_steps=1
    )
    traj_b = run_composed_cascade_rollout(
        seed=42, policy=COMPOSITION_CANDIDATE_POLICIES[2], sync_cadence_steps=1
    )
    assert len(traj_a) == 24
    assert [r["net_step_value"] for r in traj_a] == [r["net_step_value"] for r in traj_b]
    assert "internal_fraud_rule_score_buffer" not in contract.exported_resource_ids


def test_boundary_contract_epistemic_and_unit_gates_fail_closed() -> None:
    """Correlational upstream exports and unit mismatches must fail closed with `CompositionAssumptionError`."""
    with pytest.raises(CompositionAssumptionError, match="epistemic incompatibility"):
        run_composed_cascade_rollout(
            seed=42,
            policy=COMPOSITION_CANDIDATE_POLICIES[2],
            export_correlational_only=True,
            query_kind=DependencyQueryKind.INTERVENTIONAL_DO,
            enforce_boundary_gate=True,
        )

    with pytest.raises(CompositionAssumptionError, match="unit/cohort mismatch"):
        run_composed_cascade_rollout(
            seed=42,
            policy=COMPOSITION_CANDIDATE_POLICIES[2],
            export_misaligned_cohort_units=True,
            enforce_boundary_gate=True,
        )


def test_domain_composition_experiment_measures_all_five_costs_and_negatives() -> None:
    """Verify all 5 vision-named composition costs and negative findings `C-N1`..`C-N3`."""
    report = run_domain_composition_experiment(seed=42, n_seeds=5)
    assert report.epistemic_status == COMPOSITION_EPISTEMIC_STATUS
    assert report.monolithic_spec_components == 11
    assert report.composed_spec_components == 16
    assert report.specification_overhead_pct > 40.0
    assert report.synchronous_trajectory_max_abs_diff == 0.0

    # 1. Error propagation
    assert report.error_propagation_audit["nonlinear_boundary_amplification_ratio"] > 1.5
    assert report.error_propagation_audit["downstream_post_shift_net_cash_mae"] > 1000.0

    # 2. Assumption incompatibility
    assert (
        report.assumption_incompatibility_audit["correlational_upstream_blocked_for_intervention"]
        is True
    )
    assert report.assumption_incompatibility_audit["cohort_unit_mismatch_blocked"] is True
    assert (
        report.assumption_incompatibility_audit[
            "ungated_cohort_mismatch_silent_corruption_cost_usd"
        ]
        > 500.0
    )

    # 3. Temporal desynchronization
    assert float(report.temporal_desync_audit["desynchronized_utility_loss_usd"]) > 1000.0
    assert float(report.temporal_desync_audit["desynchronized_extra_sla_breach_steps"]) >= 1.0

    # 4. Uncertainty accumulation
    assert report.uncertainty_accumulation_audit["naive_interval_width_inflation_pct"] > 30.0

    # 5. Negative findings
    neg_ids = [nf["finding_id"] for nf in report.negative_findings]
    assert neg_ids == ["C-N1", "C-N2", "C-N3"]
    assert all(nf["confirmed"] is True for nf in report.negative_findings)


def test_swe_ecosystem_static_vs_dynamic_and_bitemporal_discipline() -> None:
    """Verify static-graph blindness to DB pool contention, bitemporal guard, and 5-rung ladder."""
    static_deps = compute_static_transitive_dependents("auth_lib")
    assert static_deps == {"auth_lib", "checkout_service", "api_gateway"}
    assert "payment_worker" not in static_deps
    assert "orders_db" not in static_deps

    env = SWEEcosystemEnvironment(seed=42)
    ledger = env.build_bitemporal_ci_ledger()
    pit = ledger.as_of(5.0)
    assert pit.withheld_future_knowledge_count == 2
    visible_ids = {r.record_id for r in pit.visible_records}
    assert "postmortem_root_cause_t5" not in visible_ids
    assert "canary_dlq_alert_t5" not in visible_ids

    report = run_swe_ecosystem_experiment(seed=42, n_seeds=5)
    assert report.epistemic_status == SWE_EPISTEMIC_STATUS
    assert "payment_worker" in report.true_dynamic_impacted_under_burst_rollout
    assert "orders_db" in report.true_dynamic_impacted_under_burst_rollout

    by_rung = {r.rung_id: r for r in report.ladder_results}
    r1 = by_rung["rung1_static_dependency_graph"]
    r3 = by_rung["rung3_ml_plus_static_graph_features"]
    r4 = by_rung["rung4_dynamic_sim_no_context_or_gate"]
    r5 = by_rung["rung5_experimental_swe_world_model"]

    assert r1.detected_payment_worker_contention is False
    assert r3.detected_payment_worker_contention is False
    assert r4.detected_payment_worker_contention is False
    assert r5.detected_payment_worker_contention is True

    assert r5.change_impact_f1 == 1.0
    assert r5.test_selection_fault_recall == 1.0
    assert r5.root_cause_precision_at_1 == 1.0
    assert r5.causal_gate_refused_spurious_correlation is True
    assert r5.deployment_decision_regret_usd == 0.0
    assert r1.deployment_decision_regret_usd > 50000.0

    # Ablations and negative findings
    assert len(report.ablation_results) == 4
    neg_ids = [nf["finding_id"] for nf in report.negative_findings_and_transfer_limits]
    assert neg_ids == ["SWE-N1", "SWE-N2", "SWE-N3"]
    assert all(nf["confirmed"] is True for nf in report.negative_findings_and_transfer_limits)


def test_experimental_artifact_matches_live_execution() -> None:
    """Ensure `docs/results/artifacts/experimental_composition_and_swe.json` matches live computation."""
    artifact_path = (
        ROOT / "docs" / "results" / "artifacts" / "experimental_composition_and_swe.json"
    )
    assert artifact_path.exists()
    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    comp_live = run_domain_composition_experiment(seed=42, n_seeds=5)
    swe_live = run_swe_ecosystem_experiment(seed=42, n_seeds=5)

    assert (
        data["line1_domain_composition"]["error_propagation_audit"]
        == comp_live.to_dict()["error_propagation_audit"]
    )
    assert (
        data["line1_domain_composition"]["temporal_desync_audit"]
        == comp_live.to_dict()["temporal_desync_audit"]
    )
    assert data["line2_swe_ecosystem"]["output_fingerprint"] == swe_live.output_fingerprint
