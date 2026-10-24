"""Comprehensive tests for the Flagship Fraud -> Declines -> Call-Center Cascade."""

from __future__ import annotations

import pytest

from ewm_engine.core import (
    Action,
    BitemporalLeakageError,
    BitemporalLedger,
    BitemporalRecord,
    ContextSnapshot,
    DataLeakageCertificate,
    DependencyQueryKind,
    TemporalGranularity,
    UnqualifiedCausalClaimError,
    WorldState,
)
from ewm_engine.flagship import (
    FLAGSHIP_EPISTEMIC_DISCLAIMER,
    FLAGSHIP_EPISTEMIC_LABEL,
    FraudCascadeEnvironment,
    FraudCascadeParameters,
    ObservationalRidgeBaseline,
    build_default_fraud_context,
    build_initial_fraud_cascade_state,
    project_cascade_state_to_feasible,
    run_four_capability_queries,
    verify_known_mechanisms_recovery,
)


def test_environment_correctness_vs_known_mechanisms() -> None:
    """Generator reproduces its own ground-truth analytical mechanism statistics."""
    recovery = verify_known_mechanisms_recovery(seed=42, horizon=40)
    assert recovery["all_passed"] is True
    for check_name, check_data in recovery["checks"].items():
        assert check_data["passed"] is True, f"Mechanism check failed: {check_name} -> {check_data}"


def test_cascade_propagation_across_all_three_layers() -> None:
    """Tightening the fraud threshold provably propagates across Layers 1 -> 2 -> 3 -> 4."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)

    # Baseline rollout (tau = 0.65)
    base_recs, _, _, _ = env.rollout(horizon=10, seed=77)
    # Tightened rule rollout (tau = 0.38, no SMS rescue)
    tight_ctx = build_default_fraud_context(
        params, auth_threshold=0.38, policy_version="v2_tight", context_version="2.0.0"
    )
    tight_recs, _, _, _ = env.rollout(horizon=10, seed=77, context_schedule={1: tight_ctx})

    # Layer 1: Direct declines increase and missed fraud decreases
    assert sum(r.net_fp_declines for r in tight_recs) > sum(r.net_fp_declines for r in base_recs)
    assert sum(r.missed_fraud for r in tight_recs) < sum(r.missed_fraud for r in base_recs)

    # Layer 2: Triggered workflow account locks and open dispute cases increase
    assert sum(r.locked_accounts for r in tight_recs) > sum(r.locked_accounts for r in base_recs)
    assert sum(r.open_dispute_cases for r in tight_recs) > sum(
        r.open_dispute_cases for r in base_recs
    )

    # Layer 3: Secondary customer retries, channel switches, and contact-center call arrivals increase
    assert sum(r.next_retry_queue for r in tight_recs) > sum(r.next_retry_queue for r in base_recs)
    assert sum(r.new_call_arrivals for r in tight_recs) > sum(
        r.new_call_arrivals for r in base_recs
    )

    # Layer 4: Downstream wait time, SLA breaches, and customer churn increase
    assert max(r.avg_wait_seconds for r in tight_recs) > max(r.avg_wait_seconds for r in base_recs)
    assert sum(1 for r in tight_recs if r.sla_breached) > sum(
        1 for r in base_recs if r.sla_breached
    )
    assert tight_recs[-1].cumulative_churned_customers > base_recs[-1].cumulative_churned_customers


def test_bitemporal_leakage_guard_and_point_in_time_views() -> None:
    """Attempts to read future-knowledge-time or future-event-time data at decision time t fail."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)
    _recs, ledger, _ontic, _obs = env.rollout(horizon=16, seed=42)

    # 1. Acausal record (knowledge_time < event_time) is rejected on construction
    with pytest.raises(BitemporalLeakageError, match="Acausal bitemporal record"):
        BitemporalRecord(
            record_id="bad_acausal",
            stream="auth_telemetry",
            event_time=10.0,
            knowledge_time=8.0,
            payload={"val": 1.0},
        )

    # 2. Reading a step-5 settled fraud record (knowledge_time = 5 + 4 = 9.0) at decision_time = 5.0 raises BitemporalLeakageError
    with pytest.raises(BitemporalLeakageError, match="Bitemporal leakage guard triggered"):
        ledger.read_record_at("fraud_final_step_5", decision_time=5.0)
    assert ledger.blocked_leakage_attempts == 1

    # At decision_time = 9.0, the same record is now visible!
    settled_rec = ledger.read_record_at("fraud_final_step_5", decision_time=9.0)
    assert settled_rec.payload["status"] == "settled"

    # 3. PointInTimeView at t=6.0 contains preliminary step-5 record (knowledge_time=6.0)
    #    and automatically supersedes it at t=9.0 when the revision arrives
    view_t6 = ledger.as_of(6.0)
    assert view_t6.withheld_future_knowledge_count > 0
    fraud_recs_t6 = [
        r for r in view_t6.effective_records("fraud_settlement") if r.event_time == 5.0
    ]
    assert len(fraud_recs_t6) == 1
    assert fraud_recs_t6[0].payload["status"] == "preliminary"

    view_t9 = ledger.as_of(9.0)
    fraud_recs_t9 = [
        r for r in view_t9.effective_records("fraud_settlement") if r.event_time == 5.0
    ]
    assert len(fraud_recs_t9) == 1
    assert fraud_recs_t9[0].payload["status"] == "settled"

    # 4. Definition shift at step 12 is canonicalized by PointInTimeView.latest_value
    view_t14 = ledger.as_of(14.0)
    raw_v2 = view_t14.latest_value(
        "auth_telemetry", "reported_declines", normalize_definition=False
    )
    canon_v2 = view_t14.latest_value(
        "auth_telemetry", "reported_declines", normalize_definition=True
    )
    assert pytest.approx(canon_v2, rel=1e-6) == raw_v2 * 1.25

    # 5. Multiple granularities aggregation
    step_sum = view_t14.aggregate_window(
        "auth_telemetry", "approved_legit_tx", granularity=TemporalGranularity.STEP
    )
    shift_sum = view_t14.aggregate_window(
        "auth_telemetry", "approved_legit_tx", granularity=TemporalGranularity.SHIFT_BLOCK
    )
    daily_sum = view_t14.aggregate_window(
        "auth_telemetry", "approved_legit_tx", granularity=TemporalGranularity.DAILY_SUMMARY
    )
    assert daily_sum > shift_sum > step_sum > 0.0


def test_conservation_and_capacity_invariants_and_feasible_projection() -> None:
    """Conservation and capacity invariants hold under interventions and infeasible residual projection."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)
    state0 = build_initial_fraud_cascade_state(params)

    # 1. Test feasible projection when a perturbed/learned residual proposes infeasible targets
    infeasible_targets = {
        "active_customers": 9800.0,
        "churned_customers": 500.0,  # Sum = 10,300 != 10,000
        "call_backlog": -45.0,  # Negative backlog
        "staffed_agents": 250.0,  # Exceeds max_staffed_agents=120
        "latent_churn_propensity": 1.4,  # Exceeds [0, 1]
    }
    projected, feas_rec = project_cascade_state_to_feasible(state0, infeasible_targets, params)
    assert feas_rec.is_feasible is False
    assert feas_rec.l2_projection_distance > 0.0
    assert len(feas_rec.violations) == 4
    assert pytest.approx(projected["active_customers"] + projected["churned_customers"]) == 10000.0
    assert projected["call_backlog"] == 0.0
    assert projected["staffed_agents"] == params.max_staffed_agents
    assert projected["latent_churn_propensity"] == 1.0

    # 2. Verify invariants hold across every step under extreme staffing and threshold interventions
    extreme_ctx = build_default_fraud_context(params, auth_threshold=0.25, context_version="9.0.0")
    extreme_actions = {
        k: (
            Action(
                id=f"act_{k}",
                actor_id="ops_controller",
                type="set_call_center_capacity",
                parameters={"staffed_agents": 500.0},  # Above max bound
            ),
        )
        for k in range(1, 15)
    }
    recs, _, _, obs_states = env.rollout(
        horizon=14,
        seed=99,
        context_schedule={3: extreme_ctx},
        action_schedule=extreme_actions,
    )
    for r in recs:
        assert pytest.approx(r.active_customers + r.cumulative_churned_customers) == 10000.0
        assert pytest.approx(r.total_queue_demand) == (
            r.handled_calls + r.abandoned_calls + r.ending_call_backlog
        )
        assert params.min_staffed_agents <= r.staffed_agents <= params.max_staffed_agents
        assert 0.0 <= r.ending_call_backlog <= params.max_call_backlog

    # 3. Verify partial observability: hidden ontic resources are absent from O_t
    for obs in obs_states:
        assert "latent_churn_propensity" not in obs.resources
        assert "instantaneous_fp_declines" not in obs.resources
        assert "instantaneous_missed_fraud" not in obs.resources
        assert "active_customers" in obs.resources
        assert "avg_wait_seconds" in obs.resources


def test_context_k_evolution_and_snapshot_immutability() -> None:
    """Context K_t evolution alters downstream dynamics without mutating prior snapshots."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)

    ctx_v1 = build_default_fraud_context(params, auth_threshold=0.65, context_version="1.0.0")
    v1_fp_before = ctx_v1.fingerprint

    ctx_v2 = ctx_v1.evolve(
        new_version="2.0.0",
        parameters={"auth_threshold": 0.42, "sla_wait_target_seconds": 120.0},
        policy_versions={"fraud_auth_policy": "v2_tight"},
    )

    assert ctx_v1.fingerprint == v1_fp_before
    assert ctx_v1.parameters["auth_threshold"] == 0.65
    assert ctx_v2.parent_version == "1.0.0"
    assert ctx_v2.parameters["auth_threshold"] == 0.42
    assert ctx_v1.fingerprint != ctx_v2.fingerprint

    recs_v1, _, states_v1, _ = env.rollout(horizon=8, seed=55)
    recs_v2, _, states_v2, _ = env.rollout(horizon=8, seed=55, context_schedule={5: ctx_v2})

    # Steps 1..4 before context change at step 5 are bit-identical
    for k in range(4):
        assert recs_v1[k].ontic_state_fingerprint == recs_v2[k].ontic_state_fingerprint
        assert states_v1[k + 1].ontic_fingerprint == states_v2[k + 1].ontic_fingerprint

    # Step 5+ diverges after context change while past states in states_v2[0..4] remain untouched
    assert recs_v1[4].ontic_state_fingerprint != recs_v2[4].ontic_state_fingerprint
    assert states_v2[4].context["context_version"] == "1.0.0"
    assert states_v2[5].context["context_version"] == "2.0.0"


def test_determinism_and_serialization_round_trip() -> None:
    """Identical seeds produce bit-identical trajectories and states round-trip cleanly."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)

    recs_a, ledger_a, states_a, _ = env.rollout(horizon=12, seed=12345)
    recs_b, ledger_b, _states_b, _ = env.rollout(horizon=12, seed=12345)

    assert [r.ontic_state_fingerprint for r in recs_a] == [
        r.ontic_state_fingerprint for r in recs_b
    ]
    assert ledger_a.as_of(10.0).fingerprint == ledger_b.as_of(10.0).fingerprint

    # Round-trip WorldState, ContextSnapshot, FraudCascadeParameters, and BitemporalLedger
    final_state = states_a[-1]
    state_rt = WorldState.model_validate_json(final_state.model_dump_json())
    assert state_rt.ontic_fingerprint == final_state.ontic_fingerprint

    ctx_rt = ContextSnapshot.from_dict(env.initial_context.to_dict())
    assert ctx_rt.fingerprint == env.initial_context.fingerprint

    params_rt = FraudCascadeParameters.model_validate_json(params.model_dump_json())
    assert params_rt.fingerprint == params.fingerprint

    ledger_rt = BitemporalLedger.from_dict(ledger_a.to_dict())
    assert ledger_rt.as_of(10.0).fingerprint == ledger_a.as_of(10.0).fingerprint


def test_four_capability_queries_and_leakage_certificate() -> None:
    """The 4 capability queries execute cleanly, emit a passing DataLeakageCertificate, and enforce evidence gating."""
    assert "CONTROLLED_SYNTHETIC" in FLAGSHIP_EPISTEMIC_LABEL
    assert "NOT proven efficacy" in FLAGSHIP_EPISTEMIC_DISCLAIMER

    out = run_four_capability_queries(train_seed=101, eval_seed=202, horizon=20)
    cert = DataLeakageCertificate.model_validate(out["leakage_certificate"])
    assert cert.passed is True
    assert cert.state_fingerprint_overlap_count == 0
    assert cert.temporal_boundary_verified is True
    assert cert.config_boundary_verified is True

    # Planning demonstrates cascade-aware EWM outperforms myopic Layer-1 optimization
    plan = out["planning"]
    assert plan["cascade_ewm_decision_regret"] == 0.0
    assert plan["myopic_decision_regret"] > 100000.0
    assert plan["regret_reduction_by_ewm"] > 100000.0

    # ObservationalRidgeBaseline refuses uncertified causal surgery
    ridge = ObservationalRidgeBaseline()
    with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
        ridge.validate_for_query(DependencyQueryKind.INTERVENTIONAL_DO)
