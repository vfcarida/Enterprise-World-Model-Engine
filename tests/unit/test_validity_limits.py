"""Unit, contract, and property tests for Validity Limits.

Covers:
- Section A: Performativity & endogeneity (`DriftKind.POLICY_INDUCED_SHIFT`,
  `DriftKind.ENDOGENOUS_FEEDBACK_SHIFT`, `check_plan_performative_validity`, and the
  worked Flagship performativity scenario measuring time-to-invalidation).
- Section B: Stylized adversarial adaptation (`DriftKind.ADVERSARIAL_ADAPTATION_SHIFT`,
  `STYLIZED_SYNTHETIC_ADVERSARY_NOT_REAL_FRAUD_RING_CLAIM`, policy robustness and
  time-to-degradation).
- Section C: Evidence-nature-gated inference across the full 5 x 4
  (`DependencyEvidenceNature` x `DependencyQueryKind`) matrix and multi-link chains.
- Section D: Verifiable mechanistic explainability (`VerifiableMechanisticExplanation`)
  and anti-narrative fidelity guard (`verify_narrative_explanation_fidelity`).
- Section E: Uncertainty-typed abstention (`ValidityLimitAbstentionGate`) reporting
  exact `UncertaintySourceType` triggers, human-review routing, and fallback policies.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ewm_engine.core import (
    DRIFT_KIND_TO_ORIGIN,
    EVIDENCE_QUERY_MATRIX_CONTRACT,
    EXOGENOUS_DRIFT_KINDS,
    MODEL_INDUCED_DRIFT_KINDS,
    Action,
    ContextSnapshot,
    DependencyEvidenceNature,
    DependencyQueryKind,
    DriftKind,
    DriftOrigin,
    EndogenousDecisionFeedbackSpec,
    Entity,
    EvidenceLevel,
    InsufficientCausalEvidenceWarning,
    PerformativeCouplingSpec,
    Relationship,
    Resource,
    StructuralAssignment,
    UnqualifiedCausalClaimError,
    WorldState,
    check_plan_performative_validity,
    evaluate_dependency_chain_for_query,
    evaluate_evidence_nature_gate,
)
from ewm_engine.dynamics import (
    CompositeDynamics,
    DeterministicDemandDynamics,
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
    ValidityDomain,
)
from ewm_engine.evaluation import (
    AbstentionResolutionAction,
    AbstentionTriggerKind,
    CompetenceEstimator,
    UncertaintySourceType,
    UncertaintyTaxonomyBreakdown,
    ValidityLimitAbstentionGate,
)
from ewm_engine.flagship import (
    ADVERSARIAL_EPISTEMIC_LABEL,
    VALIDITY_LIMITS_EPISTEMIC_LABEL,
    run_adversarial_adaptation_scenario,
    run_performativity_endogeneity_scenario,
)
from ewm_engine.provenance import (
    CandidateNarrativeExplanation,
    NarrativeFabricationError,
    SystemicTrace,
    VerifiableMechanisticExplanation,
    build_verifiable_explanation,
    verify_narrative_explanation_fidelity,
)

# ==============================================================================
# Section A: Performativity & Endogeneity
# ==============================================================================


@pytest.mark.contract
def test_drift_taxonomy_distinguishes_exogenous_vs_model_induced_shifts() -> None:
    """Verify policy-induced, endogenous, and adversarial drift categories are distinct from exogenous drift."""
    assert EXOGENOUS_DRIFT_KINDS.isdisjoint(MODEL_INDUCED_DRIFT_KINDS)
    assert set(DriftKind) == (EXOGENOUS_DRIFT_KINDS | MODEL_INDUCED_DRIFT_KINDS)
    assert MODEL_INDUCED_DRIFT_KINDS == {
        DriftKind.POLICY_INDUCED_SHIFT,
        DriftKind.ENDOGENOUS_FEEDBACK_SHIFT,
        DriftKind.ADVERSARIAL_ADAPTATION_SHIFT,
    }
    assert DRIFT_KIND_TO_ORIGIN[DriftKind.POLICY_INDUCED_SHIFT] == (
        DriftOrigin.POLICY_INDUCED_PERFORMATIVE
    )
    assert DRIFT_KIND_TO_ORIGIN[DriftKind.ENDOGENOUS_FEEDBACK_SHIFT] == (
        DriftOrigin.ENDOGENOUS_CLOSED_LOOP
    )
    assert DRIFT_KIND_TO_ORIGIN[DriftKind.ADVERSARIAL_ADAPTATION_SHIFT] == (
        DriftOrigin.ADVERSARIAL_STRATEGIC
    )

    coupling = PerformativeCouplingSpec(
        coupling_id="price_elasticity_lucas_shift",
        policy_key="auth_threshold",
        baseline_policy_value=0.65,
        affected_behavioral_parameters=("retry_fraction", "call_escalation_rate"),
        sensitivity_elasticities={"retry_fraction": 1.2, "call_escalation_rate": 2.0},
        assumed_static_mechanisms=("static_des_queue_model",),
        invalidation_threshold=0.05,
    )
    endog_fb = EndogenousDecisionFeedbackSpec(
        feedback_id="operator_wait_feedback",
        action_key="staffed_agents",
        conditioning_state_keys=("call_backlog", "wait_seconds"),
        unobserved_confounder_keys=("supervisor_urgency",),
        feedback_gain=0.4,
        invalidates_static_association_mechanisms=("ols_wait_predictor",),
    )

    audit = check_plan_performative_validity(
        deployed_policy_parameters={"auth_threshold": 0.48, "staffed_agents": 38.0},
        performative_couplings=(coupling,),
        endogenous_feedback_specs=(endog_fb,),
        active_mechanism_ids=("static_des_queue_model", "ols_wait_predictor"),
    )
    assert audit.is_self_invalidating is True
    assert audit.performative_shift_magnitude > 0.05
    assert audit.endogenous_confounding_detected is True
    assert set(audit.invalidated_mechanism_ids) == {
        "static_des_queue_model",
        "ols_wait_predictor",
    }

    kinds = {s.kind for s in audit.drift_signals}
    assert DriftKind.POLICY_INDUCED_SHIFT in kinds
    assert DriftKind.ENDOGENOUS_FEEDBACK_SHIFT in kinds
    for sig in audit.drift_signals:
        assert sig.is_model_induced is True
        assert sig.is_exogenous is False


def test_flagship_performativity_and_endogeneity_validity_limit_scenario() -> None:
    """Worked scenario: deploying the optimal static policy shifts customer behavior and invalidates static dynamics."""
    report = run_performativity_endogeneity_scenario(seed=42, horizon_steps=18)

    assert report.epistemic_label == VALIDITY_LIMITS_EPISTEMIC_LABEL
    assert report.deployed_policy_id == "coordinated_ewm_tau_0.48_agents_38"
    assert len(report.step_comparisons) == 18

    # Performative shift: retry propensity decays and call escalation per FP decline rises
    assert report.terminal_retry_fraction < report.baseline_retry_fraction
    assert report.terminal_call_escalation_rate > report.baseline_call_escalation_rate

    # Static dynamics become wrong within a measured number of steps
    assert report.time_to_forecast_invalidation_steps is not None
    assert 1 <= report.time_to_forecast_invalidation_steps < report.horizon_steps
    assert report.time_to_sla_breach_steps is not None
    assert 1 <= report.time_to_sla_breach_steps < report.horizon_steps

    # Static EWM over-predicts utility and suffers positive regret vs performativity-aware policy
    assert report.performative_utility_erosion_usd > 25000.0
    assert report.performative_policy_regret_usd > 10000.0
    assert report.endogenous_state_action_correlation > 0.90

    # Plan self-invalidation and abstention gate fire
    assert report.plan_invalidation_audit.is_self_invalidating is True
    assert report.abstention_decision.do_not_use_me_here is True
    assert report.abstention_decision.status_banner == "DO_NOT_USE_ME_HERE"
    assert (
        AbstentionTriggerKind.PERFORMATIVE_INVALIDATION
        in report.abstention_decision.triggered_validity_limits
    )
    assert (
        AbstentionTriggerKind.ENDOGENOUS_CONFOUNDING
        in report.abstention_decision.triggered_validity_limits
    )


# ==============================================================================
# Section B: Stylized Adversarial Adaptation
# ==============================================================================


def test_flagship_stylized_adversarial_adaptation_and_time_to_degradation() -> None:
    """Verify stylized adversary adaptation erodes static policies and measures time-to-degradation."""
    report = run_adversarial_adaptation_scenario(seed=42, horizon_steps=18)

    assert report.epistemic_label == ADVERSARIAL_EPISTEMIC_LABEL
    assert "Stylized strategic adversary" in report.epistemic_disclaimer
    assert "NOT an empirical claim about real-world fraud rings" in report.epistemic_disclaimer

    by_id = {r.policy_id: r for r in report.policy_records}
    myopic = by_id["myopic_aggressive_tau_0.40_agents_32"]
    coord_static = by_id["coordinated_ewm_tau_0.48_agents_38"]
    adaptive_def = by_id["adaptive_rolling_recalibration_tau_0.48_agents_38"]

    # Static policies suffer >= 15% catch-rate degradation within the 18-step horizon
    assert myopic.time_to_degradation_steps is not None
    assert coord_static.time_to_degradation_steps is not None
    assert myopic.time_to_degradation_steps <= coord_static.time_to_degradation_steps
    assert coord_static.terminal_fraud_catch_rate < coord_static.initial_fraud_catch_rate
    assert coord_static.excess_unblocked_fraud_loss_usd > 10000.0
    assert coord_static.adversarial_utility_degradation_usd > 10000.0

    # Rolling recalibration defense limits catch-rate drop and retains higher utility
    assert adaptive_def.realized_adversarial_utility_usd > (
        coord_static.realized_adversarial_utility_usd
    )
    assert adaptive_def.relative_catch_rate_drop < coord_static.relative_catch_rate_drop

    # Drift and abstention signals fire
    assert any(s.kind == DriftKind.ADVERSARIAL_ADAPTATION_SHIFT for s in report.drift_signals)
    assert report.abstention_decision.do_not_use_me_here is True
    assert (
        AbstentionTriggerKind.ADVERSARIAL_DEGRADATION
        in report.abstention_decision.triggered_validity_limits
    )


# ==============================================================================
# Section C: Evidence-Nature-Gated Inference (Full 5 x 4 Matrix & Chains)
# ==============================================================================


@pytest.mark.contract
def test_full_evidence_nature_by_query_kind_matrix_contract() -> None:
    """Exhaustively verify all 20 (DependencyEvidenceNature x DependencyQueryKind) matrix cells."""
    assert len(EVIDENCE_QUERY_MATRIX_CONTRACT) == 20

    for nature in DependencyEvidenceNature:
        for query_kind in DependencyQueryKind:
            cond = EVIDENCE_QUERY_MATRIX_CONTRACT[(nature, query_kind)]

            # Case 1: Unidentified, no overlap, no provenance
            if cond == "always_permitted":
                dec = evaluate_evidence_nature_gate(
                    nature,
                    query_kind,
                    is_identifiable=False,
                    overlap_satisfied=False,
                    identification_provenance=None,
                    strict=True,
                )
                assert dec.permitted is True
                assert dec.downgraded is False
            else:
                with pytest.raises(UnqualifiedCausalClaimError):
                    evaluate_evidence_nature_gate(
                        nature,
                        query_kind,
                        is_identifiable=False,
                        overlap_satisfied=False,
                        identification_provenance=None,
                        strict=True,
                    )
                with pytest.warns(InsufficientCausalEvidenceWarning):
                    dec_down = evaluate_evidence_nature_gate(
                        nature,
                        query_kind,
                        is_identifiable=False,
                        overlap_satisfied=False,
                        identification_provenance=None,
                        strict=False,
                    )
                assert dec_down.permitted is False
                assert dec_down.downgraded is True
                assert dec_down.effective_evidence_level == EvidenceLevel.PREDICTIVE

            # Case 2: Overlap satisfied only (no formal identification provenance)
            if cond in ("always_permitted", "requires_overlap"):
                dec_ovl = evaluate_evidence_nature_gate(
                    nature,
                    query_kind,
                    is_identifiable=False,
                    overlap_satisfied=True,
                    identification_provenance=None,
                    strict=True,
                )
                assert dec_ovl.permitted is True
            else:
                with pytest.raises(UnqualifiedCausalClaimError):
                    evaluate_evidence_nature_gate(
                        nature,
                        query_kind,
                        is_identifiable=False,
                        overlap_satisfied=True,
                        identification_provenance=None,
                        strict=True,
                    )

            # Case 3: Full identification evidence (identifiable + overlap + provenance)
            if cond == "forbidden":
                # Even with all flags True, STATISTICAL_ASSOCIATION (for do/twin) and
                # CAUSAL_HYPOTHESIS (for twin) must fail closed!
                with pytest.raises(UnqualifiedCausalClaimError):
                    evaluate_evidence_nature_gate(
                        nature,
                        query_kind,
                        is_identifiable=True,
                        overlap_satisfied=True,
                        identification_provenance="Backdoor adjustment + RCT",
                        strict=True,
                    )
            else:
                dec_full = evaluate_evidence_nature_gate(
                    nature,
                    query_kind,
                    is_identifiable=True,
                    overlap_satisfied=True,
                    identification_provenance="Backdoor adjustment + RCT",
                    strict=True,
                )
                assert dec_full.permitted is True
                assert dec_full.downgraded is False


def test_dependency_chain_fail_closed_weakest_link_propagation() -> None:
    """Verify multi-link dependency chains fail closed when any link cannot support the query."""
    link_rule = Relationship(
        source="policy_tau",
        target="declines",
        type="governs",
    ).with_evidence_nature(DependencyEvidenceNature.BUSINESS_RULE_MECHANISM)
    link_exp = StructuralAssignment(
        target="resource:call_volume",
        parents=("resource:declines",),
        mechanism_id="rct_decline_to_call",
        evidence_nature=DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        overlap_satisfied=True,
    )
    link_assoc = Relationship(
        source="call_volume",
        target="churn",
        type="correlates_with",
    ).with_evidence_nature(DependencyEvidenceNature.STATISTICAL_ASSOCIATION)

    # Chain [rule, exp] permits INTERVENTIONAL_DO (bounded by INTERVENTIONAL)
    chain_ok = evaluate_dependency_chain_for_query(
        [link_rule, link_exp],
        DependencyQueryKind.INTERVENTIONAL_DO,
        strict=True,
    )
    assert chain_ok.permitted is True
    assert chain_ok.effective_evidence_level == EvidenceLevel.INTERVENTIONAL

    # Adding a STATISTICAL_ASSOCIATION link blocks INTERVENTIONAL_DO fail-closed
    with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
        evaluate_dependency_chain_for_query(
            [link_rule, link_exp, link_assoc],
            DependencyQueryKind.INTERVENTIONAL_DO,
            strict=True,
        )

    # Non-strict mode downgrades the composite chain to PREDICTIVE with warning
    with pytest.warns(InsufficientCausalEvidenceWarning):
        chain_downgraded = evaluate_dependency_chain_for_query(
            [link_rule, link_exp, link_assoc],
            DependencyQueryKind.INTERVENTIONAL_DO,
            strict=False,
        )
    assert chain_downgraded.permitted is False
    assert chain_downgraded.downgraded is True
    assert chain_downgraded.effective_evidence_level == EvidenceLevel.PREDICTIVE


@pytest.mark.property
@given(
    nature=st.sampled_from(list(DependencyEvidenceNature)),
    query_kind=st.sampled_from(list(DependencyQueryKind)),
    is_identifiable=st.booleans(),
    overlap_satisfied=st.booleans(),
    has_provenance=st.booleans(),
)
def test_property_evidence_nature_gate_matrix_invariants(
    nature: DependencyEvidenceNature,
    query_kind: DependencyQueryKind,
    is_identifiable: bool,
    overlap_satisfied: bool,
    has_provenance: bool,
) -> None:
    """Property: evaluate_evidence_nature_gate strictly obeys EVIDENCE_QUERY_MATRIX_CONTRACT."""
    prov = "cert:verified_id" if has_provenance else None
    cond = EVIDENCE_QUERY_MATRIX_CONTRACT[(nature, query_kind)]

    expected_permitted = (
        True
        if cond == "always_permitted"
        else (
            overlap_satisfied
            if cond == "requires_overlap"
            else (
                (is_identifiable and overlap_satisfied and has_provenance)
                if cond
                in (
                    "requires_full_identification",
                    "requires_overlap_and_functional_identification",
                )
                else False
            )
        )
    )

    if expected_permitted:
        dec = evaluate_evidence_nature_gate(
            nature,
            query_kind,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=prov,
            strict=True,
        )
        assert dec.permitted is True
        assert dec.downgraded is False
    else:
        with pytest.raises(UnqualifiedCausalClaimError):
            evaluate_evidence_nature_gate(
                nature,
                query_kind,
                is_identifiable=is_identifiable,
                overlap_satisfied=overlap_satisfied,
                identification_provenance=prov,
                strict=True,
            )


# ==============================================================================
# Section D: Verifiable Explainability vs. Narrative Fabrication Guard
# ==============================================================================


def test_verifiable_explainability_and_anti_narrative_fabrication_guard() -> None:
    """Verify explanations reference only participating mechanisms/context and reject fabricated narratives."""
    state = WorldState(
        entities=[Entity(id="ops", type="department")],
        resources=[
            Resource(
                id="call_backlog", entity_id="ops", current=50.0, min_value=0.0, max_value=500.0
            ),
            Resource(
                id="net_cash", entity_id="ops", current=1000.0, min_value=0.0, max_value=10000.0
            ),
            Resource(
                id="unmodified_vault",
                entity_id="ops",
                current=500.0,
                min_value=0.0,
                max_value=1000.0,
            ),
        ],
    )
    ctx = ContextSnapshot(
        context_id="ctx_ops",
        version="2.3.0",
        environment_regime="nominal",
        parameters={"demand_scale": 1.5, "unused_secret_param": 99.0},
    )

    m_queue = DeterministicDemandDynamics(
        resource_id="call_backlog",
        demand=10.0,
        name="queue_drain_mechanism",
        version="1.4.0",
        context_demand_multiplier_key="demand_scale",
    )
    m_cash = DeterministicDemandDynamics(
        resource_id="net_cash",
        demand=20.0,
        name="cash_settlement_mechanism",
        version="1.1.0",
    )
    comp = CompositeDynamics(
        models=[m_queue, m_cash],
        name="ops_composite_pipeline",
        version="2.0.0",
    )

    rng = np.random.default_rng(42)
    tr = comp.transition(state, [], [], rng, runtime_context=ctx)

    trace = SystemicTrace()
    trace.add_node("node_queue", step=0, category="state_change", label="Queue drained")
    trace.add_node("node_cash", step=0, category="state_change", label="Cash settled")
    trace.add_edge("node_queue", "node_cash", step=0, relation="influences")

    explanation = build_verifiable_explanation(
        tr,
        step=0,
        prior_state=state,
        runtime_context=ctx,
        trace=trace,
        explanation_id="exp_step_0",
    )

    # 1. Explanation contains ONLY mechanisms, context keys, and resources that actually participated
    assert explanation.is_standalone_narrative_evidence is False
    assert explanation.participating_mechanism_ids == {
        "queue_drain_mechanism",
        "cash_settlement_mechanism",
    }
    assert "unused_phantom_mechanism" not in explanation.participating_mechanism_ids
    assert explanation.active_context_versions == ("2.3.0",)
    assert explanation.active_context_keys_used == ("parameters.demand_scale",)
    assert "parameters.unused_secret_param" not in explanation.active_context_keys_used
    assert explanation.modified_resources == ("call_backlog", "net_cash")
    assert "unmodified_vault" not in explanation.modified_resources

    # 2. Faithful summary passes verification without any violations
    faithful_text = explanation.render_faithful_summary()
    report_ok = verify_narrative_explanation_fidelity(
        explanation,
        faithful_text,
        known_universe_mechanisms=(
            "queue_drain_mechanism",
            "cash_settlement_mechanism",
            "phantom_llm_hallucinated_mechanism",
        ),
        known_universe_resources=("call_backlog", "net_cash", "unmodified_vault"),
        strict=True,
    )
    assert report_ok.is_faithful is True
    assert report_ok.violations == ()

    # 3. Narrative citing an unexecuted mechanism is rejected
    with pytest.raises(NarrativeFabricationError, match="phantom_llm_hallucinated_mechanism"):
        verify_narrative_explanation_fidelity(
            explanation,
            "The state transition was driven by phantom_llm_hallucinated_mechanism under ctx_version:2.3.0.",
            known_universe_mechanisms=(
                "queue_drain_mechanism",
                "cash_settlement_mechanism",
                "phantom_llm_hallucinated_mechanism",
            ),
            strict=True,
        )

    # 4. Narrative citing an unused context key or unmodified resource is rejected
    rep_bad = verify_narrative_explanation_fidelity(
        explanation,
        CandidateNarrativeExplanation(
            narrative_text="Transition modified res:unmodified_vault using ctx_key:parameters.unused_secret_param.",
            claimed_mechanisms=("queue_drain_mechanism",),
        ),
        strict=False,
    )
    assert rep_bad.is_faithful is False
    assert "unmodified_vault" in rep_bad.fabricated_resources
    assert "parameters.unused_secret_param" in rep_bad.fabricated_context_keys

    # 5. Narrative asserting causal proof on a PREDICTIVE (STATISTICAL_ASSOCIATION) trace is rejected
    pred_prov = MechanismProvenance(
        mechanism_id="observational_gbt_forecaster",
        version="1.0.0",
        kind=MechanismKind.LEARNED,
        evidence_level=EvidenceLevel.PREDICTIVE,
    )
    tr_pred = TransitionResult(
        next_state=tr.next_state,
        applied_changes={"call_backlog": -15.0},
        evidence_level=EvidenceLevel.PREDICTIVE,
        diagnostics={"mechanism_provenance": pred_prov.model_dump(mode="json")},
        model_name="observational_gbt_forecaster",
    )
    exp_pred = VerifiableMechanisticExplanation.from_transition_result(tr_pred, prior_state=state)
    assert exp_pred.effective_evidence_level == EvidenceLevel.PREDICTIVE

    with pytest.raises(NarrativeFabricationError, match="Rhetorical causal overclaim"):
        verify_narrative_explanation_fidelity(
            exp_pred,
            "mech:observational_gbt_forecaster proved causal effect and caused by the queue change.",
            strict=True,
        )

    # 6. Treating narrative as standalone proof is always rejected
    with pytest.raises(NarrativeFabricationError, match="Anti-narrative guardrail violation"):
        verify_narrative_explanation_fidelity(
            explanation,
            CandidateNarrativeExplanation(
                narrative_text=faithful_text,
                treats_narrative_as_standalone_proof=True,
            ),
            strict=True,
        )


# ==============================================================================
# Section E: Abstention & Limits-of-Validity Reporting Tied to Uncertainty Taxonomy
# ==============================================================================


def test_abstention_gate_reports_exact_uncertainty_taxonomy_sources_and_fallback() -> None:
    """Verify ValidityLimitAbstentionGate reports which UncertaintySourceType triggered abstention."""
    state_ok = WorldState(
        entities=[Entity(id="cc", type="center")],
        resources=[
            Resource(
                id="call_backlog", entity_id="cc", current=40.0, min_value=0.0, max_value=500.0
            )
        ],
    )
    state_ood = state_ok.update_resource("call_backlog", delta=200.0)  # 240.0 > 150.0 bound
    ctx_ok = ContextSnapshot(context_id="ctx", version="1.0.0", environment_regime="nominal")
    ctx_bad_regime = ContextSnapshot(
        context_id="ctx", version="2.0.0", environment_regime="crisis_unseen"
    )

    fallback_act = Action(
        id="act_safe_staffing",
        actor_id="safety_controller",
        type="set_staffing",
        parameters={"staffed_agents": 42.0},
    )
    gate = ValidityLimitAbstentionGate(
        max_aleatoric_std=5.0,
        max_parameter_std=2.0,
        max_structural_model_std=3.0,
        max_runtime_context_std=1.5,
        max_monte_carlo_se=0.8,
        max_model_discrepancy=4.0,
        validity_domain=ValidityDomain(
            allowed_regimes=("nominal",),
            resource_bounds={"call_backlog": (0.0, 150.0)},
        ),
        competence_estimator=CompetenceEstimator(max_verified_horizon=10),
        fallback_policy_id="safe_conservative_staffing_policy",
        fallback_actions=(fallback_act,),
    )

    # 1. Within validity limits and low uncertainty -> does not abstain
    unc_low = UncertaintyTaxonomyBreakdown(
        aleatoric_std=2.0,
        parameter_std=0.5,
        structural_model_std=1.0,
        runtime_context_std=0.4,
        monte_carlo_standard_error=0.2,
        model_discrepancy_vs_reality=1.0,
    )
    dec_ok = gate.evaluate(
        state=state_ok,
        runtime_context=ctx_ok,
        uncertainty_breakdown=unc_low,
        step=3,
    )
    assert dec_ok.abstain is False
    assert dec_ok.do_not_use_me_here is False
    assert dec_ok.status_banner == "WITHIN_DECLARED_VALIDITY_LIMITS"
    assert dec_ok.recommended_resolution == (AbstentionResolutionAction.PROCEED_WITH_MONITORING)

    # 2. Specific uncertainty sources exceed threshold -> reports exact triggered sources
    unc_high = UncertaintyTaxonomyBreakdown(
        aleatoric_std=2.0,
        parameter_std=4.5,  # > 2.0
        structural_model_std=1.0,
        runtime_context_std=0.4,
        monte_carlo_standard_error=0.2,
        model_discrepancy_vs_reality=9.2,  # > 4.0
    )
    dec_unc = gate.evaluate(
        state=state_ok,
        runtime_context=ctx_ok,
        uncertainty_breakdown=unc_high,
        step=3,
    )
    assert dec_unc.abstain is True
    assert dec_unc.do_not_use_me_here is True
    assert dec_unc.status_banner == "DO_NOT_USE_ME_HERE"
    assert dec_unc.triggered_uncertainty_sources == (
        UncertaintySourceType.PARAMETER,
        UncertaintySourceType.MODEL_DISCREPANCY_VS_REALITY,
    )
    assert dec_unc.fallback_policy_id == "safe_conservative_staffing_policy"
    assert dec_unc.fallback_actions == (fallback_act,)

    # 3. Out-of-domain state + unsupported context regime -> requests human review + fallback
    dec_ood = gate.evaluate(
        state=state_ood,
        runtime_context=ctx_bad_regime,
        uncertainty_breakdown=unc_low,
        step=3,
    )
    assert dec_ood.abstain is True
    assert dec_ood.requires_human_review is True
    assert dec_ood.recommended_resolution == (AbstentionResolutionAction.HUMAN_REVIEW_REQUIRED)
    assert AbstentionTriggerKind.OUT_OF_DOMAIN_STATE in dec_ood.triggered_validity_limits
    assert AbstentionTriggerKind.UNSUPPORTED_REGIME_CONTEXT in dec_ood.triggered_validity_limits
    assert UncertaintySourceType.RUNTIME_CONTEXT in dec_ood.triggered_uncertainty_sources
    assert UncertaintySourceType.STRUCTURAL_MODEL in dec_ood.triggered_uncertainty_sources
    assert dec_ood.human_review_brief is not None
    assert "DO NOT USE MODEL FOR AUTOMATED ACTUATION" in dec_ood.human_review_brief
