"""Validity Limits in the Flagship Domain: Performativity, Endogeneity & Stylized Adversarial Adaptation.

Explicit Epistemic Status:
    - Performativity & Endogeneity (`CONTROLLED_SYNTHETIC_VALIDITY_LIMITS_NOT_REAL_WORLD_EFFICACY`):
      Demonstrates in a controlled synthetic environment how deploying an "optimal" policy
      alters the behavioral elasticities it was optimized on (Lucas Critique / Performativity)
      and how closed-loop state-action feedback confounds static associations.
    - Adversarial Adaptation (`STYLIZED_SYNTHETIC_ADVERSARY_NOT_REAL_FRAUD_RING_CLAIM`):
      Models a stylized strategic adversary in a controlled synthetic environment — NOT an
      empirical claim about real-world fraud rings.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.drift import (
    DriftSignal,
    EndogenousDecisionFeedbackSpec,
    PerformativeCouplingSpec,
    PlanInvalidationAudit,
    check_plan_performative_validity,
    detect_world_drift,
)
from ewm_engine.dynamics.base import ValidityDomain
from ewm_engine.evaluation.abstention import (
    CompetenceEstimator,
    ValidityLimitAbstentionDecision,
    ValidityLimitAbstentionGate,
)
from ewm_engine.evaluation.uncertainty import UncertaintyTaxonomyBreakdown
from ewm_engine.flagship.fraud_cascade import (
    FLAGSHIP_SCENARIO_ID,
    FraudCascadeParameters,
    build_default_fraud_context,
    build_initial_fraud_cascade_state,
)

VALIDITY_LIMITS_EPISTEMIC_LABEL = "CONTROLLED_SYNTHETIC_VALIDITY_LIMITS_NOT_REAL_WORLD_EFFICACY"
ADVERSARIAL_EPISTEMIC_LABEL = "STYLIZED_SYNTHETIC_ADVERSARY_NOT_REAL_FRAUD_RING_CLAIM"
ADVERSARIAL_EPISTEMIC_DISCLAIMER = (
    "Stylized strategic adversary in a controlled synthetic environment — "
    "NOT an empirical claim about real-world fraud rings."
)


class PerformativeCustomerShiftParams(BaseModel):
    """Parameters governing policy-induced customer behavioral shift (Lucas Critique)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_policy_tau: float = Field(
        default=0.65,
        description="Historical authorization threshold tau_0 under which elasticities were calibrated.",
    )
    retry_decay_rate_per_step: float = Field(
        default=0.35,
        ge=0.0,
        description="Rate at which customer retry propensity decays per unit (tau_0 - tau) per step.",
    )
    step_up_rescue_decay_rate_per_step: float = Field(
        default=0.55,
        ge=0.0,
        description="Rate at which SMS step-up rescue effectiveness decays due to customer verification fatigue.",
    )
    call_escalation_learning_rate: float = Field(
        default=0.22,
        ge=0.0,
        description="Rate at which customer call-escalation propensity per FP decline rises per unit (tau_0 - tau) per step.",
    )
    churn_sensitivity_growth_rate: float = Field(
        default=0.55,
        ge=0.0,
        description="Rate at which customer churn sensitivity to false declines grows per unit (tau_0 - tau) per step.",
    )
    endogenous_supervisor_gain: float = Field(
        default=0.25,
        ge=0.0,
        description="Closed-loop endogenous operator response gain when wait time exceeds nominal levels.",
    )


class PerformativeStepComparison(BaseModel):
    """Step-by-step comparison between static-assumptive EWM and performative ground truth."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    post_deploy_step: int
    static_predicted_call_volume: float
    realized_performative_call_volume: float
    call_volume_abs_error: float
    static_predicted_wait_seconds: float
    realized_performative_wait_seconds: float
    realized_retry_fraction: float
    realized_call_escalation_rate: float
    realized_churn_multiplier: float
    static_predicted_step_utility_usd: float
    realized_performative_step_utility_usd: float
    forecast_invalidated_at_step: bool
    sla_breached_at_step: bool


class PerformativityValidityLimitReport(BaseModel):
    """Measured validity-limit report when a deployed optimal policy alters customer behavior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = FLAGSHIP_SCENARIO_ID
    epistemic_label: str = VALIDITY_LIMITS_EPISTEMIC_LABEL
    deployed_policy_id: str
    deployed_tau: float
    deployed_agents: float
    horizon_steps: int
    forecast_invalidation_tolerance_calls: float
    sla_wait_target_seconds: float
    time_to_forecast_invalidation_steps: int | None = Field(
        description="First post-deployment step where call-volume error exceeds tolerance."
    )
    time_to_sla_breach_steps: int | None = Field(
        description="First post-deployment step where realized wait time breaches the 120s SLA."
    )
    static_predicted_total_utility_usd: float
    realized_performative_total_utility_usd: float
    performative_utility_erosion_usd: float
    performative_aware_policy_id: str
    performative_aware_total_utility_usd: float
    performative_policy_regret_usd: float
    baseline_call_escalation_rate: float
    terminal_call_escalation_rate: float
    baseline_retry_fraction: float
    terminal_retry_fraction: float
    endogenous_state_action_correlation: float
    plan_invalidation_audit: PlanInvalidationAudit
    abstention_decision: ValidityLimitAbstentionDecision
    step_comparisons: tuple[PerformativeStepComparison, ...]
    fingerprint: str


class StylizedAdversaryParams(BaseModel):
    """Parameters for a stylized strategic fraud adversary in a controlled environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_policy_tau: float = Field(default=0.65)
    evasion_decay_rate: float = Field(
        default=0.028,
        ge=0.0,
        description="Base per-step score-evasion decay rate of fraud recall.",
    )
    policy_tightness_evasion_multiplier: float = Field(
        default=2.2,
        ge=0.0,
        description="Additional evasion acceleration when the defender tightens tau below tau_0.",
    )
    velocity_splitting_rate: float = Field(
        default=0.08,
        ge=0.0,
        description="Rate at which attackers split high-value attempts into higher-volume attack waves.",
    )
    degradation_relative_drop_threshold: float = Field(
        default=0.15,
        gt=0.0,
        lt=1.0,
        description="Relative drop in fraud catch rate (e.g. 15%) that defines time-to-degradation.",
    )


class AdversarialPolicyRobustnessRecord(BaseModel):
    """Evaluation of a candidate policy along the Adversarial Robustness Axis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str
    tau_fraud: float
    staffed_agents: float
    is_adaptive_recalibrated: bool = False
    initial_fraud_catch_rate: float
    terminal_fraud_catch_rate: float
    relative_catch_rate_drop: float
    time_to_degradation_steps: int | None = Field(
        description="Exact post-deployment step where fraud catch rate drops by >= 15% from initial."
    )
    static_assumed_utility_usd: float
    realized_adversarial_utility_usd: float
    adversarial_utility_degradation_usd: float
    excess_unblocked_fraud_loss_usd: float
    robustness_retention_ratio: float = Field(
        description="Ratio of realized adversarial utility to static assumed utility."
    )


class AdversarialAdaptationReport(BaseModel):
    """Report measuring policy degradation and robustness under stylized adversarial adaptation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str = FLAGSHIP_SCENARIO_ID
    epistemic_label: str = ADVERSARIAL_EPISTEMIC_LABEL
    epistemic_disclaimer: str = ADVERSARIAL_EPISTEMIC_DISCLAIMER
    horizon_steps: int
    degradation_threshold_pct: float
    policy_records: tuple[AdversarialPolicyRobustnessRecord, ...]
    drift_signals: tuple[DriftSignal, ...]
    abstention_decision: ValidityLimitAbstentionDecision
    fingerprint: str


def _simulate_cascade_step_custom(
    *,
    params: FraudCascadeParameters,
    active_customers: float,
    call_backlog: float,
    churn_propensity: float,
    pending_retries: float,
    tau_fraud: float,
    staffed_agents: float,
    retry_fraction: float,
    call_escalation_per_fp: float,
    churn_sensitivity_mult: float,
    fraud_catch_rate_override: float | None,
    fraud_volume_mult: float,
    rng: np.random.Generator,
    step_up_rescue_fraction: float | None = None,
) -> dict[str, float]:
    """Execute one step of the 4-layer Fraud -> Declines -> Call-Center cascade with custom elasticities."""
    tx_noise = float(rng.normal(0.0, 12.0))
    fraud_noise = float(rng.normal(0.0, 2.5))
    call_noise = float(rng.normal(0.0, 3.0))

    legit_attempts = max(
        0.0,
        active_customers * params.tx_attempt_rate_per_customer + pending_retries + tx_noise,
    )
    fraud_attempts = max(
        0.0,
        (params.base_fraud_attempt_volume * 1.65) * fraud_volume_mult + fraud_noise,
    )

    if fraud_catch_rate_override is not None:
        p_catch = float(np.clip(fraud_catch_rate_override, 0.05, 0.99))
    else:
        p_catch = float(np.clip(1.0 - params.fraud_catch_slope * (tau_fraud**2), 0.05, 0.99))

    p_fp_raw = float(
        np.clip(
            params.false_positive_base_rate
            * math.exp(-params.false_positive_decay * (tau_fraud - 0.20)),
            0.005,
            0.40,
        )
    )

    blocked_fraud = fraud_attempts * p_catch
    unblocked_fraud = max(0.0, fraud_attempts - blocked_fraud)

    raw_fp_declines = legit_attempts * p_fp_raw
    eff_rescue = (
        step_up_rescue_fraction
        if step_up_rescue_fraction is not None
        else params.step_up_sms_rescue_fraction
    )
    rescued_fp = raw_fp_declines * eff_rescue
    unrescued_fp = max(0.0, raw_fp_declines - rescued_fp)
    approved_legit = max(0.0, legit_attempts - unrescued_fp)

    locked_accounts = unrescued_fp * params.account_lock_trigger_fraction
    next_retries = unrescued_fp * retry_fraction
    switched_tx = unrescued_fp * params.channel_switch_fraction

    new_calls = max(
        0.0,
        active_customers * params.routine_call_rate_per_customer
        + unrescued_fp * call_escalation_per_fp
        + locked_accounts * params.call_escalation_per_locked_account
        + call_noise,
    )

    total_demand = call_backlog + new_calls
    capacity = staffed_agents * params.calls_per_agent_per_step
    handled_calls = min(total_demand, capacity)
    unmet = max(0.0, total_demand - handled_calls)
    abandoned_calls = unmet * params.queue_abandonment_rate
    next_backlog = min(params.max_call_backlog, max(0.0, unmet - abandoned_calls))

    utilization = total_demand / max(capacity, 1.0)
    overload = max(0.0, utilization - 0.85)
    wait_seconds = (
        params.base_wait_seconds
        + params.wait_congestion_coeff * (utilization**2)
        + 260.0 * (overload**2)
    )
    sla_breached = 1.0 if wait_seconds > params.sla_wait_target_seconds else 0.0

    fp_rate_realized = unrescued_fp / max(legit_attempts, 1.0)
    wait_excess_ratio = max(
        0.0,
        (wait_seconds - params.sla_wait_target_seconds) / params.sla_wait_target_seconds,
    )
    next_churn_propensity = float(
        np.clip(
            0.82 * churn_propensity
            + churn_sensitivity_mult * (0.28 * fp_rate_realized + 0.045 * wait_excess_ratio),
            0.005,
            0.35,
        )
    )
    newly_churned = min(active_customers, active_customers * next_churn_propensity * 0.04)
    next_active_customers = max(0.0, active_customers - newly_churned)

    gross_margin_rev = approved_legit * params.avg_order_value * params.gross_margin_rate
    fraud_loss = unblocked_fraud * params.avg_order_value * params.fraud_chargeback_Cost_multiplier
    sms_cost = raw_fp_declines * params.step_up_sms_cost_per_attempt
    switch_cost = switched_tx * params.channel_switch_fee
    staff_cost = staffed_agents * params.cost_per_agent_per_step
    call_cost = handled_calls * params.cost_per_handled_call
    sla_penalty = sla_breached * params.sla_breach_penalty_per_step
    churn_ltv_loss = newly_churned * params.churn_ltv_loss_per_customer

    step_utility = (
        gross_margin_rev
        - fraud_loss
        - sms_cost
        - switch_cost
        - staff_cost
        - call_cost
        - sla_penalty
        - churn_ltv_loss
    )

    return {
        "new_calls": new_calls,
        "wait_seconds": wait_seconds,
        "sla_breached": sla_breached,
        "next_backlog": next_backlog,
        "next_active_customers": next_active_customers,
        "next_churn_propensity": next_churn_propensity,
        "next_retries": next_retries,
        "p_catch": p_catch,
        "fraud_loss": fraud_loss,
        "step_utility": step_utility,
    }


def run_performativity_endogeneity_scenario(
    *,
    seed: int = 42,
    horizon_steps: int = 18,
    deployed_tau: float = 0.48,
    deployed_agents: float = 38.0,
    forecast_invalidation_tolerance_calls: float = 12.0,
    shift_params: PerformativeCustomerShiftParams | None = None,
) -> PerformativityValidityLimitReport:
    """Run the Performativity & Endogeneity worked scenario where deploying the optimal policy shifts customer behavior.

    Compares the Static-Assumptive EWM (which assumes customer retry, call-escalation, and
    churn elasticities remain invariant at their historical pi_0 values) against the
    Performative Ground Truth over ``horizon_steps`` post-deployment steps.
    """
    params = FraudCascadeParameters()
    sp = shift_params or PerformativeCustomerShiftParams()
    tau_delta = max(0.0, sp.baseline_policy_tau - deployed_tau)

    rng_static = np.random.default_rng(seed)
    rng_perf = np.random.default_rng(seed)
    rng_aware = np.random.default_rng(seed)

    # Track state for Static-Assumptive EWM
    s_cust = params.initial_active_customers
    s_backlog = 18.0
    s_churn = params.initial_churn_propensity
    s_retries = 12.0

    # Track state for Realized Performative Environment
    p_cust = params.initial_active_customers
    p_backlog = 18.0
    p_churn = params.initial_churn_propensity
    p_retries = 12.0

    # Track state for Performativity-Aware Policy (tau=0.53, agents=42)
    aware_tau = 0.53
    aware_agents = 42.0
    aware_tau_delta = max(0.0, sp.baseline_policy_tau - aware_tau)
    a_cust = params.initial_active_customers
    a_backlog = 18.0
    a_churn = params.initial_churn_propensity
    a_retries = 12.0

    comparisons: list[PerformativeStepComparison] = []
    time_to_forecast_inv: int | None = None
    time_to_sla_breach: int | None = None

    static_total_util = 0.0
    perf_total_util = 0.0
    aware_total_util = 0.0

    wait_series: list[float] = []
    endog_action_series: list[float] = []

    for step in range(horizon_steps):
        # 1. Static-Assumptive EWM rollout step (assumes constant historical elasticities)
        out_static = _simulate_cascade_step_custom(
            params=params,
            active_customers=s_cust,
            call_backlog=s_backlog,
            churn_propensity=s_churn,
            pending_retries=s_retries,
            tau_fraud=deployed_tau,
            staffed_agents=deployed_agents,
            retry_fraction=params.retry_fraction,
            call_escalation_per_fp=params.call_escalation_per_fp_decline,
            churn_sensitivity_mult=1.0,
            fraud_catch_rate_override=None,
            fraud_volume_mult=1.0,
            rng=rng_static,
        )
        s_cust = out_static["next_active_customers"]
        s_backlog = out_static["next_backlog"]
        s_churn = out_static["next_churn_propensity"]
        s_retries = out_static["next_retries"]
        static_total_util += out_static["step_utility"]

        # 2. Performative Ground Truth step (policy-induced elasticity shift + endogenous feedback)
        realized_retry = float(
            np.clip(
                params.retry_fraction * math.exp(-sp.retry_decay_rate_per_step * tau_delta * step),
                0.05,
                params.retry_fraction,
            )
        )
        realized_rescue = float(
            np.clip(
                params.step_up_sms_rescue_fraction
                * math.exp(-sp.step_up_rescue_decay_rate_per_step * tau_delta * step),
                0.10,
                params.step_up_sms_rescue_fraction,
            )
        )
        realized_call_esc = float(
            np.clip(
                params.call_escalation_per_fp_decline
                + sp.call_escalation_learning_rate * tau_delta * step,
                params.call_escalation_per_fp_decline,
                0.85,
            )
        )
        realized_churn_mult = 1.0 + sp.churn_sensitivity_growth_rate * tau_delta * step

        out_perf = _simulate_cascade_step_custom(
            params=params,
            active_customers=p_cust,
            call_backlog=p_backlog,
            churn_propensity=p_churn,
            pending_retries=p_retries,
            tau_fraud=deployed_tau,
            staffed_agents=deployed_agents,
            retry_fraction=realized_retry,
            call_escalation_per_fp=realized_call_esc,
            churn_sensitivity_mult=realized_churn_mult,
            fraud_catch_rate_override=None,
            fraud_volume_mult=1.0,
            rng=rng_perf,
            step_up_rescue_fraction=realized_rescue,
        )
        p_cust = out_perf["next_active_customers"]
        p_backlog = out_perf["next_backlog"]
        p_churn = out_perf["next_churn_propensity"]
        p_retries = out_perf["next_retries"]
        perf_total_util += out_perf["step_utility"]

        # Record endogenous closed-loop supervisor pressure signal
        w_t = out_perf["wait_seconds"]
        wait_series.append(w_t)
        endog_override = deployed_agents + sp.endogenous_supervisor_gain * max(
            0.0, (w_t - 60.0) / 10.0
        )
        endog_action_series.append(endog_override)

        # 3. Performativity-Aware Policy rollout step
        aware_retry = float(
            np.clip(
                params.retry_fraction
                * math.exp(-sp.retry_decay_rate_per_step * aware_tau_delta * step),
                0.05,
                params.retry_fraction,
            )
        )
        aware_rescue = float(
            np.clip(
                params.step_up_sms_rescue_fraction
                * math.exp(-sp.step_up_rescue_decay_rate_per_step * aware_tau_delta * step),
                0.10,
                params.step_up_sms_rescue_fraction,
            )
        )
        aware_call_esc = float(
            np.clip(
                params.call_escalation_per_fp_decline
                + sp.call_escalation_learning_rate * aware_tau_delta * step,
                params.call_escalation_per_fp_decline,
                0.85,
            )
        )
        aware_churn_mult = 1.0 + sp.churn_sensitivity_growth_rate * aware_tau_delta * step

        out_aware = _simulate_cascade_step_custom(
            params=params,
            active_customers=a_cust,
            call_backlog=a_backlog,
            churn_propensity=a_churn,
            pending_retries=a_retries,
            tau_fraud=aware_tau,
            staffed_agents=aware_agents,
            retry_fraction=aware_retry,
            call_escalation_per_fp=aware_call_esc,
            churn_sensitivity_mult=aware_churn_mult,
            fraud_catch_rate_override=None,
            fraud_volume_mult=1.0,
            rng=rng_aware,
            step_up_rescue_fraction=aware_rescue,
        )
        a_cust = out_aware["next_active_customers"]
        a_backlog = out_aware["next_backlog"]
        a_churn = out_aware["next_churn_propensity"]
        a_retries = out_aware["next_retries"]
        aware_total_util += out_aware["step_utility"]

        call_err = abs(out_perf["new_calls"] - out_static["new_calls"])
        inv_now = call_err > forecast_invalidation_tolerance_calls
        sla_now = out_perf["sla_breached"] > 0.5

        if inv_now and time_to_forecast_inv is None:
            time_to_forecast_inv = step
        if sla_now and time_to_sla_breach is None:
            time_to_sla_breach = step

        comparisons.append(
            PerformativeStepComparison(
                post_deploy_step=step,
                static_predicted_call_volume=round(out_static["new_calls"], 4),
                realized_performative_call_volume=round(out_perf["new_calls"], 4),
                call_volume_abs_error=round(call_err, 4),
                static_predicted_wait_seconds=round(out_static["wait_seconds"], 4),
                realized_performative_wait_seconds=round(out_perf["wait_seconds"], 4),
                realized_retry_fraction=round(realized_retry, 4),
                realized_call_escalation_rate=round(realized_call_esc, 4),
                realized_churn_multiplier=round(realized_churn_mult, 4),
                static_predicted_step_utility_usd=round(out_static["step_utility"], 2),
                realized_performative_step_utility_usd=round(out_perf["step_utility"], 2),
                forecast_invalidated_at_step=inv_now,
                sla_breached_at_step=sla_now,
            )
        )

    # Compute endogenous state-action correlation
    w_arr = np.asarray(wait_series, dtype=float)
    a_arr = np.asarray(endog_action_series, dtype=float)
    if float(np.std(w_arr)) > 1e-9 and float(np.std(a_arr)) > 1e-9:
        endog_corr = float(np.corrcoef(w_arr, a_arr)[0, 1])
    else:
        endog_corr = 0.0

    coupling = PerformativeCouplingSpec(
        coupling_id="fraud_threshold_customer_adaptation",
        policy_key="auth_threshold",
        baseline_policy_value=sp.baseline_policy_tau,
        affected_behavioral_parameters=(
            "retry_fraction",
            "call_escalation_per_fp_decline",
            "churn_sensitivity_mult",
        ),
        sensitivity_elasticities={
            "retry_fraction": 1.25,
            "call_escalation_per_fp_decline": 2.40,
            "churn_sensitivity_mult": 3.50,
        },
        assumed_static_mechanisms=(
            "layer3_customer_behavior_transition",
            "layer4_call_center_queue_dynamics",
        ),
        invalidation_threshold=0.05,
        coupling_kind="performative_customer_response",
    )
    endog_spec = EndogenousDecisionFeedbackSpec(
        feedback_id="supervisor_queue_override_loop",
        action_key="staffed_agents",
        conditioning_state_keys=("call_backlog", "wait_seconds"),
        unobserved_confounder_keys=("shift_supervisor_urgency",),
        feedback_gain=sp.endogenous_supervisor_gain,
        invalidates_static_association_mechanisms=("observational_ridge_wait_predictor",),
    )

    baseline_beh = {
        "retry_fraction": params.retry_fraction,
        "call_escalation_per_fp_decline": params.call_escalation_per_fp_decline,
        "churn_sensitivity_mult": 1.0,
    }
    realized_beh = {
        "retry_fraction": comparisons[-1].realized_retry_fraction,
        "call_escalation_per_fp_decline": comparisons[-1].realized_call_escalation_rate,
        "churn_sensitivity_mult": comparisons[-1].realized_churn_multiplier,
    }

    plan_audit = check_plan_performative_validity(
        deployed_policy_parameters={
            "auth_threshold": deployed_tau,
            "staffed_agents": deployed_agents,
        },
        performative_couplings=(coupling,),
        endogenous_feedback_specs=(endog_spec,),
        active_mechanism_ids=(
            "layer3_customer_behavior_transition",
            "layer4_call_center_queue_dynamics",
            "observational_ridge_wait_predictor",
        ),
        baseline_behavioral_parameters=baseline_beh,
        realized_behavioral_parameters=realized_beh,
    )

    # Evaluate abstention gate under performative discrepancy
    state_init = build_initial_fraud_cascade_state(params)
    ctx_init = build_default_fraud_context(params)
    terminal_call_discrepancy = comparisons[-1].call_volume_abs_error
    unc_breakdown = UncertaintyTaxonomyBreakdown.from_simulation_rollouts(
        [c.static_predicted_call_volume for c in comparisons],
        structural_model_std=terminal_call_discrepancy * 0.5,
        model_discrepancy_vs_reality=terminal_call_discrepancy,
    )
    gate = ValidityLimitAbstentionGate(
        max_structural_model_std=10.0,
        max_model_discrepancy=forecast_invalidation_tolerance_calls,
        validity_domain=ValidityDomain(
            allowed_regimes=("nominal", "shift"),
            resource_bounds={"call_backlog": (0.0, 120.0)},
        ),
        competence_estimator=CompetenceEstimator(max_verified_horizon=15),
        fallback_policy_id="performative_safe_fallback_tau_0.55_agents_40",
        fallback_actions=(
            Action(
                id="fallback_relax_tau_and_staff",
                actor_id="governance_circuit_breaker",
                type="set_authorization_and_staffing",
                parameters={"auth_threshold": 0.55, "staffed_agents": 40.0},
            ),
        ),
    )
    abstention = gate.evaluate(
        state=state_init,
        runtime_context=ctx_init,
        uncertainty_breakdown=unc_breakdown,
        drift_signals=plan_audit.drift_signals,
        plan_invalidation=plan_audit,
        step=horizon_steps,
    )

    erosion = static_total_util - perf_total_util
    regret_vs_aware = max(0.0, aware_total_util - perf_total_util)

    fp_payload = {
        "scenario_id": FLAGSHIP_SCENARIO_ID,
        "deployed_tau": deployed_tau,
        "deployed_agents": deployed_agents,
        "time_to_forecast_inv": time_to_forecast_inv,
        "time_to_sla_breach": time_to_sla_breach,
        "static_total_util": round(static_total_util, 2),
        "perf_total_util": round(perf_total_util, 2),
        "aware_total_util": round(aware_total_util, 2),
    }
    fp = hashlib.sha256(
        json.dumps(fp_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return PerformativityValidityLimitReport(
        deployed_policy_id="coordinated_ewm_tau_0.48_agents_38",
        deployed_tau=deployed_tau,
        deployed_agents=deployed_agents,
        horizon_steps=horizon_steps,
        forecast_invalidation_tolerance_calls=forecast_invalidation_tolerance_calls,
        sla_wait_target_seconds=params.sla_wait_target_seconds,
        time_to_forecast_invalidation_steps=time_to_forecast_inv,
        time_to_sla_breach_steps=time_to_sla_breach,
        static_predicted_total_utility_usd=round(static_total_util, 2),
        realized_performative_total_utility_usd=round(perf_total_util, 2),
        performative_utility_erosion_usd=round(erosion, 2),
        performative_aware_policy_id="performative_aware_tau_0.53_agents_42",
        performative_aware_total_utility_usd=round(aware_total_util, 2),
        performative_policy_regret_usd=round(regret_vs_aware, 2),
        baseline_call_escalation_rate=params.call_escalation_per_fp_decline,
        terminal_call_escalation_rate=comparisons[-1].realized_call_escalation_rate,
        baseline_retry_fraction=params.retry_fraction,
        terminal_retry_fraction=comparisons[-1].realized_retry_fraction,
        endogenous_state_action_correlation=round(endog_corr, 4),
        plan_invalidation_audit=plan_audit,
        abstention_decision=abstention,
        step_comparisons=tuple(comparisons),
        fingerprint=fp,
    )


def run_adversarial_adaptation_scenario(
    *,
    seed: int = 42,
    horizon_steps: int = 18,
    adversary_params: StylizedAdversaryParams | None = None,
) -> AdversarialAdaptationReport:
    """Run the Stylized Adversarial Adaptation stylized adversarial adaptation evaluation across candidate policies."""
    params = FraudCascadeParameters()
    ap = adversary_params or StylizedAdversaryParams()

    candidate_policies: list[tuple[str, float, float, bool]] = [
        ("status_quo_tau_0.55_agents_32", 0.55, 32.0, False),
        ("myopic_aggressive_tau_0.40_agents_32", 0.40, 32.0, False),
        ("uncoordinated_aggressive_tau_0.40_agents_35", 0.40, 35.0, False),
        ("coordinated_ewm_tau_0.48_agents_38", 0.48, 38.0, False),
        ("adaptive_rolling_recalibration_tau_0.48_agents_38", 0.48, 38.0, True),
    ]

    policy_records: list[AdversarialPolicyRobustnessRecord] = []
    max_evasion_mag = 0.0

    for pol_id, tau, agents, is_adaptive in candidate_policies:
        rng_static = np.random.default_rng(seed)
        rng_adv = np.random.default_rng(seed)

        s_cust = params.initial_active_customers
        s_backlog = 18.0
        s_churn = params.initial_churn_propensity
        s_retries = 12.0

        a_cust = params.initial_active_customers
        a_backlog = 18.0
        a_churn = params.initial_churn_propensity
        a_retries = 12.0

        static_util = 0.0
        adv_util = 0.0
        static_fraud_loss = 0.0
        adv_fraud_loss = 0.0

        base_p_catch = float(np.clip(1.0 - params.fraud_catch_slope * (tau**2), 0.05, 0.99))
        tightness = max(0.0, ap.baseline_policy_tau - tau)
        step_decay_rate = ap.evasion_decay_rate * (
            1.0 + ap.policy_tightness_evasion_multiplier * tightness
        )

        time_to_degrade: int | None = None
        terminal_catch = base_p_catch

        for step in range(horizon_steps):
            # 1. Non-adaptive static adversary assumption
            out_s = _simulate_cascade_step_custom(
                params=params,
                active_customers=s_cust,
                call_backlog=s_backlog,
                churn_propensity=s_churn,
                pending_retries=s_retries,
                tau_fraud=tau,
                staffed_agents=agents,
                retry_fraction=params.retry_fraction,
                call_escalation_per_fp=params.call_escalation_per_fp_decline,
                churn_sensitivity_mult=1.0,
                fraud_catch_rate_override=base_p_catch,
                fraud_volume_mult=1.0,
                rng=rng_static,
            )
            s_cust = out_s["next_active_customers"]
            s_backlog = out_s["next_backlog"]
            s_churn = out_s["next_churn_propensity"]
            s_retries = out_s["next_retries"]
            static_util += out_s["step_utility"]
            static_fraud_loss += out_s["fraud_loss"]

            # 2. Stylized strategic adversary adaptation
            # If adaptive_rolling_recalibration is enabled, the defender refreshes features every 5 steps
            effective_step_since_refresh = (step % 5) if is_adaptive else step
            realized_catch = float(
                np.clip(
                    base_p_catch * math.exp(-step_decay_rate * effective_step_since_refresh),
                    0.10,
                    0.99,
                )
            )
            vol_mult = 1.0 + ap.velocity_splitting_rate * tightness * effective_step_since_refresh

            rel_drop = (base_p_catch - realized_catch) / max(base_p_catch, 1e-9)
            if rel_drop >= ap.degradation_relative_drop_threshold and time_to_degrade is None:
                time_to_degrade = step
            terminal_catch = realized_catch

            out_a = _simulate_cascade_step_custom(
                params=params,
                active_customers=a_cust,
                call_backlog=a_backlog,
                churn_propensity=a_churn,
                pending_retries=a_retries,
                tau_fraud=tau,
                staffed_agents=agents,
                retry_fraction=params.retry_fraction,
                call_escalation_per_fp=params.call_escalation_per_fp_decline,
                churn_sensitivity_mult=1.0,
                fraud_catch_rate_override=realized_catch,
                fraud_volume_mult=vol_mult,
                rng=rng_adv,
            )
            a_cust = out_a["next_active_customers"]
            a_backlog = out_a["next_backlog"]
            a_churn = out_a["next_churn_propensity"]
            a_retries = out_a["next_retries"]
            adv_util += out_a["step_utility"]
            adv_fraud_loss += out_a["fraud_loss"]

        total_rel_drop = (base_p_catch - terminal_catch) / max(base_p_catch, 1e-9)
        max_evasion_mag = max(max_evasion_mag, total_rel_drop)
        retention = adv_util / max(static_util, 1.0)

        policy_records.append(
            AdversarialPolicyRobustnessRecord(
                policy_id=pol_id,
                tau_fraud=tau,
                staffed_agents=agents,
                is_adaptive_recalibrated=is_adaptive,
                initial_fraud_catch_rate=round(base_p_catch, 4),
                terminal_fraud_catch_rate=round(terminal_catch, 4),
                relative_catch_rate_drop=round(total_rel_drop, 4),
                time_to_degradation_steps=time_to_degrade,
                static_assumed_utility_usd=round(static_util, 2),
                realized_adversarial_utility_usd=round(adv_util, 2),
                adversarial_utility_degradation_usd=round(static_util - adv_util, 2),
                excess_unblocked_fraud_loss_usd=round(
                    max(0.0, adv_fraud_loss - static_fraud_loss), 2
                ),
                robustness_retention_ratio=round(retention, 4),
            )
        )

    adv_signals = detect_world_drift(
        adversarial_evasion_magnitude=max_evasion_mag,
        adversarial_affected_keys=(
            "fraud_catch_slope",
            "base_fraud_attempt_volume",
        ),
    )

    unc_breakdown = UncertaintyTaxonomyBreakdown.from_simulation_rollouts(
        [r.realized_adversarial_utility_usd for r in policy_records],
        model_discrepancy_vs_reality=max_evasion_mag,
    )
    gate = ValidityLimitAbstentionGate(
        max_model_discrepancy=ap.degradation_relative_drop_threshold,
        fallback_policy_id="adaptive_rolling_recalibration_tau_0.48_agents_38",
    )
    abstention = gate.evaluate(
        uncertainty_breakdown=unc_breakdown,
        drift_signals=adv_signals,
        step=horizon_steps,
    )

    fp_payload = {
        "scenario_id": FLAGSHIP_SCENARIO_ID,
        "epistemic_label": ADVERSARIAL_EPISTEMIC_LABEL,
        "policies": [r.model_dump(mode="json") for r in policy_records],
    }
    fp = hashlib.sha256(
        json.dumps(fp_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return AdversarialAdaptationReport(
        horizon_steps=horizon_steps,
        degradation_threshold_pct=ap.degradation_relative_drop_threshold * 100.0,
        policy_records=tuple(policy_records),
        drift_signals=tuple(adv_signals),
        abstention_decision=abstention,
        fingerprint=fp,
    )


__all__ = [
    "ADVERSARIAL_EPISTEMIC_DISCLAIMER",
    "ADVERSARIAL_EPISTEMIC_LABEL",
    "VALIDITY_LIMITS_EPISTEMIC_LABEL",
    "AdversarialAdaptationReport",
    "AdversarialPolicyRobustnessRecord",
    "PerformativeCustomerShiftParams",
    "PerformativeStepComparison",
    "PerformativityValidityLimitReport",
    "StylizedAdversaryParams",
    "run_adversarial_adaptation_scenario",
    "run_performativity_endogeneity_scenario",
]
