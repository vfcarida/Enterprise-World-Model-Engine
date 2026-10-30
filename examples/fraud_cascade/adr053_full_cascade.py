"""Full 6-Part ADR-053 State-Model Integration for the Fraud -> Declines -> Call-Center Cascade.

Demonstrates explicit per-step evolution of all 6 ADR-053 components:
1. Ontic State (S_t): True transaction cohorts, call queue backlog, and net cash in WorldState
2. Partial Observation (O_t): Actor-filtered view via ObservationFilter (masking latent fraud rate)
3. Belief State (B_t): Estimated latent resources and epistemic uncertainty via BeliefState
4. Epistemic State (P_t): Provenance, EvidenceLevel, and DriftSignal tracking via EpistemicState
5. Runtime Context (K_t): Versioned policy/staffing configuration via ContextSnapshot
6. Action / Intervention (A_t): Fraud threshold intervention do(tau) via Action
"""

from __future__ import annotations

from dataclasses import dataclass

from ewm_engine.core import (
    Action,
    AuditMetadataContextDiscovery,
    BeliefState,
    ContextSnapshot,
    DriftKind,
    DriftOrigin,
    DriftSignal,
    Entity,
    EpistemicState,
    ExponentialSmoothingBeliefUpdater,
    ObservationFilter,
    ObservationSpec,
    Relationship,
    Resource,
    WorldState,
    separate_ontic_epistemic,
)


@dataclass(frozen=True)
class ADR053FraudCascadeStepState:
    """Explicit 6-part ADR-053 state tuple (S_t, O_t, B_t, P_t, K_t, A_t) at step t."""

    step: int
    ontic_state_s_t: WorldState
    observation_o_t: WorldState
    belief_state_b_t: BeliefState
    epistemic_state_p_t: EpistemicState
    context_k_t: ContextSnapshot
    action_a_t: Action


@dataclass(frozen=True)
class ADR053FraudCascadeShowcaseResult:
    """Summary of comparing aggressive vs. balanced fraud thresholds under the 6-part ADR-053 model."""

    steps_simulated: int
    myopic_tau: float
    balanced_tau: float
    myopic_regret_usd: float
    balanced_regret_usd: float
    cohort_conservation_max_drift: float
    latent_fraud_masked_in_o_t: bool
    trajectory_balanced: tuple[ADR053FraudCascadeStepState, ...]


def _simulate_adr053_policy(
    tau: float,
    horizon: int = 12,
) -> tuple[float, float, bool, list[ADR053FraudCascadeStepState]]:
    discovery = AuditMetadataContextDiscovery()
    obs_filter = ObservationFilter(
        ObservationSpec(
            actor_id="fraud_ops_planner",
            observable_entity_ids=frozenset({"payment_gateway", "call_center"}),
            observable_resource_ids=frozenset(
                {"approved_tx", "declined_tx", "call_queue_backlog", "enterprise_net_cash"}
            ),
        )
    )
    updater = ExponentialSmoothingBeliefUpdater(
        prior_resource_estimates={"latent_true_fraud_attempts": 45.0},
        prior_uncertainty=2.0,
        uncertainty_growth_rate=0.15,
    )

    belief: BeliefState | None = None
    steps_out: list[ADR053FraudCascadeStepState] = []
    max_conservation_drift = 0.0
    latent_masked_all = True
    cumulative_utility = 0.0
    queue_backlog = 20.0

    for t in range(horizon):
        k_t = discovery.discover_context(
            {
                "context_id": f"fraud_cascade_ctx_t{t}",
                "version": "1.0.0" if t < 6 else "1.1.0",
                "environment_regime": "nominal" if t < 6 else "holiday_traffic_surge",
                "parameters": {
                    "submitted_tx": 1000.0 if t < 6 else 1250.0,
                    "agent_service_capacity": 85.0,
                    "sla_penalty_per_overflow": 220.0,
                },
                "feature_flags": {"auto_callback_queue": True},
            }
        )
        submitted = float(k_t.parameters["submitted_tx"])
        service_cap = float(k_t.parameters["agent_service_capacity"])
        sla_penalty = float(k_t.parameters["sla_penalty_per_overflow"])

        true_fraud = 0.045 * submitted
        legit_tx = submitted - true_fraud

        # Aggressive tau=0.40 blocks more fraud but triggers high false-positive declines
        if tau <= 0.45:
            blocked_fraud = 0.92 * true_fraud
            false_declines = 0.14 * legit_tx
        else:
            blocked_fraud = 0.80 * true_fraud
            false_declines = 0.025 * legit_tx

        unblocked_fraud = true_fraud - blocked_fraud
        approved_legit = legit_tx - false_declines
        total_approved = approved_legit + unblocked_fraud
        total_declined = blocked_fraud + false_declines

        # Exact cohort conservation: submitted == total_approved + total_declined
        drift = abs((total_approved + total_declined) - submitted)
        max_conservation_drift = max(max_conservation_drift, drift)

        call_arrivals = 0.65 * false_declines
        queue_backlog = max(0.0, queue_backlog + call_arrivals - service_cap)
        sla_overflow = max(0.0, queue_backlog - 30.0)

        step_net_cash = (
            12.0 * approved_legit
            - 95.0 * unblocked_fraud
            - 18.0 * false_declines
            - sla_penalty * sla_overflow
        )
        cumulative_utility += step_net_cash

        raw_state = WorldState(
            step=t,
            timestamp=float(t),
            entities={
                "payment_gateway": Entity(
                    id="payment_gateway",
                    type="gateway",
                    attributes={"active_tau": tau},
                ),
                "call_center": Entity(
                    id="call_center",
                    type="support_queue",
                    attributes={"service_capacity": service_cap},
                ),
            },
            relationships=(
                Relationship(
                    source="payment_gateway",
                    target="call_center",
                    type="false_declines_trigger_calls",
                    attributes={"call_conversion_rate": 0.65},
                ),
            ),
            resources={
                "approved_tx": Resource(
                    id="approved_tx", current=total_approved, min_value=0.0, max_value=5000.0
                ),
                "declined_tx": Resource(
                    id="declined_tx", current=total_declined, min_value=0.0, max_value=5000.0
                ),
                "call_queue_backlog": Resource(
                    id="call_queue_backlog",
                    current=queue_backlog,
                    min_value=0.0,
                    max_value=5000.0,
                ),
                "enterprise_net_cash": Resource(
                    id="enterprise_net_cash",
                    current=cumulative_utility,
                    min_value=-1e7,
                    max_value=1e7,
                ),
                "latent_true_fraud_attempts": Resource(
                    id="latent_true_fraud_attempts",
                    current=true_fraud,
                    min_value=0.0,
                    max_value=5000.0,
                ),
            },
        )

        obs_o_t = obs_filter.observe(raw_state, actor_id="fraud_ops_planner")
        if "latent_true_fraud_attempts" in obs_o_t.resources:
            latent_masked_all = False

        belief = updater.update_belief(belief, obs_o_t, actor_id="fraud_ops_planner")

        drift_signals: list[DriftSignal] = []
        if tau <= 0.45 and sla_overflow > 0.0:
            drift_signals.append(
                DriftSignal(
                    kind=DriftKind.POLICY_INDUCED_SHIFT,
                    magnitude=float(sla_overflow / max(1.0, service_cap)),
                    affected_keys=("call_queue_backlog",),
                    details=(
                        f"Aggressive fraud threshold do(tau=0.40) induced downstream overload "
                        f"({DriftOrigin.POLICY_INDUCED_PERFORMATIVE})."
                    ),
                )
            )

        ontic_s_t, epistemic_p_t = separate_ontic_epistemic(
            raw_state,
            context_versions=(k_t.version,),
            context_fingerprints=(k_t.fingerprint,),
            drift_signals=drift_signals,
        )
        action_a_t = Action(
            id=f"set_tau_{t}",
            type="set_fraud_threshold",
            parameters={"tau": tau},
        )

        steps_out.append(
            ADR053FraudCascadeStepState(
                step=t,
                ontic_state_s_t=ontic_s_t,
                observation_o_t=obs_o_t,
                belief_state_b_t=belief,
                epistemic_state_p_t=epistemic_p_t,
                context_k_t=k_t,
                action_a_t=action_a_t,
            )
        )

    return cumulative_utility, max_conservation_drift, latent_masked_all, steps_out


def run_adr053_fraud_cascade_showcase(horizon: int = 12) -> ADR053FraudCascadeShowcaseResult:
    """Compare myopic do(tau=0.40) against balanced EWM do(tau=0.55) with full 6-part ADR-053 state."""
    u_myopic, drift_m, masked_m, _ = _simulate_adr053_policy(tau=0.40, horizon=horizon)
    u_balanced, drift_b, masked_b, traj_b = _simulate_adr053_policy(tau=0.55, horizon=horizon)
    best_u = max(u_myopic, u_balanced)

    return ADR053FraudCascadeShowcaseResult(
        steps_simulated=horizon,
        myopic_tau=0.40,
        balanced_tau=0.55,
        myopic_regret_usd=max(0.0, best_u - u_myopic),
        balanced_regret_usd=max(0.0, best_u - u_balanced),
        cohort_conservation_max_drift=max(drift_m, drift_b),
        latent_fraud_masked_in_o_t=(masked_m and masked_b),
        trajectory_balanced=tuple(traj_b),
    )


if __name__ == "__main__":  # pragma: no cover
    res = run_adr053_fraud_cascade_showcase()
    print(f"Steps simulated (6-part ADR-053): {res.steps_simulated}")
    print(f"Myopic do(tau=0.40) regret:       ${res.myopic_regret_usd:,.2f}")
    print(f"Balanced EWM do(tau=0.55) regret: ${res.balanced_regret_usd:,.2f}")
    print(f"Max cohort conservation drift:    {res.cohort_conservation_max_drift:.3e}")
    print(f"Latent fraud masked in O_t:       {res.latent_fraud_masked_in_o_t}")
