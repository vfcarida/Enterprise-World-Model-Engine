"""Executable ADR-053 Deployment-Shift Walkthrough (Context K_t, Observation O_t, Belief B_t, Drift & Recovery)."""

from __future__ import annotations

from dataclasses import dataclass

from ewm_engine.core import (
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
    Resource,
    WorldState,
    separate_ontic_epistemic,
)


@dataclass(frozen=True)
class DeploymentShiftWalkthroughSummary:
    """Summary of the ADR-053 deployment-shift walkthrough execution."""

    pre_shift_mae: float
    peak_post_shift_mae: float
    recovered_post_shift_mae: float
    static_unrecovered_mae: float
    audit_discovery_delay_steps: int
    steps_until_recovery: int
    hidden_resource_masked_in_observation: bool
    ontic_fingerprint_invariant_to_epistemic: bool


def run_deployment_shift_walkthrough(
    horizon: int = 30,
    shift_step: int = 12,
    audit_delay_steps: int = 2,
) -> DeploymentShiftWalkthroughSummary:
    """Execute an end-to-end ADR-053 deployment-shift simulation and recovery walkthrough."""
    discovery = AuditMetadataContextDiscovery()
    k_nominal: ContextSnapshot = discovery.discover_context(
        {
            "context_id": "ctx_routing_v1",
            "version": "1.0.0",
            "environment_regime": "nominal",
            "parameters": {"routing_multiplier": 1.0, "base_inflow": 50.0},
            "feature_flags": {"aggressive_retry": False},
        }
    )
    k_shifted: ContextSnapshot = discovery.discover_context(
        {
            "context_id": "ctx_routing_v2",
            "version": "2.0.0",
            "parent_version": "1.0.0",
            "environment_regime": "post_deployment_shift",
            "parameters": {"routing_multiplier": 1.7, "base_inflow": 50.0},
            "feature_flags": {"aggressive_retry": True},
        }
    )

    obs_filter = ObservationFilter(
        ObservationSpec(
            actor_id="ops_controller",
            observable_entity_ids=frozenset({"gateway"}),
            observable_resource_ids=frozenset({"visible_queue"}),
        )
    )
    belief_updater = ExponentialSmoothingBeliefUpdater(
        prior_resource_estimates={"visible_queue": 50.0, "hidden_retry_buffer": 10.0},
        prior_uncertainty=1.0,
        uncertainty_growth_rate=0.1,
    )

    belief: BeliefState | None = None
    hybrid_errors: list[float] = []
    static_errors: list[float] = []
    hidden_masked_all = True
    ontic_invariant_all = True

    for step in range(horizon):
        true_k = k_nominal if step < shift_step else k_shifted
        discovered_k = k_nominal if step < (shift_step + audit_delay_steps) else k_shifted

        true_mult = float(true_k.parameters["routing_multiplier"])
        base_inflow = float(true_k.parameters["base_inflow"])
        true_queue = base_inflow * true_mult
        hidden_buffer = (true_mult - 1.0) * 35.0

        state = WorldState(
            step=step,
            timestamp=float(step),
            entities={"gateway": Entity(id="gateway", type="service")},
            resources={
                "visible_queue": Resource(
                    id="visible_queue", current=true_queue, min_value=0.0, max_value=500.0
                ),
                "hidden_retry_buffer": Resource(
                    id="hidden_retry_buffer",
                    current=hidden_buffer,
                    min_value=0.0,
                    max_value=500.0,
                ),
            },
            context={"context_version": discovered_k.version},
        )

        observed_o_t = obs_filter.observe(state, actor_id="ops_controller")
        if "hidden_retry_buffer" in observed_o_t.resources:
            hidden_masked_all = False

        belief = belief_updater.update_belief(belief, observed_o_t, actor_id="ops_controller")

        drift_signals: list[DriftSignal] = []
        if step >= shift_step:
            drift_signals.append(
                DriftSignal(
                    kind=DriftKind.STRUCTURAL_MECHANISM_CHANGE,
                    magnitude=abs(true_mult - 1.0),
                    affected_keys=("feature_flags.aggressive_retry", "parameters.routing_multiplier"),
                    baseline_version=k_nominal.version,
                    candidate_version=k_shifted.version,
                    details=f"Silent deployment flag aggressive_retry=True ({DriftOrigin.EXOGENOUS}).",
                )
            )

        ontic_s_t, epistemic_p_t = separate_ontic_epistemic(
            state,
            context_versions=(discovered_k.version,),
            context_fingerprints=(discovered_k.fingerprint,),
            drift_signals=drift_signals,
        )
        ontic_s_t_2, _ = separate_ontic_epistemic(
            state,
            context_versions=("9.9.9",),
            assumptions=("extra_audit_note",),
        )
        if ontic_s_t.fingerprint != ontic_s_t_2.fingerprint or not isinstance(
            epistemic_p_t, EpistemicState
        ):
            ontic_invariant_all = False

        # Static baseline ignores K_t discovery; context-aware hybrid adapts once audit delay elapses
        static_pred = base_inflow * float(k_nominal.parameters["routing_multiplier"])
        hybrid_pred = base_inflow * float(discovered_k.parameters["routing_multiplier"])

        static_errors.append(abs(true_queue - static_pred))
        hybrid_errors.append(abs(true_queue - hybrid_pred))

    pre_mae = sum(hybrid_errors[:shift_step]) / float(shift_step)
    peak_post = max(hybrid_errors[shift_step : shift_step + audit_delay_steps + 1])
    rec_post = sum(hybrid_errors[shift_step + audit_delay_steps :]) / float(
        len(hybrid_errors[shift_step + audit_delay_steps :])
    )
    static_post = sum(static_errors[shift_step + audit_delay_steps :]) / float(
        len(static_errors[shift_step + audit_delay_steps :])
    )

    return DeploymentShiftWalkthroughSummary(
        pre_shift_mae=pre_mae,
        peak_post_shift_mae=peak_post,
        recovered_post_shift_mae=rec_post,
        static_unrecovered_mae=static_post,
        audit_discovery_delay_steps=audit_delay_steps,
        steps_until_recovery=audit_delay_steps,
        hidden_resource_masked_in_observation=hidden_masked_all,
        ontic_fingerprint_invariant_to_epistemic=ontic_invariant_all,
    )


if __name__ == "__main__":
    summary = run_deployment_shift_walkthrough()
    print(f"Pre-shift MAE:              {summary.pre_shift_mae:.4f}")
    print(f"Peak post-shift MAE:        {summary.peak_post_shift_mae:.4f}")
    print(f"Recovered post-shift MAE:   {summary.recovered_post_shift_mae:.4f}")
    print(f"Static unrecovered MAE:     {summary.static_unrecovered_mae:.4f}")
    print(f"Steps until recovery:       {summary.steps_until_recovery}")
    print(f"Hidden resource masked:     {summary.hidden_resource_masked_in_observation}")
    print(f"Ontic fingerprint invariant:{summary.ontic_fingerprint_invariant_to_epistemic}")
