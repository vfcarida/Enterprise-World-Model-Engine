"""Mechanism, configuration, performative, endogenous, and adversarial drift taxonomy.

Distinguishes six exogenous non-stationarity categories and three model/policy-induced
validity-erosion categories in enterprise world models:

Exogenous Categories:
1. COVARIATE_SHIFT: Input state/resource distribution shift (P(S_t) changes).
2. PARAMETER_DRIFT: Continuous parameter changes in K_t or T.
3. STRUCTURAL_MECHANISM_CHANGE: Workflow definitions, automations, or mechanism versions change.
4. RULE_CONSTRAINT_CHANGE: Policy versions, active rules, or SLA contracts change.
5. TOPOLOGY_CHANGE: Service dependencies, routing topology, or entity graph structure changes.
6. OBJECTIVE_CHANGE: Evaluation weights or target utility/loss definitions change.

Model / Policy-Induced Categories:
7. POLICY_INDUCED_SHIFT: Performativity / Lucas critique — deploying a policy pi alters the
   environment's behavioral response distribution P(S_{t+1} | S_t, A_t; pi) != P(S_{t+1} | S_t, A_t; pi_0).
8. ENDOGENOUS_FEEDBACK_SHIFT: Endogeneity — actions A_t = pi(S_t, O_t, U_t) are taken in response
   to state and latent operator signals, creating closed-loop feedback that invalidates static
   observational associations.
9. ADVERSARIAL_ADAPTATION_SHIFT: Strategic adversary adaptation — intelligent actors (e.g. fraud
   adversaries) adapt their behavior in response to the deployed policy or threshold.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.state import WorldState


class DriftOrigin(StrEnum):
    """Causal origin of a detected distribution or mechanism shift."""

    EXOGENOUS = "exogenous"
    POLICY_INDUCED_PERFORMATIVE = "policy_induced_performative"
    ENDOGENOUS_CLOSED_LOOP = "endogenous_closed_loop"
    ADVERSARIAL_STRATEGIC = "adversarial_strategic"


class DriftKind(StrEnum):
    """Formal taxonomy of enterprise world-model drift, performativity, and regime shifts."""

    # Exogenous shifts
    COVARIATE_SHIFT = "covariate_shift"
    PARAMETER_DRIFT = "parameter_drift"
    STRUCTURAL_MECHANISM_CHANGE = "structural_mechanism_change"
    RULE_CONSTRAINT_CHANGE = "rule_constraint_change"
    TOPOLOGY_CHANGE = "topology_change"
    OBJECTIVE_CHANGE = "objective_change"

    # Model / policy-induced shifts
    POLICY_INDUCED_SHIFT = "policy_induced_shift"
    ENDOGENOUS_FEEDBACK_SHIFT = "endogenous_feedback_shift"
    ADVERSARIAL_ADAPTATION_SHIFT = "adversarial_adaptation_shift"


EXOGENOUS_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
    {
        DriftKind.COVARIATE_SHIFT,
        DriftKind.PARAMETER_DRIFT,
        DriftKind.STRUCTURAL_MECHANISM_CHANGE,
        DriftKind.RULE_CONSTRAINT_CHANGE,
        DriftKind.TOPOLOGY_CHANGE,
        DriftKind.OBJECTIVE_CHANGE,
    }
)

MODEL_INDUCED_DRIFT_KINDS: frozenset[DriftKind] = frozenset(
    {
        DriftKind.POLICY_INDUCED_SHIFT,
        DriftKind.ENDOGENOUS_FEEDBACK_SHIFT,
        DriftKind.ADVERSARIAL_ADAPTATION_SHIFT,
    }
)

DRIFT_KIND_TO_ORIGIN: dict[DriftKind, DriftOrigin] = {
    DriftKind.COVARIATE_SHIFT: DriftOrigin.EXOGENOUS,
    DriftKind.PARAMETER_DRIFT: DriftOrigin.EXOGENOUS,
    DriftKind.STRUCTURAL_MECHANISM_CHANGE: DriftOrigin.EXOGENOUS,
    DriftKind.RULE_CONSTRAINT_CHANGE: DriftOrigin.EXOGENOUS,
    DriftKind.TOPOLOGY_CHANGE: DriftOrigin.EXOGENOUS,
    DriftKind.OBJECTIVE_CHANGE: DriftOrigin.EXOGENOUS,
    DriftKind.POLICY_INDUCED_SHIFT: DriftOrigin.POLICY_INDUCED_PERFORMATIVE,
    DriftKind.ENDOGENOUS_FEEDBACK_SHIFT: DriftOrigin.ENDOGENOUS_CLOSED_LOOP,
    DriftKind.ADVERSARIAL_ADAPTATION_SHIFT: DriftOrigin.ADVERSARIAL_STRATEGIC,
}


class DriftSignal(BaseModel):
    """Structured diagnostic signal representing a detected drift category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DriftKind
    magnitude: float = Field(ge=0.0)
    threshold: float = Field(default=0.0, ge=0.0)
    affected_keys: tuple[str, ...] = Field(default_factory=tuple)
    baseline_version: str | None = None
    candidate_version: str | None = None
    details: str = ""

    @property
    def is_drifted(self) -> bool:
        """Return True if the drift magnitude strictly exceeds the threshold."""
        return self.magnitude > self.threshold

    @property
    def drift_origin(self) -> DriftOrigin:
        """Return the causal origin classification of this drift signal."""
        return DRIFT_KIND_TO_ORIGIN[self.kind]

    @property
    def is_model_induced(self) -> bool:
        """Return True if this shift is induced by the model's own policy, feedback, or adversary response."""
        return self.kind in MODEL_INDUCED_DRIFT_KINDS

    @property
    def is_exogenous(self) -> bool:
        """Return True if this shift originates exogenously to the deployed decision policy."""
        return self.kind in EXOGENOUS_DRIFT_KINDS


class PerformativeCouplingSpec(BaseModel):
    """Declares how deploying a decision policy alters environment behavioral parameters.

    Operationalizes Performativity (Perdomo et al., 2020) and the Lucas (1976) Critique:
    when a policy parameter ``policy_key`` moves away from ``baseline_policy_value``,
    downstream behavioral elasticities or mechanism parameters shift by
    ``sensitivity_elasticities[param] * |deployed - baseline|``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    coupling_id: str = Field(description="Unique identifier of this performative coupling.")
    policy_key: str = Field(
        description="Policy or action parameter key whose deployment alters the environment (e.g. 'auth_threshold')."
    )
    baseline_policy_value: float = Field(
        description="Historical policy value pi_0 under which behavioral parameters were calibrated."
    )
    affected_behavioral_parameters: tuple[str, ...] = Field(
        description="Environment behavioral parameters altered when policy_key deviates from baseline."
    )
    sensitivity_elasticities: dict[str, float] = Field(
        default_factory=dict,
        description="Rate of behavioral parameter shift per unit policy change.",
    )
    assumed_static_mechanisms: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Mechanism IDs in the simulator/planner that assume these behavioral parameters remain fixed.",
    )
    invalidation_threshold: float = Field(
        default=0.02,
        ge=0.0,
        description="Performative parameter shift magnitude above which static plans are flagged as self-invalidating.",
    )
    coupling_kind: Literal["performative_customer_response", "adversarial_evasion"] = Field(
        default="performative_customer_response",
        description="Whether the response arises from non-adversarial agent adaptation or strategic evasion.",
    )


class EndogenousDecisionFeedbackSpec(BaseModel):
    """Declares closed-loop endogeneity where decisions are taken in response to state.

    When actions ``action_key`` are selected as a function of observed/latent state
    ``conditioning_state_keys`` (and optional ``unobserved_confounder_keys``), static
    observational regressions P(Y | A, S) suffer simultaneity and confounding bias.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feedback_id: str = Field(description="Unique identifier for this endogenous feedback loop.")
    action_key: str = Field(
        description="Endogenous decision/action variable (e.g. 'staffed_agents')."
    )
    conditioning_state_keys: tuple[str, ...] = Field(
        description="State variables that drive endogenous action changes (e.g. ('call_backlog', 'wait_seconds'))."
    )
    unobserved_confounder_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Latent operational signals that simultaneously affect action selection and outcomes.",
    )
    feedback_gain: float = Field(
        default=1.0,
        ge=0.0,
        description="Strength of state-to-action closed-loop responsiveness.",
    )
    invalidates_static_association_mechanisms: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Observational/statistical mechanisms invalidated by this closed-loop feedback.",
    )


class PlanInvalidationAudit(BaseModel):
    """Audit report flagging whether a simulation or plan invalidates its own assumptions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_self_invalidating: bool = Field(
        description="True if deploying the plan induces performative, endogenous, or adversarial shifts that invalidate assumed relationships."
    )
    performative_shift_magnitude: float = Field(
        default=0.0,
        ge=0.0,
        description="Total L1 magnitude of policy-induced behavioral parameter shift.",
    )
    endogenous_confounding_detected: bool = Field(
        default=False,
        description="True if closed-loop state-action feedback invalidates static observational relationships.",
    )
    adversarial_shift_detected: bool = Field(
        default=False,
        description="True if strategic adversarial adaptation is triggered by the plan.",
    )
    invalidated_mechanism_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Mechanism IDs whose static invariance assumptions are violated by the plan.",
    )
    invalidated_assumptions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Human-readable explanations of which assumptions the plan invalidates.",
    )
    drift_signals: tuple[DriftSignal, ...] = Field(
        default_factory=tuple,
        description="Structured model-induced DriftSignals emitted during the audit.",
    )
    recommendation: str = Field(
        default="Plan remains within declared non-performative validity bounds.",
        description="Actionable governance recommendation.",
    )


def _changed_rule_keys(
    baseline_rules: Mapping[str, object], candidate_rules: Mapping[str, object]
) -> list[str]:
    changed: list[str] = []
    for rkey in sorted(set(baseline_rules.keys()) | set(candidate_rules.keys())):
        if baseline_rules.get(rkey) != candidate_rules.get(rkey):
            changed.append(f"active_rules.{rkey}")
    return changed


def detect_world_drift(
    *,
    baseline_context: ContextSnapshot | None = None,
    candidate_context: ContextSnapshot | None = None,
    baseline_state: WorldState | None = None,
    candidate_state: WorldState | None = None,
    baseline_mechanism_version: str | None = None,
    candidate_mechanism_version: str | None = None,
    baseline_objectives: Mapping[str, float] | None = None,
    candidate_objectives: Mapping[str, float] | None = None,
    baseline_behavioral_parameters: Mapping[str, float] | None = None,
    realized_behavioral_parameters: Mapping[str, float] | None = None,
    performative_couplings: Sequence[PerformativeCouplingSpec] | None = None,
    deployed_policy_parameters: Mapping[str, float] | None = None,
    endogenous_feedback_specs: Sequence[EndogenousDecisionFeedbackSpec] | None = None,
    adversarial_evasion_magnitude: float | None = None,
    adversarial_affected_keys: Sequence[str] = (),
    covariate_threshold: float = 0.0,
    parameter_threshold: float = 0.0,
    performative_threshold: float = 0.0,
) -> list[DriftSignal]:
    """Detect and classify exogenous, performative, endogenous, and adversarial drift."""
    signals: list[DriftSignal] = []

    b_ver = baseline_context.version if baseline_context is not None else baseline_mechanism_version
    c_ver = (
        candidate_context.version if candidate_context is not None else candidate_mechanism_version
    )

    # 1. Covariate / State Distribution Shift
    if baseline_state is not None and candidate_state is not None:
        all_rids = sorted(
            set(baseline_state.resources.keys()) | set(candidate_state.resources.keys())
        )
        sq_diff = 0.0
        changed_rids: list[str] = []
        for rid in all_rids:
            b_res = baseline_state.resources.get(rid)
            c_res = candidate_state.resources.get(rid)
            b_val = b_res.current if b_res is not None else 0.0
            c_val = c_res.current if c_res is not None else 0.0
            b_max = (
                b_res.max_value if b_res is not None and not math.isinf(b_res.max_value) else 1.0
            )
            c_max = (
                c_res.max_value if c_res is not None and not math.isinf(c_res.max_value) else 1.0
            )
            scale = max(b_max, c_max, 1.0)
            norm_delta = abs(c_val - b_val) / scale
            if norm_delta > 1e-12:
                changed_rids.append(rid)
                sq_diff += norm_delta * norm_delta
        mag = math.sqrt(sq_diff)
        if mag > covariate_threshold:
            signals.append(
                DriftSignal(
                    kind=DriftKind.COVARIATE_SHIFT,
                    magnitude=mag,
                    threshold=covariate_threshold,
                    affected_keys=tuple(changed_rids),
                    baseline_version=b_ver,
                    candidate_version=c_ver,
                    details=f"Resource state shift across {len(changed_rids)} resources.",
                )
            )

    # 2. Context-driven Drift Categories
    if baseline_context is not None and candidate_context is not None:
        # 2a. Parameter Drift
        param_keys = sorted(
            set(baseline_context.parameters.keys()) | set(candidate_context.parameters.keys())
        )
        changed_params: list[str] = []
        param_mag = 0.0
        for pkey in param_keys:
            b_p = baseline_context.parameters.get(pkey)
            c_p = candidate_context.parameters.get(pkey)
            if b_p != c_p:
                changed_params.append(f"parameters.{pkey}")
                if isinstance(b_p, (int, float)) and isinstance(c_p, (int, float)):
                    param_mag += abs(float(c_p) - float(b_p))
                else:
                    param_mag += 1.0
        if changed_params and param_mag > parameter_threshold:
            signals.append(
                DriftSignal(
                    kind=DriftKind.PARAMETER_DRIFT,
                    magnitude=param_mag,
                    threshold=parameter_threshold,
                    affected_keys=tuple(changed_params),
                    baseline_version=baseline_context.version,
                    candidate_version=candidate_context.version,
                    details="Context scalar parameters drifted.",
                )
            )

        # 2b. Structural Mechanism Change (workflows, automations, feature flags, mechanism version)
        struct_keys: list[str] = []
        for wid in sorted(
            set(baseline_context.workflow_definitions) | set(candidate_context.workflow_definitions)
        ):
            if baseline_context.workflow_definitions.get(
                wid
            ) != candidate_context.workflow_definitions.get(wid):
                struct_keys.append(f"workflow_definitions.{wid}")
        for aid in sorted(
            set(baseline_context.enabled_automations) | set(candidate_context.enabled_automations)
        ):
            if baseline_context.enabled_automations.get(
                aid
            ) != candidate_context.enabled_automations.get(aid):
                struct_keys.append(f"enabled_automations.{aid}")
        for fid in sorted(
            set(baseline_context.feature_flags) | set(candidate_context.feature_flags)
        ):
            if baseline_context.feature_flags.get(fid) != candidate_context.feature_flags.get(fid):
                struct_keys.append(f"feature_flags.{fid}")
        if (
            baseline_mechanism_version is not None
            and candidate_mechanism_version is not None
            and baseline_mechanism_version != candidate_mechanism_version
        ):
            struct_keys.append("mechanism_version")

        if struct_keys:
            signals.append(
                DriftSignal(
                    kind=DriftKind.STRUCTURAL_MECHANISM_CHANGE,
                    magnitude=float(len(struct_keys)),
                    threshold=0.0,
                    affected_keys=tuple(struct_keys),
                    baseline_version=baseline_context.version,
                    candidate_version=candidate_context.version,
                    details="Workflow definitions, automations, feature flags, or mechanism versions changed.",
                )
            )

        # 2c. Rule / Constraint Change (policy_versions, active_contracts, state.active_rules)
        rule_keys: list[str] = []
        for pid in sorted(
            set(baseline_context.policy_versions) | set(candidate_context.policy_versions)
        ):
            if baseline_context.policy_versions.get(pid) != candidate_context.policy_versions.get(
                pid
            ):
                rule_keys.append(f"policy_versions.{pid}")
        for cid in sorted(
            set(baseline_context.active_contracts) | set(candidate_context.active_contracts)
        ):
            if baseline_context.active_contracts.get(cid) != candidate_context.active_contracts.get(
                cid
            ):
                rule_keys.append(f"active_contracts.{cid}")
        if baseline_state is not None and candidate_state is not None:
            rule_keys.extend(
                _changed_rule_keys(baseline_state.active_rules, candidate_state.active_rules)
            )

        if rule_keys:
            signals.append(
                DriftSignal(
                    kind=DriftKind.RULE_CONSTRAINT_CHANGE,
                    magnitude=float(len(rule_keys)),
                    threshold=0.0,
                    affected_keys=tuple(rule_keys),
                    baseline_version=baseline_context.version,
                    candidate_version=candidate_context.version,
                    details="Policy versions, SLA contracts, or active rules changed.",
                )
            )

        # 2d. Topology Change (topology_config, external_services, state.relationships)
        topo_keys: list[str] = []
        for tid in sorted(
            set(baseline_context.topology_config) | set(candidate_context.topology_config)
        ):
            if baseline_context.topology_config.get(tid) != candidate_context.topology_config.get(
                tid
            ):
                topo_keys.append(f"topology_config.{tid}")
        for sid in sorted(
            set(baseline_context.external_services) | set(candidate_context.external_services)
        ):
            if baseline_context.external_services.get(
                sid
            ) != candidate_context.external_services.get(sid):
                topo_keys.append(f"external_services.{sid}")
        if baseline_state is not None and candidate_state is not None:
            b_edges = {(r.source, r.target, r.type) for r in baseline_state.relationships}
            c_edges = {(r.source, r.target, r.type) for r in candidate_state.relationships}
            if b_edges != c_edges:
                topo_keys.append("state.relationships")

        if topo_keys:
            signals.append(
                DriftSignal(
                    kind=DriftKind.TOPOLOGY_CHANGE,
                    magnitude=float(len(topo_keys)),
                    threshold=0.0,
                    affected_keys=tuple(topo_keys),
                    baseline_version=baseline_context.version,
                    candidate_version=candidate_context.version,
                    details="Topology configuration, external services, or entity relationships changed.",
                )
            )
    else:
        # Check mechanism/rule/topology changes even when ContextSnapshot is not provided
        if (
            baseline_mechanism_version is not None
            and candidate_mechanism_version is not None
            and baseline_mechanism_version != candidate_mechanism_version
        ):
            signals.append(
                DriftSignal(
                    kind=DriftKind.STRUCTURAL_MECHANISM_CHANGE,
                    magnitude=1.0,
                    threshold=0.0,
                    affected_keys=("mechanism_version",),
                    baseline_version=baseline_mechanism_version,
                    candidate_version=candidate_mechanism_version,
                    details="Transition mechanism version changed.",
                )
            )
        if baseline_state is not None and candidate_state is not None:
            diff_rules = _changed_rule_keys(
                baseline_state.active_rules, candidate_state.active_rules
            )
            if diff_rules:
                signals.append(
                    DriftSignal(
                        kind=DriftKind.RULE_CONSTRAINT_CHANGE,
                        magnitude=float(len(diff_rules)),
                        threshold=0.0,
                        affected_keys=tuple(diff_rules),
                        details="Active state rules changed.",
                    )
                )
            b_edges = {(r.source, r.target, r.type) for r in baseline_state.relationships}
            c_edges = {(r.source, r.target, r.type) for r in candidate_state.relationships}
            if b_edges != c_edges:
                signals.append(
                    DriftSignal(
                        kind=DriftKind.TOPOLOGY_CHANGE,
                        magnitude=float(len(b_edges ^ c_edges)),
                        threshold=0.0,
                        affected_keys=("state.relationships",),
                        details="Entity relationship graph topology changed.",
                    )
                )

    # 3. Objective Change
    if baseline_objectives is not None and candidate_objectives is not None:
        obj_keys = sorted(set(baseline_objectives.keys()) | set(candidate_objectives.keys()))
        changed_obj: list[str] = []
        obj_mag = 0.0
        for okey in obj_keys:
            b_w = baseline_objectives.get(okey, 0.0)
            c_w = candidate_objectives.get(okey, 0.0)
            if (
                abs(c_w - b_w) > 1e-12
                or (okey not in baseline_objectives)
                or (okey not in candidate_objectives)
            ):
                changed_obj.append(okey)
                obj_mag += abs(c_w - b_w) if abs(c_w - b_w) > 0.0 else 1.0
        if changed_obj:
            signals.append(
                DriftSignal(
                    kind=DriftKind.OBJECTIVE_CHANGE,
                    magnitude=obj_mag,
                    threshold=0.0,
                    affected_keys=tuple(changed_obj),
                    baseline_version=b_ver,
                    candidate_version=c_ver,
                    details="Evaluation objective weights or targets changed.",
                )
            )

    # 4. Performative / Policy-Induced Behavioral Distribution Shift
    perf_keys: list[str] = []
    perf_mag = 0.0
    if baseline_behavioral_parameters is not None and realized_behavioral_parameters is not None:
        for bkey in sorted(
            set(baseline_behavioral_parameters.keys()) | set(realized_behavioral_parameters.keys())
        ):
            b_val = float(baseline_behavioral_parameters.get(bkey, 0.0))
            r_val = float(realized_behavioral_parameters.get(bkey, 0.0))
            delta = abs(r_val - b_val)
            if delta > 1e-12:
                perf_keys.append(f"behavioral.{bkey}")
                perf_mag += delta

    if performative_couplings and deployed_policy_parameters is not None:
        for coupling in performative_couplings:
            if coupling.policy_key in deployed_policy_parameters:
                pol_delta = abs(
                    float(deployed_policy_parameters[coupling.policy_key])
                    - coupling.baseline_policy_value
                )
                if pol_delta > 1e-12:
                    for bparam in coupling.affected_behavioral_parameters:
                        elast = abs(coupling.sensitivity_elasticities.get(bparam, 1.0))
                        induced = pol_delta * elast
                        if induced > 1e-12:
                            key_tag = f"performative.{coupling.policy_key}->{bparam}"
                            if key_tag not in perf_keys:
                                perf_keys.append(key_tag)
                            if coupling.coupling_kind == "performative_customer_response":
                                perf_mag += induced

    if perf_keys and perf_mag > performative_threshold:
        signals.append(
            DriftSignal(
                kind=DriftKind.POLICY_INDUCED_SHIFT,
                magnitude=perf_mag,
                threshold=performative_threshold,
                affected_keys=tuple(perf_keys),
                baseline_version=b_ver,
                candidate_version=c_ver,
                details=(
                    "Policy-induced performative shift: deployed policy altered environmental "
                    "behavioral elasticities relative to historical calibration."
                ),
            )
        )

    # 5. Endogenous Closed-Loop Decision Feedback Shift
    if endogenous_feedback_specs:
        endog_keys: list[str] = []
        endog_mag = 0.0
        for fb in endogenous_feedback_specs:
            if fb.feedback_gain > 0.0:
                endog_keys.append(
                    f"endogenous.{fb.action_key}<-({','.join(fb.conditioning_state_keys)})"
                )
                endog_mag += float(fb.feedback_gain) * (
                    1.0 + 0.5 * float(len(fb.unobserved_confounder_keys))
                )
        if endog_keys and endog_mag > 0.0:
            signals.append(
                DriftSignal(
                    kind=DriftKind.ENDOGENOUS_FEEDBACK_SHIFT,
                    magnitude=endog_mag,
                    threshold=0.0,
                    affected_keys=tuple(endog_keys),
                    baseline_version=b_ver,
                    candidate_version=c_ver,
                    details=(
                        "Endogenous state-to-action feedback loop active: decisions respond to "
                        "state/confounders, invalidating static observational correlations."
                    ),
                )
            )

    # 6. Strategic Adversarial Adaptation Shift
    adv_mag = float(adversarial_evasion_magnitude or 0.0)
    adv_keys: list[str] = list(adversarial_affected_keys)
    if performative_couplings and deployed_policy_parameters is not None:
        for coupling in performative_couplings:
            if (
                coupling.coupling_kind == "adversarial_evasion"
                and coupling.policy_key in deployed_policy_parameters
            ):
                pol_delta = abs(
                    float(deployed_policy_parameters[coupling.policy_key])
                    - coupling.baseline_policy_value
                )
                for bparam in coupling.affected_behavioral_parameters:
                    elast = abs(coupling.sensitivity_elasticities.get(bparam, 1.0))
                    induced_adv = pol_delta * elast
                    if induced_adv > 1e-12:
                        adv_mag += induced_adv
                        tag = f"adversarial.{coupling.policy_key}->{bparam}"
                        if tag not in adv_keys:
                            adv_keys.append(tag)

    if adv_mag > 0.0:
        if not adv_keys:
            adv_keys = ["adversarial.strategic_response"]
        signals.append(
            DriftSignal(
                kind=DriftKind.ADVERSARIAL_ADAPTATION_SHIFT,
                magnitude=adv_mag,
                threshold=0.0,
                affected_keys=tuple(adv_keys),
                baseline_version=b_ver,
                candidate_version=c_ver,
                details=(
                    "Strategic adversary adaptation shift: intelligent adversary altered attack "
                    "distribution in response to deployed policy."
                ),
            )
        )

    return signals


def check_plan_performative_validity(
    *,
    deployed_policy_parameters: Mapping[str, float],
    performative_couplings: Sequence[PerformativeCouplingSpec] = (),
    endogenous_feedback_specs: Sequence[EndogenousDecisionFeedbackSpec] = (),
    active_mechanism_ids: Sequence[str] = (),
    baseline_behavioral_parameters: Mapping[str, float] | None = None,
    realized_behavioral_parameters: Mapping[str, float] | None = None,
    adversarial_evasion_magnitude: float | None = None,
) -> PlanInvalidationAudit:
    """Evaluate whether a candidate plan or simulation invalidates its own assumed relationships.

    Flags when:
    1. A deployed policy parameter alters environmental behavioral parameters that an active
       simulator/planner mechanism assumes are static (Performativity / Lucas critique).
    2. Closed-loop endogenous decision feedback invalidates static observational mechanisms.
    3. Strategic adversarial adaptation erodes the policy's assumed recall or effectiveness.
    """
    active_set = set(active_mechanism_ids)
    invalidated_mechs: list[str] = []
    invalidated_assumptions: list[str] = []
    total_perf_mag = 0.0
    adv_detected = False

    for coupling in performative_couplings:
        if coupling.policy_key not in deployed_policy_parameters:
            continue
        deployed_val = float(deployed_policy_parameters[coupling.policy_key])
        pol_delta = abs(deployed_val - coupling.baseline_policy_value)
        if pol_delta <= 1e-12:
            continue

        coupling_shift = 0.0
        for bparam in coupling.affected_behavioral_parameters:
            elast = abs(coupling.sensitivity_elasticities.get(bparam, 1.0))
            coupling_shift += pol_delta * elast

        if coupling.coupling_kind == "adversarial_evasion" and coupling_shift > 0.0:
            adv_detected = True

        total_perf_mag += coupling_shift
        if coupling_shift > coupling.invalidation_threshold:
            matched_mechs = [
                m for m in coupling.assumed_static_mechanisms if (not active_set or m in active_set)
            ]
            for m in matched_mechs:
                if m not in invalidated_mechs:
                    invalidated_mechs.append(m)
            invalidated_assumptions.append(
                f"Coupling '{coupling.coupling_id}': policy '{coupling.policy_key}' changed from "
                f"{coupling.baseline_policy_value:.3f} to {deployed_val:.3f} (|delta|={pol_delta:.3f}), "
                f"shifting {list(coupling.affected_behavioral_parameters)} by {coupling_shift:.4f} "
                f"(> threshold {coupling.invalidation_threshold:.4f}) and invalidating static "
                f"invariance in {matched_mechs or list(coupling.assumed_static_mechanisms)}."
            )

    if baseline_behavioral_parameters is not None and realized_behavioral_parameters is not None:
        emp_shift = sum(
            abs(
                float(realized_behavioral_parameters.get(k, 0.0))
                - float(baseline_behavioral_parameters.get(k, 0.0))
            )
            for k in set(baseline_behavioral_parameters) | set(realized_behavioral_parameters)
        )
        total_perf_mag = max(total_perf_mag, emp_shift)

    endog_detected = False
    for fb in endogenous_feedback_specs:
        if fb.feedback_gain > 0.0:
            endog_detected = True
            matched_mechs = [
                m
                for m in fb.invalidates_static_association_mechanisms
                if (not active_set or m in active_set)
            ]
            for m in matched_mechs:
                if m not in invalidated_mechs:
                    invalidated_mechs.append(m)
            invalidated_assumptions.append(
                f"Endogenous feedback '{fb.feedback_id}': action '{fb.action_key}' responds to "
                f"state {list(fb.conditioning_state_keys)} (confounders={list(fb.unobserved_confounder_keys)}), "
                f"invalidating static associations in {matched_mechs or list(fb.invalidates_static_association_mechanisms)}."
            )

    if adversarial_evasion_magnitude is not None and adversarial_evasion_magnitude > 0.0:
        adv_detected = True
        invalidated_assumptions.append(
            f"Adversarial adaptation shift magnitude {adversarial_evasion_magnitude:.4f} > 0: "
            "static attack distribution assumption violated."
        )

    signals = detect_world_drift(
        baseline_behavioral_parameters=baseline_behavioral_parameters,
        realized_behavioral_parameters=realized_behavioral_parameters,
        performative_couplings=performative_couplings,
        deployed_policy_parameters=deployed_policy_parameters,
        endogenous_feedback_specs=endogenous_feedback_specs,
        adversarial_evasion_magnitude=adversarial_evasion_magnitude,
    )

    is_self_invalidating = len(invalidated_assumptions) > 0 or endog_detected or adv_detected
    if is_self_invalidating:
        rec = (
            "DO NOT rely on static-elasticity rollout predictions: re-estimate or condition on "
            "performative behavioral response and closed-loop intervention dynamics, or abstain."
        )
    else:
        rec = "Plan remains within declared non-performative validity bounds."

    return PlanInvalidationAudit(
        is_self_invalidating=is_self_invalidating,
        performative_shift_magnitude=total_perf_mag,
        endogenous_confounding_detected=endog_detected,
        adversarial_shift_detected=adv_detected,
        invalidated_mechanism_ids=tuple(invalidated_mechs),
        invalidated_assumptions=tuple(invalidated_assumptions),
        drift_signals=tuple(signals),
        recommendation=rec,
    )
