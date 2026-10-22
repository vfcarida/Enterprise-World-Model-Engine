"""Mechanism and configuration drift taxonomy and detection API.

Distinguishes six formal non-stationarity categories in enterprise world models:
1. COVARIATE_SHIFT: Input state/resource distribution shift (P(S_t) changes).
2. PARAMETER_DRIFT: Continuous parameter changes in K_t or T.
3. STRUCTURAL_MECHANISM_CHANGE: Workflow definitions, automations, or mechanism versions change.
4. RULE_CONSTRAINT_CHANGE: Policy versions, active rules, or SLA contracts change.
5. TOPOLOGY_CHANGE: Service dependencies, routing topology, or entity graph structure changes.
6. OBJECTIVE_CHANGE: Evaluation weights or target utility/loss definitions change.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.state import WorldState


class DriftKind(StrEnum):
    """Formal taxonomy of enterprise world-model drift and regime shifts."""

    COVARIATE_SHIFT = "covariate_shift"
    PARAMETER_DRIFT = "parameter_drift"
    STRUCTURAL_MECHANISM_CHANGE = "structural_mechanism_change"
    RULE_CONSTRAINT_CHANGE = "rule_constraint_change"
    TOPOLOGY_CHANGE = "topology_change"
    OBJECTIVE_CHANGE = "objective_change"


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
    covariate_threshold: float = 0.0,
    parameter_threshold: float = 0.0,
) -> list[DriftSignal]:
    """Detect and classify drift across context, state, mechanism, and objectives."""
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

    return signals
