"""Verifiable mechanistic explainability and anti-narrative fidelity guards.

Core Doctrine:
1. Explanations in EWM Engine are a **verifiable deterministic function** of the
   mechanisms (`MechanismProvenance`), runtime context (`ContextSnapshot` version and
   fingerprint), systemic trace (`SystemicTrace`), and state deltas that **actually
   participated** in producing a prediction, trajectory, or policy recommendation.
2. **Anti-Narrative Guardrail**: Free-text or LLM-generated prose is *never* accepted
   as standalone evidence that a valid mechanism was used (`is_standalone_narrative_evidence=False`).
3. If an LLM or reporting layer renders an explanation in natural language, its
   fidelity against the recorded `VerifiableMechanisticExplanation` must be checkable
   via `verify_narrative_explanation_fidelity(...)`, which deterministically rejects:
   - fabricated mechanism attributions not present in the trace,
   - unrecorded context versions or context keys,
   - unrecorded resource modifications or trace edges, and
   - rhetorical causal overclaiming beyond the trace's fail-closed `EvidenceLevel`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.provenance.evidence import (
    DependencyEvidenceNature,
    EvidenceLevel,
    evidence_level_to_nature,
)
from ewm_engine.provenance.trace import SystemicTrace

if TYPE_CHECKING:
    from ewm_engine.core.context import ContextSnapshot
    from ewm_engine.core.epistemic import EpistemicState
    from ewm_engine.core.state import WorldState
    from ewm_engine.dynamics.base import TransitionResult
    from ewm_engine.simulation.trajectory import Trajectory

_EVIDENCE_RANK_ORDER: dict[EvidenceLevel, int] = {
    EvidenceLevel.CONFOUNDED: 0,
    EvidenceLevel.UNIDENTIFIED: 1,
    EvidenceLevel.HYPOTHETICAL: 2,
    EvidenceLevel.ASSUMED: 2,
    EvidenceLevel.DIRECTIONAL_UNCERTAIN: 3,
    EvidenceLevel.PROVISIONAL: 4,
    EvidenceLevel.PREDICTIVE: 5,
    EvidenceLevel.CALIBRATED: 6,
    EvidenceLevel.QUASI_CAUSAL: 7,
    EvidenceLevel.INTERVENTIONAL: 8,
    EvidenceLevel.ASSERTED: 9,
    EvidenceLevel.STRUCTURAL: 10,
    EvidenceLevel.COUNTERFACTUAL: 11,
}

__all__ = [
    "CandidateNarrativeExplanation",
    "ExplanationFidelityReport",
    "MechanismAttributionRecord",
    "NarrativeFabricationError",
    "VerifiableMechanisticExplanation",
    "build_verifiable_explanation",
    "verify_narrative_explanation_fidelity",
]

_CAUSAL_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bproved\s+causal\b", re.IGNORECASE),
    re.compile(r"\bproven\s+causal\b", re.IGNORECASE),
    re.compile(r"\bguaranteed\s+counterfactual\b", re.IGNORECASE),
    re.compile(r"\bdo-calculus\s+verified\b", re.IGNORECASE),
    re.compile(r"\broot\s+cause\s+proved\b", re.IGNORECASE),
    re.compile(r"\bcaused\s+by\b", re.IGNORECASE),
)

_MECH_TAG_PATTERN = re.compile(r"\bmech:([A-Za-z0-9_.\-]+)")
_CTX_VER_TAG_PATTERN = re.compile(r"\bctx_version:([A-Za-z0-9_.\-]+)")
_CTX_KEY_TAG_PATTERN = re.compile(r"\bctx_key:([A-Za-z0-9_.\-]+)")
_RES_TAG_PATTERN = re.compile(r"\bres:([A-Za-z0-9_.\-]+)")
_EDGE_TAG_PATTERN = re.compile(r"\bedge:([A-Za-z0-9_.\-]+)->([A-Za-z0-9_.\-]+)")


class NarrativeFabricationError(ValueError):
    """Raised when a narrative or LLM explanation asserts mechanisms or claims not in the trace."""

    def __init__(self, message: str, violations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.violations = tuple(violations)


class MechanismAttributionRecord(BaseModel):
    """Verifiable record of a concrete mechanism that participated in a computation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism_id: str = Field(description="Concrete identifier of the executed mechanism.")
    mechanism_version: str = Field(
        default="1.0.0", description="Semantic version of the executed mechanism."
    )
    mechanism_kind: str = Field(
        default="known_deterministic",
        description="MechanismKind taxonomy classification.",
    )
    evidence_nature: DependencyEvidenceNature = Field(
        default=DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        description="Supporting evidence nature of this mechanism.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Fail-closed epistemic standing of this mechanism.",
    )
    context_version: str | None = Field(
        default=None,
        description="Runtime ContextSnapshot version active during mechanism execution.",
    )
    context_fingerprint: str | None = Field(
        default=None,
        description="Canonical SHA-256 fingerprint of the active ContextSnapshot.",
    )
    required_context_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Declared context keys actually read by this mechanism.",
    )
    participated_steps: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Simulation step indices at which this mechanism executed.",
    )
    trace_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="SystemicTrace node IDs produced or touched by this mechanism.",
    )
    trace_edges: tuple[tuple[str, str, str], ...] = Field(
        default_factory=tuple,
        description="Directed trace edges (source, target, relation) recorded during execution.",
    )
    state_resource_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Non-zero resource deltas enacted by this mechanism.",
    )
    residual_feasibility_action: str | None = Field(
        default=None,
        description="Feasible-projection policy action if a learned residual was projected/rejected.",
    )


class VerifiableMechanisticExplanation(BaseModel):
    """Deterministic explanation derived exclusively from recorded mechanisms, context, and trace.

    Epistemic Invariant:
        ``is_standalone_narrative_evidence`` is permanently ``False``: an explanation
        is a verifiable projection of the recorded computation trace, never a free-standing
        rhetorical justification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    explanation_id: str = Field(description="Unique identifier of this mechanistic explanation.")
    is_standalone_narrative_evidence: bool = Field(
        default=False,
        description="Always False: prose/narrative is never accepted as standalone mechanistic proof.",
    )
    source_epistemic_fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint of the underlying EpistemicState or trace."
    )
    effective_evidence_level: EvidenceLevel = Field(
        description="Fail-closed composite EvidenceLevel of the explained computation."
    )
    effective_evidence_nature: DependencyEvidenceNature = Field(
        description="Canonical DependencyEvidenceNature corresponding to effective_evidence_level."
    )
    participating_mechanisms: tuple[MechanismAttributionRecord, ...] = Field(
        default_factory=tuple,
        description="Ordered tuple of mechanisms that actually executed during the computation.",
    )
    active_context_versions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="ContextSnapshot versions active during the computation.",
    )
    active_context_fingerprints: tuple[str, ...] = Field(
        default_factory=tuple,
        description="ContextSnapshot SHA-256 fingerprints active during the computation.",
    )
    active_context_keys_used: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Union of required_context_keys read by participating mechanisms.",
    )
    trace_node_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="SystemicTrace node IDs recorded during the computation.",
    )
    trace_edges: tuple[tuple[str, str, str], ...] = Field(
        default_factory=tuple,
        description="SystemicTrace directed edges (source, target, relation) recorded.",
    )
    modified_resources: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Resource IDs whose values changed during the explained computation.",
    )
    drift_signals: tuple[Any, ...] = Field(
        default_factory=tuple,
        description="DriftSignals recorded in the epistemic state.",
    )
    explanation_fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint sealing the mechanistic attribution structure."
    )

    @property
    def participating_mechanism_ids(self) -> frozenset[str]:
        """Return the exact set of mechanism IDs that participated in the computation."""
        return frozenset(m.mechanism_id for m in self.participating_mechanisms)

    def render_faithful_summary(self) -> str:
        """Render a deterministic text summary referencing ONLY recorded trace elements."""
        mech_parts = [
            f"mech:{m.mechanism_id}@v{m.mechanism_version}({m.evidence_nature.value})"
            for m in self.participating_mechanisms
        ]
        ctx_parts = [f"ctx_version:{v}" for v in self.active_context_versions]
        key_parts = [f"ctx_key:{k}" for k in self.active_context_keys_used]
        res_parts = [f"res:{r}" for r in self.modified_resources]
        edge_parts = [f"edge:{u}->{v}" for (u, v, _rel) in self.trace_edges]

        sections = [
            f"VerifiableExplanation[{self.explanation_id}]",
            f"effective_evidence_level={self.effective_evidence_level.value}",
            f"mechanisms=[{', '.join(mech_parts)}]",
            f"context_versions=[{', '.join(ctx_parts)}]",
            f"context_keys=[{', '.join(key_parts)}]",
            f"modified_resources=[{', '.join(res_parts)}]",
            f"trace_edges=[{', '.join(edge_parts)}]",
            f"fingerprint={self.explanation_fingerprint[:16]}",
        ]
        return " | ".join(sections)

    @classmethod
    def from_transition_result(
        cls,
        result: TransitionResult,
        *,
        step: int = 0,
        prior_state: WorldState | None = None,
        runtime_context: ContextSnapshot | None = None,
        trace: SystemicTrace | None = None,
        explanation_id: str = "transition_explanation",
    ) -> VerifiableMechanisticExplanation:
        """Construct a verifiable explanation directly from an executed TransitionResult."""
        mp = result.mechanism_provenance
        cm = result.context_metadata or {}
        ctx_ver = (
            runtime_context.version
            if runtime_context is not None
            else (str(cm["version"]) if "version" in cm and cm["version"] is not None else None)
        )
        ctx_fp = (
            runtime_context.fingerprint
            if runtime_context is not None
            else (
                str(cm["fingerprint"])
                if "fingerprint" in cm and cm["fingerprint"] is not None
                else None
            )
        )

        deltas: dict[str, float] = {}
        if prior_state is not None:
            for rid, res_after in sorted(result.next_state.resources.items()):
                res_before = prior_state.resources.get(rid)
                before_val = res_before.current if res_before is not None else 0.0
                diff = float(res_after.current - before_val)
                if abs(diff) > 1e-12:
                    deltas[rid] = diff
        else:
            for k, v in sorted(result.applied_changes.items()):
                if isinstance(v, (int, float)) and abs(float(v)) > 1e-12:
                    deltas[str(k)] = float(v)

        rf = result.residual_feasibility
        rf_action = rf.action_taken if rf is not None else None

        active_trace = trace if trace is not None else SystemicTrace()
        node_ids = tuple(sorted(active_trace.nodes.keys()))
        edges = tuple(sorted((e.source, e.target, e.relation) for e in active_trace.edges))

        # Check if composite child provenances or substep diagnostics were recorded
        child_provs_list: list[Mapping[str, Any]] = []
        raw_child_provs = result.diagnostics.get("child_mechanism_provenances")
        if isinstance(raw_child_provs, Sequence):
            for item in raw_child_provs:
                if isinstance(item, Mapping):
                    child_provs_list.append(item)
        for d_key, d_val in sorted(result.diagnostics.items()):
            if str(d_key).startswith("substep_") and isinstance(d_val, Mapping):
                sub_mp = d_val.get("mechanism_provenance")
                if isinstance(sub_mp, Mapping):
                    child_provs_list.append(sub_mp)

        attributions: list[MechanismAttributionRecord] = []

        if len(child_provs_list) > 0:
            for raw_cp in child_provs_list:
                if isinstance(raw_cp, Mapping):
                    c_id = str(raw_cp.get("mechanism_id", "unknown_child"))
                    c_ver = str(raw_cp.get("version", "1.0.0"))
                    c_kind = str(raw_cp.get("kind", "known_deterministic"))
                    c_lvl_raw = raw_cp.get("evidence_level", result.evidence_level)
                    c_lvl = (
                        c_lvl_raw
                        if isinstance(c_lvl_raw, EvidenceLevel)
                        else EvidenceLevel(str(c_lvl_raw))
                    )
                    c_keys = tuple(str(k) for k in raw_cp.get("required_context_keys", ()))
                    attributions.append(
                        MechanismAttributionRecord(
                            mechanism_id=c_id,
                            mechanism_version=c_ver,
                            mechanism_kind=c_kind,
                            evidence_nature=evidence_level_to_nature(c_lvl),
                            evidence_level=c_lvl,
                            context_version=ctx_ver,
                            context_fingerprint=ctx_fp,
                            required_context_keys=c_keys,
                            participated_steps=(step,),
                            trace_node_ids=node_ids,
                            trace_edges=edges,
                            state_resource_deltas=deltas,
                            residual_feasibility_action=rf_action,
                        )
                    )
        else:
            mech_id = (
                mp.mechanism_id
                if mp is not None
                else (result.model_name or "unnamed_transition_mechanism")
            )
            mech_ver = mp.version if mp is not None else "1.0.0"
            mech_kind = mp.kind.value if mp is not None else "known_deterministic"
            mech_lvl = mp.evidence_level if mp is not None else result.evidence_level
            req_keys = mp.required_context_keys if mp is not None else ()
            attributions.append(
                MechanismAttributionRecord(
                    mechanism_id=mech_id,
                    mechanism_version=mech_ver,
                    mechanism_kind=mech_kind,
                    evidence_nature=evidence_level_to_nature(mech_lvl),
                    evidence_level=mech_lvl,
                    context_version=ctx_ver,
                    context_fingerprint=ctx_fp,
                    required_context_keys=tuple(req_keys),
                    participated_steps=(step,),
                    trace_node_ids=node_ids,
                    trace_edges=edges,
                    state_resource_deltas=deltas,
                    residual_feasibility_action=rf_action,
                )
            )

        from ewm_engine.core.epistemic import (
            EpistemicState,
            resolve_fail_closed_evidence_level,
        )

        eff_level = resolve_fail_closed_evidence_level(
            [a.evidence_level for a in attributions] or [result.evidence_level]
        )
        ctx_versions = tuple(sorted({a.context_version for a in attributions if a.context_version}))
        ctx_fps = tuple(
            sorted({a.context_fingerprint for a in attributions if a.context_fingerprint})
        )
        used_keys = tuple(sorted({k for a in attributions for k in a.required_context_keys}))
        mod_res = tuple(sorted(deltas.keys()))

        epistemic = EpistemicState.from_world_state(
            result.next_state,
            trace=active_trace,
            effective_evidence_level=eff_level,
            mechanism_provenances=[mp.model_dump(mode="json")] if mp is not None else [],
            context_versions=ctx_versions,
            context_fingerprints=ctx_fps,
        )

        fp_payload = {
            "explanation_id": explanation_id,
            "source_epistemic_fingerprint": epistemic.epistemic_fingerprint,
            "effective_evidence_level": eff_level.value,
            "mechanisms": [a.model_dump(mode="json") for a in attributions],
            "context_versions": ctx_versions,
            "context_fingerprints": ctx_fps,
            "context_keys": used_keys,
            "trace_nodes": node_ids,
            "trace_edges": edges,
            "modified_resources": mod_res,
        }
        exp_fp = canonical_sha256(fp_payload)

        return cls(
            explanation_id=explanation_id,
            is_standalone_narrative_evidence=False,
            source_epistemic_fingerprint=epistemic.epistemic_fingerprint,
            effective_evidence_level=eff_level,
            effective_evidence_nature=evidence_level_to_nature(eff_level),
            participating_mechanisms=tuple(attributions),
            active_context_versions=ctx_versions,
            active_context_fingerprints=ctx_fps,
            active_context_keys_used=used_keys,
            trace_node_ids=node_ids,
            trace_edges=edges,
            modified_resources=mod_res,
            drift_signals=(),
            explanation_fingerprint=exp_fp,
        )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        explanation_id: str | None = None,
    ) -> VerifiableMechanisticExplanation:
        """Construct a verifiable explanation from a completed simulation Trajectory."""
        exp_id = explanation_id or f"trajectory_{trajectory.sample_id}_seed_{trajectory.seed}"
        epistemic: EpistemicState = trajectory.epistemic_state
        trace = trajectory.systemic_trace

        mech_steps: dict[str, list[int]] = {}
        mech_meta: dict[str, dict[str, Any]] = {}
        mech_deltas: dict[str, dict[str, float]] = {}
        all_mod_res: set[str] = set()

        prev_state = trajectory.initial_state
        for step_rec in trajectory.steps:
            tr = step_rec.transition_result
            mp = tr.mechanism_provenance
            cm = tr.context_metadata or {}
            m_id = (
                mp.mechanism_id
                if mp is not None
                else (tr.model_name or f"step_{step_rec.step}_mechanism")
            )
            mech_steps.setdefault(m_id, []).append(step_rec.step)
            if m_id not in mech_meta:
                mech_meta[m_id] = {
                    "version": mp.version if mp is not None else "1.0.0",
                    "kind": mp.kind.value if mp is not None else "known_deterministic",
                    "evidence_level": mp.evidence_level if mp is not None else tr.evidence_level,
                    "context_version": cm.get("version"),
                    "context_fingerprint": cm.get("fingerprint"),
                    "required_context_keys": (mp.required_context_keys if mp is not None else ()),
                    "rf_action": (
                        tr.residual_feasibility.action_taken
                        if tr.residual_feasibility is not None
                        else None
                    ),
                }
            step_deltas = mech_deltas.setdefault(m_id, {})
            for rid, r_after in tr.next_state.resources.items():
                r_before = prev_state.resources.get(rid)
                b_val = r_before.current if r_before is not None else 0.0
                d_val = float(r_after.current - b_val)
                if abs(d_val) > 1e-12:
                    step_deltas[rid] = step_deltas.get(rid, 0.0) + d_val
                    all_mod_res.add(rid)
            prev_state = tr.next_state

        node_ids = tuple(sorted(trace.nodes.keys()))
        edges = tuple(sorted((e.source, e.target, e.relation) for e in trace.edges))

        attributions: list[MechanismAttributionRecord] = []
        for m_id in sorted(mech_meta.keys()):
            meta = mech_meta[m_id]
            lvl: EvidenceLevel = meta["evidence_level"]
            attributions.append(
                MechanismAttributionRecord(
                    mechanism_id=m_id,
                    mechanism_version=str(meta["version"]),
                    mechanism_kind=str(meta["kind"]),
                    evidence_nature=evidence_level_to_nature(lvl),
                    evidence_level=lvl,
                    context_version=(
                        str(meta["context_version"])
                        if meta["context_version"] is not None
                        else None
                    ),
                    context_fingerprint=(
                        str(meta["context_fingerprint"])
                        if meta["context_fingerprint"] is not None
                        else None
                    ),
                    required_context_keys=tuple(meta["required_context_keys"]),
                    participated_steps=tuple(mech_steps.get(m_id, ())),
                    trace_node_ids=node_ids,
                    trace_edges=edges,
                    state_resource_deltas=mech_deltas.get(m_id, {}),
                    residual_feasibility_action=meta["rf_action"],
                )
            )

        ctx_versions = tuple(sorted(set(epistemic.context_versions)))
        ctx_fps = tuple(sorted(set(epistemic.context_fingerprints)))
        used_keys = tuple(sorted({k for a in attributions for k in a.required_context_keys}))
        mod_res = tuple(sorted(all_mod_res))

        fp_payload = {
            "explanation_id": exp_id,
            "source_epistemic_fingerprint": epistemic.epistemic_fingerprint,
            "effective_evidence_level": epistemic.effective_evidence_level.value,
            "mechanisms": [a.model_dump(mode="json") for a in attributions],
            "context_versions": ctx_versions,
            "context_fingerprints": ctx_fps,
            "context_keys": used_keys,
            "trace_nodes": node_ids,
            "trace_edges": edges,
            "modified_resources": mod_res,
        }
        exp_fp = canonical_sha256(fp_payload)

        return cls(
            explanation_id=exp_id,
            is_standalone_narrative_evidence=False,
            source_epistemic_fingerprint=epistemic.epistemic_fingerprint,
            effective_evidence_level=epistemic.effective_evidence_level,
            effective_evidence_nature=evidence_level_to_nature(epistemic.effective_evidence_level),
            participating_mechanisms=tuple(attributions),
            active_context_versions=ctx_versions,
            active_context_fingerprints=ctx_fps,
            active_context_keys_used=used_keys,
            trace_node_ids=node_ids,
            trace_edges=edges,
            modified_resources=mod_res,
            drift_signals=epistemic.drift_signals,
            explanation_fingerprint=exp_fp,
        )


def build_verifiable_explanation(
    source: TransitionResult | Trajectory,
    *,
    step: int = 0,
    prior_state: WorldState | None = None,
    runtime_context: ContextSnapshot | None = None,
    trace: SystemicTrace | None = None,
    explanation_id: str | None = None,
) -> VerifiableMechanisticExplanation:
    """Build a verifiable mechanistic explanation from a TransitionResult or Trajectory."""
    from ewm_engine.dynamics.base import TransitionResult as _TR

    if isinstance(source, _TR):
        return VerifiableMechanisticExplanation.from_transition_result(
            source,
            step=step,
            prior_state=prior_state,
            runtime_context=runtime_context,
            trace=trace,
            explanation_id=explanation_id or "transition_explanation",
        )
    return VerifiableMechanisticExplanation.from_trajectory(
        source,
        explanation_id=explanation_id,
    )


class CandidateNarrativeExplanation(BaseModel):
    """A natural-language or LLM-rendered explanation candidate to be checked against the trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    narrative_text: str = Field(description="Rendered natural-language explanation text.")
    claimed_mechanisms: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Mechanism IDs explicitly cited as having participated.",
    )
    claimed_context_versions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="ContextSnapshot versions cited as governing the transition.",
    )
    claimed_context_keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Context parameter/policy keys cited as having influenced the transition.",
    )
    claimed_modified_resources: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Resource IDs claimed to have been modified.",
    )
    claimed_trace_edges: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Directed trace edges (source, target) claimed to have propagated effects.",
    )
    claimed_evidence_level: EvidenceLevel | None = Field(
        default=None,
        description="Epistemic standing claimed by the narrative explanation.",
    )
    treats_narrative_as_standalone_proof: bool = Field(
        default=False,
        description="If True, attempts to assert narrative plausibility as standalone proof (always rejected).",
    )


class ExplanationFidelityReport(BaseModel):
    """Audit result verifying a candidate narrative explanation against the mechanistic trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_faithful: bool
    ground_truth_explanation_fingerprint: str
    fabricated_mechanisms: tuple[str, ...] = Field(default_factory=tuple)
    fabricated_context_versions: tuple[str, ...] = Field(default_factory=tuple)
    fabricated_context_keys: tuple[str, ...] = Field(default_factory=tuple)
    fabricated_resources: tuple[str, ...] = Field(default_factory=tuple)
    fabricated_trace_edges: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    epistemic_overclaims: tuple[str, ...] = Field(default_factory=tuple)
    standalone_narrative_violation: bool = False
    violations: tuple[str, ...] = Field(default_factory=tuple)


def verify_narrative_explanation_fidelity(
    ground_truth: VerifiableMechanisticExplanation | None,
    candidate: CandidateNarrativeExplanation | str,
    *,
    known_universe_mechanisms: Sequence[str] = (),
    known_universe_resources: Sequence[str] = (),
    known_universe_context_keys: Sequence[str] = (),
    strict: bool = True,
) -> ExplanationFidelityReport:
    """Verify that a narrative/LLM explanation only attributes mechanisms and context in the trace.

    Enforces four hard anti-narrative invariants:
    1. Narrative prose is never accepted without a grounded `VerifiableMechanisticExplanation`
       or when `treats_narrative_as_standalone_proof=True`.
    2. Every mechanism, context version, context key, modified resource, and trace edge
       claimed in structured fields or inline prose tags (`mech:...`, `ctx_version:...`,
       `ctx_key:...`, `res:...`, `edge:A->B`) or matched against the known domain universe
       must exist in `ground_truth`.
    3. Any reference to a mechanism in `known_universe_mechanisms` that did NOT participate
       in `ground_truth` is flagged as a fabricated attribution.
    4. Narrative prose cannot claim causal/counterfactual proof when `ground_truth`'s
       `effective_evidence_level` is non-causal (e.g. `PREDICTIVE` / `STATISTICAL_ASSOCIATION`).
    """
    cand = (
        CandidateNarrativeExplanation(narrative_text=candidate)
        if isinstance(candidate, str)
        else candidate
    )

    violations: list[str] = []
    if ground_truth is None or len(ground_truth.participating_mechanisms) == 0:
        violations.append(
            "Standalone narrative explainability violation: no recorded mechanistic trace "
            "exists to ground the narrative explanation."
        )
        if strict:
            raise NarrativeFabricationError(violations[0], violations=violations)
        return ExplanationFidelityReport(
            is_faithful=False,
            ground_truth_explanation_fingerprint="",
            standalone_narrative_violation=True,
            violations=tuple(violations),
        )

    standalone_viol = False
    if cand.treats_narrative_as_standalone_proof:
        standalone_viol = True
        violations.append(
            "Anti-narrative guardrail violation: LLM/narrative justification cannot be "
            "accepted as standalone evidence of mechanism validity."
        )

    actual_mechs = ground_truth.participating_mechanism_ids
    actual_ctx_vers = set(ground_truth.active_context_versions)
    actual_ctx_keys = set(ground_truth.active_context_keys_used)
    actual_resources = set(ground_truth.modified_resources)
    actual_edges = {(u, v) for (u, v, _rel) in ground_truth.trace_edges}

    # Extract inline tags from narrative_text (stripping sentence-ending periods)
    text = cand.narrative_text
    inline_mechs = {m.rstrip(".") for m in _MECH_TAG_PATTERN.findall(text)}
    inline_ctx_vers = {m.rstrip(".") for m in _CTX_VER_TAG_PATTERN.findall(text)}
    inline_ctx_keys = {m.rstrip(".") for m in _CTX_KEY_TAG_PATTERN.findall(text)}
    inline_resources = {m.rstrip(".") for m in _RES_TAG_PATTERN.findall(text)}
    inline_edges = {(u.rstrip("."), v.rstrip(".")) for (u, v) in _EDGE_TAG_PATTERN.findall(text)}

    # Also check any known universe identifiers mentioned verbatim in narrative_text
    for um in known_universe_mechanisms:
        if um and re.search(rf"\b{re.escape(um)}\b", text):
            inline_mechs.add(um)
    for ur in known_universe_resources:
        if ur and re.search(rf"\b{re.escape(ur)}\b", text):
            inline_resources.add(ur)
    for uk in known_universe_context_keys:
        if uk and re.search(rf"\b{re.escape(uk)}\b", text):
            inline_ctx_keys.add(uk)

    claimed_mechs = set(cand.claimed_mechanisms) | inline_mechs
    claimed_ctx_vers = set(cand.claimed_context_versions) | inline_ctx_vers
    claimed_ctx_keys = set(cand.claimed_context_keys) | inline_ctx_keys
    claimed_resources = set(cand.claimed_modified_resources) | inline_resources
    claimed_edges = set(cand.claimed_trace_edges) | inline_edges

    fab_mechs = tuple(sorted(claimed_mechs - actual_mechs))
    fab_ctx_vers = tuple(sorted(claimed_ctx_vers - actual_ctx_vers))
    fab_ctx_keys = tuple(sorted(claimed_ctx_keys - actual_ctx_keys))
    fab_resources = tuple(sorted(claimed_resources - actual_resources))
    fab_edges = tuple(sorted(claimed_edges - actual_edges))

    if fab_mechs:
        violations.append(
            f"Fabricated mechanism attribution(s) not present in trace: {list(fab_mechs)} "
            f"(actual participating mechanisms: {sorted(actual_mechs)})."
        )
    if fab_ctx_vers:
        violations.append(
            f"Fabricated context version(s) not active in trace: {list(fab_ctx_vers)} "
            f"(actual active versions: {sorted(actual_ctx_vers)})."
        )
    if fab_ctx_keys:
        violations.append(
            f"Fabricated context key attribution(s) not used by participating mechanisms: "
            f"{list(fab_ctx_keys)} (actual keys used: {sorted(actual_ctx_keys)})."
        )
    if fab_resources:
        violations.append(
            f"Fabricated state/resource modification claim(s): {list(fab_resources)} "
            f"(actual modified resources: {sorted(actual_resources)})."
        )
    if fab_edges:
        violations.append(
            f"Fabricated trace dependency edge(s) not present in SystemicTrace: {list(fab_edges)}."
        )

    # Check epistemic overclaiming
    overclaims: list[str] = []
    gt_rank = _EVIDENCE_RANK_ORDER.get(ground_truth.effective_evidence_level, 0)
    causal_threshold_rank = _EVIDENCE_RANK_ORDER[EvidenceLevel.QUASI_CAUSAL]

    if cand.claimed_evidence_level is not None:
        cand_rank = _EVIDENCE_RANK_ORDER.get(cand.claimed_evidence_level, 0)
        if cand_rank > gt_rank:
            msg = (
                f"Narrative claims evidence_level='{cand.claimed_evidence_level.value}' "
                f"exceeding trace fail-closed ceiling '{ground_truth.effective_evidence_level.value}'."
            )
            overclaims.append(msg)
            violations.append(msg)

    if gt_rank < causal_threshold_rank:
        for pat in _CAUSAL_PROSE_PATTERNS:
            match = pat.search(text)
            if match is not None:
                phrase = match.group(0)
                msg = (
                    f"Rhetorical causal overclaim '{phrase}' in narrative text is forbidden "
                    f"when trace effective_evidence_level is '{ground_truth.effective_evidence_level.value}' "
                    f"({ground_truth.effective_evidence_nature.value})."
                )
                overclaims.append(msg)
                violations.append(msg)

    is_faithful = len(violations) == 0
    if not is_faithful and strict:
        raise NarrativeFabricationError(
            "Narrative explanation failed mechanistic fidelity check: " + " | ".join(violations),
            violations=violations,
        )

    return ExplanationFidelityReport(
        is_faithful=is_faithful,
        ground_truth_explanation_fingerprint=ground_truth.explanation_fingerprint,
        fabricated_mechanisms=fab_mechs,
        fabricated_context_versions=fab_ctx_vers,
        fabricated_context_keys=fab_ctx_keys,
        fabricated_resources=fab_resources,
        fabricated_trace_edges=fab_edges,
        epistemic_overclaims=tuple(overclaims),
        standalone_narrative_violation=standalone_viol,
        violations=tuple(violations),
    )
