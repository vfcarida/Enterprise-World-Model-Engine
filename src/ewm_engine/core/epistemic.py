"""Epistemic state (P_t) and fail-closed causal evidence resolution.

Separates ONTIC STATE ("what is true in the modeled enterprise world", S_t)
from EPISTEMIC STATE ("what the engine knows, infers, assumes, or has causal
identification evidence for", P_t) without breaking the v1 WorldState schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.drift import DriftSignal
from ewm_engine.core.state import WorldState
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace

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


def resolve_fail_closed_evidence_level(
    levels: Sequence[EvidenceLevel],
    *,
    has_unidentified_learned_or_discovered: bool = False,
) -> EvidenceLevel:
    """Resolve the effective causal EvidenceLevel under fail-closed semantics.

    1. The composite evidence level is bounded above by the weakest component
       in ``levels`` according to ``_EVIDENCE_RANK_ORDER``.
    2. If any component is a learned or runtime-discovered mechanism lacking an
       explicit causal identification certificate, the effective evidence level
       is capped at ``EvidenceLevel.PREDICTIVE``.
    """
    if not levels:
        base = EvidenceLevel.PREDICTIVE
    else:
        base = min(levels, key=lambda lvl: _EVIDENCE_RANK_ORDER.get(lvl, 0))
    if (
        has_unidentified_learned_or_discovered
        and _EVIDENCE_RANK_ORDER.get(base, 0) > _EVIDENCE_RANK_ORDER[EvidenceLevel.PREDICTIVE]
    ):
        return EvidenceLevel.PREDICTIVE
    return base


class EpistemicState(BaseModel):
    """Epistemic state P_t capturing provenance, evidence ceilings, and assumptions.

    Decoupled from the ontic WorldState S_t so that epistemic audit updates
    (such as trace annotations, mechanism provenance records, or drift signals)
    never alter the ontic state fingerprint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace: SystemicTrace = Field(default_factory=SystemicTrace)
    effective_evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL
    mechanism_provenances: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    context_versions: tuple[str, ...] = Field(default_factory=tuple)
    context_fingerprints: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[str, ...] = Field(default_factory=tuple)
    drift_signals: tuple[DriftSignal, ...] = Field(default_factory=tuple)

    @property
    def epistemic_fingerprint(self) -> str:
        """Compute canonical SHA-256 hex digest of the epistemic state P_t."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_world_state(
        cls,
        state: WorldState,
        *,
        trace: SystemicTrace | None = None,
        effective_evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
        mechanism_provenances: Sequence[dict[str, Any]] = (),
        context_versions: Sequence[str] = (),
        context_fingerprints: Sequence[str] = (),
        assumptions: Sequence[str] = (),
        drift_signals: Sequence[DriftSignal] = (),
    ) -> EpistemicState:
        """Extract the epistemic state P_t from a WorldState and optional SystemicTrace."""
        extracted_assumptions: list[str] = list(assumptions)
        for k, v in sorted(state.memory.items()):
            if str(k).startswith("_epistemic_"):
                extracted_assumptions.append(f"{k}={v}")
        for k, v in sorted(state.context.items()):
            if str(k).startswith("_epistemic_"):
                extracted_assumptions.append(f"{k}={v}")

        return cls(
            trace=copy.deepcopy(trace) if trace is not None else SystemicTrace(),
            effective_evidence_level=effective_evidence_level,
            mechanism_provenances=tuple(copy.deepcopy(list(mechanism_provenances))),
            context_versions=tuple(context_versions),
            context_fingerprints=tuple(context_fingerprints),
            assumptions=tuple(extracted_assumptions),
            drift_signals=tuple(drift_signals),
        )


def separate_ontic_epistemic(
    state: WorldState,
    *,
    trace: SystemicTrace | None = None,
    effective_evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
    mechanism_provenances: Sequence[dict[str, Any]] = (),
    context_versions: Sequence[str] = (),
    context_fingerprints: Sequence[str] = (),
    assumptions: Sequence[str] = (),
    drift_signals: Sequence[DriftSignal] = (),
) -> tuple[WorldState, EpistemicState]:
    """Split a WorldState into a pure ontic WorldState and an EpistemicState."""
    epistemic = EpistemicState.from_world_state(
        state,
        trace=trace,
        effective_evidence_level=effective_evidence_level,
        mechanism_provenances=mechanism_provenances,
        context_versions=context_versions,
        context_fingerprints=context_fingerprints,
        assumptions=assumptions,
        drift_signals=drift_signals,
    )
    ontic = state.to_ontic_state()
    return ontic, epistemic
