"""Provenance, metadata, and systemic trace tracking for EWM Engine."""

from __future__ import annotations

from ewm_engine.provenance.evidence import (
    EVIDENCE_LEVEL_TO_NATURE,
    EVIDENCE_NATURE_TO_DEFAULT_LEVEL,
    PERMISSIBLE_QUERIES_BY_NATURE,
    DependencyEvidenceNature,
    DependencyQueryKind,
    EvidenceLevel,
    EvidenceNature,
    evidence_level_to_nature,
    evidence_nature_to_level,
)
from ewm_engine.provenance.explainability import (
    CandidateNarrativeExplanation,
    ExplanationFidelityReport,
    MechanismAttributionRecord,
    NarrativeFabricationError,
    VerifiableMechanisticExplanation,
    build_verifiable_explanation,
    verify_narrative_explanation_fidelity,
)
from ewm_engine.provenance.metadata import ComponentVersion, Provenance, SimulationMetadata
from ewm_engine.provenance.trace import (
    ExogenousNoiseRecord,
    SystemicTrace,
    TraceDiff,
    TraceEdge,
    TraceNode,
)

__all__ = [
    "EVIDENCE_LEVEL_TO_NATURE",
    "EVIDENCE_NATURE_TO_DEFAULT_LEVEL",
    "PERMISSIBLE_QUERIES_BY_NATURE",
    "CandidateNarrativeExplanation",
    "ComponentVersion",
    "DependencyEvidenceNature",
    "DependencyQueryKind",
    "EvidenceLevel",
    "EvidenceNature",
    "ExogenousNoiseRecord",
    "ExplanationFidelityReport",
    "MechanismAttributionRecord",
    "NarrativeFabricationError",
    "Provenance",
    "SimulationMetadata",
    "SystemicTrace",
    "TraceDiff",
    "TraceEdge",
    "TraceNode",
    "VerifiableMechanisticExplanation",
    "build_verifiable_explanation",
    "evidence_level_to_nature",
    "evidence_nature_to_level",
    "verify_narrative_explanation_fidelity",
]
