"""Trajectory verification subsystem: Oracle Graph and Signal Temporal Logic.

Conforms to Track T3 in 01_EXPANDED_ROADMAP.md.
"""

from __future__ import annotations

from ewm_engine.verification.oracle import (
    OracleEdge,
    OracleGraph,
    OracleNode,
    OracleVerificationResult,
    VerificationViolation,
    evaluate_oracle_graph,
)
from ewm_engine.verification.rtamt_adapter import (
    RobustnessDistribution,
    RTAMTEvaluationBackend,
    is_rtamt_available,
)
from ewm_engine.verification.spec import (
    PropertySpec,
    compute_property_hash,
    fold_properties_into_fingerprint,
)
from ewm_engine.verification.stubs import (
    LLMSoftCheckAdapter,
    MoonLightSTRELAdapter,
)
from ewm_engine.verification.temporal import (
    AlwaysFormula,
    AndFormula,
    EventuallyFormula,
    ImpliesFormula,
    NotFormula,
    OrFormula,
    PredicateFormula,
    STLFormula,
    STLVerdict,
    UntilFormula,
    always,
    evaluate_stl_bounded,
    eventually,
    extract_trajectory_signals,
    implies,
    predicate,
    until,
)

__all__ = [
    "AlwaysFormula",
    "AndFormula",
    "EventuallyFormula",
    "ImpliesFormula",
    "LLMSoftCheckAdapter",
    "MoonLightSTRELAdapter",
    "NotFormula",
    "OrFormula",
    "OracleEdge",
    "OracleGraph",
    "OracleNode",
    "OracleVerificationResult",
    "PredicateFormula",
    "PropertySpec",
    "RTAMTEvaluationBackend",
    "RobustnessDistribution",
    "STLFormula",
    "STLVerdict",
    "UntilFormula",
    "VerificationViolation",
    "always",
    "compute_property_hash",
    "evaluate_oracle_graph",
    "evaluate_stl_bounded",
    "eventually",
    "extract_trajectory_signals",
    "fold_properties_into_fingerprint",
    "implies",
    "is_rtamt_available",
    "predicate",
    "until",
]
