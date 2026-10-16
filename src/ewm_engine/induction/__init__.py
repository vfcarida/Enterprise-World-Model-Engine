"""World-model induction subsystem for data-driven enterprise twins.

Provides point-it-at-your-data capabilities:
    - Tabular & SQL connectors (Arrow boundary, point-in-time correctness)
    - Relational schema induction (tables -> entities, asserted FKs -> edges, RDL latent link prediction)
    - Object-Centric Event Log (OCEL 2.0) process induction (Inductive Miner, bounded-resource invariants, conformance-derived EvidenceLevel)
    - Strict out-of-process AGPL license isolation for PM4Py
    - Multi-method ensemble causal-structure induction (PC/FCI/DAGMA/DirectLiNGAM/DYNOTEARS -> DiBS CPDAG/PAG)
    - Strict out-of-process GPL license isolation for Tigramite
    - Dynamics & rate-law induction via SINDy and symbolic regression
    - Conservation-law / invariant induction via AI-Poincaré nullspace and Monte Carlo stress testing
"""

from __future__ import annotations

from ewm_engine.induction.causal import (
    CausalEdge,
    CausalEdgeType,
    CausalEquivalenceClass,
    induce_causal,
    reconcile_ensemble,
    run_dagma_notears,
    run_direct_lingam,
    run_fci_algorithm,
    run_pc_algorithm,
    run_temporal_discovery,
)
from ewm_engine.induction.connectors import (
    DataFrameSchemaMapping,
    induce_from_dataframe,
    induce_from_sql,
)
from ewm_engine.induction.dynamics import (
    InducedRateLaw,
    induce_dynamics,
    run_sindy_stlsq,
)
from ewm_engine.induction.invariants import (
    InvariantHypothesis,
    estimate_conservation_nullspace,
    induce_invariants,
    stress_test_candidate_invariant,
)
from ewm_engine.induction.pm4py_bridge import (
    is_pm4py_available,
    run_pm4py_inductive_miner_out_of_process,
)
from ewm_engine.induction.process import (
    OcelEvent,
    OcelLog,
    OcelObject,
    induce_process,
    parse_ocel_json,
)
from ewm_engine.induction.report import (
    CandidateInvariant,
    CausalEquivalenceClassSummary,
    ConformanceScore,
    InducedRateLawSummary,
    InductionReport,
    ProvisionalLink,
)
from ewm_engine.induction.schema import (
    TableSchema,
    induce_schema,
    parse_ddl_to_tables,
)
from ewm_engine.induction.tigramite_bridge import (
    is_tigramite_available,
    run_tigramite_out_of_process,
)
from ewm_engine.induction.validation import (
    HeldOutReproductionResult,
    InductionValidationGate,
    InductionValidationReport,
    InvariantStressResult,
    OpenQuestion,
    ProcessConformanceResult,
    RefusedElement,
    RefuterResult,
    RefuterType,
    refute_data_subset,
    refute_placebo_treatment,
    refute_random_common_cause,
    run_causal_refutations,
    validate_dynamics_held_out,
    validate_induction,
)

__all__ = [
    "CandidateInvariant",
    "CausalEdge",
    "CausalEdgeType",
    "CausalEquivalenceClass",
    "CausalEquivalenceClassSummary",
    "ConformanceScore",
    "DataFrameSchemaMapping",
    "HeldOutReproductionResult",
    "InducedRateLaw",
    "InducedRateLawSummary",
    "InductionReport",
    "InductionValidationGate",
    "InductionValidationReport",
    "InvariantHypothesis",
    "InvariantStressResult",
    "OcelEvent",
    "OcelLog",
    "OcelObject",
    "OpenQuestion",
    "ProcessConformanceResult",
    "ProvisionalLink",
    "RefusedElement",
    "RefuterResult",
    "RefuterType",
    "TableSchema",
    "estimate_conservation_nullspace",
    "induce_causal",
    "induce_dynamics",
    "induce_from_dataframe",
    "induce_from_sql",
    "induce_invariants",
    "induce_process",
    "induce_schema",
    "is_pm4py_available",
    "is_tigramite_available",
    "parse_ddl_to_tables",
    "parse_ocel_json",
    "reconcile_ensemble",
    "refute_data_subset",
    "refute_placebo_treatment",
    "refute_random_common_cause",
    "run_causal_refutations",
    "run_dagma_notears",
    "run_direct_lingam",
    "run_fci_algorithm",
    "run_pc_algorithm",
    "run_pm4py_inductive_miner_out_of_process",
    "run_sindy_stlsq",
    "run_temporal_discovery",
    "run_tigramite_out_of_process",
    "stress_test_candidate_invariant",
    "validate_dynamics_held_out",
    "validate_induction",
]
