"""World-model induction subsystem for data-driven enterprise twins.

Provides point-it-at-your-data capabilities:
    - Tabular & SQL connectors (Arrow boundary, point-in-time correctness)
    - Relational schema induction (tables -> entities, asserted FKs -> edges, RDL latent link prediction)
    - Object-Centric Event Log (OCEL 2.0) process induction (Inductive Miner, bounded-resource invariants, conformance-derived EvidenceLevel)
    - Strict out-of-process AGPL license isolation for PM4Py
"""

from __future__ import annotations

from ewm_engine.induction.connectors import (
    DataFrameSchemaMapping,
    induce_from_dataframe,
    induce_from_sql,
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
    ConformanceScore,
    InductionReport,
    ProvisionalLink,
)
from ewm_engine.induction.schema import (
    TableSchema,
    induce_schema,
    parse_ddl_to_tables,
)

__all__ = [
    "CandidateInvariant",
    "ConformanceScore",
    "DataFrameSchemaMapping",
    "InductionReport",
    "OcelEvent",
    "OcelLog",
    "OcelObject",
    "ProvisionalLink",
    "TableSchema",
    "induce_from_dataframe",
    "induce_from_sql",
    "induce_process",
    "induce_schema",
    "is_pm4py_available",
    "parse_ddl_to_tables",
    "parse_ocel_json",
    "run_pm4py_inductive_miner_out_of_process",
]
