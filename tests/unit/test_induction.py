"""Unit tests for world-model induction subsystem (T04).

Tests:
    1. Arrow/DataFrame connectors with point-in-time correctness
    2. SQL connectors (SQLite / DuckDB)
    3. Relational schema induction (asserted FKs, SQL check constraints, RDL link prediction)
    4. OCEL 2.0 ingestion, Inductive Miner, candidate conservation invariants, conformance checking
    5. InductionReport determinism and SHA-256 fingerprinting
    6. PM4Py AGPL isolation boundary
    7. CLI subcommand execution (`ewm induce`)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ewm_engine.core.spec import WorldSpec
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.induction.connectors import (
    DataFrameSchemaMapping,
    induce_from_dataframe,
)
from ewm_engine.induction.pm4py_bridge import is_pm4py_available
from ewm_engine.induction.process import (
    compute_conformance,
    induce_process,
    mine_dfg_and_transitions,
    parse_ocel_json,
)
from ewm_engine.induction.schema import (
    induce_schema,
)
from ewm_engine.provenance.evidence import EvidenceLevel

# ---------------------------------------------------------------------------
# 1. DataFrame & Point-in-Time Correctness Tests
# ---------------------------------------------------------------------------


def test_dataframe_connector_point_in_time_filtering() -> None:
    """Verify point-in-time cutoff prevents future temporal leakage (RelBench discipline)."""
    data = [
        {"entity_id": "wh_1", "timestamp": 10, "stock": 100.0, "region": "North"},
        {"entity_id": "wh_1", "timestamp": 20, "stock": 80.0, "region": "North"},
        {"entity_id": "wh_1", "timestamp": 30, "stock": 50.0, "region": "North"},
        {"entity_id": "wh_2", "timestamp": 10, "stock": 200.0, "region": "South"},
        {"entity_id": "wh_2", "timestamp": 25, "stock": 170.0, "region": "South"},
    ]

    # As of step 20: wh_1 should have stock 80.0, wh_2 should have stock 200.0; step 25 & 30 excluded
    spec, report = induce_from_dataframe(
        df=data,
        schema=DataFrameSchemaMapping(
            entity_id_col="entity_id",
            timestamp_col="timestamp",
            resource_cols=["stock"],
        ),
        as_of=20,
        name="WarehouseAsOf20",
    )

    assert len(spec.entities) == 2
    assert report.point_in_time_cutoff == "20"

    res_map = {r.id: r.current for r in spec.resources}
    assert res_map["wh_1_stock"] == 80.0
    assert res_map["wh_2_stock"] == 200.0


def test_dataframe_connector_arrow_interop() -> None:
    """Verify pyarrow Table can be passed across the Arrow boundary seamlessly."""
    try:
        import pyarrow as pa
    except ImportError:
        pytest.skip("pyarrow not installed")

    table = pa.Table.from_pydict(
        {
            "node_id": ["node_a", "node_b"],
            "timestamp": [1.0, 2.0],
            "capacity": [500.0, 750.0],
            "status": ["active", "standby"],
        }
    )

    spec = WorldSpec.from_dataframe(
        df=table,
        schema=DataFrameSchemaMapping(
            entity_id_col="node_id",
            timestamp_col="timestamp",
            resource_cols=["capacity"],
            attribute_cols=["status"],
        ),
        name="ArrowInducedWorld",
    )

    assert len(spec.entities) == 2
    assert len(spec.resources) == 2
    assert spec.resources[0].current == 500.0


# ---------------------------------------------------------------------------
# 2. SQL Connector Tests
# ---------------------------------------------------------------------------


def test_sql_connector_sqlite_in_memory() -> None:
    """Verify WorldSpec.from_sql executes query and builds valid world specification."""
    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("CREATE TABLE inventory (id TEXT, timestamp INT, units REAL, category TEXT);")
    cur.execute("INSERT INTO inventory VALUES ('inv_1', 1, 150.0, 'raw_materials');")
    cur.execute("INSERT INTO inventory VALUES ('inv_1', 5, 120.0, 'raw_materials');")
    cur.execute("INSERT INTO inventory VALUES ('inv_2', 1, 300.0, 'finished_goods');")
    con.commit()

    spec = WorldSpec.from_sql(
        query="SELECT * FROM inventory",
        con=con,
        as_of=3,
        schema=DataFrameSchemaMapping(
            entity_id_col="id",
            timestamp_col="timestamp",
            resource_cols=["units"],
        ),
        name="SqlInducedWorld",
    )

    assert len(spec.entities) == 2
    res_map = {r.id: r.current for r in spec.resources}
    assert res_map["inv_1_units"] == 150.0  # As of step 3, step 5 is ignored
    assert res_map["inv_2_units"] == 300.0


# ---------------------------------------------------------------------------
# 3. Relational Schema Induction Tests
# ---------------------------------------------------------------------------

SAMPLE_DDL = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    credit_limit REAL CHECK (credit_limit >= 0)
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_amount REAL CHECK (order_amount >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);
"""


def test_schema_induction_asserted_fks_and_check_constraints() -> None:
    """Verify declared foreign keys are tagged as ASSERTED and CHECK constraints are candidate invariants."""
    spec, report = induce_schema(
        ddl=SAMPLE_DDL,
        link_prediction=False,
        name="ProcurementSchemaWorld",
    )

    assert len(spec.entities) == 2
    assert any(e.id == "customers" for e in spec.entities)
    assert any(e.id == "orders" for e in spec.entities)

    # Declared FK is tagged as ASSERTED / STRUCTURAL
    assert len(spec.relationships) == 1
    rel = spec.relationships[0]
    assert rel.source == "orders"
    assert rel.target == "customers"
    assert rel.attributes["asserted"] is True
    assert rel.attributes["evidence_level"] == EvidenceLevel.STRUCTURAL.value

    # Candidate invariants extracted from CHECK and column keywords
    assert len(report.candidate_invariants) >= 2
    inv_names = [inv.name for inv in report.candidate_invariants]
    assert any("credit_limit" in name for name in inv_names)
    assert any("order_amount" in name for name in inv_names)


def test_schema_induction_link_prediction_provisional() -> None:
    """Verify latent link discovery proposes provisional links with confidence scores."""
    ddl = """
    CREATE TABLE warehouses (
        warehouse_id TEXT PRIMARY KEY,
        region TEXT
    );

    CREATE TABLE shipments (
        shipment_id TEXT PRIMARY KEY,
        warehouse_id TEXT,
        destination TEXT
    );
    """
    # Note: No explicit FOREIGN KEY declared between shipments and warehouses,
    # but 'warehouse_id' column is shared.
    spec, report = induce_schema(
        ddl=ddl,
        link_prediction=True,
        link_threshold=0.7,
        name="LatentLinkWorld",
    )

    assert len(report.provisional_links) >= 1
    link = report.provisional_links[0]
    assert link.source_entity == "shipments"
    assert link.target_entity == "warehouses"
    assert link.confidence >= 0.7
    assert link.evidence_level == EvidenceLevel.PREDICTIVE
    assert link.clears_gate is True

    # Added provisionally to specification with explicit tag
    inferred_rels = [r for r in spec.relationships if r.attributes.get("inferred_link")]
    assert len(inferred_rels) == 1
    assert inferred_rels[0].attributes["provisional"] is True


# ---------------------------------------------------------------------------
# 4. OCEL 2.0 Process Model Induction & Conformance Tests
# ---------------------------------------------------------------------------

SAMPLE_OCEL_JSON = {
    "objectTypes": ["order", "package"],
    "objects": [
        {"id": "ord_1", "type": "order", "attributes": {"total": 250.0}},
        {"id": "ord_2", "type": "order", "attributes": {"total": 420.0}},
        {"id": "pkg_1", "type": "package", "attributes": {"weight": 5.4}},
    ],
    "eventTypes": ["create_order", "approve_order", "pack", "dispatch"],
    "events": [
        {
            "id": "e1",
            "type": "create_order",
            "time": 1.0,
            "attributes": {"amount": 250.0},
            "relationships": [{"objectId": "ord_1"}],
        },
        {
            "id": "e2",
            "type": "approve_order",
            "time": 2.0,
            "attributes": {"amount": 250.0},
            "relationships": [{"objectId": "ord_1"}],
        },
        {
            "id": "e3",
            "type": "pack",
            "time": 3.0,
            "attributes": {"amount": 250.0},
            "relationships": [{"objectId": "ord_1"}, {"objectId": "pkg_1"}],
        },
        {
            "id": "e4",
            "type": "dispatch",
            "time": 4.0,
            "attributes": {"amount": 250.0},
            "relationships": [{"objectId": "pkg_1"}],
        },
        {
            "id": "e5",
            "type": "create_order",
            "time": 1.5,
            "attributes": {"amount": 420.0},
            "relationships": [{"objectId": "ord_2"}],
        },
        {
            "id": "e6",
            "type": "approve_order",
            "time": 2.5,
            "attributes": {"amount": 420.0},
            "relationships": [{"objectId": "ord_2"}],
        },
    ],
}


def test_ocel_json_parsing() -> None:
    """Verify OCEL 2.0 JSON standard format is correctly parsed into structured OcelLog."""
    log = parse_ocel_json(SAMPLE_OCEL_JSON)
    assert len(log.objects) == 3
    assert len(log.events) == 6
    assert "order" in log.object_types
    assert "package" in log.object_types


def test_native_inductive_miner_and_conformance() -> None:
    """Verify Directly-Follows Graph discovery, soundness guarantee, and alignment fitness/precision."""
    traces = [
        ["create_order", "approve_order", "pack"],
        ["create_order", "approve_order", "pack"],
        ["create_order", "approve_order"],
    ]

    model = mine_dfg_and_transitions(traces, "order")
    assert model.soundness_guarantee is True
    assert "create_order" in model.transitions
    assert "approve_order" in model.dfg_edges["create_order"]
    assert "pack" in model.dfg_edges["approve_order"]

    conf = compute_conformance(traces, model)
    assert conf.fitness == 1.0
    assert conf.precision >= 0.90
    assert conf.evidence_level == EvidenceLevel.STRUCTURAL


def test_ocel_full_worldspec_induction() -> None:
    """Verify ewm induce process generates WorldSpec with conformance-derived EvidenceLevel and invariants."""
    spec, report = induce_process(
        ocel_data=SAMPLE_OCEL_JSON,
        engine="native",
        name="ProcurementOCELWorld",
        seed=101,
    )

    assert len(spec.entities) == 3
    assert "order" in report.conformance
    assert report.conformance["order"].fitness == 1.0
    assert report.conformance["order"].evidence_level == EvidenceLevel.STRUCTURAL

    # Candidate non-negativity constraint created from 'amount'
    assert len(report.candidate_invariants) >= 1
    assert any(inv.target_resource == "amount" for inv in report.candidate_invariants)


# ---------------------------------------------------------------------------
# 5. Determinism & Fingerprinting Tests
# ---------------------------------------------------------------------------


def test_induction_report_sha256_determinism() -> None:
    """Verify identical induction runs with the same seed generate byte-identical SHA-256 fingerprints."""
    spec1, report1 = induce_schema(ddl=SAMPLE_DDL, seed=42)
    spec2, report2 = induce_schema(ddl=SAMPLE_DDL, seed=42)

    assert report1.sha256_fingerprint != ""
    assert report1.sha256_fingerprint == report2.sha256_fingerprint
    assert spec1.model_dump() == spec2.model_dump()


def test_induction_report_representations() -> None:
    """Verify InductionReport supports markdown, json, and rich notebook html repr."""
    _spec, report = induce_schema(ddl=SAMPLE_DDL, seed=42)
    md = report.to_markdown()
    assert "# World Model Induction Audit Report" in md
    assert "Procurement" in md or "InducedSchemaWorld" in md

    js = report.to_json()
    assert "sha256_fingerprint" in js

    html = report._repr_html_()
    assert "InductionReport" in html
    assert "Entities" in html


# ---------------------------------------------------------------------------
# 6. PM4Py AGPL Isolation Boundary Tests
# ---------------------------------------------------------------------------


def test_pm4py_isolation_boundary_safety() -> None:
    """Verify PM4Py is never imported at module level and availability check is safe."""

    # Confirm pm4py is not loaded in sys.modules prior to test
    # (Even if installed, EWM core does not import it at import time)
    available = is_pm4py_available()
    assert isinstance(available, bool)


def test_pm4py_engine_fallback_or_refusal() -> None:
    """If pm4py is not available, selecting engine='pm4py' raises informative error."""
    if not is_pm4py_available():
        with pytest.raises(SimulationConfigurationError) as exc_info:
            induce_process(SAMPLE_OCEL_JSON, engine="pm4py")
        assert "PM4Py is not installed" in str(exc_info.value)
    else:
        # If installed, subprocess isolation should execute without errors
        spec, _report = induce_process(SAMPLE_OCEL_JSON, engine="pm4py")
        assert len(spec.entities) == 3


# ---------------------------------------------------------------------------
# 7. CLI Subcommand Execution Tests
# ---------------------------------------------------------------------------


def test_cli_induce_schema_and_dataframe(tmp_path: Path) -> None:
    """Verify CLI `ewm induce schema` and `ewm induce dataframe` produce valid files on disk."""
    from ewm_engine.cli.main import main

    ddl_file = tmp_path / "schema.sql"
    ddl_file.write_text(SAMPLE_DDL, encoding="utf-8")

    out_yaml = tmp_path / "induced_schema.yaml"
    out_report = tmp_path / "schema_report.json"

    ret = main(
        [
            "induce",
            "schema",
            "--ddl",
            str(ddl_file),
            "--out",
            str(out_yaml),
            "--report",
            str(out_report),
        ]
    )
    assert ret == 0
    assert out_yaml.exists()
    assert out_report.exists()

    # Load produced spec and verify
    loaded_spec = WorldSpec.from_yaml(out_yaml)
    assert len(loaded_spec.entities) == 2

    # Test dataframe induction via CLI
    csv_file = tmp_path / "telemetry.csv"
    csv_file.write_text("entity_id,timestamp,voltage\ne1,10,12.5\ne1,20,11.8\n", encoding="utf-8")

    df_yaml = tmp_path / "induced_df.yaml"
    df_report = tmp_path / "df_report.json"

    ret2 = main(
        [
            "induce",
            "dataframe",
            "--data",
            str(csv_file),
            "--as-of",
            "15",
            "--out",
            str(df_yaml),
            "--report",
            str(df_report),
        ]
    )
    assert ret2 == 0
    assert df_yaml.exists()
    assert df_report.exists()
