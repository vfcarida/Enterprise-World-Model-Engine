"""Verification suite for Prompt E5: Data Interfaces, Interoperability, and PM4Py.

Verifies:
1. `export_to_ocel_json(filepath)` utilizing `pm4py.write_ocel2_json`, exporting a populated
   SQLite OCEL 2.0 state to JSON, parsing the JSON in a fresh `StateEncoder` instance, and
   asserting 100% deep structural equality (`equals()` and `to_canonical_snapshot()`).
2. ISO-8601 timestamp precision (including microsecond fidelity) across SQLite -> JSON -> SQLite.
3. External OCEL 2.0 JSON schema validation (`validate_ocel2_json` via `jsonschema` and
   `pm4py.objects.ocel.validation.jsonocel.apply`).
4. CSV event stream ingestion (`ingest_csv_event_stream` / `StateEncoder.ingest_csv`) mapping
   standard CSV streams into relational E2O and O2O tables.
5. Legacy XES flat log ingestion (`ingest_xes_event_log` / `StateEncoder.ingest_xes`) and
   OCEL 2.0 XML export (`export_to_ocel_xml`).
6. Operational safeguards: chunked streaming export and `max_payload_bytes` limit enforcement.
7. Pandas DataFrame interoperability for downstream data science workflows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pm4py
import pytest

from ewm_engine.core.ocel import (
    OcelEventRecord,
    OcelObjectRelation,
    OcelObjectTarget,
    StateEncoder,
    StateEncodingError,
    export_to_ocel_json,
    export_to_ocel_xml,
    ingest_csv_event_stream,
    ingest_xes_event_log,
    validate_ocel2_json,
    validate_ocel2_sqlite,
)


def _populate_sample_ocel_state(encoder: StateEncoder) -> None:
    """Populate `encoder` with multi-object events, O2O links, and microsecond ISO-8601 timestamps."""
    encoder.encode_event(
        OcelEventRecord(
            event_id="evt_e5_001",
            event_type="ConfirmCheckout",
            timestamp="2026-11-17T13:50:22.123456Z",
            objects=(
                OcelObjectTarget(
                    object_id="usr_501",
                    object_type="User",
                    qualifier="initiator",
                    attributes={"tier": "platinum", "country": "US"},
                ),
                OcelObjectTarget(
                    object_id="ord_601",
                    object_type="Order",
                    qualifier="subject",
                    attributes={"items_count": 3, "priority": "urgent"},
                ),
                OcelObjectTarget(
                    object_id="inv_701",
                    object_type="Invoice",
                    qualifier="generated",
                    attributes={"amount": 1850.75, "currency": "USD"},
                ),
            ),
            attributes={
                "total_amount": 1850.75,
                "payment_method": "wire",
                "channel": "enterprise_api",
            },
            object_relations=(
                OcelObjectRelation(
                    source_id="usr_501",
                    target_id="ord_601",
                    qualifier="places",
                ),
                OcelObjectRelation(
                    source_id="ord_601",
                    target_id="inv_701",
                    qualifier="billed_by",
                ),
            ),
        )
    )
    encoder.encode_event(
        OcelEventRecord(
            event_id="evt_e5_002",
            event_type="DispatchShipment",
            timestamp="2026-11-17T13:55:10.987654Z",
            objects=(
                OcelObjectTarget(
                    object_id="ord_601",
                    object_type="Order",
                    qualifier="dispatched_order",
                ),
                OcelObjectTarget(
                    object_id="shp_801",
                    object_type="Shipment",
                    qualifier="carrier_leg",
                    attributes={"carrier": "freight_air", "weight_kg": 42.5},
                ),
            ),
            attributes={
                "warehouse_code": "WH-EAST-01",
                "expedited_code": 1,
            },
            object_relations=(
                OcelObjectRelation(
                    source_id="ord_601",
                    target_id="shp_801",
                    qualifier="fulfilled_by",
                ),
            ),
        )
    )
    encoder.encode_event(
        OcelEventRecord(
            event_id="evt_e5_003",
            event_type="SettleInvoice",
            timestamp="2026-11-17T14:00:00Z",
            objects=(
                OcelObjectTarget(
                    object_id="usr_501",
                    object_type="User",
                    qualifier="payer",
                ),
                OcelObjectTarget(
                    object_id="inv_701",
                    object_type="Invoice",
                    qualifier="settled_invoice",
                ),
            ),
            attributes={
                "settled_amount": 1850.75,
                "settlement_ref": "TXN-99012",
            },
        )
    )


def test_export_to_ocel_json_roundtrip_deep_equality_and_schema_validation(
    tmp_path: Path,
) -> None:
    """Export populated SQLite state to JSON, parse in fresh instance, and assert deep equality."""
    src_db = tmp_path / "source_state.sqlite"
    exported_json = tmp_path / "exported_state.jsonocel"
    reloaded_db = tmp_path / "reloaded_state.sqlite"

    with StateEncoder(src_db) as src_encoder:
        _populate_sample_ocel_state(src_encoder)
        out_path = src_encoder.export_to_ocel_json(exported_json)
        assert out_path == exported_json
        assert exported_json.exists()
        assert exported_json.stat().st_size > 500

        # Validate exported JSON against external OCEL 2.0 JSON Schema and PM4Py jsonocel validator
        assert validate_ocel2_json(exported_json) is True

        # Parse the exported JSON in a fresh StateEncoder instance
        with StateEncoder(reloaded_db) as dst_encoder:
            ingested_count = dst_encoder.encode_json(exported_json)
            assert ingested_count == 3

            # Assert 24/24 OCEL 2.0 relational constraints on the reloaded SQLite store
            sat, unsat = dst_encoder.validate_schema()
            assert len(sat) == 24
            assert unsat == []

            # Assert deep structural equality between source and reloaded instances
            src_snapshot = src_encoder.to_canonical_snapshot()
            dst_snapshot = dst_encoder.to_canonical_snapshot()
            assert src_snapshot == dst_snapshot
            assert src_encoder.equals(dst_encoder)

            # Verify exact ISO-8601 microsecond timestamp precision was preserved
            assert dst_snapshot["events"]["evt_e5_001"]["time"] == "2026-11-17T13:50:22.123456Z"
            assert dst_snapshot["events"]["evt_e5_002"]["time"] == "2026-11-17T13:55:10.987654Z"
            assert dst_snapshot["events"]["evt_e5_003"]["time"] == "2026-11-17T14:00:00Z"


def test_csv_event_stream_ingestion_into_e2o_and_o2o_schema(tmp_path: Path) -> None:
    """Verify CSV event stream ingestion maps rows safely into relational E2O and O2O tables."""
    csv_file = tmp_path / "events_stream.csv"
    csv_content = (
        "event_id,event_type,timestamp,user_id,order_id,invoice_id,objects,o2o,amount,channel\n"
        "csv_e1,CreateOrder,2026-11-17T09:15:30.250000Z,u_10,o_20,i_30,"
        "Warehouse:wh_1:origin,o_20->wh_1:routed_through,499.95,web\n"
        "csv_e2,ApproveOrder,2026-11-17T09:16:45.500000Z,u_10,o_20,i_30,"
        "Warehouse:wh_1:origin,,499.95,api\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    db_path = tmp_path / "csv_ingested.sqlite"
    count = ingest_csv_event_stream(
        csv_file,
        db_path,
        object_columns={
            "user_id": "User",
            "order_id": "Order",
            "invoice_id": "Invoice",
        },
        o2o_rules=[
            ("user_id", "order_id", "places"),
            ("order_id", "invoice_id", "bills"),
        ],
        batch_size=1,
    )
    assert count == 2

    sat, unsat = validate_ocel2_sqlite(db_path)
    assert len(sat) == 24 and unsat == []

    json_out = tmp_path / "csv_exported.jsonocel"
    export_to_ocel_json(db_path, json_out)
    assert validate_ocel2_json(json_out) is True

    with StateEncoder(db_path) as encoder:
        snap = encoder.to_canonical_snapshot()
        assert snap["events"]["csv_e1"]["time"] == "2026-11-17T09:15:30.250000Z"
        assert snap["events"]["csv_e2"]["time"] == "2026-11-17T09:16:45.500000Z"
        assert len(snap["objects"]) == 4  # u_10, o_20, i_30, wh_1
        assert ("u_10", "o_20", "places") in snap["o2o"]
        assert ("o_20", "i_30", "bills") in snap["o2o"]
        assert ("o_20", "wh_1", "routed_through") in snap["o2o"]


def test_legacy_xes_ingestion_and_xml_json_export(tmp_path: Path) -> None:
    """Verify legacy XES flat log ingestion into OCEL 2.0 SQLite and export to JSON/XML."""
    xes_file = tmp_path / "legacy_log.xes"
    xes_xml = """<?xml version="1.0" encoding="UTF-8" ?>
    <log xes.version="1.0" xmlns="http://www.xes-standard.org/">
        <trace>
            <string key="concept:name" value="CASE-1001"/>
            <string key="customer_segment" value="enterprise"/>
            <event>
                <string key="ocel:eid" value="xes_e1"/>
                <string key="concept:name" value="ReceiveRequest"/>
                <date key="time:timestamp" value="2026-11-17T08:00:00.500000Z"/>
                <string key="org:resource" value="Agent-Alice"/>
                <float key="cost" value="45.0"/>
            </event>
            <event>
                <string key="ocel:eid" value="xes_e2"/>
                <string key="concept:name" value="CompleteRequest"/>
                <date key="time:timestamp" value="2026-11-17T08:30:00Z"/>
                <string key="org:resource" value="Agent-Bob"/>
                <float key="cost" value="120.5"/>
            </event>
        </trace>
    </log>
    """
    xes_file.write_text(xes_xml, encoding="utf-8")

    db_path = tmp_path / "from_xes.sqlite"
    ingested = ingest_xes_event_log(xes_file, db_path)
    assert ingested == 2

    sat, unsat = validate_ocel2_sqlite(db_path)
    assert len(sat) == 24 and unsat == []

    json_path = tmp_path / "xes_to_ocel.jsonocel"
    xml_path = tmp_path / "xes_to_ocel.xmlocel"
    export_to_ocel_json(db_path, json_path)
    export_to_ocel_xml(db_path, xml_path)

    assert validate_ocel2_json(json_path) is True
    assert xml_path.exists() and xml_path.stat().st_size > 200


def test_export_payload_size_safeguard_and_pandas_loading(tmp_path: Path) -> None:
    """Verify max_payload_bytes safeguard prevents oversized exports and test Pandas loading."""
    db_path = tmp_path / "safeguard.sqlite"
    json_ok = tmp_path / "valid.jsonocel"
    json_too_large = tmp_path / "oversized.jsonocel"

    with StateEncoder(db_path) as encoder:
        _populate_sample_ocel_state(encoder)

        # 1. Exceeding max_payload_bytes raises StateEncodingError and removes partial file
        with pytest.raises(StateEncodingError, match="max_payload_bytes"):
            encoder.export_to_ocel_json(json_too_large, max_payload_bytes=128)
        assert not json_too_large.exists()

        # 2. Normal export succeeds and can be loaded into Pandas DataFrames
        encoder.export_to_ocel_json(json_ok)

    # Load into Pandas via pm4py and json_normalize (as documented for data scientists)
    ocel = pm4py.read_ocel2_json(str(json_ok))
    assert isinstance(ocel.events, pd.DataFrame)
    assert isinstance(ocel.objects, pd.DataFrame)
    assert isinstance(ocel.relations, pd.DataFrame)
    assert len(ocel.events) == 3
    assert len(ocel.objects) == 4

    raw_payload = json.loads(json_ok.read_text(encoding="utf-8"))
    flat_events_df = pd.json_normalize(raw_payload["events"])
    assert len(flat_events_df) == 3
    assert list(flat_events_df["id"]) == ["evt_e5_001", "evt_e5_002", "evt_e5_003"]
