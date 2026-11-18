"""Unit and integration tests for Prompt E2: Core State, Time, Action, and OCEL 2.0 Semantics.

Verifies:
1. SQLite schema initialization strictly adhering to the 24 OCEL 2.0 relational constraints.
2. `StateEncoder` encoding discrete events with $N:M$ Event-to-Object (E2O) and
   Object-to-Object (O2O) mappings (e.g. a single event affecting `User`, `Order`, `Invoice`)
   and lossless reconstruction via `pm4py.read_ocel2_sqlite`.
3. JSON, JSONL, and XML (`xmlocel`) event stream ingestion into normalized SQLite transactions.
4. Core invariant enforcement: no event can exist without a valid finite timestamp and
   at least one mapped object with a valid object type.
5. Parameterized SQL query safety against SQL injection payloads and context-managed cleanup.
6. Deprecation of legacy flat-log parsers (`parse_flat_event_log`, `FlatLogParser`).
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List
from uuid import UUID

import pytest

from ewm_engine.core.ocel import (
    OCEL2_BASE_TABLES,
    FlatLogParser,
    OcelEventRecord,
    OcelObjectRelation,
    OcelObjectTarget,
    StateEncoder,
    StateEncodingError,
    parse_flat_event_log,
    read_ocel2_sqlite,
    validate_ocel2_sqlite,
)
from ewm_engine.core.types import EngineError


def test_single_event_three_objects_e2o_o2o_and_pm4py_reconstruction() -> None:
    """Demonstrate a single event affecting User, Order, and Invoice is reconstructed via PM4Py without data loss."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "ocel2_three_objects.sqlite"

        with StateEncoder(db_path) as encoder:
            event_record = OcelEventRecord(
                event_id="evt_checkout_001",
                event_type="ConfirmCheckout",
                timestamp="2026-11-17T13:49:41Z",
                objects=(
                    OcelObjectTarget(
                        object_id="usr_101",
                        object_type="User",
                        qualifier="initiator",
                        attributes={"tier": "enterprise", "region": "NA"},
                    ),
                    OcelObjectTarget(
                        object_id="ord_202",
                        object_type="Order",
                        qualifier="subject",
                        attributes={"items_count": 4, "priority": "high"},
                    ),
                    OcelObjectTarget(
                        object_id="inv_303",
                        object_type="Invoice",
                        qualifier="generated",
                        attributes={"tax_rate": 0.08, "status": "issued"},
                    ),
                ),
                attributes={
                    "total_amount": 1250.50,
                    "currency": "USD",
                    "channel": "b2b_portal",
                },
                object_relations=(
                    OcelObjectRelation(
                        source_id="usr_101",
                        target_id="ord_202",
                        qualifier="places",
                    ),
                    OcelObjectRelation(
                        source_id="ord_202",
                        target_id="inv_303",
                        qualifier="billed_by",
                    ),
                ),
            )
            encoder.encode_event(event_record)

            tables = encoder.list_tables()
            for base_tbl in OCEL2_BASE_TABLES:
                assert base_tbl in tables
            assert "event_ConfirmCheckout" in tables
            assert "object_User" in tables
            assert "object_Order" in tables
            assert "object_Invoice" in tables

            # Verify all 24 standard OCEL 2.0 relational constraints pass
            satisfied, unsatisfied = encoder.validate_schema()
            assert len(satisfied) == 24
            assert unsatisfied == []

        # Reconstruct via pm4py.read_ocel2_sqlite after context manager closes DB
        ocel = read_ocel2_sqlite(db_path, validate=True)

        # 1. Verify Event reconstruction without data loss
        assert len(ocel.events) == 1
        ev_row = ocel.events.iloc[0]
        assert str(ev_row["ocel:eid"]) == "evt_checkout_001"
        assert str(ev_row["ocel:activity"]) == "ConfirmCheckout"
        assert float(ev_row["total_amount"]) == pytest.approx(1250.50)
        assert str(ev_row["currency"]) == "USD"
        assert str(ev_row["channel"]) == "b2b_portal"

        # 2. Verify all 3 distinct Objects (User, Order, Invoice) reconstructed without data loss
        assert len(ocel.objects) == 3
        obj_by_id: Dict[str, Any] = {
            str(row["ocel:oid"]): row for _, row in ocel.objects.iterrows()
        }
        assert set(obj_by_id.keys()) == {"usr_101", "ord_202", "inv_303"}
        assert str(obj_by_id["usr_101"]["ocel:type"]) == "User"
        assert str(obj_by_id["usr_101"]["tier"]) == "enterprise"
        assert str(obj_by_id["ord_202"]["ocel:type"]) == "Order"
        assert int(obj_by_id["ord_202"]["items_count"]) == 4
        assert str(obj_by_id["inv_303"]["ocel:type"]) == "Invoice"
        assert float(obj_by_id["inv_303"]["tax_rate"]) == pytest.approx(0.08)

        # 3. Verify E2O relations mapping the single event to all 3 objects
        assert len(ocel.relations) == 3
        e2o_triples = {
            (
                str(r["ocel:eid"]),
                str(r["ocel:oid"]),
                str(r["ocel:type"]),
                str(r["ocel:qualifier"]),
            )
            for _, r in ocel.relations.iterrows()
        }
        assert e2o_triples == {
            ("evt_checkout_001", "usr_101", "User", "initiator"),
            ("evt_checkout_001", "ord_202", "Order", "subject"),
            ("evt_checkout_001", "inv_303", "Invoice", "generated"),
        }

        # 4. Verify O2O relations
        assert ocel.o2o is not None
        assert len(ocel.o2o) == 2
        o2o_triples = {
            (
                str(r["ocel:oid"]),
                str(r["ocel:oid_2"]),
                str(r["ocel:qualifier"]),
            )
            for _, r in ocel.o2o.iterrows()
        }
        assert o2o_triples == {
            ("usr_101", "ord_202", "places"),
            ("ord_202", "inv_303", "billed_by"),
        }


def test_nm_cardinality_convergence_and_divergence_resolution(tmp_path: Path) -> None:
    """Verify N:M Event-to-Object cardinality resolves convergence without duplicating event records."""
    db_path = tmp_path / "ocel2_nm_cardinality.sqlite"

    events_stream: List[Dict[str, Any]] = [
        {
            "event_id": "e1_create",
            "event_type": "CreateOrder",
            "timestamp": 1700000000.0,
            "objects": [
                {"object_id": "u_1", "object_type": "User", "qualifier": "buyer"},
                {"object_id": "o_1", "object_type": "Order", "qualifier": "created"},
                {"object_id": "i_1", "object_type": "Invoice", "qualifier": "drafted"},
            ],
            "attributes": {"step_idx": 1},
        },
        {
            "event_id": "e2_fulfill",
            "event_type": "FulfillSplitShipment",
            "timestamp": 1700000060.0,
            "objects": [
                {"object_id": "o_1", "object_type": "Order", "qualifier": "fulfilled"},
                {"object_id": "i_1", "object_type": "Invoice", "qualifier": "finalized"},
                {"object_id": "s_1", "object_type": "Shipment", "qualifier": "parcel_a"},
                {"object_id": "s_2", "object_type": "Shipment", "qualifier": "parcel_b"},
            ],
            "attributes": {"step_idx": 2},
        },
        {
            "event_id": "e3_settle",
            "event_type": "SettlePayment",
            "timestamp": 1700000120.0,
            "objects": [
                {"object_id": "u_1", "object_type": "User", "qualifier": "payer"},
                {"object_id": "i_1", "object_type": "Invoice", "qualifier": "settled"},
            ],
            "attributes": {"step_idx": 3},
        },
    ]

    with StateEncoder(db_path) as encoder:
        encoded = encoder.encode_events(events_stream)
        assert len(encoded) == 3

        # Confirm each event appears once in `event` while `event_object` stores all 9 N:M edges
        cur = encoder.connection.cursor()
        cur.execute('SELECT Count(*) FROM "event";')
        assert cur.fetchone()[0] == 3
        cur.execute('SELECT Count(*) FROM "object";')
        assert cur.fetchone()[0] == 5
        cur.execute('SELECT Count(*) FROM "event_object";')
        assert cur.fetchone()[0] == 9

    satisfied, unsatisfied = validate_ocel2_sqlite(db_path)
    assert len(satisfied) == 24
    assert unsatisfied == []

    ocel = read_ocel2_sqlite(db_path)
    assert len(ocel.events) == 3
    assert len(ocel.objects) == 5
    assert len(ocel.relations) == 9


def test_json_and_xml_stream_ingestion_and_validation(tmp_path: Path) -> None:
    """Verify JSON, JSONL, and XML OCEL 2.0 streams encode into valid SQLite databases."""
    # 1. JSON stream test
    json_db = tmp_path / "from_json.sqlite"
    ocel_json_payload = {
        "objectTypes": [{"name": "User"}, {"name": "Order"}, {"name": "Invoice"}],
        "eventTypes": [{"name": "SubmitOrder"}, {"name": "IssueInvoice"}],
        "objects": [
            {
                "id": "usr_1",
                "type": "User",
                "attributes": [{"name": "role", "value": "procurement"}],
                "relationships": [{"objectId": "ord_1", "qualifier": "owns"}],
            },
            {
                "id": "ord_1",
                "type": "Order",
                "attributes": [{"name": "total", "value": 500.0}],
                "relationships": [{"objectId": "inv_1", "qualifier": "generates"}],
            },
            {
                "id": "inv_1",
                "type": "Invoice",
                "attributes": [{"name": "due_days", "value": 30}],
            },
        ],
        "events": [
            {
                "id": "ev_submit",
                "type": "SubmitOrder",
                "time": "2026-11-17T10:00:00Z",
                "attributes": [{"name": "portal", "value": "web"}],
                "relationships": [
                    {"objectId": "usr_1", "qualifier": "actor"},
                    {"objectId": "ord_1", "qualifier": "target"},
                    {"objectId": "inv_1", "qualifier": "linked_invoice"},
                ],
            },
            {
                "id": "ev_invoice",
                "type": "IssueInvoice",
                "time": "2026-11-17T10:05:00Z",
                "attributes": [{"name": "auto_approved", "value": True}],
                "relationships": [
                    {"objectId": "ord_1", "qualifier": "source_order"},
                    {"objectId": "inv_1", "qualifier": "issued_doc"},
                ],
            },
        ],
    }

    with StateEncoder(json_db) as encoder:
        count = encoder.encode_json(json.dumps(ocel_json_payload))
        assert count == 2
        sat, unsat = encoder.validate_schema()
        assert len(sat) == 24 and unsat == []

    ocel_from_json = read_ocel2_sqlite(json_db)
    assert len(ocel_from_json.events) == 2
    assert len(ocel_from_json.objects) == 3
    assert len(ocel_from_json.relations) == 5

    # 2. XML stream test
    xml_db = tmp_path / "from_xml.sqlite"
    ocel_xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <log>
        <objects>
            <object id="u_xml" type="User">
                <attributes>
                    <attribute name="department" value="finance"/>
                </attributes>
                <objects>
                    <relationship object-id="o_xml" qualifier="creates"/>
                </objects>
            </object>
            <object id="o_xml" type="Order">
                <attributes>
                    <attribute name="amount" value="900.00"/>
                </attributes>
            </object>
            <object id="i_xml" type="Invoice">
                <attributes>
                    <attribute name="currency" value="EUR"/>
                </attributes>
            </object>
        </objects>
        <events>
            <event id="e_xml_1" type="ProcessOrder" time="2026-11-17T12:00:00Z">
                <attributes>
                    <attribute name="verified" value="true"/>
                </attributes>
                <objects>
                    <relationship object-id="u_xml" qualifier="initiator"/>
                    <relationship object-id="o_xml" qualifier="order_ref"/>
                    <relationship object-id="i_xml" qualifier="invoice_ref"/>
                </objects>
            </event>
        </events>
    </log>
    """

    with StateEncoder(xml_db) as encoder:
        xml_count = encoder.encode_xml(ocel_xml_payload)
        assert xml_count == 1
        sat_xml, unsat_xml = encoder.validate_schema()
        assert len(sat_xml) == 24 and unsat_xml == []

    ocel_from_xml = read_ocel2_sqlite(xml_db)
    assert len(ocel_from_xml.events) == 1
    assert len(ocel_from_xml.objects) == 3
    assert len(ocel_from_xml.relations) == 3


def test_core_invariants_reject_missing_timestamp_or_unmapped_objects(tmp_path: Path) -> None:
    """Verify that no event can exist without a valid timestamp and mapped object type."""
    db_path = tmp_path / "invariants.sqlite"

    with StateEncoder(db_path) as encoder:
        # 1. Missing or invalid timestamps must raise StateEncodingError (subclass of EngineError)
        for bad_ts in (None, "", "   ", "not-a-date", float("nan"), float("inf"), float("-inf")):
            with pytest.raises(StateEncodingError) as exc_info:
                encoder.encode_event(
                    {
                        "event_id": "evt_bad_ts",
                        "event_type": "BadTimestampEvent",
                        "timestamp": bad_ts,
                        "objects": [{"object_id": "o1", "object_type": "Order"}],
                    }
                )
            assert isinstance(exc_info.value, EngineError)

        # 2. Event with no mapped objects must be rejected
        with pytest.raises(StateEncodingError, match="at least one mapped object"):
            encoder.encode_event(
                {
                    "event_id": "evt_no_objs",
                    "event_type": "OrphanEvent",
                    "timestamp": "2026-11-17T13:00:00Z",
                    "objects": [],
                }
            )

        # 3. Event with object lacking object_type must be rejected
        with pytest.raises(StateEncodingError, match="unmapped object_type"):
            encoder.encode_event(
                {
                    "event_id": "evt_untyped_obj",
                    "event_type": "UntypedObjectEvent",
                    "timestamp": "2026-11-17T13:00:00Z",
                    "objects": [{"object_id": "o_untyped", "object_type": ""}],
                }
            )

        # Verify failed transactions rolled back cleanly leaving 0 events in DB
        cur = encoder.connection.execute('SELECT Count(*) FROM "event";')
        assert cur.fetchone()[0] == 0


def test_sql_injection_prevention_and_attribute_serialization_edge_cases(tmp_path: Path) -> None:
    """Verify parameterized SQL prevents injection and complex/edge-case attributes are handled."""
    db_path = tmp_path / "security_and_edge_cases.sqlite"
    malicious_id = "ord_1'); DROP TABLE event; --"
    malicious_note = "100%; DELETE FROM object; --"

    with StateEncoder(db_path) as encoder:
        encoder.encode_event(
            {
                "event_id": "evt_sec_1",
                "event_type": "SecurityAudit",
                "timestamp": datetime(2026, 11, 17, 13, 49, 41, tzinfo=UTC),
                "objects": [
                    {
                        "object_id": malicious_id,
                        "object_type": "Order",
                        "qualifier": "audited",
                        "attributes": {
                            "decimal_amount": Decimal("9999.95"),
                            "trace_uuid": UUID("12345678-1234-5678-1234-567812345678"),
                            "nested_meta": {"tags": ["a", "b"], "verified": True},
                        },
                    }
                ],
                "attributes": {
                    "note": malicious_note,
                    "is_flagged": False,
                },
            }
        )

        # Non-finite float or Decimal in attributes must be rejected
        with pytest.raises(StateEncodingError, match="Non-finite"):
            encoder.encode_event(
                {
                    "event_id": "evt_nan_attr",
                    "event_type": "SecurityAudit",
                    "timestamp": "2026-11-17T14:00:00Z",
                    "objects": [{"object_id": "o_ok", "object_type": "Order"}],
                    "attributes": {"bad_metric": float("nan")},
                }
            )

        with pytest.raises(StateEncodingError, match="Non-finite Decimal"):
            encoder.encode_event(
                {
                    "event_id": "evt_nan_dec",
                    "event_type": "SecurityAudit",
                    "timestamp": "2026-11-17T14:00:00Z",
                    "objects": [{"object_id": "o_ok", "object_type": "Order"}],
                    "attributes": {"bad_dec": Decimal("NaN")},
                }
            )

    ocel = read_ocel2_sqlite(db_path)
    assert len(ocel.events) == 1
    assert len(ocel.objects) == 1
    assert str(ocel.objects.iloc[0]["ocel:oid"]) == malicious_id
    assert str(ocel.events.iloc[0]["note"]) == malicious_note


def test_deprecated_legacy_flat_log_parsers_emit_warning() -> None:
    """Verify legacy flat-log parsers are retained for compatibility and marked @deprecated."""
    raw_flat = [
        {"event_id": "e1", "case_id": "c1", "activity": "start", "timestamp": 1700000000.0},
    ]
    with pytest.deprecated_call():
        parsed = parse_flat_event_log(raw_flat)
    assert len(parsed) == 1
    assert parsed[0]["case_id"] == "c1"

    parser = FlatLogParser()
    with pytest.deprecated_call():
        parsed_via_cls = parser.parse(raw_flat)
    assert parsed_via_cls == parsed
