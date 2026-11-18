"""Object-Centric Event Log (OCEL 2.0) relational state schemas, encoder, and PM4Py retrieval.

Implements the OCEL 2.0 relational specification (arXiv:2403.01975) backed by SQLite
to represent multi-dimensional enterprise state transitions without the divergence and
convergence errors of traditional case-centric flat logs.

Relational Schema & Mathematical Cardinality Contracts (OCEL 2.0)
-----------------------------------------------------------------
1. **Type Mapping Tables (`event_map_type`, `object_map_type`)**:
   - ``event_map_type(ocel_type TEXT PRIMARY KEY, ocel_type_map TEXT NOT NULL UNIQUE)``
   - ``object_map_type(ocel_type TEXT PRIMARY KEY, ocel_type_map TEXT NOT NULL UNIQUE)``
   Maps domain event/object types to normalized SQLite table suffixes (`event_<map>`, `object_<map>`).

2. **Entity Tables (`event`, `object`)**:
   - ``event(ocel_id TEXT PRIMARY KEY, ocel_type TEXT NOT NULL REFERENCES event_map_type(ocel_type))``
   - ``object(ocel_id TEXT PRIMARY KEY, ocel_type TEXT NOT NULL REFERENCES object_map_type(ocel_type))``

3. **Event-to-Object (E2O) Relation (`event_object`) — $N:M$ Cardinality**:
   - ``event_object(ocel_event_id TEXT, ocel_object_id TEXT, ocel_qualifier TEXT, PRIMARY KEY (ocel_event_id, ocel_object_id, ocel_qualifier))``
   - Foreign keys enforce ``ocel_event_id -> event(ocel_id)`` and ``ocel_object_id -> object(ocel_id)``.
   - Mathematically represents a bipartite relation $R_{E2O} \\subseteq E \\times O \\times Q$ with
     many-to-many ($N:M$) cardinality: a single event $e \\in E$ may simultaneously bind to
     $M \\ge 1$ distinct objects $(o_1, \\dots, o_M)$ across heterogeneous types (e.g., `User`,
     `Order`, `Invoice`), and each object $o \\in O$ may participate in $N \\ge 1$ events over
     its lifecycle, eliminating convergence (duplicate event replication) and divergence errors.

4. **Object-to-Object (O2O) Relation (`object_object`) — $N:M$ Cardinality**:
   - ``object_object(ocel_source_id TEXT, ocel_target_id TEXT, ocel_qualifier TEXT, PRIMARY KEY (ocel_source_id, ocel_target_id, ocel_qualifier))``
   - Foreign keys enforce ``ocel_source_id -> object(ocel_id)`` and ``ocel_target_id -> object(ocel_id)``.
   - Represents a directed labeled multi-graph $R_{O2O} \\subseteq O \\times O \\times Q$ over objects.

5. **Type-Specific Attribute & Temporal Tables (`event_<type>`, `object_<type>`)**:
   - ``event_<map>(ocel_id TEXT PRIMARY KEY REFERENCES event(ocel_id), ocel_time TIMESTAMP NOT NULL, ...)``
   - ``object_<map>(ocel_id TEXT NOT NULL REFERENCES object(ocel_id), ocel_time TIMESTAMP, ocel_changed_field TEXT, ...)``

6. **Core State Invariant**:
   No event may exist in the OCEL 2.0 store without:
   (a) a valid, finite timestamp (`ocel_time`), and
   (b) at least one mapped object with a valid registered `object_type` in `event_object`.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import sqlite3
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.deprecation import deprecated
from ewm_engine.exceptions import InvalidWorldStateError

_IDENTIFIER_SANITIZE_RE = re.compile(r"[^a-zA-Z0-9_]+")
_VALID_COLUMN_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

OCEL2_BASE_TABLES: Tuple[str, ...] = (
    "event_map_type",
    "object_map_type",
    "event",
    "object",
    "event_object",
    "object_object",
)


class StateEncodingError(InvalidWorldStateError):
    """Raised when an event or object violates OCEL 2.0 relational invariants or serialization rules."""


class OcelObjectTarget(BaseModel):
    """Typed specification of an object participating in an OCEL 2.0 event (E2O binding)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    object_id: str = Field(min_length=1, description="Unique identifier of the target object.")
    object_type: str = Field(min_length=1, description="Mapped object type name.")
    qualifier: str = Field(
        default="participant",
        description="E2O role qualifier (e.g., 'initiator', 'subject', 'target').",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional object attribute values or updates at event time.",
    )


class OcelObjectRelation(BaseModel):
    """Typed specification of a directed Object-to-Object (O2O) relationship."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1, description="Source object identifier.")
    target_id: str = Field(min_length=1, description="Target object identifier.")
    qualifier: str = Field(
        default="relates_to",
        description="O2O relationship qualifier (e.g., 'places', 'bills_for', 'contains').",
    )


class OcelEventRecord(BaseModel):
    """Immutable discrete system event record for OCEL 2.0 state encoding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(min_length=1, description="Globally unique event identifier.")
    event_type: str = Field(min_length=1, description="Event activity or classification type.")
    timestamp: Union[str, float, int, datetime] = Field(
        description="Valid ISO-8601 string, datetime, or finite epoch timestamp."
    )
    objects: Tuple[OcelObjectTarget, ...] = Field(
        default_factory=tuple,
        description="N:M Event-to-Object (E2O) bindings associated with this event.",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-level payload attributes.",
    )
    object_relations: Tuple[OcelObjectRelation, ...] = Field(
        default_factory=tuple,
        description="Optional Object-to-Object (O2O) relationships established by this event.",
    )


def _quote_sql_ident(identifier: str) -> str:
    """Safely quote a validated SQLite table or column identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _safe_parse_xml_root(xml_text: str) -> ET.Element:
    """Safely parse XML string after rejecting DOCTYPE and ENTITY declarations (XXE defense)."""
    cleaned = xml_text.strip()
    upper = cleaned.upper()
    if "<!DOCTYPE" in upper or "<!ENTITY" in upper:
        raise StateEncodingError(
            "Security violation: XML DOCTYPE and ENTITY declarations are forbidden."
        )
    return ET.fromstring(cleaned)  # nosec B314


def _sanitize_type_map_name(type_name: str, existing_maps: Dict[str, str]) -> str:
    """Normalize an event or object type name into an injective, SQLite-safe table suffix."""
    raw = str(type_name).strip()
    if not raw:
        raise StateEncodingError("OCEL 2.0 type name cannot be empty or whitespace.")
    if raw in existing_maps:
        return existing_maps[raw]

    cleaned = _IDENTIFIER_SANITIZE_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = "Type"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"

    used_lower = {v.casefold() for v in existing_maps.values()}
    candidate = cleaned
    suffix = 2
    while candidate.casefold() in used_lower or candidate.casefold() in {"map_type", "object"}:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1
    return candidate


def _sanitize_attribute_column(attr_name: str) -> str:
    """Validate and normalize an attribute key into a safe SQLite column identifier."""
    cleaned = _IDENTIFIER_SANITIZE_RE.sub("_", str(attr_name).strip()).strip("_")
    if not cleaned:
        raise StateEncodingError(f"Invalid empty attribute column name: {attr_name!r}")
    if cleaned[0].isdigit():
        cleaned = f"attr_{cleaned}"
    reserved = {"ocel_id", "ocel_time", "ocel_type", "ocel_type_map", "ocel_changed_field"}
    if cleaned.casefold() in reserved:
        cleaned = f"attr_{cleaned}"
    if not _VALID_COLUMN_RE.match(cleaned):
        raise StateEncodingError(f"Attribute name {attr_name!r} could not be normalized safely.")
    return cleaned


def _format_iso8601_utc(dt: datetime) -> str:
    """Format a datetime as a canonical ISO-8601 UTC string preserving microsecond precision."""
    dt_utc = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    if dt_utc.microsecond:
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_timestamp(raw_ts: Any) -> str:
    """Validate and normalize a timestamp into an ISO-8601 UTC string for SQLite TIMESTAMP storage.

    Enforces the OCEL 2.0 invariant that no event can exist without a valid, finite timestamp,
    and maintains full ISO-8601 sub-second precision across serialization boundaries (E5).
    """
    if raw_ts is None or isinstance(raw_ts, bool):
        raise StateEncodingError("Event timestamp cannot be None or boolean.")

    if isinstance(raw_ts, datetime):
        return _format_iso8601_utc(raw_ts)

    if isinstance(raw_ts, (int, float)):
        val = float(raw_ts)
        if not math.isfinite(val):
            raise StateEncodingError(f"Event timestamp must be finite, got {raw_ts!r}.")
        dt = datetime.fromtimestamp(val, tz=UTC)
        return _format_iso8601_utc(dt)

    if isinstance(raw_ts, str):
        s = raw_ts.strip()
        if not s:
            raise StateEncodingError("Event timestamp string cannot be empty.")
        # Check if numeric string
        try:
            num_val = float(s)
            if not math.isfinite(num_val):
                raise StateEncodingError(f"Event timestamp must be finite, got {raw_ts!r}.")
            dt = datetime.fromtimestamp(num_val, tz=UTC)
            return _format_iso8601_utc(dt)
        except ValueError:
            pass

        iso_candidate = s[:-1] + "+00:00" if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(iso_candidate)
            return _format_iso8601_utc(dt)
        except ValueError as exc:
            raise StateEncodingError(
                f"Invalid timestamp format {raw_ts!r}; expected ISO-8601 string or epoch seconds."
            ) from exc

    raise StateEncodingError(f"Unsupported timestamp type: {type(raw_ts).__name__}")


def _serialize_attribute_value(val: Any) -> Tuple[Union[str, int, float, None], str]:
    """Serialize an attribute value into a SQLite-compatible scalar and return `(value, sql_type)`.

    Handles edge cases:
    - Rejects non-finite floats (`NaN`, `Inf`, `-Inf`) with `StateEncodingError`.
    - Serializes `Decimal` and `UUID` as `TEXT`.
    - Serializes `datetime` as ISO-8601 `TEXT`.
    - Serializes nested `dict`, `list`, `tuple` structures as canonical JSON `TEXT`.
    """
    if val is None:
        return None, "TEXT"
    if isinstance(val, bool):
        return int(val), "INTEGER"
    if isinstance(val, int):
        return val, "INTEGER"
    if isinstance(val, float):
        if not math.isfinite(val):
            raise StateEncodingError(f"Non-finite float attribute value {val!r} is forbidden.")
        return val, "REAL"
    if isinstance(val, str):
        return val, "TEXT"
    if isinstance(val, (Decimal, UUID)):
        if isinstance(val, Decimal) and not val.is_finite():
            raise StateEncodingError(f"Non-finite Decimal attribute value {val!r} is forbidden.")
        return str(val), "TEXT"
    if isinstance(val, datetime):
        return _normalize_timestamp(val), "TEXT"
    if isinstance(val, (Mapping, Sequence)) and not isinstance(val, (bytes, bytearray)):
        try:
            return json.dumps(val, sort_keys=True), "TEXT"
        except (TypeError, ValueError) as exc:
            raise StateEncodingError(f"Unserializable container attribute {val!r}: {exc}") from exc

    raise StateEncodingError(
        f"Unsupported attribute value type '{type(val).__name__}' for OCEL serialization."
    )


class StateEncoder:
    """Relational OCEL 2.0 state encoder writing normalized, immutable SQLite transactions.

    Maintains the six canonical OCEL 2.0 base tables (`event_map_type`, `object_map_type`,
    `event`, `object`, `event_object`, `object_object`) plus dynamic per-type tables
    (`event_<ocel_type_map>`, `object_<ocel_type_map>`) with full primary/foreign key
    constraints and parameterized SQL queries.

    Enforces the core invariant that no event can be committed without a valid finite
    timestamp and at least one mapped object with a valid object type.
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:") -> None:
        """Initialize the OCEL 2.0 SQLite database schema at `db_path`."""
        self.db_path: Path = Path(db_path) if str(db_path) != ":memory:" else Path(":memory:")
        self._db_target: str = str(db_path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            self._db_target, check_same_thread=False
        )
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._event_type_map: Dict[str, str] = {}
        self._object_type_map: Dict[str, str] = {}
        self._initialize_base_schema()
        self._load_existing_type_maps()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the active SQLite connection or raise `StateEncodingError` if closed."""
        if self._conn is None:
            raise StateEncodingError("StateEncoder SQLite connection is already closed.")
        return self._conn

    def __enter__(self) -> Self:
        """Enter context manager returning `self`."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        """Close the SQLite connection cleanly when exiting a context manager block."""
        self.close()

    def close(self) -> None:
        """Commit any pending state and close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _initialize_base_schema(self) -> None:
        """Create the six standard OCEL 2.0 type-independent relational tables."""
        conn = self.connection
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "event_map_type" (
                    "ocel_type" TEXT PRIMARY KEY,
                    "ocel_type_map" TEXT NOT NULL UNIQUE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "object_map_type" (
                    "ocel_type" TEXT PRIMARY KEY,
                    "ocel_type_map" TEXT NOT NULL UNIQUE
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "event" (
                    "ocel_id" TEXT PRIMARY KEY,
                    "ocel_type" TEXT NOT NULL,
                    FOREIGN KEY ("ocel_type") REFERENCES "event_map_type"("ocel_type")
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "object" (
                    "ocel_id" TEXT PRIMARY KEY,
                    "ocel_type" TEXT NOT NULL,
                    FOREIGN KEY ("ocel_type") REFERENCES "object_map_type"("ocel_type")
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "event_object" (
                    "ocel_event_id" TEXT NOT NULL,
                    "ocel_object_id" TEXT NOT NULL,
                    "ocel_qualifier" TEXT NOT NULL,
                    PRIMARY KEY ("ocel_event_id", "ocel_object_id", "ocel_qualifier"),
                    FOREIGN KEY ("ocel_event_id") REFERENCES "event"("ocel_id"),
                    FOREIGN KEY ("ocel_object_id") REFERENCES "object"("ocel_id")
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS "object_object" (
                    "ocel_source_id" TEXT NOT NULL,
                    "ocel_target_id" TEXT NOT NULL,
                    "ocel_qualifier" TEXT NOT NULL,
                    PRIMARY KEY ("ocel_source_id", "ocel_target_id", "ocel_qualifier"),
                    FOREIGN KEY ("ocel_source_id") REFERENCES "object"("ocel_id"),
                    FOREIGN KEY ("ocel_target_id") REFERENCES "object"("ocel_id")
                );
                """
            )

    def _load_existing_type_maps(self) -> None:
        """Load existing `event_map_type` and `object_map_type` entries from the database."""
        cur = self.connection.cursor()
        cur.execute('SELECT "ocel_type", "ocel_type_map" FROM "event_map_type";')
        for o_type, o_map in cur.fetchall():
            self._event_type_map[str(o_type)] = str(o_map)

        cur.execute('SELECT "ocel_type", "ocel_type_map" FROM "object_map_type";')
        for o_type, o_map in cur.fetchall():
            self._object_type_map[str(o_type)] = str(o_map)

    def _ensure_event_type_table(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        attr_columns: Dict[str, str],
    ) -> str:
        """Register `event_type` in `event_map_type` and ensure `event_<map>` table and columns exist."""
        e_type = event_type.strip()
        if not e_type:
            raise StateEncodingError("Event type cannot be empty.")

        if e_type not in self._event_type_map:
            mapped = _sanitize_type_map_name(e_type, self._event_type_map)
            conn.execute(
                'INSERT OR IGNORE INTO "event_map_type" ("ocel_type", "ocel_type_map") VALUES (?, ?);',
                (e_type, mapped),
            )
            self._event_type_map[e_type] = mapped
        else:
            mapped = self._event_type_map[e_type]

        table_name = f"event_{mapped}"
        quoted_table = _quote_sql_ident(table_name)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quoted_table} (
                "ocel_id" TEXT PRIMARY KEY,
                "ocel_time" TIMESTAMP NOT NULL,
                FOREIGN KEY ("ocel_id") REFERENCES "event"("ocel_id")
            );
            """
        )
        self._ensure_columns(conn, table_name, attr_columns)
        return table_name

    def _ensure_object_type_table(
        self,
        conn: sqlite3.Connection,
        object_type: str,
        attr_columns: Dict[str, str],
    ) -> str:
        """Register `object_type` in `object_map_type` and ensure `object_<map>` table and columns exist."""
        o_type = object_type.strip()
        if not o_type:
            raise StateEncodingError("Object type cannot be empty.")

        if o_type not in self._object_type_map:
            mapped = _sanitize_type_map_name(o_type, self._object_type_map)
            conn.execute(
                'INSERT OR IGNORE INTO "object_map_type" ("ocel_type", "ocel_type_map") VALUES (?, ?);',
                (o_type, mapped),
            )
            self._object_type_map[o_type] = mapped
        else:
            mapped = self._object_type_map[o_type]

        table_name = f"object_{mapped}"
        quoted_table = _quote_sql_ident(table_name)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {quoted_table} (
                "ocel_id" TEXT NOT NULL,
                "ocel_time" TIMESTAMP,
                "ocel_changed_field" TEXT,
                FOREIGN KEY ("ocel_id") REFERENCES "object"("ocel_id")
            );
            """
        )
        self._ensure_columns(conn, table_name, attr_columns)
        return table_name

    @staticmethod
    def _ensure_columns(
        conn: sqlite3.Connection,
        table_name: str,
        attr_columns: Dict[str, str],
    ) -> None:
        """Dynamically add any new attribute columns to `table_name`."""
        if not attr_columns:
            return
        quoted_table = _quote_sql_ident(table_name)
        cur = conn.execute(f"PRAGMA table_info({quoted_table});")
        existing_cols = {str(row[1]).casefold() for row in cur.fetchall()}
        for col_name, sql_type in attr_columns.items():
            if col_name.casefold() not in existing_cols:
                quoted_col = _quote_sql_ident(col_name)
                conn.execute(f"ALTER TABLE {quoted_table} ADD COLUMN {quoted_col} {sql_type};")
                existing_cols.add(col_name.casefold())

    def _upsert_object_in_tx(
        self,
        conn: sqlite3.Connection,
        object_id: str,
        object_type: str,
        attributes: Optional[Mapping[str, Any]] = None,
        timestamp: Optional[str] = None,
        changed_field: Optional[str] = None,
    ) -> None:
        """Insert or update an object in `object` and `object_<type>` within an active transaction."""
        oid = str(object_id).strip()
        otype = str(object_type).strip()
        if not oid:
            raise StateEncodingError("Object ID cannot be empty.")
        if not otype:
            raise StateEncodingError(f"Object '{oid}' must have a non-empty object_type.")

        serialized_attrs: Dict[str, Union[str, int, float, None]] = {}
        attr_col_types: Dict[str, str] = {}
        for k, v in (attributes or {}).items():
            col_name = _sanitize_attribute_column(k)
            s_val, s_type = _serialize_attribute_value(v)
            serialized_attrs[col_name] = s_val
            attr_col_types[col_name] = s_type

        table_name = self._ensure_object_type_table(conn, otype, attr_col_types)
        quoted_table = _quote_sql_ident(table_name)

        # Check if object already exists with a conflicting type
        cur = conn.execute('SELECT "ocel_type" FROM "object" WHERE "ocel_id" = ?;', (oid,))
        existing = cur.fetchone()
        if existing is not None and str(existing[0]) != otype:
            raise StateEncodingError(
                f"Object '{oid}' already registered with type '{existing[0]}', cannot re-register as '{otype}'."
            )

        conn.execute(
            'INSERT OR IGNORE INTO "object" ("ocel_id", "ocel_type") VALUES (?, ?);',
            (oid, otype),
        )

        # Check if an initial base row (ocel_changed_field IS NULL) already exists in object_<type>
        cur_base = conn.execute(
            f'SELECT rowid, "ocel_time" FROM {quoted_table} WHERE "ocel_id" = ? AND "ocel_changed_field" IS NULL;',  # nosec B608
            (oid,),
        )
        base_row = cur_base.fetchone()

        if base_row is None and changed_field is None:
            col_names = ["ocel_id", "ocel_time", "ocel_changed_field", *serialized_attrs.keys()]
            quoted_cols = ", ".join(_quote_sql_ident(c) for c in col_names)
            placeholders = ", ".join("?" for _ in col_names)
            values: List[Any] = [oid, timestamp, None, *serialized_attrs.values()]
            conn.execute(
                f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES ({placeholders});",  # nosec B608
                tuple(values),
            )
        elif changed_field is not None:
            col_names = ["ocel_id", "ocel_time", "ocel_changed_field", *serialized_attrs.keys()]
            quoted_cols = ", ".join(_quote_sql_ident(c) for c in col_names)
            placeholders = ", ".join("?" for _ in col_names)
            values = [oid, timestamp, changed_field, *serialized_attrs.values()]
            conn.execute(
                f"INSERT INTO {quoted_table} ({quoted_cols}) VALUES ({placeholders});",  # nosec B608
                tuple(values),
            )
        elif base_row is not None:
            updates: Dict[str, Any] = dict(serialized_attrs)
            if base_row[1] is None and timestamp is not None:
                updates["ocel_time"] = timestamp
            if updates:
                set_clauses = ", ".join(f"{_quote_sql_ident(c)} = ?" for c in updates)
                update_vals: List[Any] = [*updates.values(), base_row[0]]
                conn.execute(
                    f"UPDATE {quoted_table} SET {set_clauses} WHERE rowid = ?;",  # nosec B608
                    tuple(update_vals),
                )

    def encode_object(
        self,
        object_id: str,
        object_type: str,
        attributes: Optional[Mapping[str, Any]] = None,
        relationships: Optional[Sequence[Union[Mapping[str, str], OcelObjectRelation]]] = None,
        timestamp: Optional[Union[str, float, int, datetime]] = None,
    ) -> None:
        """Register an object and optional O2O relationships in an atomic SQLite transaction."""
        norm_ts = _normalize_timestamp(timestamp) if timestamp is not None else None
        conn = self.connection
        with conn:
            self._upsert_object_in_tx(
                conn,
                object_id=object_id,
                object_type=object_type,
                attributes=attributes,
                timestamp=norm_ts,
            )
            for rel in relationships or ():
                if isinstance(rel, OcelObjectRelation):
                    src_id, tgt_id, qual = rel.source_id, rel.target_id, rel.qualifier
                else:
                    src_id = str(rel.get("source_id", object_id))
                    tgt_id = str(rel.get("target_id", rel.get("objectId", "")))
                    qual = str(rel.get("qualifier", "relates_to"))
                if not tgt_id:
                    raise StateEncodingError(
                        f"O2O relationship for '{object_id}' missing target_id."
                    )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO "object_object"
                    ("ocel_source_id", "ocel_target_id", "ocel_qualifier")
                    VALUES (?, ?, ?);
                    """,
                    (src_id, tgt_id, qual),
                )

    def encode_o2o(
        self,
        source_id: str,
        target_id: str,
        qualifier: str = "relates_to",
    ) -> None:
        """Record a directed Object-to-Object (O2O) relationship between two registered objects."""
        src = str(source_id).strip()
        tgt = str(target_id).strip()
        qual = str(qualifier).strip() or "relates_to"
        if not src or not tgt:
            raise StateEncodingError("O2O relationship requires non-empty source_id and target_id.")

        conn = self.connection
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO "object_object"
                    ("ocel_source_id", "ocel_target_id", "ocel_qualifier")
                    VALUES (?, ?, ?);
                    """,
                    (src, tgt, qual),
                )
        except sqlite3.IntegrityError as exc:
            raise StateEncodingError(
                f"O2O foreign key violation linking '{src}' -> '{tgt}': both objects must exist."
            ) from exc

    @staticmethod
    def _parse_event_dict(
        event: Union[Mapping[str, Any], OcelEventRecord],
        object_type_lookup: Mapping[str, str],
    ) -> OcelEventRecord:
        """Normalize a raw event dictionary or `OcelEventRecord` and validate E2O invariants."""
        if isinstance(event, OcelEventRecord):
            record = event
        elif isinstance(event, Mapping):
            raw_id = event.get("event_id", event.get("id", event.get("ocel:eid", "")))
            raw_type = event.get(
                "event_type",
                event.get("type", event.get("activity", event.get("ocel:activity", ""))),
            )
            raw_ts = event.get(
                "timestamp",
                event.get("time", event.get("ocel:timestamp")),
            )
            if raw_id is None or not str(raw_id).strip():
                raise StateEncodingError("Event is missing a valid non-empty 'event_id' / 'id'.")
            if raw_type is None or not str(raw_type).strip():
                raise StateEncodingError(
                    f"Event '{raw_id}' is missing a valid non-empty 'event_type' / 'type'."
                )
            if raw_ts is None:
                raise StateEncodingError(
                    f"Event '{raw_id}' violated invariant: missing mandatory 'timestamp' / 'time'."
                )

            # Extract mapped objects (supports 'objects', 'relationships', or 'omap')
            raw_objs = event.get(
                "objects",
                event.get("relationships", event.get("e2o", event.get("omap", []))),
            )
            parsed_targets: List[OcelObjectTarget] = []
            if isinstance(raw_objs, Mapping):
                # Mapping of {object_type: [id1, id2]} or {object_id: object_type}
                for k, v in raw_objs.items():
                    if isinstance(v, (list, tuple, set)):
                        for item_id in v:
                            parsed_targets.append(
                                OcelObjectTarget(
                                    object_id=str(item_id),
                                    object_type=str(k),
                                    qualifier="participant",
                                )
                            )
                    elif isinstance(v, Mapping):
                        o_type = str(v.get("type", v.get("object_type", "")))
                        qual = str(v.get("qualifier", "participant"))
                        attrs = dict(v.get("attributes", {}))
                        parsed_targets.append(
                            OcelObjectTarget(
                                object_id=str(k),
                                object_type=o_type,
                                qualifier=qual,
                                attributes=attrs,
                            )
                        )
                    else:
                        parsed_targets.append(
                            OcelObjectTarget(
                                object_id=str(k),
                                object_type=str(v),
                                qualifier="participant",
                            )
                        )
            elif isinstance(raw_objs, Sequence) and not isinstance(raw_objs, (str, bytes)):
                for item in raw_objs:
                    if isinstance(item, OcelObjectTarget):
                        parsed_targets.append(item)
                    elif isinstance(item, Mapping):
                        oid = str(
                            item.get(
                                "object_id",
                                item.get("objectId", item.get("id", item.get("ocel:oid", ""))),
                            )
                        ).strip()
                        otype = str(
                            item.get(
                                "object_type",
                                item.get(
                                    "objectType",
                                    item.get(
                                        "type",
                                        item.get("ocel:type", object_type_lookup.get(oid, "")),
                                    ),
                                ),
                            )
                        ).strip()
                        qual = str(item.get("qualifier", item.get("ocel:qualifier", "participant")))
                        attrs = dict(item.get("attributes", {}))
                        if not oid or not otype:
                            raise StateEncodingError(
                                f"Event '{raw_id}' has E2O entry with missing object_id ({oid!r}) "
                                f"or unmapped object_type ({otype!r})."
                            )
                        parsed_targets.append(
                            OcelObjectTarget(
                                object_id=oid,
                                object_type=otype,
                                qualifier=qual,
                                attributes=attrs,
                            )
                        )
                    elif isinstance(item, str):
                        oid = item.strip()
                        otype = object_type_lookup.get(oid, "")
                        if not oid or not otype:
                            raise StateEncodingError(
                                f"Event '{raw_id}' references object '{oid}' without a mapped object_type."
                            )
                        parsed_targets.append(OcelObjectTarget(object_id=oid, object_type=otype))

            raw_o2o = event.get("object_relations", event.get("o2o", []))
            parsed_o2o: List[OcelObjectRelation] = []
            for rel in raw_o2o:
                if isinstance(rel, OcelObjectRelation):
                    parsed_o2o.append(rel)
                elif isinstance(rel, Mapping):
                    parsed_o2o.append(
                        OcelObjectRelation(
                            source_id=str(rel.get("source_id", rel.get("source", ""))),
                            target_id=str(
                                rel.get("target_id", rel.get("target", rel.get("objectId", "")))
                            ),
                            qualifier=str(rel.get("qualifier", "relates_to")),
                        )
                    )

            record = OcelEventRecord(
                event_id=str(raw_id).strip(),
                event_type=str(raw_type).strip(),
                timestamp=raw_ts,
                objects=tuple(parsed_targets),
                attributes=dict(event.get("attributes", event.get("vmap", {}))),
                object_relations=tuple(parsed_o2o),
            )
        else:
            raise StateEncodingError(f"Unsupported event input type: {type(event).__name__}")

        if not record.objects:
            raise StateEncodingError(
                f"Event '{record.event_id}' violated OCEL 2.0 invariant: "
                "no event can exist without at least one mapped object and object_type."
            )
        return record

    def _encode_event_in_tx(
        self,
        conn: sqlite3.Connection,
        event: Union[Mapping[str, Any], OcelEventRecord],
        object_type_lookup: Dict[str, str],
    ) -> OcelEventRecord:
        """Validate and write a single event and its E2O/O2O links inside an active transaction."""
        record = self._parse_event_dict(event, object_type_lookup)
        norm_ts = _normalize_timestamp(record.timestamp)

        # Ensure event ID is not already committed (immutability of discrete events)
        cur = conn.execute('SELECT "ocel_id" FROM "event" WHERE "ocel_id" = ?;', (record.event_id,))
        if cur.fetchone() is not None:
            raise StateEncodingError(
                f"Immutable OCEL 2.0 violation: duplicate event_id '{record.event_id}'."
            )

        # Serialize event attributes
        serialized_attrs: Dict[str, Union[str, int, float, None]] = {}
        attr_col_types: Dict[str, str] = {}
        for k, v in record.attributes.items():
            col_name = _sanitize_attribute_column(k)
            s_val, s_type = _serialize_attribute_value(v)
            serialized_attrs[col_name] = s_val
            attr_col_types[col_name] = s_type

        event_table = self._ensure_event_type_table(conn, record.event_type, attr_col_types)
        quoted_event_table = _quote_sql_ident(event_table)

        # 1. Insert into `event`
        conn.execute(
            'INSERT INTO "event" ("ocel_id", "ocel_type") VALUES (?, ?);',
            (record.event_id, record.event_type),
        )

        # 2. Insert into `event_<ocel_type_map>`
        col_names = ["ocel_id", "ocel_time", *serialized_attrs.keys()]
        quoted_cols = ", ".join(_quote_sql_ident(c) for c in col_names)
        placeholders = ", ".join("?" for _ in col_names)
        values: List[Any] = [record.event_id, norm_ts, *serialized_attrs.values()]
        conn.execute(
            f"INSERT INTO {quoted_event_table} ({quoted_cols}) VALUES ({placeholders});",  # nosec B608
            tuple(values),
        )

        # 3. Ensure all E2O target objects exist and insert N:M rows into `event_object`
        for target in record.objects:
            self._upsert_object_in_tx(
                conn,
                object_id=target.object_id,
                object_type=target.object_type,
                attributes=target.attributes,
                timestamp=norm_ts,
            )
            object_type_lookup[target.object_id] = target.object_type
            conn.execute(
                """
                INSERT OR IGNORE INTO "event_object"
                ("ocel_event_id", "ocel_object_id", "ocel_qualifier")
                VALUES (?, ?, ?);
                """,
                (record.event_id, target.object_id, target.qualifier),
            )

        # 4. Insert any O2O relationships
        for rel in record.object_relations:
            conn.execute(
                """
                INSERT OR IGNORE INTO "object_object"
                ("ocel_source_id", "ocel_target_id", "ocel_qualifier")
                VALUES (?, ?, ?);
                """,
                (rel.source_id, rel.target_id, rel.qualifier),
            )

        return record

    def encode_event(self, event: Union[Mapping[str, Any], OcelEventRecord]) -> OcelEventRecord:
        """Accept a discrete system event and write an immutable OCEL 2.0 record atomically."""
        with self._lock:
            conn = self.connection
            cur = conn.execute('SELECT "ocel_id", "ocel_type" FROM "object";')
            lookup: Dict[str, str] = {str(r[0]): str(r[1]) for r in cur.fetchall()}
            with conn:
                return self._encode_event_in_tx(conn, event, lookup)

    def encode_events(
        self,
        events: Iterable[Union[Mapping[str, Any], OcelEventRecord]],
    ) -> List[OcelEventRecord]:
        """Batch-encode a sequence of discrete events in a single atomic SQLite transaction."""
        conn = self.connection
        cur = conn.execute('SELECT "ocel_id", "ocel_type" FROM "object";')
        lookup: Dict[str, str] = {str(r[0]): str(r[1]) for r in cur.fetchall()}
        encoded: List[OcelEventRecord] = []
        with conn:
            for ev in events:
                encoded.append(self._encode_event_in_tx(conn, ev, lookup))
        return encoded

    def encode_json(
        self,
        json_input: Union[str, Path, Mapping[str, Any], Sequence[Mapping[str, Any]]],
    ) -> int:
        """Ingest an OCEL 2.0 JSON document, JSONL stream, or event list into SQLite.

        Returns:
            Number of events encoded into the OCEL 2.0 SQLite database.
        """
        if isinstance(json_input, Path):
            raw_text = json_input.read_text(encoding="utf-8")
            return self.encode_json(raw_text)

        if isinstance(json_input, str):
            stripped = json_input.strip()
            if not stripped:
                raise StateEncodingError("Empty JSON input provided to StateEncoder.")
            if stripped.startswith("{") and "\n{" in stripped:
                records = [json.loads(line) for line in stripped.splitlines() if line.strip()]
                return self._encode_structured_payload({"records": records})
            parsed = json.loads(stripped)
            if isinstance(parsed, Mapping):
                return self._encode_structured_payload(parsed)
            if isinstance(parsed, list):
                return len(self.encode_events(parsed))
            raise StateEncodingError("JSON input must decode to an object or array of events.")

        if isinstance(json_input, Mapping):
            return self._encode_structured_payload(json_input)

        if isinstance(json_input, Sequence) and not isinstance(json_input, (str, bytes)):
            return len(self.encode_events(json_input))

        raise StateEncodingError(f"Unsupported JSON input type: {type(json_input).__name__}")

    def _encode_structured_payload(self, data: Mapping[str, Any]) -> int:
        """Encode a structured OCEL 2.0 dictionary containing objects, events, and relations."""
        conn = self.connection
        cur = conn.execute('SELECT "ocel_id", "ocel_type" FROM "object";')
        lookup: Dict[str, str] = {str(r[0]): str(r[1]) for r in cur.fetchall()}

        pending_o2o: List[Tuple[str, str, str]] = []
        events_encoded = 0

        with conn:
            # 1. Process JSONL mixed records if present
            if "records" in data and isinstance(data["records"], list):
                for rec in data["records"]:
                    if rec.get("record_type") == "object" or (
                        "id" in rec
                        and "type" in rec
                        and "time" not in rec
                        and "timestamp" not in rec
                    ):
                        oid = str(rec["id"])
                        otype = str(rec["type"])
                        self._upsert_object_in_tx(
                            conn,
                            object_id=oid,
                            object_type=otype,
                            attributes=rec.get("attributes", {}),
                        )
                        lookup[oid] = otype
                    else:
                        self._encode_event_in_tx(conn, rec, lookup)
                        events_encoded += 1
                return events_encoded

            # 2. Pre-register declared objects
            raw_objects = data.get("objects", data.get("ocel:objects", []))
            if isinstance(raw_objects, Mapping):
                for oid, odata in raw_objects.items():
                    otype = str(odata.get("type", odata.get("ocel:type", "")))
                    attrs = odata.get("attributes", odata.get("ocel:ovmap", {}))
                    self._upsert_object_in_tx(conn, str(oid), otype, attributes=attrs)
                    lookup[str(oid)] = otype
                    for rel in odata.get("relationships", []):
                        tgt = str(rel.get("objectId", rel.get("target_id", "")))
                        qual = str(rel.get("qualifier", "relates_to"))
                        if tgt:
                            pending_o2o.append((str(oid), tgt, qual))
            elif isinstance(raw_objects, Sequence):
                for obj in raw_objects:
                    oid = str(obj.get("id", obj.get("object_id", "")))
                    otype = str(obj.get("type", obj.get("object_type", "")))
                    attrs = obj.get("attributes", {})
                    obj_ts: Optional[str] = None
                    if isinstance(attrs, list):
                        # Standard OCEL 2.0 JSON attribute list: [{"name": ..., "value": ..., "time": ...}]
                        for a in attrs:
                            if (
                                isinstance(a, Mapping)
                                and a.get("time")
                                and str(a.get("time")) != "1970-01-01T00:00:00Z"
                                and obj_ts is None
                            ):
                                obj_ts = _normalize_timestamp(a["time"])
                        attrs = {str(a["name"]): a.get("value") for a in attrs if "name" in a}
                    self._upsert_object_in_tx(conn, oid, otype, attributes=attrs, timestamp=obj_ts)
                    lookup[oid] = otype
                    for rel in obj.get("relationships", []):
                        tgt = str(rel.get("objectId", rel.get("target_id", "")))
                        qual = str(rel.get("qualifier", "relates_to"))
                        if tgt:
                            pending_o2o.append((oid, tgt, qual))

            # 3. Insert deferred O2O relations after all objects exist
            for src_id, tgt_id, qual in pending_o2o:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO "object_object"
                    ("ocel_source_id", "ocel_target_id", "ocel_qualifier")
                    VALUES (?, ?, ?);
                    """,
                    (src_id, tgt_id, qual),
                )

            # 4. Encode all events
            raw_events = data.get("events", data.get("ocel:events", []))
            if isinstance(raw_events, Mapping):
                for eid, edata in raw_events.items():
                    ev_copy = dict(edata)
                    ev_copy.setdefault("event_id", str(eid))
                    self._encode_event_in_tx(conn, ev_copy, lookup)
                    events_encoded += 1
            elif isinstance(raw_events, Sequence):
                for ev in raw_events:
                    ev_copy = dict(ev)
                    if isinstance(ev_copy.get("attributes"), list):
                        ev_copy["attributes"] = {
                            str(a["name"]): a.get("value")
                            for a in ev_copy["attributes"]
                            if "name" in a
                        }
                    self._encode_event_in_tx(conn, ev_copy, lookup)
                    events_encoded += 1

        return events_encoded

    def encode_xml(self, xml_input: Union[str, Path]) -> int:
        """Parse an OCEL 2.0 XML event stream and write normalized SQLite transactions.

        Returns:
            Number of events encoded into the OCEL 2.0 SQLite database.
        """
        if isinstance(xml_input, Path):
            xml_text = xml_input.read_text(encoding="utf-8")
        else:
            p = Path(xml_input) if "\n" not in xml_input and len(xml_input) < 260 else None
            xml_text = (
                p.read_text(encoding="utf-8") if (p is not None and p.is_file()) else xml_input
            )

        try:
            root = _safe_parse_xml_root(xml_text)
        except ET.ParseError as exc:
            raise StateEncodingError(f"Malformed OCEL 2.0 XML input: {exc}") from exc

        def _strip_ns(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        objects_list: List[Dict[str, Any]] = []
        events_list: List[Dict[str, Any]] = []

        for child in root:
            cname = _strip_ns(child.tag).lower()
            if cname == "objects":
                for obj_el in child:
                    if _strip_ns(obj_el.tag).lower() != "object":
                        continue
                    oid = obj_el.attrib.get("id", "")
                    otype = obj_el.attrib.get("type", "")
                    attrs: Dict[str, Any] = {}
                    rels: List[Dict[str, str]] = []
                    for sub in obj_el:
                        sname = _strip_ns(sub.tag).lower()
                        if sname == "attributes":
                            for attr_el in sub:
                                aname = attr_el.attrib.get("name", "")
                                aval = attr_el.attrib.get("value", (attr_el.text or "").strip())
                                if aname:
                                    attrs[aname] = aval
                        elif sname in ("objects", "relationships"):
                            for rel_el in sub:
                                tgt_id = rel_el.attrib.get(
                                    "object-id", rel_el.attrib.get("objectId", "")
                                )
                                qual = rel_el.attrib.get("qualifier", "relates_to")
                                if tgt_id:
                                    rels.append({"objectId": tgt_id, "qualifier": qual})
                    objects_list.append(
                        {"id": oid, "type": otype, "attributes": attrs, "relationships": rels}
                    )
            elif cname == "events":
                for ev_el in child:
                    if _strip_ns(ev_el.tag).lower() != "event":
                        continue
                    eid = ev_el.attrib.get("id", "")
                    etype = ev_el.attrib.get("type", "")
                    etime = ev_el.attrib.get("time", ev_el.attrib.get("timestamp", ""))
                    attrs = {}
                    e2o_rels: List[Dict[str, str]] = []
                    for sub in ev_el:
                        sname = _strip_ns(sub.tag).lower()
                        if sname == "attributes":
                            for attr_el in sub:
                                aname = attr_el.attrib.get("name", "")
                                aval = attr_el.attrib.get("value", (attr_el.text or "").strip())
                                if aname:
                                    attrs[aname] = aval
                        elif sname in ("objects", "relationships"):
                            for rel_el in sub:
                                tgt_id = rel_el.attrib.get(
                                    "object-id", rel_el.attrib.get("objectId", "")
                                )
                                tgt_type = rel_el.attrib.get(
                                    "object-type", rel_el.attrib.get("type", "")
                                )
                                qual = rel_el.attrib.get("qualifier", "participant")
                                entry: Dict[str, str] = {"objectId": tgt_id, "qualifier": qual}
                                if tgt_type:
                                    entry["type"] = tgt_type
                                e2o_rels.append(entry)
                    events_list.append(
                        {
                            "id": eid,
                            "type": etype,
                            "time": etime,
                            "attributes": attrs,
                            "relationships": e2o_rels,
                        }
                    )

        return self._encode_structured_payload({"objects": objects_list, "events": events_list})

    def list_tables(self) -> List[str]:
        """Return sorted list of all relational tables created in the SQLite database."""
        cur = self.connection.execute(
            'SELECT tbl_name FROM sqlite_master WHERE type = "table" ORDER BY tbl_name ASC;'
        )
        return [str(row[0]) for row in cur.fetchall()]

    def validate_schema(self) -> Tuple[List[str], List[str]]:
        """Validate the SQLite database against the 24 OCEL 2.0 relational constraints."""
        if self._conn is not None:
            return _validate_connection_constraints(self._conn)
        if self._db_target != ":memory:" and self.db_path.exists():
            return validate_ocel2_sqlite(self.db_path)
        raise StateEncodingError("Cannot validate closed in-memory OCEL 2.0 SQLite database.")

    def read_ocel2(self, validate: bool = True) -> Any:
        """Retrieve the encoded OCEL 2.0 object via `pm4py.read_ocel2_sqlite`.

        Supports both filesystem-backed and `:memory:` `StateEncoder` instances (via an
        automatic temporary SQLite backup when `:memory:` is active).
        """
        if self._db_target == ":memory:":
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_db = Path(tmp_dir) / "ocel2_mem_snapshot.sqlite"
                dst_conn = sqlite3.connect(str(tmp_db))
                try:
                    self.connection.backup(dst_conn)
                finally:
                    dst_conn.close()
                return read_ocel2_sqlite(tmp_db, validate=validate)
        return read_ocel2_sqlite(self.db_path, validate=validate)

    def to_canonical_snapshot(self) -> Dict[str, Any]:
        """Return a canonical, deterministic dictionary snapshot of all internal OCEL 2.0 state.

        Used to assert deep structural equality across SQLite -> JSON -> SQLite roundtrips.
        """
        conn = self.connection
        event_types = sorted(
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                'SELECT "ocel_type", "ocel_type_map" FROM "event_map_type" ORDER BY "ocel_type" ASC;'
            ).fetchall()
        )
        object_types = sorted(
            (str(r[0]), str(r[1]))
            for r in conn.execute(
                'SELECT "ocel_type", "ocel_type_map" FROM "object_map_type" ORDER BY "ocel_type" ASC;'
            ).fetchall()
        )
        events_map: Dict[str, Dict[str, Any]] = {}
        for etype, emap in event_types:
            tbl = _quote_sql_ident(f"event_{emap}")
            cur = conn.execute(
                f'SELECT * FROM {tbl} ORDER BY "ocel_id" ASC;'  # nosec B608
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                row_dict = dict(zip(cols, row, strict=False))
                eid = str(row_dict.pop("ocel_id"))
                etime = str(row_dict.pop("ocel_time"))
                attrs = {k: v for k, v in sorted(row_dict.items()) if v is not None}
                events_map[eid] = {
                    "type": etype,
                    "time": etime,
                    "attributes": attrs,
                }

        objects_map: Dict[str, Dict[str, Any]] = {}
        for otype, omap in object_types:
            tbl = _quote_sql_ident(f"object_{omap}")
            cur = conn.execute(
                f'SELECT * FROM {tbl} WHERE "ocel_changed_field" IS NULL ORDER BY "ocel_id" ASC;'  # nosec B608
            )
            cols = [d[0] for d in cur.description]
            for row in cur.fetchall():
                row_dict = dict(zip(cols, row, strict=False))
                oid = str(row_dict.pop("ocel_id"))
                otime = row_dict.pop("ocel_time")
                row_dict.pop("ocel_changed_field", None)
                attrs = {k: v for k, v in sorted(row_dict.items()) if v is not None}
                objects_map[oid] = {
                    "type": otype,
                    "time": str(otime) if otime is not None else None,
                    "attributes": attrs,
                }

        e2o_list = sorted(
            (str(r[0]), str(r[1]), str(r[2]))
            for r in conn.execute(
                'SELECT "ocel_event_id", "ocel_object_id", "ocel_qualifier" FROM "event_object";'
            ).fetchall()
        )
        o2o_list = sorted(
            (str(r[0]), str(r[1]), str(r[2]))
            for r in conn.execute(
                'SELECT "ocel_source_id", "ocel_target_id", "ocel_qualifier" FROM "object_object";'
            ).fetchall()
        )

        return {
            "event_types": event_types,
            "object_types": object_types,
            "events": dict(sorted(events_map.items())),
            "objects": dict(sorted(objects_map.items())),
            "e2o": e2o_list,
            "o2o": o2o_list,
        }

    def equals(self, other: Any) -> bool:
        """Return True if this `StateEncoder` and `other` have deeply equal internal OCEL 2.0 structures."""
        if not isinstance(other, StateEncoder):
            return False
        return self.to_canonical_snapshot() == other.to_canonical_snapshot()

    def export_to_ocel_json(
        self,
        filepath: Union[str, Path],
        *,
        max_payload_bytes: int = 64 * 1024 * 1024,
        chunk_size: int = 2048,
    ) -> Path:
        """Export the internal OCEL 2.0 state to a standard JSONOCEL 2.0 file via `pm4py.write_ocel2_json`.

        Preserves exact ISO-8601 timestamp precision (including microseconds) and streams
        serialized chunks to disk with an operational memory/payload bound (`max_payload_bytes`)
        to prevent OOM errors on large event logs.

        Pandas Data Scientist Usage Example
        -----------------------------------
        ```python
        import pandas as pd
        import pm4py

        # 1. Load exported OCEL 2.0 JSON via PM4Py into Pandas DataFrames
        ocel = pm4py.read_ocel2_json("exported_state.jsonocel")
        events_df: pd.DataFrame = ocel.events
        objects_df: pd.DataFrame = ocel.objects
        e2o_relations_df: pd.DataFrame = ocel.relations
        extended_df: pd.DataFrame = ocel.get_extended_table()

        # 2. Or load raw JSON directly with pandas.json_normalize
        import json
        with open("exported_state.jsonocel", encoding="utf-8") as f:
            payload = json.load(f)
        flat_events_df = pd.json_normalize(payload["events"])
        ```
        """
        import warnings as _warnings

        import pm4py

        if max_payload_bytes <= 0:
            raise StateEncodingError(
                f"max_payload_bytes must be positive, got {max_payload_bytes}."
            )

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        ocel_obj = self.read_ocel2(validate=True)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            _warnings.simplefilter("ignore", FutureWarning)
            _warnings.simplefilter("ignore", DeprecationWarning)
            pm4py.write_ocel2_json(ocel_obj, str(target_path))

        # Enrich exported JSON with exact ISO-8601 sub-second timestamps and exact scalar types
        # from the SQLite tables while enforcing max_payload_bytes and streaming in chunks.
        snapshot = self.to_canonical_snapshot()
        raw_doc = json.loads(target_path.read_text(encoding="utf-8"))

        # Preserve exact ISO-8601 event timestamps and scalar attribute values
        snap_events = snapshot["events"]
        for ev_entry in raw_doc.get("events", []):
            eid = str(ev_entry.get("id", ""))
            if eid in snap_events:
                ev_snap = snap_events[eid]
                ev_entry["time"] = ev_snap["time"]
                snap_attrs = ev_snap["attributes"]
                if snap_attrs:
                    ev_entry["attributes"] = [
                        {"name": k, "value": v} for k, v in sorted(snap_attrs.items())
                    ]

        # Preserve exact ISO-8601 object timestamps and scalar attribute values
        snap_objects = snapshot["objects"]
        for obj_entry in raw_doc.get("objects", []):
            oid = str(obj_entry.get("id", ""))
            if oid in snap_objects:
                obj_snap = snap_objects[oid]
                obj_time = obj_snap["time"] or "1970-01-01T00:00:00Z"
                snap_attrs = obj_snap["attributes"]
                if snap_attrs:
                    obj_entry["attributes"] = [
                        {"name": k, "time": obj_time, "value": v}
                        for k, v in sorted(snap_attrs.items())
                    ]

        # Stream to disk in bounded chunks while enforcing max_payload_bytes
        bytes_written = 0
        with open(target_path, "w", encoding="utf-8") as out_f:
            encoder_iter = json.JSONEncoder(indent=2, sort_keys=False).iterencode(raw_doc)
            buf: List[str] = []
            buf_len = 0
            for piece in encoder_iter:
                piece_bytes = len(piece.encode("utf-8"))
                bytes_written += piece_bytes
                if bytes_written > max_payload_bytes:
                    out_f.close()
                    target_path.unlink(missing_ok=True)
                    raise StateEncodingError(
                        f"OCEL 2.0 JSON export exceeded max_payload_bytes limit "
                        f"({bytes_written} > {max_payload_bytes} bytes)."
                    )
                buf.append(piece)
                buf_len += len(piece)
                if buf_len >= chunk_size:
                    out_f.write("".join(buf))
                    buf.clear()
                    buf_len = 0
            if buf:
                out_f.write("".join(buf))

        return target_path

    def export_to_ocel_xml(
        self,
        filepath: Union[str, Path],
        *,
        max_payload_bytes: int = 64 * 1024 * 1024,
    ) -> Path:
        """Export the internal OCEL 2.0 state to an XML-OCEL 2.0 file via `pm4py.write_ocel2_xml`."""
        import warnings as _warnings

        import pm4py

        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        ocel_obj = self.read_ocel2(validate=True)

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            _warnings.simplefilter("ignore", FutureWarning)
            _warnings.simplefilter("ignore", DeprecationWarning)
            pm4py.write_ocel2_xml(ocel_obj, str(target_path))

        size = target_path.stat().st_size
        if size > max_payload_bytes:
            target_path.unlink(missing_ok=True)
            raise StateEncodingError(
                f"OCEL 2.0 XML export exceeded max_payload_bytes limit ({size} > {max_payload_bytes} bytes)."
            )
        return target_path

    def ingest_csv(
        self,
        csv_input: Union[str, Path, Iterable[str]],
        *,
        event_id_col: str = "event_id",
        event_type_col: str = "event_type",
        timestamp_col: str = "timestamp",
        objects_col: str = "objects",
        o2o_col: str = "o2o",
        object_columns: Optional[Mapping[str, str]] = None,
        o2o_rules: Optional[Sequence[Tuple[str, str, str]]] = None,
        batch_size: int = 500,
    ) -> int:
        """Stream and safely map a standard CSV event stream into the relational E2O and O2O schema.

        Supports both:
        1. Explicit E2O/O2O columns (`objects` formatted as JSON or `Type:id[:qualifier];...`
           and `o2o` formatted as JSON or `src_id->tgt_id[:qualifier];...`), and
        2. Column-mapped object bindings via `object_columns={"user_id": "User", "order_id": "Order"}`
           and `o2o_rules=[("user_id", "order_id", "places")]`.

        Streams rows in bounded batches (`batch_size`) to prevent OOM errors on large CSV files.

        Returns:
            Total number of events ingested into the OCEL 2.0 SQLite store.
        """
        if isinstance(csv_input, Path):
            with open(csv_input, encoding="utf-8", newline="") as f:
                return self._ingest_csv_reader(
                    csv.DictReader(f),
                    event_id_col=event_id_col,
                    event_type_col=event_type_col,
                    timestamp_col=timestamp_col,
                    objects_col=objects_col,
                    o2o_col=o2o_col,
                    object_columns=object_columns,
                    o2o_rules=o2o_rules,
                    batch_size=batch_size,
                )
        if isinstance(csv_input, str):
            p = Path(csv_input) if "\n" not in csv_input and len(csv_input) < 260 else None
            if p is not None and p.is_file():
                with open(p, encoding="utf-8", newline="") as f:
                    return self._ingest_csv_reader(
                        csv.DictReader(f),
                        event_id_col=event_id_col,
                        event_type_col=event_type_col,
                        timestamp_col=timestamp_col,
                        objects_col=objects_col,
                        o2o_col=o2o_col,
                        object_columns=object_columns,
                        o2o_rules=o2o_rules,
                        batch_size=batch_size,
                    )
            return self._ingest_csv_reader(
                csv.DictReader(io.StringIO(csv_input.strip())),
                event_id_col=event_id_col,
                event_type_col=event_type_col,
                timestamp_col=timestamp_col,
                objects_col=objects_col,
                o2o_col=o2o_col,
                object_columns=object_columns,
                o2o_rules=o2o_rules,
                batch_size=batch_size,
            )

        return self._ingest_csv_reader(
            csv.DictReader(csv_input),
            event_id_col=event_id_col,
            event_type_col=event_type_col,
            timestamp_col=timestamp_col,
            objects_col=objects_col,
            o2o_col=o2o_col,
            object_columns=object_columns,
            o2o_rules=o2o_rules,
            batch_size=batch_size,
        )

    def _ingest_csv_reader(
        self,
        reader: csv.DictReader[str],
        *,
        event_id_col: str,
        event_type_col: str,
        timestamp_col: str,
        objects_col: str,
        o2o_col: str,
        object_columns: Optional[Mapping[str, str]],
        o2o_rules: Optional[Sequence[Tuple[str, str, str]]],
        batch_size: int,
    ) -> int:
        obj_col_map = dict(object_columns or {})
        reserved_cols = {
            event_id_col,
            event_type_col,
            timestamp_col,
            objects_col,
            o2o_col,
            "id",
            "type",
            "activity",
            "time",
            "e2o",
            "object_relations",
            *obj_col_map.keys(),
        }

        total_ingested = 0
        batch: List[OcelEventRecord] = []

        for idx, raw_row in enumerate(reader):
            row = {
                str(k).strip(): (v.strip() if isinstance(v, str) else v)
                for k, v in raw_row.items()
                if k is not None
            }
            eid = str(
                row.get(event_id_col)
                or row.get("id")
                or row.get("ocel:eid")
                or f"csv_evt_{idx + 1}"
            ).strip()
            etype = str(
                row.get(event_type_col)
                or row.get("activity")
                or row.get("type")
                or row.get("ocel:activity")
                or row.get("concept:name")
                or ""
            ).strip()
            raw_ts = (
                row.get(timestamp_col)
                or row.get("time")
                or row.get("ocel:timestamp")
                or row.get("time:timestamp")
            )
            if not etype:
                raise StateEncodingError(
                    f"CSV row {idx + 1} (event '{eid}') is missing event_type/activity."
                )
            if not raw_ts:
                raise StateEncodingError(f"CSV row {idx + 1} (event '{eid}') is missing timestamp.")

            targets: List[OcelObjectTarget] = []
            # 1. Parse explicit objects column if present
            raw_objs_str = row.get(objects_col) or row.get("e2o") or ""
            if raw_objs_str:
                targets.extend(_parse_csv_e2o_cell(str(raw_objs_str)))

            # 2. Parse mapped object_columns (e.g. {"user_id": "User", "order_id": "Order"})
            for col_key, otype_name in obj_col_map.items():
                cell_val = row.get(col_key, "")
                if cell_val:
                    for oid_part in str(cell_val).split(";"):
                        oid_clean = oid_part.strip()
                        if oid_clean:
                            targets.append(
                                OcelObjectTarget(
                                    object_id=oid_clean,
                                    object_type=otype_name,
                                    qualifier=col_key,
                                )
                            )

            # 3. Auto-detect columns prefixed with 'obj:' (e.g., 'obj:User', 'obj:Order')
            for col_key, cell_val in row.items():
                if col_key.startswith("obj:") and cell_val:
                    otype_name = col_key.split(":", 1)[1].strip()
                    for oid_part in str(cell_val).split(";"):
                        oid_clean = oid_part.strip()
                        if oid_clean and otype_name:
                            targets.append(
                                OcelObjectTarget(
                                    object_id=oid_clean,
                                    object_type=otype_name,
                                    qualifier="participant",
                                )
                            )

            if not targets:
                raise StateEncodingError(
                    f"CSV row {idx + 1} (event '{eid}') has no mapped E2O objects."
                )

            # 4. Parse O2O relationships from explicit o2o column or o2o_rules
            relations: List[OcelObjectRelation] = []
            raw_o2o_str = row.get(o2o_col) or row.get("object_relations") or ""
            if raw_o2o_str:
                relations.extend(_parse_csv_o2o_cell(str(raw_o2o_str)))

            for src_col, tgt_col, qual in o2o_rules or ():
                src_val = str(row.get(src_col, "")).strip()
                tgt_val = str(row.get(tgt_col, "")).strip()
                if src_val and tgt_val:
                    relations.append(
                        OcelObjectRelation(
                            source_id=src_val,
                            target_id=tgt_val,
                            qualifier=qual,
                        )
                    )

            # 5. Extract remaining columns as typed event attributes
            attrs: Dict[str, Any] = {}
            for col_key, cell_val in row.items():
                if (
                    col_key not in reserved_cols
                    and not col_key.startswith("obj:")
                    and cell_val is not None
                    and cell_val != ""
                ):
                    attrs[col_key] = _coerce_csv_scalar(str(cell_val))

            batch.append(
                OcelEventRecord(
                    event_id=eid,
                    event_type=etype,
                    timestamp=raw_ts,
                    objects=tuple(targets),
                    attributes=attrs,
                    object_relations=tuple(relations),
                )
            )
            if len(batch) >= max(1, batch_size):
                self.encode_events(batch)
                total_ingested += len(batch)
                batch.clear()

        if batch:
            self.encode_events(batch)
            total_ingested += len(batch)
            batch.clear()

        return total_ingested

    def ingest_xes(
        self,
        xes_input: Union[str, Path],
        *,
        case_object_type: str = "Case",
        resource_object_type: str = "Resource",
    ) -> int:
        """Ingest a legacy IEEE 1849 XES flat event log into the relational OCEL 2.0 schema.

        Maps each `<trace>` to a case object (`case_object_type`) and optional organizational
        resource (`org:resource` -> `resource_object_type`), creating E2O and O2O links for
        every `<event>` while preserving ISO-8601 timestamp precision.

        Returns:
            Total number of events ingested into the OCEL 2.0 SQLite store.
        """
        if isinstance(xes_input, Path):
            xes_text = xes_input.read_text(encoding="utf-8")
        else:
            p = Path(xes_input) if "\n" not in xes_input and len(xes_input) < 260 else None
            xes_text = (
                p.read_text(encoding="utf-8") if (p is not None and p.is_file()) else xes_input
            )

        try:
            root = _safe_parse_xml_root(xes_text)
        except ET.ParseError as exc:
            raise StateEncodingError(f"Malformed XES XML input: {exc}") from exc

        def _strip_ns(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        def _extract_xes_kv(elem: ET.Element) -> Dict[str, Any]:
            kv: Dict[str, Any] = {}
            for child in elem:
                ctag = _strip_ns(child.tag).lower()
                key = child.attrib.get("key", "")
                val_raw = child.attrib.get("value", "")
                if not key:
                    continue
                if ctag == "int":
                    kv[key] = int(val_raw)
                elif ctag == "float":
                    kv[key] = float(val_raw)
                elif ctag == "boolean":
                    kv[key] = val_raw.strip().lower() in ("true", "1")
                else:
                    kv[key] = val_raw
            return kv

        records: List[OcelEventRecord] = []
        ev_counter = 0

        for trace_idx, trace_el in enumerate(root):
            if _strip_ns(trace_el.tag).lower() != "trace":
                continue
            trace_kv = _extract_xes_kv(trace_el)
            case_id = str(
                trace_kv.pop("concept:name", trace_kv.pop("case:id", f"case_{trace_idx + 1}"))
            )
            trace_attrs = {_sanitize_attribute_column(k): v for k, v in trace_kv.items()}

            for child_el in trace_el:
                if _strip_ns(child_el.tag).lower() != "event":
                    continue
                ev_counter += 1
                ev_kv = _extract_xes_kv(child_el)
                eid = str(
                    ev_kv.pop(
                        "ocel:eid",
                        ev_kv.pop("event_id", ev_kv.pop("id", f"xes_evt_{ev_counter}")),
                    )
                )
                activity = str(ev_kv.pop("concept:name", ev_kv.pop("activity", "XesActivity")))
                raw_ts = ev_kv.pop("time:timestamp", ev_kv.pop("timestamp", None))
                if raw_ts is None:
                    raise StateEncodingError(
                        f"XES event '{eid}' in case '{case_id}' is missing 'time:timestamp'."
                    )
                org_res = ev_kv.pop("org:resource", None)

                targets: List[OcelObjectTarget] = [
                    OcelObjectTarget(
                        object_id=case_id,
                        object_type=case_object_type,
                        qualifier="case",
                        attributes=trace_attrs,
                    )
                ]
                relations: List[OcelObjectRelation] = []
                if org_res is not None and str(org_res).strip():
                    res_id = str(org_res).strip()
                    targets.append(
                        OcelObjectTarget(
                            object_id=res_id,
                            object_type=resource_object_type,
                            qualifier="actor",
                        )
                    )
                    relations.append(
                        OcelObjectRelation(
                            source_id=res_id,
                            target_id=case_id,
                            qualifier="handles",
                        )
                    )

                clean_attrs = {_sanitize_attribute_column(k): v for k, v in ev_kv.items()}
                records.append(
                    OcelEventRecord(
                        event_id=eid,
                        event_type=activity,
                        timestamp=raw_ts,
                        objects=tuple(targets),
                        attributes=clean_attrs,
                        object_relations=tuple(relations),
                    )
                )

        self.encode_events(records)
        return len(records)


def _validate_connection_constraints(
    conn: sqlite3.Connection,
) -> Tuple[List[str], List[str]]:
    """Evaluate all 24 standard OCEL 2.0 relational integrity constraints on an open SQLite connection."""
    from pm4py.objects.ocel.validation import ocel20_rel_validation

    curs = conn.cursor()
    satisfied: List[str] = []
    unsatisfied: List[str] = []
    try:
        for name, check_fn in ocel20_rel_validation.constraints.items():
            if check_fn(curs):
                satisfied.append(str(name))
            else:
                unsatisfied.append(str(name))
    finally:
        curs.close()
    return satisfied, unsatisfied


def validate_ocel2_sqlite(db_path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """Validate an OCEL 2.0 SQLite database file against the 24 OCEL 2.0 relational constraints.

    Returns:
        Tuple of `(satisfied_constraints, unsatisfied_constraints)`.
    """
    path = Path(db_path)
    if not path.exists():
        raise StateEncodingError(f"OCEL 2.0 SQLite file does not exist: {path}")

    conn = sqlite3.connect(str(path))
    try:
        return _validate_connection_constraints(conn)
    finally:
        conn.close()


def read_ocel2_sqlite(
    db_path: Union[str, Path],
    *,
    validate: bool = True,
) -> Any:
    """Retrieve an OCEL 2.0 log from a SQLite database using `pm4py.read_ocel2_sqlite`.

    Suppresses spurious pandas/pm4py internal timestamp parsing warnings and optionally
    enforces the 24 OCEL 2.0 relational schema constraints prior to loading.
    """
    import warnings as _warnings

    path = Path(db_path)
    if not path.exists():
        raise StateEncodingError(f"OCEL 2.0 SQLite database file not found: {path}")

    if validate:
        _satisfied, unsatisfied = validate_ocel2_sqlite(path)
        if unsatisfied:
            raise StateEncodingError(
                f"OCEL 2.0 SQLite schema validation failed on constraints: {unsatisfied}"
            )

    import pm4py

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        _warnings.simplefilter("ignore", FutureWarning)
        _warnings.simplefilter("ignore", DeprecationWarning)
        return pm4py.read_ocel2_sqlite(str(path))


# ==============================================================================
# Legacy Flat-Log Parsers (Retained for Backwards Compatibility with @deprecated)
# ==============================================================================


@deprecated(
    since="1.5.0",
    removed_in="2.0.0",
    alternative="Use StateEncoder and OCEL 2.0 relational SQLite schemas instead.",
)
def parse_flat_event_log(
    flat_records: Sequence[Mapping[str, Any]],
    *,
    case_id_key: str = "case_id",
    activity_key: str = "activity",
    timestamp_key: str = "timestamp",
) -> List[Dict[str, Any]]:
    """Legacy case-centric flat event log parser (deprecated in favor of OCEL 2.0 `StateEncoder`)."""
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(flat_records):
        normalized.append(
            {
                "event_id": str(row.get("event_id", f"flat_evt_{idx}")),
                "case_id": str(row.get(case_id_key, "default_case")),
                "activity": str(row.get(activity_key, "noop")),
                "timestamp": _normalize_timestamp(row.get(timestamp_key, 0.0)),
            }
        )
    return normalized


class FlatLogParser:
    """Legacy case-centric flat log parser retained for backwards compatibility."""

    @deprecated(
        since="1.5.0",
        removed_in="2.0.0",
        alternative="Use StateEncoder for OCEL 2.0 relational event encoding.",
    )
    def parse(self, records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Parse flat case-centric event records into normalized dictionaries."""
        return parse_flat_event_log(records)


def _coerce_csv_scalar(val: str) -> Any:
    """Safely convert a CSV string value into int, float, bool, or str."""
    s = val.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if "." not in s and "e" not in s.lower():
            return int(s)
        fval = float(s)
        if math.isfinite(fval):
            return fval
    except ValueError:
        pass
    return s


def _parse_csv_e2o_cell(cell: str) -> List[OcelObjectTarget]:
    """Parse a CSV E2O cell formatted as JSON or `ObjectType:object_id[:qualifier];...`."""
    s = cell.strip()
    if not s:
        return []
    if s.startswith("[") or s.startswith("{"):
        parsed = json.loads(s)
        items = parsed if isinstance(parsed, list) else [parsed]
        out: List[OcelObjectTarget] = []
        for item in items:
            if isinstance(item, Mapping):
                out.append(
                    OcelObjectTarget(
                        object_id=str(
                            item.get("object_id", item.get("objectId", item.get("id", "")))
                        ),
                        object_type=str(
                            item.get("object_type", item.get("objectType", item.get("type", "")))
                        ),
                        qualifier=str(item.get("qualifier", "participant")),
                        attributes=dict(item.get("attributes", {})),
                    )
                )
        return out

    targets: List[OcelObjectTarget] = []
    for token in re.split(r"[;|]", s):
        tok = token.strip()
        if not tok:
            continue
        parts = [p.strip() for p in tok.split(":")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise StateEncodingError(
                f"Invalid CSV E2O token '{tok}'; expected 'ObjectType:object_id[:qualifier]'."
            )
        otype = parts[0]
        oid = parts[1]
        qual = parts[2] if len(parts) >= 3 and parts[2] else "participant"
        targets.append(OcelObjectTarget(object_id=oid, object_type=otype, qualifier=qual))
    return targets


def _parse_csv_o2o_cell(cell: str) -> List[OcelObjectRelation]:
    """Parse a CSV O2O cell formatted as JSON or `src_id->tgt_id[:qualifier];...`."""
    s = cell.strip()
    if not s:
        return []
    if s.startswith("[") or s.startswith("{"):
        parsed = json.loads(s)
        items = parsed if isinstance(parsed, list) else [parsed]
        out: List[OcelObjectRelation] = []
        for item in items:
            if isinstance(item, Mapping):
                out.append(
                    OcelObjectRelation(
                        source_id=str(item.get("source_id", item.get("source", ""))),
                        target_id=str(
                            item.get("target_id", item.get("objectId", item.get("target", "")))
                        ),
                        qualifier=str(item.get("qualifier", "relates_to")),
                    )
                )
        return out

    relations: List[OcelObjectRelation] = []
    for token in re.split(r"[;|]", s):
        tok = token.strip()
        if not tok:
            continue
        if "->" not in tok:
            raise StateEncodingError(
                f"Invalid CSV O2O token '{tok}'; expected 'source_id->target_id[:qualifier]'."
            )
        lhs, rhs = tok.split("->", 1)
        src_id = lhs.strip()
        rhs_parts = [p.strip() for p in rhs.split(":", 1)]
        tgt_id = rhs_parts[0]
        qual = rhs_parts[1] if len(rhs_parts) > 1 and rhs_parts[1] else "relates_to"
        relations.append(OcelObjectRelation(source_id=src_id, target_id=tgt_id, qualifier=qual))
    return relations


OCEL2_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OCEL 2.0 JSON Exchange Schema",
    "type": "object",
    "required": ["objectTypes", "eventTypes", "objects", "events"],
    "additionalProperties": False,
    "properties": {
        "objectTypes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "attributes"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "type": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
        "eventTypes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "attributes"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "type": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "minLength": 1},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "time", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "time": {"type": "string", "minLength": 1},
                                "value": {},
                            },
                        },
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["objectId", "qualifier"],
                            "additionalProperties": False,
                            "properties": {
                                "objectId": {"type": "string", "minLength": 1},
                                "qualifier": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "time"],
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "minLength": 1},
                    "time": {"type": "string", "minLength": 1},
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "value"],
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "value": {},
                            },
                        },
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["objectId", "qualifier"],
                            "additionalProperties": False,
                            "properties": {
                                "objectId": {"type": "string", "minLength": 1},
                                "qualifier": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
        },
    },
}


def validate_ocel2_json(
    json_path: Union[str, Path],
    *,
    schema_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Validate an exported JSONOCEL 2.0 file against the official OCEL 2.0 JSON Schema and PM4Py validator.

    Verifies:
    1. Structural compliance against `OCEL2_JSON_SCHEMA` via `jsonschema.validate` and
       `pm4py.objects.ocel.validation.jsonocel.apply`.
    2. ISO-8601 timestamp integrity across all events and object attribute entries.
    """
    import jsonschema
    from pm4py.objects.ocel.validation import jsonocel as pm4py_jsonocel

    path = Path(json_path)
    if not path.exists():
        raise StateEncodingError(f"OCEL 2.0 JSON file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=OCEL2_JSON_SCHEMA)

    # Verify ISO-8601 timestamp formatting on all events and object attributes
    for ev in payload.get("events", []):
        _normalize_timestamp(ev.get("time"))
    for obj in payload.get("objects", []):
        for attr in obj.get("attributes", []):
            _normalize_timestamp(attr.get("time"))

    if schema_path is not None:
        return bool(pm4py_jsonocel.apply(str(path), str(schema_path)))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_schema = Path(tmp_dir) / "ocel20_schema.json"
        tmp_schema.write_text(json.dumps(OCEL2_JSON_SCHEMA), encoding="utf-8")
        return bool(pm4py_jsonocel.apply(str(path), str(tmp_schema)))


def export_to_ocel_json(
    source: Union[StateEncoder, str, Path],
    filepath: Union[str, Path],
    *,
    max_payload_bytes: int = 64 * 1024 * 1024,
    chunk_size: int = 2048,
) -> Path:
    """Export an OCEL 2.0 `StateEncoder` or SQLite database path to a compliant OCEL 2.0 JSON file."""
    if isinstance(source, StateEncoder):
        return source.export_to_ocel_json(
            filepath,
            max_payload_bytes=max_payload_bytes,
            chunk_size=chunk_size,
        )
    with StateEncoder(source) as encoder:
        return encoder.export_to_ocel_json(
            filepath,
            max_payload_bytes=max_payload_bytes,
            chunk_size=chunk_size,
        )


def export_to_ocel_xml(
    source: Union[StateEncoder, str, Path],
    filepath: Union[str, Path],
    *,
    max_payload_bytes: int = 64 * 1024 * 1024,
) -> Path:
    """Export an OCEL 2.0 `StateEncoder` or SQLite database path to a compliant OCEL 2.0 XML file."""
    if isinstance(source, StateEncoder):
        return source.export_to_ocel_xml(filepath, max_payload_bytes=max_payload_bytes)
    with StateEncoder(source) as encoder:
        return encoder.export_to_ocel_xml(filepath, max_payload_bytes=max_payload_bytes)


def ingest_csv_event_stream(
    csv_input: Union[str, Path, Iterable[str]],
    db_path: Union[str, Path, StateEncoder],
    *,
    event_id_col: str = "event_id",
    event_type_col: str = "event_type",
    timestamp_col: str = "timestamp",
    objects_col: str = "objects",
    o2o_col: str = "o2o",
    object_columns: Optional[Mapping[str, str]] = None,
    o2o_rules: Optional[Sequence[Tuple[str, str, str]]] = None,
    batch_size: int = 500,
) -> int:
    """Ingest a standard CSV event stream into an OCEL 2.0 SQLite database or `StateEncoder`."""
    if isinstance(db_path, StateEncoder):
        return db_path.ingest_csv(
            csv_input,
            event_id_col=event_id_col,
            event_type_col=event_type_col,
            timestamp_col=timestamp_col,
            objects_col=objects_col,
            o2o_col=o2o_col,
            object_columns=object_columns,
            o2o_rules=o2o_rules,
            batch_size=batch_size,
        )
    with StateEncoder(db_path) as encoder:
        return encoder.ingest_csv(
            csv_input,
            event_id_col=event_id_col,
            event_type_col=event_type_col,
            timestamp_col=timestamp_col,
            objects_col=objects_col,
            o2o_col=o2o_col,
            object_columns=object_columns,
            o2o_rules=o2o_rules,
            batch_size=batch_size,
        )


def ingest_xes_event_log(
    xes_input: Union[str, Path],
    db_path: Union[str, Path, StateEncoder],
    *,
    case_object_type: str = "Case",
    resource_object_type: str = "Resource",
) -> int:
    """Ingest a legacy XES flat log into an OCEL 2.0 SQLite database or `StateEncoder`."""
    if isinstance(db_path, StateEncoder):
        return db_path.ingest_xes(
            xes_input,
            case_object_type=case_object_type,
            resource_object_type=resource_object_type,
        )
    with StateEncoder(db_path) as encoder:
        return encoder.ingest_xes(
            xes_input,
            case_object_type=case_object_type,
            resource_object_type=resource_object_type,
        )


__all__ = [
    "OCEL2_BASE_TABLES",
    "OCEL2_JSON_SCHEMA",
    "FlatLogParser",
    "OcelEventRecord",
    "OcelObjectRelation",
    "OcelObjectTarget",
    "StateEncoder",
    "StateEncodingError",
    "export_to_ocel_json",
    "export_to_ocel_xml",
    "ingest_csv_event_stream",
    "ingest_xes_event_log",
    "parse_flat_event_log",
    "read_ocel2_sqlite",
    "validate_ocel2_json",
    "validate_ocel2_sqlite",
]
