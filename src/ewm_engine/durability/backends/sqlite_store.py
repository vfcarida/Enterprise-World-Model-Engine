"""Standard-library SQLite backend for EventStore.

Provides embedded relational persistence for transition events, stream metadata,
and DAG forks with zero external dependencies and WAL-mode concurrency.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.durability.protocol import EventStore
from ewm_engine.durability.replay import fold_events


class SqliteEventStore(EventStore):
    """EventStore implementation backed by Python's built-in sqlite3."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """Initialize SQLite event store.

        Args:
            db_path: Path to database file or ':memory:' for an in-process SQLite instance.
        """
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._lock = threading.RLock()

        # For in-memory, preserve a single connection across threads; for file, open per thread/op
        if self._is_memory:
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
        else:
            self._shared_conn = None

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS streams (
                        stream_id TEXT PRIMARY KEY,
                        parent_stream_id TEXT,
                        fork_fingerprint TEXT,
                        created_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        stream_id TEXT NOT NULL,
                        step INTEGER NOT NULL,
                        parent_fingerprint TEXT NOT NULL,
                        child_fingerprint TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        seed INTEGER,
                        payload TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        FOREIGN KEY(stream_id) REFERENCES streams(stream_id)
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_stream_step ON events(stream_id, step);
                    CREATE INDEX IF NOT EXISTS idx_events_parent ON events(parent_fingerprint);
                    CREATE INDEX IF NOT EXISTS idx_events_child ON events(child_fingerprint);
                    """
                )
                conn.commit()
            finally:
                if self._shared_conn is None:
                    conn.close()

    def register_stream(
        self,
        stream_id: str,
        parent_stream_id: str | None = None,
        fork_fingerprint: str | None = None,
    ) -> None:
        """Register a stream metadata header with optional parent fork reference."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO streams (stream_id, parent_stream_id, fork_fingerprint, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (stream_id, parent_stream_id, fork_fingerprint, time.time()),
                )
                conn.commit()
            finally:
                if self._shared_conn is None:
                    conn.close()

    def append(self, stream_id: str, events: Sequence[TransitionEvent]) -> None:
        """Append one or more transition events to the stream in a single transaction."""
        if not events:
            return

        with self._lock:
            conn = self._get_connection()
            try:
                now = time.time()
                # Ensure stream exists
                conn.execute(
                    "INSERT OR IGNORE INTO streams (stream_id, parent_stream_id, fork_fingerprint, created_at) VALUES (?, NULL, NULL, ?)",
                    (stream_id, now),
                )
                cursor = conn.cursor()
                rows = [
                    (
                        ev.event_id,
                        stream_id,
                        ev.step,
                        ev.parent_fingerprint,
                        ev.child_fingerprint,
                        ev.timestamp,
                        ev.seed,
                        json.dumps(ev.model_dump(mode="json")),
                        now,
                    )
                    for ev in events
                ]
                cursor.executemany(
                    """
                    INSERT INTO events (event_id, stream_id, step, parent_fingerprint, child_fingerprint, timestamp, seed, payload, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
            finally:
                if self._shared_conn is None:
                    conn.close()

    def read_stream(
        self,
        stream_id: str,
        from_step: int = 0,
        to_step: int | None = None,
    ) -> list[TransitionEvent]:
        """Read ordered transition events for a given stream with optional step range."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                if to_step is None:
                    cursor.execute(
                        "SELECT payload FROM events WHERE stream_id = ? AND step >= ? ORDER BY step ASC",
                        (stream_id, from_step),
                    )
                else:
                    cursor.execute(
                        "SELECT payload FROM events WHERE stream_id = ? AND step >= ? AND step <= ? ORDER BY step ASC",
                        (stream_id, from_step, to_step),
                    )
                results: list[TransitionEvent] = []
                for (payload_str,) in cursor.fetchall():
                    data = json.loads(payload_str)
                    results.append(TransitionEvent.model_validate(data))
                return results
            finally:
                if self._shared_conn is None:
                    conn.close()

    def get_event(self, event_id: str) -> TransitionEvent | None:
        """Fetch an individual transition event by globally unique ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT payload FROM events WHERE event_id = ?", (event_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                data = json.loads(row[0])
                return TransitionEvent.model_validate(data)
            finally:
                if self._shared_conn is None:
                    conn.close()

    def list_streams(self) -> list[str]:
        """List all distinct stream identifiers registered in SQLite."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT stream_id FROM streams ORDER BY stream_id ASC")
                return [row[0] for row in cursor.fetchall()]
            finally:
                if self._shared_conn is None:
                    conn.close()

    def fold(self, initial_state: WorldState, stream_id: str) -> WorldState:
        """Reconstruct the final world state of a stream by folding its events."""
        events = self.read_stream(stream_id)
        if not events:
            return initial_state
        return fold_events(initial_state=initial_state, events=events)

    def close(self) -> None:
        """Close shared connection if open."""
        with self._lock:
            if self._shared_conn is not None:
                self._shared_conn.close()
                self._shared_conn = None
