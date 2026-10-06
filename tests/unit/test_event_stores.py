"""Unit tests for EventStore backends (InMemory, JsonFile, Sqlite)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.durability.backends.in_memory import InMemoryEventStore
from ewm_engine.durability.backends.json_store import JsonFileEventStore
from ewm_engine.durability.backends.sqlite_store import SqliteEventStore
from ewm_engine.durability.events import TransitionEvent


def _make_state(step: int = 0, current: float = 100.0) -> WorldState:
    return WorldState(
        step=step,
        entities=[Entity(id="e1", type="node")],
        resources=[Resource(id="r1", current=current)],
    )


def _make_events(
    stream_id: str, s0: WorldState, s1: WorldState, s2: WorldState
) -> list[TransitionEvent]:
    return [
        TransitionEvent(
            stream_id=stream_id,
            step=0,
            parent_fingerprint=s0.fingerprint,
            child_fingerprint=s1.fingerprint,
            timestamp=0.0,
            applied_changes={"r1": -10.0},
            state_snapshot=s1,
        ),
        TransitionEvent(
            stream_id=stream_id,
            step=1,
            parent_fingerprint=s1.fingerprint,
            child_fingerprint=s2.fingerprint,
            timestamp=1.0,
            applied_changes={"r1": -15.0},
            state_snapshot=s2,
        ),
    ]


@pytest.mark.parametrize("store_type", ["in_memory", "json", "yaml", "sqlite_mem", "sqlite_file"])
def test_event_stores_append_read_fold(store_type: str, tmp_path: Path) -> None:
    s0 = _make_state(0, 100.0)
    s1 = _make_state(1, 90.0)
    s2 = _make_state(2, 75.0)

    events = _make_events("stream-1", s0, s1, s2)

    store: InMemoryEventStore | JsonFileEventStore | SqliteEventStore
    if store_type == "in_memory":
        store = InMemoryEventStore()
    elif store_type == "json":
        store = JsonFileEventStore(base_dir=tmp_path / "json_events", format="json")
    elif store_type == "yaml":
        store = JsonFileEventStore(base_dir=tmp_path / "yaml_events", format="yaml")
    elif store_type == "sqlite_mem":
        store = SqliteEventStore(":memory:")
    elif store_type == "sqlite_file":
        store = SqliteEventStore(tmp_path / "events.db")

    # 1. Append
    store.append("stream-1", events)

    # 2. List streams
    streams = store.list_streams()
    assert "stream-1" in streams

    # 3. Read stream
    read_evs = store.read_stream("stream-1")
    assert len(read_evs) == 2
    assert read_evs[0].step == 0
    assert read_evs[0].parent_fingerprint == s0.fingerprint
    assert read_evs[1].step == 1
    assert read_evs[1].child_fingerprint == s2.fingerprint

    # 4. Filtered read
    step1_only = store.read_stream("stream-1", from_step=1, to_step=1)
    assert len(step1_only) == 1
    assert step1_only[0].step == 1

    # 5. Get individual event
    first_ev_id = events[0].event_id
    retrieved = store.get_event(first_ev_id)
    assert retrieved is not None
    assert retrieved.event_id == first_ev_id
    assert retrieved.child_fingerprint == s1.fingerprint

    # 6. Fold over events
    folded_state = store.fold(initial_state=s0, stream_id="stream-1")
    assert folded_state.fingerprint == s2.fingerprint
    assert folded_state.step == 2
    assert folded_state.get_resource("r1").current == 75.0

    if isinstance(store, SqliteEventStore):
        store.close()
