"""In-memory reference backends for EventStore and ResultStore.

Thread-safe, dependency-free reference implementations suitable for testing,
development, and transient in-process execution.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Mapping, Sequence
from typing import Any

from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.durability.protocol import EventStore, ResultStore
from ewm_engine.durability.replay import fold_events
from ewm_engine.simulation.trajectory import SimulationResult


class InMemoryEventStore(EventStore):
    """Thread-safe in-memory implementation of the EventStore protocol."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._streams: dict[str, list[TransitionEvent]] = {}
        self._events_by_id: dict[str, TransitionEvent] = {}
        self._stream_metadata: dict[str, dict[str, Any]] = {}

    def append(self, stream_id: str, events: Sequence[TransitionEvent]) -> None:
        """Append one or more transition events to the specified stream."""
        if not events:
            return
        with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = []
                self._stream_metadata[stream_id] = {"parent_stream_id": None}

            for ev in events:
                if ev.stream_id != stream_id:
                    raise ValueError(
                        f"Event stream_id '{ev.stream_id}' does not match target stream '{stream_id}'."
                    )
                self._streams[stream_id].append(ev)
                self._events_by_id[ev.event_id] = ev

    def read_stream(
        self,
        stream_id: str,
        from_step: int = 0,
        to_step: int | None = None,
    ) -> list[TransitionEvent]:
        """Read ordered transition events for a given stream with optional step range."""
        with self._lock:
            events = self._streams.get(stream_id, [])
            filtered = [
                ev
                for ev in events
                if ev.step >= from_step and (to_step is None or ev.step <= to_step)
            ]
            return list(filtered)

    def get_event(self, event_id: str) -> TransitionEvent | None:
        """Fetch an individual transition event by globally unique ID."""
        with self._lock:
            return self._events_by_id.get(event_id)

    def list_streams(self) -> list[str]:
        """List all distinct stream identifiers registered in this event store."""
        with self._lock:
            return sorted(self._streams.keys())

    def fold(self, initial_state: WorldState, stream_id: str) -> WorldState:
        """Reconstruct the final world state of a stream by folding its events."""
        events = self.read_stream(stream_id)
        if not events:
            return initial_state
        return fold_events(initial_state=initial_state, events=events)


class InMemoryResultStore(ResultStore):
    """Thread-safe in-memory implementation of the ResultStore protocol."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._results: dict[str, SimulationResult] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def get(self, fingerprint: str) -> SimulationResult | None:
        """Retrieve a cached SimulationResult matching the canonical fingerprint."""
        with self._lock:
            res = self._results.get(fingerprint)
            return copy.deepcopy(res) if res is not None else None

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a SimulationResult keyed by its canonical SHA-256 fingerprint."""
        with self._lock:
            self._results[fingerprint] = copy.deepcopy(result)
            self._metadata[fingerprint] = dict(metadata or {})

    def contains(self, fingerprint: str) -> bool:
        """Check whether a result is cached for the given fingerprint."""
        with self._lock:
            return fingerprint in self._results

    def delete(self, fingerprint: str) -> bool:
        """Delete a cached simulation result by fingerprint. Returns True if deleted."""
        with self._lock:
            if fingerprint in self._results:
                del self._results[fingerprint]
                self._metadata.pop(fingerprint, None)
                return True
            return False

    def list_fingerprints(self) -> list[str]:
        """List all cached result fingerprints in the store."""
        with self._lock:
            return sorted(self._results.keys())
