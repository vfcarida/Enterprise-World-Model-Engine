"""Protocols for EventStore and ResultStore.

Conforms to Tracks T1 & T2 in 01_EXPANDED_ROADMAP.md:
- EventStore: append/read/fold abstractions enabling in-memory, SQLite, and future
  adapters (e.g. eventsourcing, KurrentDB) without core modifications.
- ResultStore: get/put/contains/delete keyed by canonical simulation fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.simulation.trajectory import SimulationResult


@runtime_checkable
class EventStore(Protocol):
    """Protocol for event-sourced persistence of simulation state transitions."""

    def append(self, stream_id: str, events: Sequence[TransitionEvent]) -> None:
        """Append one or more transition events to the specified stream."""
        ...

    def read_stream(
        self,
        stream_id: str,
        from_step: int = 0,
        to_step: int | None = None,
    ) -> list[TransitionEvent]:
        """Read ordered transition events for a given stream with optional step range."""
        ...

    def get_event(self, event_id: str) -> TransitionEvent | None:
        """Fetch an individual transition event by globally unique ID."""
        ...

    def list_streams(self) -> list[str]:
        """List all distinct stream identifiers registered in this event store."""
        ...

    def fold(self, initial_state: WorldState, stream_id: str) -> WorldState:
        """Reconstruct the final world state of a stream by folding its events."""
        ...


@runtime_checkable
class ResultStore(Protocol):
    """Protocol for fingerprint-keyed simulation result caching and memoization."""

    def get(self, fingerprint: str) -> SimulationResult | None:
        """Retrieve a cached SimulationResult matching the canonical fingerprint."""
        ...

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a SimulationResult keyed by its canonical SHA-256 fingerprint."""
        ...

    def contains(self, fingerprint: str) -> bool:
        """Check whether a result is cached for the given fingerprint."""
        ...

    def delete(self, fingerprint: str) -> bool:
        """Delete a cached simulation result by fingerprint. Returns True if deleted."""
        ...

    def list_fingerprints(self) -> list[str]:
        """List all cached result fingerprints in the store."""
        ...
