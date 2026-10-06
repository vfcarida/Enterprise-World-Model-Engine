"""Durability, event-sourcing, replay, and result store facilities for EWM Engine.

Conforms to Tracks T1 & T2 in 01_EXPANDED_ROADMAP.md.
"""

from __future__ import annotations

from ewm_engine.durability.backends import (
    FilesystemResultStore,
    InMemoryEventStore,
    InMemoryResultStore,
    JsonFileEventStore,
    RedisResultStore,
    S3ResultStore,
    SqliteEventStore,
)
from ewm_engine.durability.events import (
    TraceLog,
    TransitionEvent,
)
from ewm_engine.durability.memoize import (
    MemoizedSimulationRunner,
    compute_simulation_fingerprint,
    run_memoized,
)
from ewm_engine.durability.protocol import (
    EventStore,
    ResultStore,
)
from ewm_engine.durability.replay import (
    fold_events,
    replay_trajectory,
    verify_trajectory_replay,
)

__all__ = [
    "EventStore",
    "FilesystemResultStore",
    "InMemoryEventStore",
    "InMemoryResultStore",
    "JsonFileEventStore",
    "MemoizedSimulationRunner",
    "RedisResultStore",
    "ResultStore",
    "S3ResultStore",
    "SqliteEventStore",
    "TraceLog",
    "TransitionEvent",
    "compute_simulation_fingerprint",
    "fold_events",
    "replay_trajectory",
    "run_memoized",
    "verify_trajectory_replay",
]
