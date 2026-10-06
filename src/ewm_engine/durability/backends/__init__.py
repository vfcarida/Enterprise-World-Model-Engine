"""Storage backends for EventStore and ResultStore."""

from __future__ import annotations

from ewm_engine.durability.backends.filesystem_result_store import FilesystemResultStore
from ewm_engine.durability.backends.in_memory import InMemoryEventStore, InMemoryResultStore
from ewm_engine.durability.backends.json_store import JsonFileEventStore
from ewm_engine.durability.backends.sqlite_store import SqliteEventStore
from ewm_engine.durability.backends.stubs import RedisResultStore, S3ResultStore

__all__ = [
    "FilesystemResultStore",
    "InMemoryEventStore",
    "InMemoryResultStore",
    "JsonFileEventStore",
    "RedisResultStore",
    "S3ResultStore",
    "SqliteEventStore",
]
