"""Filesystem JSON/YAML backend for EventStore.

Persists transition events to disk in structured JSON or YAML files with
atomic file updates and no external dependencies.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml

from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.durability.protocol import EventStore
from ewm_engine.durability.replay import fold_events


class JsonFileEventStore(EventStore):
    """EventStore backend persisting event streams to JSON or YAML files on disk."""

    def __init__(
        self,
        base_dir: str | Path,
        format: Literal["json", "yaml"] = "json",
    ) -> None:
        """Initialize filesystem event store at the given directory."""
        self.base_dir = Path(base_dir)
        self.format = format
        self.streams_dir = self.base_dir / "streams"
        self.streams_dir.mkdir(parents=True, exist_ok=True)

    def _stream_path(self, stream_id: str) -> Path:
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in stream_id)
        ext = "yaml" if self.format == "yaml" else "json"
        return self.streams_dir / f"{sanitized}.{ext}"

    def append(self, stream_id: str, events: Sequence[TransitionEvent]) -> None:
        """Append events to a stream file on disk atomically."""
        if not events:
            return
        path = self._stream_path(stream_id)
        existing_events = self.read_stream(stream_id)
        all_events = existing_events + list(events)

        data = [ev.model_dump(mode="json") for ev in all_events]

        tmp_path = path.with_suffix(f".tmp_{os.getpid()}")
        if self.format == "yaml":
            tmp_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        else:
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        tmp_path.replace(path)

    def read_stream(
        self,
        stream_id: str,
        from_step: int = 0,
        to_step: int | None = None,
    ) -> list[TransitionEvent]:
        """Read ordered transition events from stream file."""
        path = self._stream_path(stream_id)
        if not path.exists():
            return []

        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []

        if self.format == "yaml":
            raw_list = yaml.safe_load(text) or []
        else:
            raw_list = json.loads(text)

        events: list[TransitionEvent] = []
        for item in raw_list:
            ev = TransitionEvent.model_validate(item)
            if ev.step >= from_step and (to_step is None or ev.step <= to_step):
                events.append(ev)

        return sorted(events, key=lambda e: e.step)

    def get_event(self, event_id: str) -> TransitionEvent | None:
        """Search all streams for an event with matching event_id."""
        for stream_id in self.list_streams():
            for ev in self.read_stream(stream_id):
                if ev.event_id == event_id:
                    return ev
        return None

    def list_streams(self) -> list[str]:
        """List all distinct stream identifiers registered in this directory."""
        ext = ".yaml" if self.format == "yaml" else ".json"
        streams: list[str] = []
        for file in self.streams_dir.glob(f"*{ext}"):
            streams.append(file.stem)
        return sorted(streams)

    def fold(self, initial_state: WorldState, stream_id: str) -> WorldState:
        """Reconstruct the final world state of a stream by folding its events."""
        events = self.read_stream(stream_id)
        if not events:
            return initial_state
        return fold_events(initial_state=initial_state, events=events)
