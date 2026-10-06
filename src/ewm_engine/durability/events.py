"""Event-sourced TransitionEvent schema and TraceLog DAG representation.

Conforms to Track T1: Event-Sourced TraceLog & Replay.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import ConstraintResult
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.metadata import ComponentVersion
from ewm_engine.simulation.trajectory import StepRecord, Trajectory


class TransitionEvent(BaseModel):
    """Immutable, versioned canonical transition event in the simulation event store.

    Captures the exact parent state fingerprint, child state fingerprint, actions,
    shocks, applied state changes, constraint violations, and component versions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic schema version.",
    )
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique transition event identifier.",
    )
    stream_id: str = Field(
        description="Stream or rollout trajectory identifier.",
    )
    step: int = Field(
        ge=0,
        description="Discrete simulation time step index.",
    )
    parent_fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint of the world state prior to transition.",
    )
    child_fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint of the world state resulting from transition.",
    )
    timestamp: float = Field(
        ge=0.0,
        description="Continuous simulation time.",
    )
    seed: int | None = Field(
        default=None,
        description="Deterministic RNG seed assigned to this rollout step.",
    )
    actions_proposed: tuple[Action, ...] = Field(
        default_factory=tuple,
        description="Actions proposed by policy or actors prior to constraint validation.",
    )
    actions_accepted: tuple[Action, ...] = Field(
        default_factory=tuple,
        description="Actions accepted after constraint validation and dispatched to dynamics.",
    )
    exogenous_events: tuple[ExogenousEvent, ...] = Field(
        default_factory=tuple,
        description="Exogenous shocks or environment events sampled during this step.",
    )
    applied_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of state deltas and resource flows enacted during transition.",
    )
    constraint_violations: tuple[ConstraintResult, ...] = Field(
        default_factory=tuple,
        description="Constraint evaluations or invariant violations recorded during step.",
    )
    transition_result: TransitionResult | None = Field(
        default=None,
        description="Complete transition result including dynamics output and applied changes.",
    )
    step_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Scalar numerical metrics logged during step.",
    )
    state_snapshot: WorldState | None = Field(
        default=None,
        description="Optional full world state snapshot for instant reconstruction or validation.",
    )
    component_versions: tuple[ComponentVersion, ...] = Field(
        default_factory=tuple,
        description="Versions of dynamics, actors, and constraints governing this transition.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata, scenario context, or execution annotations.",
    )

    @classmethod
    def from_step_record(
        cls,
        stream_id: str,
        record: StepRecord,
        parent_fingerprint: str,
        child_state: WorldState | None = None,
        seed: int | None = None,
        component_versions: Sequence[ComponentVersion] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> TransitionEvent:
        """Create a TransitionEvent from an executed StepRecord."""
        child_fp = (
            child_state.fingerprint
            if child_state is not None
            else (
                record.transition_result.next_state.fingerprint
                if record.transition_result is not None
                else parent_fingerprint
            )
        )
        applied = (
            record.transition_result.applied_changes if record.transition_result is not None else {}
        )
        return cls(
            stream_id=stream_id,
            step=record.step,
            parent_fingerprint=parent_fingerprint,
            child_fingerprint=child_fp,
            timestamp=record.timestamp,
            seed=seed,
            actions_proposed=record.actions_proposed,
            actions_accepted=record.actions_accepted,
            exogenous_events=record.exogenous_events,
            applied_changes=dict(applied),
            transition_result=record.transition_result,
            constraint_violations=record.constraint_violations,
            step_metrics=dict(record.step_metrics),
            state_snapshot=child_state,
            component_versions=tuple(component_versions),
            metadata=dict(metadata or {}),
        )


TransitionEvent.model_rebuild()


class TraceLog:
    """Append-only, branch-aware event log representing a simulation execution DAG.

    Every transition is an immutable TransitionEvent pointing from parent_fingerprint
    to child_fingerprint. Branches are represented as forks in the event DAG, preserving
    strict isolation between lineages.
    """

    def __init__(
        self,
        stream_id: str,
        parent_stream_id: str | None = None,
        fork_fingerprint: str | None = None,
        events: Sequence[TransitionEvent] | None = None,
        initial_state: WorldState | None = None,
    ) -> None:
        """Initialize an append-only TraceLog stream."""
        self.stream_id = stream_id
        self.parent_stream_id = parent_stream_id
        self.fork_fingerprint = fork_fingerprint
        self.initial_state = initial_state
        self._events: list[TransitionEvent] = list(events or [])

    @property
    def events(self) -> tuple[TransitionEvent, ...]:
        """Immutable view of events in this stream."""
        return tuple(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[TransitionEvent]:
        return iter(self._events)

    def append(self, event: TransitionEvent) -> None:
        """Append a new transition event to the log."""
        if event.stream_id != self.stream_id:
            raise ValueError(
                f"Cannot append event with stream_id '{event.stream_id}' "
                f"to TraceLog for stream '{self.stream_id}'."
            )
        if self._events:
            last_event = self._events[-1]
            if event.step <= last_event.step:
                raise ValueError(
                    f"Non-monotonic step index in TraceLog: incoming step {event.step} "
                    f"<= previous step {last_event.step}."
                )
            if event.parent_fingerprint != last_event.child_fingerprint:
                raise ValueError(
                    f"Discontinuous state chain in TraceLog: event parent_fingerprint "
                    f"'{event.parent_fingerprint}' does not match prior child_fingerprint "
                    f"'{last_event.child_fingerprint}'."
                )
        elif self.initial_state is not None:
            if event.parent_fingerprint != self.initial_state.fingerprint:
                raise ValueError(
                    f"Initial event parent_fingerprint '{event.parent_fingerprint}' "
                    f"does not match declared initial_state fingerprint "
                    f"'{self.initial_state.fingerprint}'."
                )
        self._events.append(event)

    def extend(self, events: Sequence[TransitionEvent]) -> None:
        """Append multiple transition events sequentially."""
        for ev in events:
            self.append(ev)

    def branch(
        self,
        new_stream_id: str,
        at_step: int | None = None,
        at_fingerprint: str | None = None,
    ) -> TraceLog:
        """Fork a new TraceLog branch from a specific step or fingerprint in this stream.

        Preserves strict branch isolation: subsequent events appended to the new branch
        do not affect this stream, and vice-versa.
        """
        if new_stream_id == self.stream_id:
            raise ValueError("New branch stream_id must differ from parent stream_id.")

        fork_state: WorldState | None = None
        fork_fp: str | None = None
        copied_events: list[TransitionEvent] = []

        if at_fingerprint is not None:
            matching_idx = None
            if self.initial_state and self.initial_state.fingerprint == at_fingerprint:
                fork_state = self.initial_state
                fork_fp = at_fingerprint
                copied_events = []
            else:
                for idx, ev in enumerate(self._events):
                    if ev.child_fingerprint == at_fingerprint:
                        matching_idx = idx
                        break
                if matching_idx is None:
                    raise KeyError(
                        f"Fingerprint '{at_fingerprint}' not found in TraceLog stream '{self.stream_id}'."
                    )
                fork_fp = at_fingerprint
                fork_state = self._events[matching_idx].state_snapshot
                copied_events = [
                    ev.model_copy(update={"stream_id": new_stream_id})
                    for ev in self._events[: matching_idx + 1]
                ]
        elif at_step is not None:
            if at_step < 0 or at_step >= len(self._events):
                raise IndexError(
                    f"Step index {at_step} out of bounds for TraceLog with {len(self._events)} events."
                )
            target_ev = self._events[at_step]
            fork_fp = target_ev.child_fingerprint
            fork_state = target_ev.state_snapshot
            copied_events = [
                ev.model_copy(update={"stream_id": new_stream_id})
                for ev in self._events[: at_step + 1]
            ]
        else:
            # Fork from the latest event
            if self._events:
                last_ev = self._events[-1]
                fork_fp = last_ev.child_fingerprint
                fork_state = last_ev.state_snapshot
                copied_events = [
                    ev.model_copy(update={"stream_id": new_stream_id}) for ev in self._events
                ]
            else:
                fork_fp = self.initial_state.fingerprint if self.initial_state else ""
                fork_state = self.initial_state
                copied_events = []

        return TraceLog(
            stream_id=new_stream_id,
            parent_stream_id=self.stream_id,
            fork_fingerprint=fork_fp,
            events=copied_events,
            initial_state=fork_state or self.initial_state,
        )

    def get_dag(self) -> dict[str, Any]:
        """Extract nodes and edges representing the event transition DAG."""
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        seen_nodes: set[str] = set()

        if self.initial_state:
            fp = self.initial_state.fingerprint
            nodes.append({"id": fp, "type": "initial_state", "step": 0})
            seen_nodes.add(fp)

        for ev in self._events:
            if ev.parent_fingerprint not in seen_nodes:
                nodes.append({"id": ev.parent_fingerprint, "type": "state", "step": ev.step})
                seen_nodes.add(ev.parent_fingerprint)
            if ev.child_fingerprint not in seen_nodes:
                nodes.append({"id": ev.child_fingerprint, "type": "state", "step": ev.step + 1})
                seen_nodes.add(ev.child_fingerprint)
            edges.append(
                {
                    "source": ev.parent_fingerprint,
                    "target": ev.child_fingerprint,
                    "event_id": ev.event_id,
                    "step": str(ev.step),
                }
            )

        return {
            "stream_id": self.stream_id,
            "parent_stream_id": self.parent_stream_id,
            "fork_fingerprint": self.fork_fingerprint,
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        stream_id: str | None = None,
        component_versions: Sequence[ComponentVersion] = (),
    ) -> TraceLog:
        """Construct a TraceLog from an existing Trajectory object."""
        sid = stream_id or f"trajectory_{trajectory.sample_id}_{trajectory.seed}"
        log = cls(stream_id=sid, initial_state=trajectory.initial_state)

        current_fp = trajectory.initial_state.fingerprint
        for step_idx, step_rec in enumerate(trajectory.steps):
            is_last = step_idx == len(trajectory.steps) - 1
            if is_last and trajectory.final_state is not None:
                child_state = trajectory.final_state
            else:
                child_state = step_rec.transition_result.next_state.advance_time(1.0)
            event = TransitionEvent.from_step_record(
                stream_id=sid,
                record=step_rec,
                parent_fingerprint=current_fp,
                child_state=child_state,
                seed=trajectory.seed,
                component_versions=component_versions,
            )
            log.append(event)
            current_fp = event.child_fingerprint

        return log
