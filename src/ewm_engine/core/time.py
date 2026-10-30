"""Discrete Temporal Semantics, Monotonic EventLog, and Leak-Proof BranchManager (EWM-001).

Formal Temporal Guarantees
--------------------------
1. **Discrete Time Domain ($t \\in \\mathbb{N}_0$)**: Every state step index $t$ is a non-negative integer.
2. **Strict Monotonicity ($t_{k+1} > t_k$)**: Appending a state transition to an `EventLog` requires
   `next_state.step > current_state.step` and `next_state.timestamp >= current_state.timestamp`.
   Retroactive modification of historical states ($t' < t$) is structurally impossible and raises
   `TemporalMonotonicityError`.
3. **Counterfactual Branch Isolation (Pearl, 2009)**: `BranchManager` maintains a Directed Acyclic Graph
   (DAG) of simulation timelines. Spawning a counterfactual branch at step $t$ deep-clones the branch
   point state so that subsequent `do(A_t)` interventions on Branch A share zero mutable object references
   (`id()`) with Branch B or with historical states $t' \\le t$.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.action import Action, Intervention, StateTransitionFn, apply_action
from ewm_engine.core.state import WorldState

__all__ = [
    "BranchManager",
    "BranchMetadata",
    "EventLog",
    "EventRecord",
    "TemporalMonotonicityError",
    "collect_mutable_container_ids",
]


class TemporalMonotonicityError(ValueError):
    """Raised when a state transition violates discrete temporal monotonicity ($t' <= t$)."""


class EventRecord(BaseModel):
    """Immutable, hash-chained record of a single state transition $S_t + A_t \\to S_{t+1}$."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_index: int = Field(ge=0, description="Monotonic 0-based index within the branch EventLog.")
    branch_id: str = Field(min_length=1, description="Identifier of the timeline branch.")
    step_from: int = Field(ge=0, description="Origin discrete time step t in N_0.")
    step_to: int = Field(ge=1, description="Target discrete time step t' > t in N_0.")
    timestamp_from: float = Field(description="Origin continuous simulation timestamp.")
    timestamp_to: float = Field(description="Target continuous simulation timestamp.")
    action_id: str | None = Field(default=None, description="Optional Action ID triggering transition.")
    prior_state_hash: str = Field(description="SHA-256 state hash of S_t.")
    next_state_hash: str = Field(description="SHA-256 state hash of S_{t+1}.")
    prev_event_hash: str = Field(description="Chained SHA-256 hash of the preceding EventRecord.")
    event_hash: str = Field(description="Cryptographic SHA-256 digest of this transition event.")


class BranchMetadata(BaseModel):
    """Immutable metadata node in the counterfactual Temporal DAG."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    branch_id: str
    parent_branch_id: str | None = None
    fork_step: int = Field(ge=0)
    intervention_id: str | None = None
    child_branch_ids: tuple[str, ...] = ()


def _fast_state_digest(state: WorldState, full_hash: bool = False) -> str:
    """Compute either the canonical SHA-256 state hash or a fast structural SHA-256 digest."""
    if full_hash:
        return state.state_hash
    raw_res = object.__getattribute__(state, "resources")
    raw_mem = object.__getattribute__(state, "memory")
    raw_ctx = object.__getattribute__(state, "context")
    raw_rules = object.__getattribute__(state, "active_rules")
    raw_ents = object.__getattribute__(state, "entities")
    payload = (
        f"s={state.step}|ts={state.timestamp:.9f}|e={len(raw_ents)}|"
        f"r={[(k, v.current) for k, v in raw_res.items()]}|"
        f"m={raw_mem!r}|c={raw_ctx!r}|ru={raw_rules!r}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EventLog:
    """Strictly monotonic, append-only event-sourced state trajectory over discrete time $t \\in \\mathbb{N}_0$.

    All stored states are protected against external mutation: `head_state` and `get_state(step)`
    return deep-cloned `WorldState` copies, and any attempt to append a state with `step <= head.step`
    or `timestamp < head.timestamp` raises `TemporalMonotonicityError`.
    """

    def __init__(
        self,
        initial_state: WorldState,
        *,
        branch_id: str = "main",
        full_canonical_hashes: bool = False,
    ) -> None:
        if initial_state.step < 0:
            raise TemporalMonotonicityError(
                f"Initial state step must be in N_0 (step >= 0), got {initial_state.step}."
            )
        if not math.isfinite(initial_state.timestamp):
            raise TemporalMonotonicityError(
                f"Initial state timestamp must be finite, got {initial_state.timestamp!r}."
            )
        self._branch_id: str = branch_id
        self._full_hashes: bool = full_canonical_hashes
        root_clone = initial_state.clone()
        self._states_by_step: dict[int, WorldState] = {root_clone.step: root_clone}
        self._ordered_steps: list[int] = [root_clone.step]
        self._events: list[EventRecord] = []
        self._head_hash: str = _fast_state_digest(root_clone, full_hash=full_canonical_hashes)

    @property
    def branch_id(self) -> str:
        """Identifier of the branch timeline tracked by this EventLog."""
        return self._branch_id

    @property
    def initial_step(self) -> int:
        """Discrete time step $t_0$ at which this EventLog was initialized."""
        return self._ordered_steps[0]

    @property
    def current_step(self) -> int:
        """Current discrete head time step $t_{\\text{head}} \\in \\mathbb{N}_0$."""
        return self._ordered_steps[-1]

    @property
    def head_state(self) -> WorldState:
        """Return an isolated deep copy of the latest WorldState at $t_{\\text{head}}$."""
        return self._states_by_step[self._ordered_steps[-1]].clone()

    @property
    def events(self) -> tuple[EventRecord, ...]:
        """Immutable sequence of transition EventRecords appended to this log."""
        return tuple(self._events)

    @property
    def steps(self) -> tuple[int, ...]:
        """Immutable sequence of discrete time steps recorded in this log."""
        return tuple(self._ordered_steps)

    def __len__(self) -> int:
        """Number of state snapshots recorded in this EventLog."""
        return len(self._ordered_steps)

    def get_state(self, step: int) -> WorldState:
        """Retrieve an isolated deep clone of the historical state at discrete time `step`."""
        if step not in self._states_by_step:
            raise KeyError(
                f"Step {step} not present in EventLog for branch '{self._branch_id}' "
                f"(available steps: {self._ordered_steps[0]}..{self._ordered_steps[-1]})."
            )
        return self._states_by_step[step].clone()

    def append(
        self,
        next_state: WorldState,
        action: Action | None = None,
        *,
        retain_intermediate_states: bool = True,
    ) -> EventRecord:
        """Append a strictly future state $S_{t'}$ ($t' > t_{\\text{head}}$) to the log.

        Parameters
        ----------
        next_state:
            The new immutable `WorldState` at step $t' > t_{\\text{head}}$.
        action:
            Optional `Action` that caused the transition.
        retain_intermediate_states:
            When `True` (default), every step's `WorldState` is retained in `_states_by_step`.
            Structural sharing inside `WorldState` keeps per-step memory overhead minimal.
        """
        head = self._states_by_step[self._ordered_steps[-1]]
        if next_state.step <= head.step:
            raise TemporalMonotonicityError(
                f"Non-monotonic step transition on branch '{self._branch_id}': "
                f"next_state.step={next_state.step} must be strictly greater than current head step={head.step}."
            )
        if not math.isfinite(next_state.timestamp) or next_state.timestamp < head.timestamp:
            raise TemporalMonotonicityError(
                f"Retrograde timestamp on branch '{self._branch_id}': "
                f"next_state.timestamp={next_state.timestamp} < current head timestamp={head.timestamp}."
            )

        prior_hash = self._head_hash
        next_hash = _fast_state_digest(next_state, full_hash=self._full_hashes)
        prev_ev_hash = self._events[-1].event_hash if self._events else ("0" * 64)
        idx = len(self._events)
        act_id = action.id if action is not None else None
        chain_payload = (
            f"{self._branch_id}:{idx}:{head.step}->{next_state.step}:"
            f"{prior_hash}:{next_hash}:{act_id}:{prev_ev_hash}"
        )
        ev_hash = hashlib.sha256(chain_payload.encode("utf-8")).hexdigest()

        record = EventRecord(
            event_index=idx,
            branch_id=self._branch_id,
            step_from=head.step,
            step_to=next_state.step,
            timestamp_from=head.timestamp,
            timestamp_to=next_state.timestamp,
            action_id=act_id,
            prior_state_hash=prior_hash,
            next_state_hash=next_hash,
            prev_event_hash=prev_ev_hash,
            event_hash=ev_hash,
        )

        if not retain_intermediate_states and len(self._ordered_steps) > 1:
            # Keep initial state and current head state only if bounded-memory mode requested
            prev_step = self._ordered_steps[-1]
            self._states_by_step.pop(prev_step, None)

        self._states_by_step[next_state.step] = next_state
        self._ordered_steps.append(next_state.step)
        self._events.append(record)
        self._head_hash = next_hash
        return record


def collect_mutable_container_ids(state: WorldState) -> set[int]:
    """Collect Python `id()` values of all internal mutable containers inside a `WorldState`.

    Used to prove zero shared mutable state across parallel simulation branches.
    """
    ids: set[int] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            ids.add(id(obj))
            for k, v in obj.items():
                _walk(k)
                _walk(v)
        elif isinstance(obj, list | set):
            ids.add(id(obj))
            for item in obj:
                _walk(item)
        elif isinstance(obj, tuple):
            for item in obj:
                _walk(item)
        elif hasattr(obj, "__dict__"):
            for attr_Name in ("entities", "resources", "relationships", "memory", "active_rules", "context", "attributes"):
                try:
                    val = object.__getattribute__(obj, attr_Name)
                except AttributeError:
                    continue
                _walk(val)

    _walk(state)
    return ids


class BranchManager:
    """Counterfactual simulation branch manager with Temporal DAG tracking and strict `id()` isolation."""

    def __init__(
        self,
        initial_state: WorldState,
        *,
        root_branch_id: str = "main",
        full_canonical_hashes: bool = False,
    ) -> None:
        self._root_branch_id = root_branch_id
        self._full_hashes = full_canonical_hashes
        root_log = EventLog(
            initial_state.clone(),
            branch_id=root_branch_id,
            full_canonical_hashes=full_canonical_hashes,
        )
        self._logs: dict[str, EventLog] = {root_branch_id: root_log}
        self._parent_map: dict[str, str | None] = {root_branch_id: None}
        self._fork_step_map: dict[str, int] = {root_branch_id: initial_state.step}
        self._intervention_map: dict[str, str | None] = {root_branch_id: None}
        self._children_map: dict[str, list[str]] = {root_branch_id: []}

    @property
    def root_branch_id(self) -> str:
        """Identifier of the root simulation branch."""
        return self._root_branch_id

    @property
    def branch_ids(self) -> tuple[str, ...]:
        """Tuple of all registered branch identifiers in creation order."""
        return tuple(self._logs.keys())

    def get_log(self, branch_id: str) -> EventLog:
        """Return the `EventLog` for `branch_id` or raise KeyError."""
        if branch_id not in self._logs:
            raise KeyError(f"Unknown simulation branch '{branch_id}'.")
        return self._logs[branch_id]

    def get_head_state(self, branch_id: str) -> WorldState:
        """Return an isolated deep copy of the current head state for `branch_id`."""
        return self.get_log(branch_id).head_state

    def get_state_at(self, branch_id: str, step: int) -> WorldState:
        """Retrieve the historical state at `step` for `branch_id` (traversing ancestors if prior to fork)."""
        curr: str | None = branch_id
        while curr is not None:
            log = self.get_log(curr)
            if step >= log.initial_step:
                return log.get_state(step)
            curr = self._parent_map[curr]
        raise KeyError(f"Step {step} precedes root initial step on branch '{branch_id}'.")

    def get_branch_metadata(self, branch_id: str) -> BranchMetadata:
        """Return the immutable Temporal DAG node metadata for `branch_id`."""
        if branch_id not in self._logs:
            raise KeyError(f"Unknown simulation branch '{branch_id}'.")
        return BranchMetadata(
            branch_id=branch_id,
            parent_branch_id=self._parent_map[branch_id],
            fork_step=self._fork_step_map[branch_id],
            intervention_id=self._intervention_map[branch_id],
            child_branch_ids=tuple(self._children_map[branch_id]),
        )

    def get_dag(self) -> dict[str, BranchMetadata]:
        """Return the complete counterfactual Temporal DAG mapping branch IDs to `BranchMetadata`."""
        return {bid: self.get_branch_metadata(bid) for bid in self._logs}

    def spawn_branch(
        self,
        branch_id: str,
        *,
        parent_branch_id: str | None = None,
        at_step: int | None = None,
        intervention: Intervention | Action | None = None,
    ) -> WorldState:
        """Spawn an isolated counterfactual branch from `parent_branch_id` at discrete time `at_step`.

        Guarantees zero shared mutable container references (`id()`) with the parent branch or any
        sibling branch.
        """
        if branch_id in self._logs:
            raise ValueError(f"Branch '{branch_id}' already exists in BranchManager.")
        parent_id = parent_branch_id or self._root_branch_id
        parent_log = self.get_log(parent_id)
        fork_step = parent_log.current_step if at_step is None else at_step
        if fork_step > parent_log.current_step:
            raise TemporalMonotonicityError(
                f"Cannot spawn branch '{branch_id}' at future step {fork_step} > parent '{parent_id}' head step {parent_log.current_step}."
            )

        base_state = self.get_state_at(parent_id, fork_step).clone()
        intervention_id: str | None = None
        if isinstance(intervention, Intervention):
            base_state = intervention.apply(base_state).clone()
            intervention_id = intervention.id
        elif isinstance(intervention, Action):
            base_state = base_state.with_memory("_fork_action_id", intervention.id).clone()
            intervention_id = intervention.id

        child_log = EventLog(
            base_state,
            branch_id=branch_id,
            full_canonical_hashes=self._full_hashes,
        )
        self._logs[branch_id] = child_log
        self._parent_map[branch_id] = parent_id
        self._fork_step_map[branch_id] = fork_step
        self._intervention_map[branch_id] = intervention_id
        self._children_map[branch_id] = []
        self._children_map[parent_id].append(branch_id)
        return child_log.head_state

    def step_branch(
        self,
        branch_id: str,
        action: Action | None = None,
        transition_fn: StateTransitionFn | None = None,
        *,
        delta_t: float = 1.0,
        retain_intermediate_states: bool = True,
    ) -> WorldState:
        """Execute a single monotonic transition $S_t + A_t \\to S_{t+1}$ on `branch_id`."""
        log = self.get_log(branch_id)
        current = log._states_by_step[log._ordered_steps[-1]]
        act = action or Action(type="noop", timestamp=current.timestamp)
        next_state = apply_action(current, act, transition_fn, delta_t=delta_t)
        log.append(
            next_state,
            act,
            retain_intermediate_states=retain_intermediate_states,
        )
        return next_state

    def verify_branch_isolation(self, branch_a: str, branch_b: str) -> bool:
        """Verify that internal head states of `branch_a` and `branch_b` share zero mutable container `id()`s."""
        if branch_a == branch_b:
            raise ValueError("Cannot compare a branch with itself for cross-branch isolation.")
        log_a = self.get_log(branch_a)
        log_b = self.get_log(branch_b)
        internal_state_a = log_a._states_by_step[log_a.current_step]
        internal_state_b = log_b._states_by_step[log_b.current_step]
        ids_a = collect_mutable_container_ids(internal_state_a)
        ids_b = collect_mutable_container_ids(internal_state_b)
        return ids_a.isdisjoint(ids_b)
