"""SWE Domain Pack state representation with NetworkX DAGs and merge queues (EWM-005).

Models repository file nodes, directed acyclic code dependency graphs (DAGs),
FIFO pull-request merge queues, downstream build/test queues, and agentic
revision observations without leaking SWE domain logic into the core engine.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.state import Entity, Resource, WorldState

if TYPE_CHECKING:
    import networkx as nx
else:
    try:
        import networkx as nx
    except ImportError:
        nx = None


class _FallbackDiGraph:
    """Lightweight pure-Python directed graph fallback when optional ``networkx`` extra is absent."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._adj: dict[str, set[str]] = {}

    def add_node(self, node_for_adding: str, **attr: Any) -> None:
        self._nodes.setdefault(str(node_for_adding), {}).update(attr)
        self._adj.setdefault(str(node_for_adding), set())

    def add_edge(self, u_of_edge: str, v_of_edge: str) -> None:
        u, v = str(u_of_edge), str(v_of_edge)
        self.add_node(u)
        self.add_node(v)
        self._adj[u].add(v)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(len(targets) for targets in self._adj.values())

    def has_edge(self, u: str, v: str) -> bool:
        return str(v) in self._adj.get(str(u), set())

    def nodes(self, data: bool = False) -> list[Any]:
        if data:
            return [(k, dict(v)) for k, v in self._nodes.items()]
        return list(self._nodes.keys())

    def edges(self) -> list[tuple[str, str]]:
        return [(u, v) for u, targets in self._adj.items() for v in sorted(targets)]


MAX_SWE_GRAPH_NODES: Final[int] = 2000
SWE_REPO_ENTITY_ID: Final[str] = "swe_repository"


class SWEGraphCapacityError(ValueError):
    """Raised when a repository dependency graph exceeds MAX_SWE_GRAPH_NODES."""


class CircularDependencyError(ValueError):
    """Raised when a code dependency graph contains a directed cycle."""

    def __init__(self, cycle_path: tuple[str, ...]) -> None:
        self.cycle_path: tuple[str, ...] = cycle_path
        cycle_str = " -> ".join(cycle_path)
        super().__init__(f"Circular dependency detected in SWE graph: {cycle_str}")


class StaleObservationError(ValueError):
    """Raised when an agent action is based on an outdated file or dependency revision."""


class PullRequestStatus(StrEnum):
    """Lifecycle status of a pull request in the SWE merge queue."""

    PENDING = "PENDING"
    MERGED = "MERGED"
    CONFLICT_STALE = "CONFLICT_STALE"
    REJECTED_CYCLE = "REJECTED_CYCLE"


class CodeFileNode(BaseModel):
    """Immutable representation of a tracked source file in the repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file_path: str = Field(..., min_length=1)
    revision: int = Field(default=1, ge=1)
    last_modified_by: str = Field(default="repo_init", min_length=1)
    content_hash: str = Field(default="sha256:init", min_length=1)
    needs_rebuild: bool = Field(default=False)


class MergeQueueItem(BaseModel):
    """Immutable pull request queued for merge into the repository branch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pr_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    modified_files: tuple[str, ...] = Field(..., min_length=1)
    base_revisions: dict[str, int] = Field(default_factory=dict)
    added_dependencies: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    removed_dependencies: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    status: PullRequestStatus = Field(default=PullRequestStatus.PENDING)
    description: str = Field(default="")


class StaleObservationRecord(BaseModel):
    """Audit record produced when an agent operates on a stale file/dependency snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(..., min_length=1)
    action_id: str = Field(..., min_length=1)
    pr_id: str | None = Field(default=None)
    file_path: str = Field(..., min_length=1)
    expected_revision: int = Field(..., ge=1)
    actual_revision: int = Field(..., ge=1)
    reason: str = Field(..., min_length=1)


def detect_cycle_iterative_o_v_plus_e(
    dependencies: Mapping[str, Sequence[str]],
    all_nodes: Iterable[str] | None = None,
) -> tuple[str, ...] | None:
    """Detect directed cycles in O(V + E) time using iterative algorithms only.

    First performs an iterative Kahn's topological pass in O(V + E) time.
    If unvisited nodes remain (indicating at least one cycle), extracts a
    deterministic cycle path ``(u, ..., v, u)`` using an explicit stack-based
    3-color depth-first search (avoiding Python's recursion limit on large graphs).

    Args:
        dependencies: Mapping of ``source_file -> sequence of target_files``
            where ``source_file`` depends on ``target_file``.
        all_nodes: Optional iterable of additional isolated node IDs.

    Returns:
        A tuple ``(n_0, n_1, ..., n_0)`` representing a directed cycle if one
        exists, or ``None`` if the graph is a valid DAG.
    """
    node_set: set[str] = set(dependencies.keys())
    if all_nodes is not None:
        node_set.update(all_nodes)
    for targets in dependencies.values():
        node_set.update(targets)

    if not node_set:
        return None

    sorted_nodes = sorted(node_set)
    in_degree: dict[str, int] = dict.fromkeys(sorted_nodes, 0)
    adj: dict[str, list[str]] = {node: [] for node in sorted_nodes}

    for src in sorted_nodes:
        raw_targets = dependencies.get(src, ())
        seen_targets: set[str] = set()
        for dst in raw_targets:
            if dst == src:
                return (src, src)
            if dst not in seen_targets:
                seen_targets.add(dst)
                adj[src].append(dst)
                in_degree[dst] = in_degree.get(dst, 0) + 1

    queue: deque[str] = deque(node for node in sorted_nodes if in_degree[node] == 0)
    visited_count = 0

    while queue:
        current = queue.popleft()
        visited_count += 1
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited_count == len(sorted_nodes):
        return None

    # Iterative 3-color DFS restricted to nodes still having in_degree > 0
    # Color coding: 0 = WHITE (unvisited), 1 = GRAY (on active stack), 2 = BLACK (completed)
    color: dict[str, int] = dict.fromkeys(sorted_nodes, 0)

    for start_node in sorted_nodes:
        if in_degree[start_node] == 0 or color[start_node] != 0:
            continue

        # Explicit stack holds (node, neighbor_index)
        stack: list[tuple[str, int]] = [(start_node, 0)]
        color[start_node] = 1

        while stack:
            curr_node, idx = stack[-1]
            neighbors = adj[curr_node]
            if idx < len(neighbors):
                stack[-1] = (curr_node, idx + 1)
                nxt = neighbors[idx]
                if color[nxt] == 1:
                    # Reconstruct cycle from active stack
                    cycle_nodes: list[str] = [nxt]
                    for stack_node, _ in reversed(stack):
                        cycle_nodes.append(stack_node)
                        if stack_node == nxt:
                            break
                    cycle_nodes.reverse()
                    return tuple(cycle_nodes)
                if color[nxt] == 0:
                    color[nxt] = 1
                    stack.append((nxt, 0))
            else:
                color[curr_node] = 2
                stack.pop()

    return None


def find_downstream_dependents_iterative(
    dependencies: Mapping[str, Sequence[str]],
    modified_files: Iterable[str],
    all_nodes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Compute all files needing rebuild when ``modified_files`` change in O(V + E) time.

    In the dependency graph, an edge ``(A, B)`` means file ``A`` depends on file ``B``.
    When file ``B`` is modified, ``B`` itself and all dependents ``A`` that transitively
    depend on ``B`` (i.e. ``A ->* B``) must be rebuilt.

    Returns:
        Tuple of affected file paths ordered deterministically so dependencies are
        rebuilt before the files that depend on them.
    """
    node_set: set[str] = set(dependencies.keys())
    if all_nodes is not None:
        node_set.update(all_nodes)
    for targets in dependencies.values():
        node_set.update(targets)

    # Build reverse adjacency list: target_dependency -> list of dependent_files
    reverse_adj: dict[str, list[str]] = {node: [] for node in node_set}
    for dependent, targets in dependencies.items():
        for target in targets:
            reverse_adj.setdefault(target, []).append(dependent)

    visited: set[str] = set()
    bfs_queue: deque[str] = deque()
    ordered_affected: list[str] = []

    for fpath in sorted(set(modified_files)):
        if fpath in node_set and fpath not in visited:
            visited.add(fpath)
            bfs_queue.append(fpath)

    while bfs_queue:
        current = bfs_queue.popleft()
        ordered_affected.append(current)
        for dependent in sorted(reverse_adj.get(current, ())):
            if dependent not in visited:
                visited.add(dependent)
                bfs_queue.append(dependent)

    return tuple(ordered_affected)


def build_swe_action(
    action_id: str,
    action_type: str,
    *,
    actor_id: str = "agent_a",
    parameters: Mapping[str, Any] | None = None,
) -> Action:
    """Create a domain-neutral core ``Action`` for an SWE repository operation."""
    params: dict[str, Any] = dict(parameters or {})
    params["action_type"] = action_type.upper()
    params.setdefault("agent_id", actor_id)
    return Action(
        id=action_id,
        type=action_type.upper(),
        actor_id=actor_id,
        parameters=params,
    )


class SWEState(BaseModel):
    """Immutable domain state for an agentic software engineering repository.

    Models:
    - A directed acyclic graph (DAG) of code file dependencies backed by ``networkx.DiGraph``
    - A FIFO queue of pending ``MergeQueueItem`` actions
    - A deterministic ``build_queue`` of files requiring compilation/testing after changes
    - Agent observation snapshots and stale observation audit records
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(default=0, ge=0)
    repository_id: str = Field(default=SWE_REPO_ENTITY_ID, min_length=1)
    files: dict[str, CodeFileNode] = Field(default_factory=dict)
    dependencies: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    merge_queue: tuple[MergeQueueItem, ...] = Field(default_factory=tuple)
    merged_prs: tuple[str, ...] = Field(default_factory=tuple)
    build_queue: tuple[str, ...] = Field(default_factory=tuple)
    agent_observations: dict[str, dict[str, int]] = Field(default_factory=dict)
    stale_observations: tuple[StaleObservationRecord, ...] = Field(default_factory=tuple)
    max_graph_nodes: int = Field(default=MAX_SWE_GRAPH_NODES, ge=1, le=MAX_SWE_GRAPH_NODES)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_graph_bounds_and_acyclicity()

    def _validate_graph_bounds_and_acyclicity(self) -> SWEState:
        all_nodes: set[str] = set(self.files.keys()) | set(self.dependencies.keys())
        for targets in self.dependencies.values():
            all_nodes.update(targets)

        if len(all_nodes) > self.max_graph_nodes:
            raise SWEGraphCapacityError(
                f"SWE dependency graph contains {len(all_nodes)} nodes, "
                f"exceeding max_graph_nodes={self.max_graph_nodes}"
            )

        missing_files = all_nodes - set(self.files.keys())
        if missing_files:
            raise ValueError(f"Dependencies reference untracked files: {sorted(missing_files)}")

        cycle = detect_cycle_iterative_o_v_plus_e(self.dependencies, all_nodes)
        if cycle is not None:
            raise CircularDependencyError(cycle)

        return self

    @property
    def dependency_graph(self) -> nx.DiGraph:
        """Build a fresh ``networkx.DiGraph`` representing the repository dependency DAG.

        Returns a new graph instance on every access so external callers cannot
        mutate ``SWEState`` in place. Falls back to ``_FallbackDiGraph`` if the
        optional ``networkx`` extra is not installed.
        """
        dag: Any = nx.DiGraph() if nx is not None else _FallbackDiGraph()
        for fpath, node in sorted(self.files.items()):
            dag.add_node(
                fpath,
                revision=node.revision,
                last_modified_by=node.last_modified_by,
                content_hash=node.content_hash,
                needs_rebuild=node.needs_rebuild,
            )
        for dependent, targets in sorted(self.dependencies.items()):
            for target in targets:
                dag.add_edge(dependent, target)
        return dag

    @classmethod
    def from_networkx_dag(
        cls,
        dag: nx.DiGraph,
        *,
        step: int = 0,
        repository_id: str = SWE_REPO_ENTITY_ID,
        merge_queue: Sequence[MergeQueueItem] = (),
        merged_prs: Sequence[str] = (),
        build_queue: Sequence[str] = (),
        agent_observations: Mapping[str, Mapping[str, int]] | None = None,
        stale_observations: Sequence[StaleObservationRecord] = (),
        max_graph_nodes: int = MAX_SWE_GRAPH_NODES,
    ) -> SWEState:
        """Construct an immutable ``SWEState`` from a ``networkx.DiGraph``."""
        if dag.number_of_nodes() > max_graph_nodes:
            raise SWEGraphCapacityError(
                f"NetworkX graph contains {dag.number_of_nodes()} nodes, "
                f"exceeding max_graph_nodes={max_graph_nodes}"
            )

        files: dict[str, CodeFileNode] = {}
        dependencies: dict[str, tuple[str, ...]] = {}

        for raw_node, attrs in dag.nodes(data=True):
            fpath = str(raw_node)
            files[fpath] = CodeFileNode(
                file_path=fpath,
                revision=int(attrs.get("revision", 1)),
                last_modified_by=str(attrs.get("last_modified_by", "repo_init")),
                content_hash=str(attrs.get("content_hash", f"sha256:{fpath}:v1")),
                needs_rebuild=bool(attrs.get("needs_rebuild", False)),
            )
            dependencies[fpath] = ()

        grouped_edges: dict[str, list[str]] = {fpath: [] for fpath in files}
        for u, v in dag.edges():
            src = str(u)
            dst = str(v)
            grouped_edges.setdefault(src, []).append(dst)

        for fpath, targets in grouped_edges.items():
            dependencies[fpath] = tuple(sorted(set(targets)))

        obs_dict: dict[str, dict[str, int]] = (
            {agent: dict(revs) for agent, revs in agent_observations.items()}
            if agent_observations is not None
            else {}
        )

        return cls(
            step=step,
            repository_id=repository_id,
            files=files,
            dependencies=dependencies,
            merge_queue=tuple(merge_queue),
            merged_prs=tuple(merged_prs),
            build_queue=tuple(build_queue),
            agent_observations=obs_dict,
            stale_observations=tuple(stale_observations),
            max_graph_nodes=max_graph_nodes,
        )

    def observe_files(self, agent_id: str, file_paths: Iterable[str] | None = None) -> SWEState:
        """Record an agent's observation of current file revisions."""
        target_paths = list(self.files.keys()) if file_paths is None else list(file_paths)
        snapshot: dict[str, int] = dict(self.agent_observations.get(agent_id, {}))
        for fpath in target_paths:
            if fpath not in self.files:
                raise KeyError(f"Unknown file '{fpath}' observed by agent '{agent_id}'")
            snapshot[fpath] = self.files[fpath].revision

        new_obs = dict(self.agent_observations)
        new_obs[agent_id] = snapshot
        return self.model_copy(update={"agent_observations": new_obs})

    def enqueue_pr(self, pr: MergeQueueItem) -> SWEState:
        """Return a new ``SWEState`` with ``pr`` appended to the FIFO ``merge_queue``."""
        for existing in self.merge_queue:
            if existing.pr_id == pr.pr_id:
                raise ValueError(f"PR '{pr.pr_id}' is already in merge_queue")
        return self.model_copy(update={"merge_queue": (*self.merge_queue, pr)})

    def relevant_dependency_closure(self, modified_files: Iterable[str]) -> set[str]:
        """Return modified files together with their direct upstream & downstream neighbors."""
        closure: set[str] = set()
        for fpath in modified_files:
            closure.add(fpath)
            closure.update(self.dependencies.get(fpath, ()))
            for dependent, targets in self.dependencies.items():
                if fpath in targets:
                    closure.add(dependent)
        return closure

    def to_world_state(self, *, step: int | None = None) -> WorldState:
        """Serialize this ``SWEState`` into a domain-neutral core ``WorldState``."""
        effective_step = self.step if step is None else step
        pending_count = sum(
            1 for item in self.merge_queue if item.status == PullRequestStatus.PENDING
        )
        swe_payload: dict[str, Any] = {
            "repository_id": self.repository_id,
            "files": {
                fpath: node.model_dump(mode="python") for fpath, node in sorted(self.files.items())
            },
            "dependencies": {
                fpath: list(targets) for fpath, targets in sorted(self.dependencies.items())
            },
            "merge_queue": [item.model_dump(mode="python") for item in self.merge_queue],
            "merged_prs": list(self.merged_prs),
            "build_queue": list(self.build_queue),
            "agent_observations": {
                agent: dict(revs) for agent, revs in sorted(self.agent_observations.items())
            },
            "stale_observations": [
                rec.model_dump(mode="python") for rec in self.stale_observations
            ],
            "max_graph_nodes": self.max_graph_nodes,
        }
        repo_entity = Entity(
            id=self.repository_id,
            type="swe_repository",
            attributes={
                "node_count": len(self.files),
                "pending_merges": pending_count,
                "build_queue_length": len(self.build_queue),
                "stale_observation_count": len(self.stale_observations),
            },
        )
        resources: dict[str, Resource] = {
            "build_queue_size": Resource(
                id="build_queue_size",
                current=float(len(self.build_queue)),
                min_value=0.0,
                max_value=float(self.max_graph_nodes),
            ),
            "pending_merge_count": Resource(
                id="pending_merge_count",
                current=float(pending_count),
                min_value=0.0,
                max_value=float(self.max_graph_nodes),
            ),
            "stale_observation_count": Resource(
                id="stale_observation_count",
                current=float(len(self.stale_observations)),
                min_value=0.0,
                max_value=float(self.max_graph_nodes * 10),
            ),
        }
        return WorldState(
            step=effective_step,
            timestamp=float(effective_step),
            entities={self.repository_id: repo_entity},
            resources=resources,
            memory={self.repository_id: swe_payload},
        )

    @classmethod
    def from_world_state(
        cls,
        world_state: WorldState,
        repository_id: str = SWE_REPO_ENTITY_ID,
    ) -> SWEState:
        """Reconstruct a validated ``SWEState`` from a core ``WorldState``."""
        raw_payload = world_state.memory.get(repository_id)
        if not isinstance(raw_payload, Mapping):
            raise KeyError(f"WorldState memory does not contain SWE repository '{repository_id}'")

        raw_files: Mapping[str, Any] = raw_payload.get("files", {})
        raw_deps: Mapping[str, Any] = raw_payload.get("dependencies", {})
        raw_queue: Sequence[Any] = raw_payload.get("merge_queue", ())
        raw_merged: Sequence[Any] = raw_payload.get("merged_prs", ())
        raw_build: Sequence[Any] = raw_payload.get("build_queue", ())
        raw_obs: Mapping[str, Any] = raw_payload.get("agent_observations", {})
        raw_stale: Sequence[Any] = raw_payload.get("stale_observations", ())
        max_nodes = int(raw_payload.get("max_graph_nodes", MAX_SWE_GRAPH_NODES))

        files = {
            str(fpath): (
                node_data
                if isinstance(node_data, CodeFileNode)
                else CodeFileNode.model_validate(node_data)
            )
            for fpath, node_data in raw_files.items()
        }
        dependencies = {
            str(fpath): tuple(str(target) for target in targets)
            for fpath, targets in raw_deps.items()
        }
        merge_queue = tuple(
            item if isinstance(item, MergeQueueItem) else MergeQueueItem.model_validate(item)
            for item in raw_queue
        )
        stale_observations = tuple(
            rec
            if isinstance(rec, StaleObservationRecord)
            else StaleObservationRecord.model_validate(rec)
            for rec in raw_stale
        )
        agent_observations = {
            str(agent): {str(fp): int(rev) for fp, rev in dict(revs).items()}
            for agent, revs in raw_obs.items()
        }

        return cls(
            step=int(world_state.step),
            repository_id=str(raw_payload.get("repository_id", repository_id)),
            files=files,
            dependencies=dependencies,
            merge_queue=merge_queue,
            merged_prs=tuple(str(pr) for pr in raw_merged),
            build_queue=tuple(str(fp) for fp in raw_build),
            agent_observations=agent_observations,
            stale_observations=stale_observations,
            max_graph_nodes=max_nodes,
        )


__all__ = [
    "MAX_SWE_GRAPH_NODES",
    "SWE_REPO_ENTITY_ID",
    "CircularDependencyError",
    "CodeFileNode",
    "MergeQueueItem",
    "PullRequestStatus",
    "SWEGraphCapacityError",
    "SWEState",
    "StaleObservationError",
    "StaleObservationRecord",
    "build_swe_action",
    "detect_cycle_iterative_o_v_plus_e",
    "find_downstream_dependents_iterative",
]
