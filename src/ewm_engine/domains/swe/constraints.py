"""SWE Domain Pack structural constraints for DAG acyclicity and stale observations (EWM-005).

Provides ``AcyclicDependencyConstraint`` which plugs into ``HybridTransitionRouter``
(EWM-002) and ``ConstraintRegistry`` without leaking SWE logic into the core engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.domains.swe.state import (
    MAX_SWE_GRAPH_NODES,
    SWE_REPO_ENTITY_ID,
    PullRequestStatus,
    StaleObservationRecord,
    SWEState,
    detect_cycle_iterative_o_v_plus_e,
)


class StructuralViolationError(ValueError):
    """Raised when an SWE action violates a hard structural constraint (cycle, capacity, or stale check)."""


def detect_stale_observations_for_action(
    swe_state: SWEState,
    *,
    agent_id: str,
    action_id: str,
    modified_files: Iterable[str],
    base_revisions: Mapping[str, int] | None = None,
    pr_id: str | None = None,
) -> tuple[StaleObservationRecord, ...]:
    """Check whether an agent's action or PR is operating on stale file/dependency revisions.

    Inspects both the directly modified files and their dependency neighborhood
    (files that ``modified_files`` depend on, and files that depend on ``modified_files``).
    If any file in that closure has advanced beyond the revision observed by ``agent_id``,
    a ``StaleObservationRecord`` is emitted.
    """
    modified_set = {str(fp) for fp in modified_files}
    observed_revs: dict[str, int] = dict(swe_state.agent_observations.get(agent_id, {}))
    if base_revisions is not None:
        for fp, rev in base_revisions.items():
            observed_revs[str(fp)] = int(rev)

    if not observed_revs:
        return ()

    closure = swe_state.relevant_dependency_closure(modified_set)
    records: list[StaleObservationRecord] = []

    for fpath in sorted(closure):
        if fpath not in observed_revs or fpath not in swe_state.files:
            continue
        expected_rev = observed_revs[fpath]
        actual_node = swe_state.files[fpath]
        if actual_node.revision != expected_rev:
            conflict_kind = (
                "direct_file_conflict"
                if fpath in modified_set
                else "dependency_revision_drift"
            )
            reason = (
                f"{conflict_kind}: agent '{agent_id}' observed '{fpath}' at "
                f"rev {expected_rev}, but current rev is {actual_node.revision} "
                f"(last modified by '{actual_node.last_modified_by}')"
            )
            records.append(
                StaleObservationRecord(
                    agent_id=agent_id,
                    action_id=action_id,
                    pr_id=pr_id,
                    file_path=fpath,
                    expected_revision=expected_rev,
                    actual_revision=actual_node.revision,
                    reason=reason,
                )
            )

    return tuple(records)


def _preview_dependencies_with_edges(
    current_dependencies: Mapping[str, Sequence[str]],
    added_edges: Iterable[tuple[str, str]],
    removed_edges: Iterable[tuple[str, str]] = (),
) -> dict[str, tuple[str, ...]]:
    """Build a candidate adjacency map after applying edge additions and removals."""
    removed_set = {(str(u), str(v)) for u, v in removed_edges}
    adj: dict[str, set[str]] = {
        str(src): {
            str(dst)
            for dst in targets
            if (str(src), str(dst)) not in removed_set
        }
        for src, targets in current_dependencies.items()
    }
    for u, v in added_edges:
        src = str(u)
        dst = str(v)
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set())
    return {src: tuple(sorted(targets)) for src, targets in adj.items()}


class AcyclicDependencyConstraint:
    """Structural constraint enforcing O(V + E) DAG acyclicity and graph bounds.

    Optionally rejects transitions when an agent attempts to commit or merge
    against a stale observation snapshot (``reject_stale_observations=True``).
    """

    def __init__(
        self,
        *,
        constraint_id: str = "swe_acyclic_dependency_constraint",
        version: str = "1.0.0",
        repository_id: str = SWE_REPO_ENTITY_ID,
        max_graph_nodes: int = MAX_SWE_GRAPH_NODES,
        reject_stale_observations: bool = False,
        raise_on_violation: bool = False,
    ) -> None:
        if max_graph_nodes < 1 or max_graph_nodes > MAX_SWE_GRAPH_NODES:
            raise ValueError(
                f"max_graph_nodes must be in [1, {MAX_SWE_GRAPH_NODES}], got {max_graph_nodes}"
            )
        self._constraint_id = constraint_id
        self._version = version
        self._repository_id = repository_id
        self._max_graph_nodes = max_graph_nodes
        self._reject_stale_observations = reject_stale_observations
        self._raise_on_violation = raise_on_violation

    @property
    def constraint_id(self) -> str:
        return self._constraint_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return (
            "Enforces O(V + E) directed acyclic dependency graph (DAG) invariants, "
            "node capacity bounds, and stale agent observation checks."
        )

    def _check_single_action_pre(
        self,
        swe_state: SWEState,
        action: Action,
    ) -> tuple[bool, str]:
        params = action.parameters
        action_type = str(params.get("action_type", action.type)).upper()

        added_edges: list[tuple[str, str]] = []
        removed_edges: list[tuple[str, str]] = []
        new_files: set[str] = set()

        if action_type == "ADD_DEPENDENCY":
            src = str(params.get("source_file", ""))
            dst = str(params.get("target_file", ""))
            if not src or not dst:
                return (
                    False,
                    "ADD_DEPENDENCY requires non-empty 'source_file' and 'target_file'",
                )
            added_edges.append((src, dst))
            new_files.update((src, dst))

        elif action_type in {"ENQUEUE_PR", "OPEN_PR"}:
            raw_added: Sequence[Any] = params.get("added_dependencies", ())
            raw_removed: Sequence[Any] = params.get("removed_dependencies", ())
            raw_modified: Sequence[Any] = params.get("modified_files", ())
            for edge in raw_added:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    u, v = str(edge[0]), str(edge[1])
                    added_edges.append((u, v))
                    new_files.update((u, v))
            for edge in raw_removed:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    removed_edges.append((str(edge[0]), str(edge[1])))
            for fp in raw_modified:
                new_files.add(str(fp))

        elif action_type == "MERGE_PR":
            target_pr_id = params.get("pr_id")
            queue_item = None
            for item in swe_state.merge_queue:
                if item.status == PullRequestStatus.PENDING and (
                    target_pr_id is None or item.pr_id == str(target_pr_id)
                ):
                    queue_item = item
                    break
            if queue_item is not None:
                added_edges.extend(queue_item.added_dependencies)
                removed_edges.extend(queue_item.removed_dependencies)
                new_files.update(queue_item.modified_files)
                for u, v in queue_item.added_dependencies:
                    new_files.update((u, v))

                strict_stale = bool(
                    params.get("strict_stale_check", self._reject_stale_observations)
                )
                if strict_stale:
                    stale_records = detect_stale_observations_for_action(
                        swe_state,
                        agent_id=queue_item.agent_id,
                        action_id=action.id,
                        modified_files=queue_item.modified_files,
                        base_revisions=queue_item.base_revisions,
                        pr_id=queue_item.pr_id,
                    )
                    if stale_records:
                        return (
                            False,
                            f"Stale observation rejected on PR '{queue_item.pr_id}': "
                            f"{stale_records[0].reason}",
                        )

        elif action_type == "MODIFY_FILE":
            target_file = str(params.get("file_path", ""))
            if target_file:
                new_files.add(target_file)
            raw_added = params.get("added_dependencies", ())
            for edge in raw_added:
                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    u, v = str(edge[0]), str(edge[1])
                    added_edges.append((u, v))
                    new_files.update((u, v))

            strict_stale = bool(
                params.get("strict_stale_check", self._reject_stale_observations)
            )
            if strict_stale and target_file:
                agent_id = str(params.get("agent_id", "agent_unknown"))
                raw_base = params.get("base_revisions")
                base_revs = (
                    {str(k): int(v) for k, v in dict(raw_base).items()}
                    if isinstance(raw_base, Mapping)
                    else None
                )
                stale_records = detect_stale_observations_for_action(
                    swe_state,
                    agent_id=agent_id,
                    action_id=action.id,
                    modified_files=(target_file,),
                    base_revisions=base_revs,
                )
                if stale_records:
                    return (
                        False,
                        f"Stale observation rejected: {stale_records[0].reason}",
                    )

        candidate_nodes = set(swe_state.files.keys()) | new_files
        limit = min(self._max_graph_nodes, swe_state.max_graph_nodes)
        if len(candidate_nodes) > limit:
            return (
                False,
                f"Action would grow SWE graph to {len(candidate_nodes)} nodes, "
                f"exceeding limit {limit}",
            )

        if added_edges or removed_edges:
            candidate_deps = _preview_dependencies_with_edges(
                swe_state.dependencies,
                added_edges=added_edges,
                removed_edges=removed_edges,
            )
            cycle = detect_cycle_iterative_o_v_plus_e(candidate_deps, candidate_nodes)
            if cycle is not None:
                cycle_str = " -> ".join(cycle)
                return (
                    False,
                    f"Action introduces circular dependency: {cycle_str}",
                )

        return True, "Pre-action SWE DAG and observation checks satisfied."

    def _check_post_state(self, state: WorldState) -> tuple[bool, str]:
        raw_payload = state.memory.get(self._repository_id)
        if not isinstance(raw_payload, Mapping):
            return (
                False,
                f"Missing SWE repository memory '{self._repository_id}' in post-state",
            )

        raw_files: Mapping[str, Any] = raw_payload.get("files", {})
        raw_deps: Mapping[str, Any] = raw_payload.get("dependencies", {})

        all_nodes: set[str] = {str(k) for k in raw_files.keys()} | {
            str(k) for k in raw_deps.keys()
        }
        deps_map: dict[str, tuple[str, ...]] = {}
        for src, targets in raw_deps.items():
            target_tuple = tuple(str(dst) for dst in targets)
            deps_map[str(src)] = target_tuple
            all_nodes.update(target_tuple)

        limit = min(
            self._max_graph_nodes,
            int(raw_payload.get("max_graph_nodes", self._max_graph_nodes)),
        )
        if len(all_nodes) > limit:
            return (
                False,
                f"Post-transition SWE graph contains {len(all_nodes)} nodes, "
                f"exceeding max_graph_nodes={limit}",
            )

        missing = all_nodes - {str(k) for k in raw_files.keys()}
        if missing:
            return (
                False,
                f"Dependencies reference untracked files: {sorted(missing)}",
            )

        cycle = detect_cycle_iterative_o_v_plus_e(deps_map, all_nodes)
        if cycle is not None:
            cycle_str = " -> ".join(cycle)
            return (
                False,
                f"Post-transition circular dependency detected: {cycle_str}",
            )

        return True, "Post-transition SWE repository graph is a valid bounded DAG."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        """Evaluate O(V + E) DAG acyclicity and stale observation constraints."""
        if phase == ConstraintPhase.PRE_ACTION:
            try:
                swe_state = SWEState.from_world_state(
                    state, repository_id=self._repository_id
                )
            except Exception as exc:
                return ConstraintResult(
                    satisfied=False,
                    constraint_id=self.constraint_id,
                    constraint_version=self.version,
                    severity=ConstraintSeverity.HARD,
                    phase=phase,
                    message=f"Invalid pre-transition SWEState: {exc}",
                    violating_entities=(self._repository_id,),
                    step=state.step,
                )

            for act in actions:
                ok, reason = self._check_single_action_pre(swe_state, act)
                if not ok:
                    if self._raise_on_violation:
                        raise StructuralViolationError(reason)
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        constraint_version=self.version,
                        severity=ConstraintSeverity.HARD,
                        phase=phase,
                        message=reason,
                        violating_entities=(self._repository_id,),
                        step=state.step,
                        preceding_action_id=act.id,
                    )

            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                constraint_version=self.version,
                severity=ConstraintSeverity.HARD,
                phase=phase,
                message="Pre-action SWE DAG constraints satisfied.",
                step=state.step,
            )

        ok, reason = self._check_post_state(state)
        if not ok and self._raise_on_violation:
            raise StructuralViolationError(reason)
        return ConstraintResult(
            satisfied=ok,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=ConstraintSeverity.HARD,
            phase=phase,
            message=reason,
            violating_entities=() if ok else (self._repository_id,),
            step=state.step,
        )

    def as_pre_rule(self, state: WorldState, actions: Sequence[Action]) -> tuple[bool, str]:
        """Callable adapter for ``HybridTransitionRouter(pre_rules=[...])``."""
        res = self.evaluate(state, actions, phase=ConstraintPhase.PRE_ACTION)
        if not res.satisfied and self._raise_on_violation:
            raise StructuralViolationError(res.message)
        return res.satisfied, res.message

    def as_post_rule(
        self,
        prev_state: WorldState,
        candidate_state: WorldState,
        actions: Sequence[Action],
    ) -> tuple[bool, str]:
        """Callable adapter for ``HybridTransitionRouter(post_rules=[...])``."""
        res = self.evaluate(candidate_state, actions, phase=ConstraintPhase.POST_TRANSITION)
        if not res.satisfied and self._raise_on_violation:
            raise StructuralViolationError(res.message)
        return res.satisfied, res.message


__all__ = [
    "AcyclicDependencyConstraint",
    "StructuralViolationError",
    "detect_stale_observations_for_action",
]
