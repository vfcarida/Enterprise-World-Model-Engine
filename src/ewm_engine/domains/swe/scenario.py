"""SWE Domain Pack dynamics and concurrency intervention scenario (EWM-005).

Implements:
- ``SWERepositoryDynamics``: Structural dynamics for repository file updates,
  PR merge queue transitions, and O(V + E) downstream build-queue propagation.
- ``run_concurrent_agent_intervention_scenario``: Counterfactual ``do()`` simulation
  answering: "What if Agent A merges PR X while Agent B modifies dependent File Y?"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.scm import CausalGraph
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.domains.swe.constraints import (
    AcyclicDependencyConstraint,
    StructuralViolationError,
    detect_stale_observations_for_action,
)
from ewm_engine.domains.swe.state import (
    MAX_SWE_GRAPH_NODES,
    SWE_REPO_ENTITY_ID,
    CodeFileNode,
    MergeQueueItem,
    PullRequestStatus,
    StaleObservationRecord,
    SWEState,
    build_swe_action,
    find_downstream_dependents_iterative,
)
from ewm_engine.dynamics.base import (
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
)
from ewm_engine.dynamics.router import HybridTransitionRouter, TransitionEngine
from ewm_engine.provenance.evidence import EvidenceLevel


def _merge_build_queues(
    existing_queue: Sequence[str],
    newly_affected: Sequence[str],
) -> tuple[str, ...]:
    """Preserve existing FIFO build order while appending newly affected files."""
    seen: set[str] = set()
    merged: list[str] = []
    for fpath in (*existing_queue, *newly_affected):
        if fpath not in seen:
            seen.add(fpath)
            merged.append(fpath)
    return tuple(merged)


class SWERepositoryDynamics:
    """Deterministic structural ``DynamicsModel`` for an agentic SWE repository."""

    def __init__(
        self,
        *,
        repository_id: str = SWE_REPO_ENTITY_ID,
        name: str = "SWERepositoryDynamics",
        version: str = "1.0.0",
    ) -> None:
        self._repository_id = repository_id
        self.name = name
        self.version = version
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            evidence_level=EvidenceLevel.STRUCTURAL,
            source="swe_dag_and_merge_queue_dynamics",
            uncertainty_contract="point_deterministic",
            invariants_affected=("swe_acyclic_dependency_constraint",),
            failure_semantics=InfeasibleResidualPolicy.REJECT,
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        return self._provenance

    def _apply_single_action(self, swe_state: SWEState, action: Action) -> SWEState:
        params = action.parameters
        action_type = str(params.get("action_type", action.type)).upper()
        default_agent = str(params.get("agent_id", action.actor_id or "agent_unknown"))

        files: dict[str, CodeFileNode] = dict(swe_state.files)
        dependencies: dict[str, tuple[str, ...]] = dict(swe_state.dependencies)
        merge_queue: list[MergeQueueItem] = list(swe_state.merge_queue)
        merged_prs: list[str] = list(swe_state.merged_prs)
        build_queue: tuple[str, ...] = swe_state.build_queue
        agent_observations: dict[str, dict[str, int]] = {
            agent: dict(revs) for agent, revs in swe_state.agent_observations.items()
        }
        stale_observations: list[StaleObservationRecord] = list(swe_state.stale_observations)

        if action_type == "OBSERVE_REPO":
            raw_paths = params.get("file_paths")
            target_paths = (
                [str(p) for p in raw_paths]
                if isinstance(raw_paths, (list, tuple))
                else list(files.keys())
            )
            snapshot = dict(agent_observations.get(default_agent, {}))
            for fp in target_paths:
                if fp in files:
                    snapshot[fp] = files[fp].revision
            agent_observations[default_agent] = snapshot

        elif action_type == "ADD_DEPENDENCY":
            src = str(params["source_file"])
            dst = str(params["target_file"])
            if src not in files:
                files[src] = CodeFileNode(file_path=src, last_modified_by=default_agent)
                dependencies.setdefault(src, ())
            if dst not in files:
                files[dst] = CodeFileNode(file_path=dst, last_modified_by=default_agent)
                dependencies.setdefault(dst, ())

            current_targets = set(dependencies.get(src, ()))
            current_targets.add(dst)
            dependencies[src] = tuple(sorted(current_targets))

            src_node = files[src]
            files[src] = src_node.model_copy(
                update={
                    "revision": src_node.revision + 1,
                    "last_modified_by": default_agent,
                    "content_hash": f"sha256:{src}:v{src_node.revision + 1}",
                }
            )
            affected = find_downstream_dependents_iterative(dependencies, (src,), files.keys())
            for af in affected:
                files[af] = files[af].model_copy(update={"needs_rebuild": True})
            build_queue = _merge_build_queues(build_queue, affected)

        elif action_type in {"ENQUEUE_PR", "OPEN_PR"}:
            pr_id = str(params["pr_id"])
            mod_files = tuple(str(fp) for fp in params["modified_files"])
            raw_added: Sequence[Any] = params.get("added_dependencies", ())
            raw_removed: Sequence[Any] = params.get("removed_dependencies", ())
            added_deps = tuple((str(edge[0]), str(edge[1])) for edge in raw_added)
            removed_deps = tuple((str(edge[0]), str(edge[1])) for edge in raw_removed)

            raw_base = params.get("base_revisions")
            if isinstance(raw_base, Mapping):
                base_revs: dict[str, int] = {str(k): int(v) for k, v in raw_base.items()}
            elif default_agent in agent_observations:
                base_revs = dict(agent_observations[default_agent])
            else:
                closure = swe_state.relevant_dependency_closure(mod_files)
                base_revs = {fp: files[fp].revision for fp in closure if fp in files}

            pr_item = MergeQueueItem(
                pr_id=pr_id,
                agent_id=default_agent,
                modified_files=mod_files,
                base_revisions=base_revs,
                added_dependencies=added_deps,
                removed_dependencies=removed_deps,
                status=PullRequestStatus.PENDING,
                description=str(params.get("description", f"PR {pr_id} by {default_agent}")),
            )
            merge_queue.append(pr_item)

        elif action_type == "MODIFY_FILE":
            target_file = str(params["file_path"])
            raw_base = params.get("base_revisions")
            opt_base_revs: dict[str, int] | None = (
                {str(k): int(v) for k, v in raw_base.items()}
                if isinstance(raw_base, Mapping)
                else None
            )

            stale_records = detect_stale_observations_for_action(
                swe_state,
                agent_id=default_agent,
                action_id=action.id,
                modified_files=(target_file,),
                base_revisions=opt_base_revs,
            )
            stale_observations.extend(stale_records)

            if target_file not in files:
                files[target_file] = CodeFileNode(
                    file_path=target_file,
                    revision=1,
                    last_modified_by=default_agent,
                    content_hash=f"sha256:{target_file}:v1",
                    needs_rebuild=True,
                )
                dependencies.setdefault(target_file, ())
            else:
                prev_node = files[target_file]
                next_rev = prev_node.revision + 1
                files[target_file] = prev_node.model_copy(
                    update={
                        "revision": next_rev,
                        "last_modified_by": default_agent,
                        "content_hash": str(
                            params.get(
                                "content_hash",
                                f"sha256:{target_file}:v{next_rev}:{default_agent}",
                            )
                        ),
                        "needs_rebuild": True,
                    }
                )

            raw_added = params.get("added_dependencies", ())
            for edge in raw_added:
                src, dst = str(edge[0]), str(edge[1])
                if dst not in files:
                    files[dst] = CodeFileNode(file_path=dst, last_modified_by=default_agent)
                    dependencies.setdefault(dst, ())
                cur = set(dependencies.get(src, ()))
                cur.add(dst)
                dependencies[src] = tuple(sorted(cur))

            affected = find_downstream_dependents_iterative(
                dependencies, (target_file,), files.keys()
            )
            for af in affected:
                files[af] = files[af].model_copy(update={"needs_rebuild": True})
            build_queue = _merge_build_queues(build_queue, affected)

            agent_snap = dict(agent_observations.get(default_agent, {}))
            agent_snap[target_file] = files[target_file].revision
            agent_observations[default_agent] = agent_snap

        elif action_type == "MERGE_PR":
            target_pr_id = params.get("pr_id")
            force_merge = bool(params.get("force_merge", False))
            target_idx: int | None = None

            for idx, item in enumerate(merge_queue):
                if item.status == PullRequestStatus.PENDING and (
                    target_pr_id is None or item.pr_id == str(target_pr_id)
                ):
                    target_idx = idx
                    break

            if target_idx is None:
                raise ValueError(
                    f"No PENDING pull request matching pr_id={target_pr_id!r} in merge_queue"
                )

            pr_item = merge_queue[target_idx]
            stale_records = detect_stale_observations_for_action(
                swe_state,
                agent_id=pr_item.agent_id,
                action_id=action.id,
                modified_files=pr_item.modified_files,
                base_revisions=pr_item.base_revisions,
                pr_id=pr_item.pr_id,
            )

            if stale_records and not force_merge:
                stale_observations.extend(stale_records)
                merge_queue[target_idx] = pr_item.model_copy(
                    update={"status": PullRequestStatus.CONFLICT_STALE}
                )
                affected = find_downstream_dependents_iterative(
                    dependencies, pr_item.modified_files, files.keys()
                )
                for af in affected:
                    if af in files:
                        files[af] = files[af].model_copy(update={"needs_rebuild": True})
                build_queue = _merge_build_queues(build_queue, affected)
            else:
                if stale_records:
                    stale_observations.extend(stale_records)

                removed_set = set(pr_item.removed_dependencies)
                for src, targets in list(dependencies.items()):
                    dependencies[src] = tuple(
                        dst for dst in targets if (src, dst) not in removed_set
                    )
                for src, dst in pr_item.added_dependencies:
                    if src not in files:
                        files[src] = CodeFileNode(file_path=src, last_modified_by=pr_item.agent_id)
                        dependencies.setdefault(src, ())
                    if dst not in files:
                        files[dst] = CodeFileNode(file_path=dst, last_modified_by=pr_item.agent_id)
                        dependencies.setdefault(dst, ())
                    cur_targets = set(dependencies.get(src, ()))
                    cur_targets.add(dst)
                    dependencies[src] = tuple(sorted(cur_targets))

                for fp in pr_item.modified_files:
                    if fp not in files:
                        files[fp] = CodeFileNode(
                            file_path=fp,
                            revision=1,
                            last_modified_by=pr_item.agent_id,
                            content_hash=f"sha256:{fp}:v1:{pr_item.pr_id}",
                            needs_rebuild=True,
                        )
                        dependencies.setdefault(fp, ())
                    else:
                        prev_node = files[fp]
                        next_rev = prev_node.revision + 1
                        files[fp] = prev_node.model_copy(
                            update={
                                "revision": next_rev,
                                "last_modified_by": pr_item.agent_id,
                                "content_hash": f"sha256:{fp}:v{next_rev}:{pr_item.pr_id}",
                                "needs_rebuild": True,
                            }
                        )

                merge_queue[target_idx] = pr_item.model_copy(
                    update={"status": PullRequestStatus.MERGED}
                )
                merged_prs.append(pr_item.pr_id)

                affected = find_downstream_dependents_iterative(
                    dependencies, pr_item.modified_files, files.keys()
                )
                for af in affected:
                    if af in files:
                        files[af] = files[af].model_copy(update={"needs_rebuild": True})
                build_queue = _merge_build_queues(build_queue, affected)

        elif action_type == "RUN_BUILD_QUEUE":
            for fp in build_queue:
                if fp in files:
                    files[fp] = files[fp].model_copy(update={"needs_rebuild": False})
            build_queue = ()

        else:
            raise ValueError(f"Unsupported SWE action_type: '{action_type}'")

        return SWEState(
            step=swe_state.step,
            repository_id=swe_state.repository_id,
            files=files,
            dependencies=dependencies,
            merge_queue=tuple(merge_queue),
            merged_prs=tuple(merged_prs),
            build_queue=build_queue,
            agent_observations=agent_observations,
            stale_observations=tuple(stale_observations),
            max_graph_nodes=swe_state.max_graph_nodes,
        )

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        """Apply accepted SWE actions and active ``do()`` causal interventions."""
        _ = (exogenous_events, rng, runtime_context)
        swe_state = SWEState.from_world_state(state, repository_id=self._repository_id)

        for act in actions:
            swe_state = self._apply_single_action(swe_state, act)

        next_world = swe_state.to_world_state(step=state.step)
        # Preserve any causal do() memory keys injected by TransitionEngine.do()
        merged_mem = dict(state.memory)
        merged_mem[self._repository_id] = next_world.memory[self._repository_id]
        next_world = next_world.model_copy(
            update={
                "step": state.step,
                "timestamp": state.timestamp,
                "memory": merged_mem,
            }
        )

        return TransitionResult(
            next_state=next_world,
            applied_changes={
                "node_count": len(swe_state.files),
                "build_queue": list(swe_state.build_queue),
                "merged_prs": list(swe_state.merged_prs),
                "stale_observation_count": len(swe_state.stale_observations),
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics={
                "build_queue": list(swe_state.build_queue),
                "stale_observation_count": len(swe_state.stale_observations),
            },
            model_name=self.name,
        )


def build_swe_agent_engine(
    initial_swe_state: SWEState | None = None,
    *,
    repository_id: str = SWE_REPO_ENTITY_ID,
    max_graph_nodes: int = MAX_SWE_GRAPH_NODES,
    reject_stale_observations: bool = False,
    raise_on_violation: bool = True,
) -> TransitionEngine:
    """Create a ``TransitionEngine`` wired with SWE DAG constraints and dynamics."""
    swe_init = (
        initial_swe_state
        if initial_swe_state is not None
        else create_default_swe_repository_state()
    )
    init_world = swe_init.to_world_state(step=0)

    constraint = AcyclicDependencyConstraint(
        repository_id=repository_id,
        max_graph_nodes=max_graph_nodes,
        reject_stale_observations=reject_stale_observations,
        raise_on_violation=raise_on_violation,
    )
    reg = ConstraintRegistry([constraint])

    swe_causal_graph = CausalGraph(
        nodes={
            "ci_scheduler",
            "agent_a_merge_pr_x",
            "agent_b_modify_file_y",
            "src/core/schema.py",
            "src/services/OrderService.py",
            "src/api/OrderController.py",
            "build_queue_size",
            "stale_observation_count",
        },
        edges={
            ("ci_scheduler", "agent_a_merge_pr_x"),
            ("ci_scheduler", "agent_b_modify_file_y"),
            ("agent_a_merge_pr_x", "src/services/OrderService.py"),
            ("agent_b_modify_file_y", "src/core/schema.py"),
            ("src/core/schema.py", "src/services/OrderService.py"),
            ("src/services/OrderService.py", "src/api/OrderController.py"),
            ("src/services/OrderService.py", "build_queue_size"),
            ("src/api/OrderController.py", "build_queue_size"),
            ("src/core/schema.py", "stale_observation_count"),
        },
    )

    router = HybridTransitionRouter(
        structural_model=SWERepositoryDynamics(repository_id=repository_id),
        constraints=reg,
        pre_rules=(constraint.as_pre_rule,),
        post_rules=(constraint.as_post_rule,),
        causal_graph=swe_causal_graph,
        raise_on_pre_rule_violation=raise_on_violation,
        name="SWERepositoryRouter",
    )
    return TransitionEngine(
        router=router,
        initial_state=init_world,
        branch_id="swe_main",
    )


@dataclass(frozen=True)
class BranchExecutionTrace:
    """Detailed execution trace of a single counterfactual SWE branch."""

    branch_id: str
    root_state_hash: str
    final_state_hash: str
    step_hashes: tuple[str, ...]
    severed_inbound_edges: tuple[tuple[str, str], ...]
    file_revisions: dict[str, int]
    pr_statuses: dict[str, str]
    merged_prs: tuple[str, ...]
    build_queue: tuple[str, ...]
    stale_observations: tuple[StaleObservationRecord, ...]


@dataclass(frozen=True)
class ConcurrencyInterventionTrace:
    """Comparison trace of parallel agentic coding counterfactual interventions."""

    shared_checkpoint_hash: str
    branch_agent_a_merges_first: BranchExecutionTrace
    branch_agent_b_modifies_first: BranchExecutionTrace
    cyclic_injection_rejected: bool
    cyclic_rejection_reason: str


def create_default_swe_repository_state() -> SWEState:
    """Build the canonical 3-tier repository DAG for the concurrency scenario.

    Dependency DAG:
    ``src/api/OrderController.py`` -> ``src/services/OrderService.py`` -> ``src/core/schema.py``

    Where:
    - ``PR-101`` (PR X) by ``agent_a`` modifies ``src/services/OrderService.py`` (File X)
    - ``agent_b`` concurrently modifies ``src/core/schema.py`` (dependent File Y)
    """
    file_y = "src/core/schema.py"
    file_x = "src/services/OrderService.py"
    file_z = "src/api/OrderController.py"

    files = {
        file_y: CodeFileNode(
            file_path=file_y,
            revision=1,
            last_modified_by="repo_init",
            content_hash="sha256:schema:v1",
        ),
        file_x: CodeFileNode(
            file_path=file_x,
            revision=1,
            last_modified_by="repo_init",
            content_hash="sha256:OrderService:v1",
        ),
        file_z: CodeFileNode(
            file_path=file_z,
            revision=1,
            last_modified_by="repo_init",
            content_hash="sha256:OrderController:v1",
        ),
    }
    dependencies: dict[str, tuple[str, ...]] = {
        file_y: (),
        file_x: (file_y,),
        file_z: (file_x,),
    }
    base_snapshot = {file_y: 1, file_x: 1, file_z: 1}
    pr_x = MergeQueueItem(
        pr_id="PR-101",
        agent_id="agent_a",
        modified_files=(file_x,),
        base_revisions={file_x: 1, file_y: 1},
        status=PullRequestStatus.PENDING,
        description="PR X: Refactor OrderService validation against core schema",
    )
    return SWEState(
        files=files,
        dependencies=dependencies,
        merge_queue=(pr_x,),
        agent_observations={
            "agent_a": dict(base_snapshot),
            "agent_b": dict(base_snapshot),
        },
    )


def run_concurrent_agent_intervention_scenario() -> ConcurrencyInterventionTrace:
    """Run the counterfactual intervention scenario:

    "What if Agent A merges PR X (PR-101 on OrderService.py) while Agent B
    modifies dependent File Y (src/core/schema.py)?"

    Executes two parallel causal branches forked from the identical checkpoint
    state via ``engine.branch().do()``, plus an adversarial cyclic dependency
    injection test.
    """
    initial_swe = create_default_swe_repository_state()
    base_engine = build_swe_agent_engine(initial_swe_state=initial_swe)
    assert base_engine.current_state is not None
    checkpoint_hash = base_engine.current_state.state_hash

    action_merge_pr_x = build_swe_action(
        "act_agent_a_merge_pr_101",
        "MERGE_PR",
        actor_id="agent_a",
        parameters={"pr_id": "PR-101"},
    )
    action_modify_file_y = build_swe_action(
        "act_agent_b_modify_schema_y",
        "MODIFY_FILE",
        actor_id="agent_b",
        parameters={
            "file_path": "src/core/schema.py",
            "content_hash": "sha256:schema:v2:agent_b_breaking_field",
        },
    )

    # Branch 1: Agent A merges PR-101 first, then Agent B modifies File Y (src/core/schema.py)
    branch_a = base_engine.branch("branch_agent_a_merges_first").do({"agent_a_merge_pr_x": 1})
    severed_a = branch_a.causal_execution_log[-1].severed_inbound_edges
    res_a1 = branch_a.step(action=action_merge_pr_x)
    res_a2 = branch_a.step(action=action_modify_file_y)
    swe_a_final = SWEState.from_world_state(res_a2.next_state)

    trace_a = BranchExecutionTrace(
        branch_id="branch_agent_a_merges_first",
        root_state_hash=checkpoint_hash,
        final_state_hash=res_a2.next_state.state_hash,
        step_hashes=(res_a1.next_state.state_hash, res_a2.next_state.state_hash),
        severed_inbound_edges=severed_a,
        file_revisions={fp: node.revision for fp, node in sorted(swe_a_final.files.items())},
        pr_statuses={item.pr_id: item.status.value for item in swe_a_final.merge_queue},
        merged_prs=swe_a_final.merged_prs,
        build_queue=swe_a_final.build_queue,
        stale_observations=swe_a_final.stale_observations,
    )

    # Branch 2: Agent B modifies dependent File Y first, then Agent A attempts to merge PR-101
    branch_b = base_engine.branch("branch_agent_b_modifies_first").do({"agent_b_modify_file_y": 1})
    severed_b = branch_b.causal_execution_log[-1].severed_inbound_edges
    res_b1 = branch_b.step(action=action_modify_file_y)
    res_b2 = branch_b.step(action=action_merge_pr_x)
    swe_b_final = SWEState.from_world_state(res_b2.next_state)

    trace_b = BranchExecutionTrace(
        branch_id="branch_agent_b_modifies_first",
        root_state_hash=checkpoint_hash,
        final_state_hash=res_b2.next_state.state_hash,
        step_hashes=(res_b1.next_state.state_hash, res_b2.next_state.state_hash),
        severed_inbound_edges=severed_b,
        file_revisions={fp: node.revision for fp, node in sorted(swe_b_final.files.items())},
        pr_statuses={item.pr_id: item.status.value for item in swe_b_final.merge_queue},
        merged_prs=swe_b_final.merged_prs,
        build_queue=swe_b_final.build_queue,
        stale_observations=swe_b_final.stale_observations,
    )

    # Adversarial cyclic dependency injection on a third branch
    branch_cycle = base_engine.branch("branch_cyclic_injection")
    action_inject_cycle = build_swe_action(
        "act_inject_cycle",
        "ADD_DEPENDENCY",
        actor_id="agent_rogue",
        parameters={
            "source_file": "src/core/schema.py",
            "target_file": "src/api/OrderController.py",
        },
    )
    cyclic_rejected = False
    cyclic_reason = ""
    try:
        branch_cycle.step(action=action_inject_cycle)
    except StructuralViolationError as exc:
        cyclic_rejected = True
        cyclic_reason = str(exc)

    return ConcurrencyInterventionTrace(
        shared_checkpoint_hash=checkpoint_hash,
        branch_agent_a_merges_first=trace_a,
        branch_agent_b_modifies_first=trace_b,
        cyclic_injection_rejected=cyclic_rejected,
        cyclic_rejection_reason=cyclic_reason,
    )


__all__ = [
    "BranchExecutionTrace",
    "ConcurrencyInterventionTrace",
    "SWERepositoryDynamics",
    "build_swe_agent_engine",
    "create_default_swe_repository_state",
    "run_concurrent_agent_intervention_scenario",
]
