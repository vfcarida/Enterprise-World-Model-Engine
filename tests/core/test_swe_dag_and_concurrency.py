"""Verification suite for SWE Domain Pack DAG state, merge queues, and concurrency (EWM-005)."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pytest

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import ConstraintPhase
from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import Entity, WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.domains.swe import (
    MAX_SWE_GRAPH_NODES,
    SWE_REPO_ENTITY_ID,
    AcyclicDependencyConstraint,
    CircularDependencyError,
    CodeFileNode,
    PullRequestStatus,
    StructuralViolationError,
    SWEGraphCapacityError,
    SWERepositoryDynamics,
    SWEState,
    build_swe_action,
    build_swe_agent_engine,
    create_default_swe_repository_state,
    detect_cycle_iterative_o_v_plus_e,
    detect_stale_observations_for_action,
    find_downstream_dependents_iterative,
    run_concurrent_agent_intervention_scenario,
)
from ewm_engine.dynamics.base import (
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
)
from ewm_engine.dynamics.router import HybridTransitionRouter, TransitionEngine
from ewm_engine.provenance.evidence import EvidenceLevel


def test_networkx_dag_roundtrip_and_immutability() -> None:
    """SWEState converts cleanly to/from networkx.DiGraph and prevents external mutation."""
    dag: nx.DiGraph = nx.DiGraph()
    dag.add_node("src/app.py", revision=2, last_modified_by="agent_a")
    dag.add_node("src/utils.py", revision=1, last_modified_by="repo_init")
    dag.add_edge("src/app.py", "src/utils.py")

    swe_state = SWEState.from_networkx_dag(
        dag,
        step=3,
        agent_observations={"agent_a": {"src/app.py": 2, "src/utils.py": 1}},
    )
    assert len(swe_state.files) == 2
    assert swe_state.dependencies["src/app.py"] == ("src/utils.py",)

    exported_dag = swe_state.dependency_graph
    assert isinstance(exported_dag, nx.DiGraph)
    assert list(exported_dag.edges()) == [("src/app.py", "src/utils.py")]

    # Mutating the exported networkx DiGraph must not mutate SWEState
    exported_dag.add_edge("src/utils.py", "src/app.py")
    assert swe_state.dependencies["src/utils.py"] == ()

    # WorldState serialization round-trip
    world_state = swe_state.to_world_state(step=3)
    restored = SWEState.from_world_state(world_state)
    assert restored == swe_state


def test_iterative_o_v_plus_e_cycle_detection_handles_deep_graphs_without_recursion_error() -> None:
    """Iterative cycle detector handles 1,800-node chains and cycles without RecursionError."""
    chain_len = 1800
    nodes = [f"mod_{i:04d}.py" for i in range(chain_len)]
    acyclic_deps: dict[str, tuple[str, ...]] = {
        nodes[i]: (nodes[i + 1],) for i in range(chain_len - 1)
    }
    acyclic_deps[nodes[-1]] = ()

    assert detect_cycle_iterative_o_v_plus_e(acyclic_deps, nodes) is None

    # Close the 1,800-node loop: mod_1799.py -> mod_0000.py
    cyclic_deps = dict(acyclic_deps)
    cyclic_deps[nodes[-1]] = (nodes[0],)
    cycle = detect_cycle_iterative_o_v_plus_e(cyclic_deps, nodes)
    assert cycle is not None
    assert cycle[0] == cycle[-1]
    assert len(cycle) == chain_len + 1

    # Self-loop detection
    self_loop = {"a.py": ("a.py",)}
    assert detect_cycle_iterative_o_v_plus_e(self_loop) == ("a.py", "a.py")


def test_swe_state_rejects_cycles_missing_files_and_node_capacity_overflow() -> None:
    """SWEState model validation enforces DAG acyclicity and MAX_SWE_GRAPH_NODES bounds."""
    with pytest.raises(CircularDependencyError) as exc_info:
        SWEState(
            files={
                "a.py": CodeFileNode(file_path="a.py"),
                "b.py": CodeFileNode(file_path="b.py"),
            },
            dependencies={
                "a.py": ("b.py",),
                "b.py": ("a.py",),
            },
        )
    assert exc_info.value.cycle_path == ("a.py", "b.py", "a.py")

    with pytest.raises(ValueError, match="untracked files"):
        SWEState(
            files={"a.py": CodeFileNode(file_path="a.py")},
            dependencies={"a.py": ("missing.py",)},
        )

    with pytest.raises(SWEGraphCapacityError):
        SWEState(
            files={f"f_{i}.py": CodeFileNode(file_path=f"f_{i}.py") for i in range(12)},
            dependencies={f"f_{i}.py": () for i in range(12)},
            max_graph_nodes=10,
        )

    oversized_dag: nx.DiGraph = nx.DiGraph()
    for i in range(MAX_SWE_GRAPH_NODES + 5):
        oversized_dag.add_node(f"node_{i}.py")
    with pytest.raises(SWEGraphCapacityError):
        SWEState.from_networkx_dag(oversized_dag)


def test_100_percent_structural_rejection_of_cyclic_dependency_injections() -> None:
    """AcyclicDependencyConstraint rejects 100% of cyclic transitions across 100 topologies."""
    engine = build_swe_agent_engine()
    base_swe = create_default_swe_repository_state()
    base_world = base_swe.to_world_state()

    # 1. Direct ADD_DEPENDENCY creating a 3-node cycle
    with pytest.raises(StructuralViolationError, match="circular dependency"):
        engine.step(
            base_world,
            build_swe_action(
                "add_cycle_1",
                "ADD_DEPENDENCY",
                actor_id="agent_x",
                parameters={
                    "source_file": "src/core/schema.py",
                    "target_file": "src/api/OrderController.py",
                },
            ),
        )

    # 2. ENQUEUE_PR containing cyclic added_dependencies
    with pytest.raises(StructuralViolationError, match="circular dependency"):
        engine.step(
            base_world,
            build_swe_action(
                "enqueue_cyclic_pr",
                "ENQUEUE_PR",
                actor_id="agent_x",
                parameters={
                    "pr_id": "PR-CYCLE",
                    "modified_files": ["src/core/schema.py"],
                    "added_dependencies": [["src/core/schema.py", "src/services/OrderService.py"]],
                },
            ),
        )

    # 3. MODIFY_FILE containing cyclic added_dependencies
    with pytest.raises(StructuralViolationError, match="circular dependency"):
        engine.step(
            base_world,
            build_swe_action(
                "modify_cyclic",
                "MODIFY_FILE",
                actor_id="agent_x",
                parameters={
                    "file_path": "src/core/schema.py",
                    "added_dependencies": [["src/core/schema.py", "src/api/OrderController.py"]],
                },
            ),
        )

    # 4. 100 procedural multi-hop cycle injections -> 100% rejection rate
    rejected_count = 0
    total_trials = 100
    for trial in range(total_trials):
        ring_size = 4 + (trial % 15)
        files = {f"pkg/m_{i}.py": CodeFileNode(file_path=f"pkg/m_{i}.py") for i in range(ring_size)}
        deps: dict[str, tuple[str, ...]] = {
            f"pkg/m_{i}.py": (f"pkg/m_{i + 1}.py",) for i in range(ring_size - 1)
        }
        deps[f"pkg/m_{ring_size - 1}.py"] = ()
        state_t = SWEState(files=files, dependencies=deps).to_world_state()

        closing_action = build_swe_action(
            f"close_ring_{trial}",
            "ADD_DEPENDENCY",
            actor_id="agent_test",
            parameters={
                "source_file": f"pkg/m_{ring_size - 1}.py",
                "target_file": "pkg/m_0.py",
            },
        )
        try:
            engine.step(state_t, closing_action)
        except StructuralViolationError:
            rejected_count += 1

    assert rejected_count == total_trials


def test_concurrent_agent_intervention_scenario_and_branch_isolation() -> None:
    """Verify 'what if Agent A merges PR X while Agent B modifies dependent File Y?'."""
    trace = run_concurrent_agent_intervention_scenario()

    assert trace.branch_agent_a_merges_first.root_state_hash == trace.shared_checkpoint_hash
    assert trace.branch_agent_b_modifies_first.root_state_hash == trace.shared_checkpoint_hash
    assert (
        trace.branch_agent_a_merges_first.final_state_hash
        != trace.branch_agent_b_modifies_first.final_state_hash
    )

    # Branch 1: Agent A merged PR-101 first (OrderService.py -> rev 2),
    # then Agent B modified dependent File Y (schema.py -> rev 2).
    branch_a = trace.branch_agent_a_merges_first
    assert branch_a.severed_inbound_edges == (("ci_scheduler", "agent_a_merge_pr_x"),)
    assert branch_a.pr_statuses["PR-101"] == PullRequestStatus.MERGED.value
    assert branch_a.merged_prs == ("PR-101",)
    assert branch_a.file_revisions == {
        "src/api/OrderController.py": 1,
        "src/core/schema.py": 2,
        "src/services/OrderService.py": 2,
    }
    assert set(branch_a.build_queue) == {
        "src/core/schema.py",
        "src/services/OrderService.py",
        "src/api/OrderController.py",
    }
    assert len(branch_a.stale_observations) == 1
    assert branch_a.stale_observations[0].agent_id == "agent_b"
    assert branch_a.stale_observations[0].file_path == "src/services/OrderService.py"
    assert "dependency_revision_drift" in branch_a.stale_observations[0].reason

    # Branch 2: Agent B modified dependent File Y (schema.py -> rev 2) first,
    # invalidating PR-101's base snapshot when Agent A attempted to merge PR-101.
    branch_b = trace.branch_agent_b_modifies_first
    assert branch_b.severed_inbound_edges == (("ci_scheduler", "agent_b_modify_file_y"),)
    assert branch_b.pr_statuses["PR-101"] == PullRequestStatus.CONFLICT_STALE.value
    assert branch_b.merged_prs == ()
    assert branch_b.file_revisions == {
        "src/api/OrderController.py": 1,
        "src/core/schema.py": 2,
        "src/services/OrderService.py": 1,
    }
    assert branch_b.build_queue == (
        "src/core/schema.py",
        "src/services/OrderService.py",
        "src/api/OrderController.py",
    )
    assert len(branch_b.stale_observations) == 1
    assert branch_b.stale_observations[0].agent_id == "agent_a"
    assert branch_b.stale_observations[0].pr_id == "PR-101"
    assert branch_b.stale_observations[0].file_path == "src/core/schema.py"
    assert branch_b.stale_observations[0].expected_revision == 1
    assert branch_b.stale_observations[0].actual_revision == 2

    # Branch 3: Cyclic injection was rejected structurally
    assert trace.cyclic_injection_rejected is True
    assert "circular dependency" in trace.cyclic_rejection_reason


def test_build_queue_propagation_and_execution_lifecycle() -> None:
    """Verify downstream build queue propagation and RUN_BUILD_QUEUE clearing."""
    swe0 = create_default_swe_repository_state().observe_files("agent_c")
    engine = build_swe_agent_engine(initial_swe_state=swe0)
    world0 = swe0.to_world_state()

    # Add a new downstream test file that depends on OrderController.py
    step1 = engine.step(
        world0,
        build_swe_action(
            "add_e2e_test",
            "ADD_DEPENDENCY",
            actor_id="agent_c",
            parameters={
                "source_file": "tests/test_orders_e2e.py",
                "target_file": "src/api/OrderController.py",
            },
        ),
    )
    swe1 = SWEState.from_world_state(step1.next_state)
    assert "tests/test_orders_e2e.py" in swe1.build_queue

    # Run build queue to clear needs_rebuild flags
    step2 = engine.step(
        step1.next_state,
        build_swe_action(
            "run_ci_build_1",
            "RUN_BUILD_QUEUE",
            actor_id="ci_runner",
        ),
    )
    swe2 = SWEState.from_world_state(step2.next_state)
    assert swe2.build_queue == ()
    assert all(not node.needs_rebuild for node in swe2.files.values())

    # Modifying root dependency src/core/schema.py must propagate across all 4 files
    step3 = engine.step(
        step2.next_state,
        build_swe_action(
            "update_schema",
            "MODIFY_FILE",
            actor_id="agent_c",
            parameters={"file_path": "src/core/schema.py"},
        ),
    )
    swe3 = SWEState.from_world_state(step3.next_state)
    assert swe3.build_queue == (
        "src/core/schema.py",
        "src/services/OrderService.py",
        "src/api/OrderController.py",
        "tests/test_orders_e2e.py",
    )
    assert all(swe3.files[fp].needs_rebuild for fp in swe3.build_queue)


def test_strict_stale_rejection_and_post_constraint_safeguards() -> None:
    """Verify strict_stale_check rejection and post-rule cycle/capacity enforcement."""
    strict_engine = build_swe_agent_engine(reject_stale_observations=True)
    world0 = create_default_swe_repository_state().to_world_state()

    # Agent B modifies schema.py (valid, fresh observation)
    step1 = strict_engine.step(
        world0,
        build_swe_action(
            "b_mod",
            "MODIFY_FILE",
            actor_id="agent_b",
            parameters={"file_path": "src/core/schema.py"},
        ),
    )

    # Now Agent A tries to merge PR-101 with reject_stale_observations=True -> StructuralViolationError
    with pytest.raises(StructuralViolationError, match="Stale observation rejected"):
        strict_engine.step(
            step1.next_state,
            build_swe_action(
                "a_merge_stale",
                "MERGE_PR",
                actor_id="agent_a",
                parameters={"pr_id": "PR-101"},
            ),
        )

    # Verify post-rule catches a cyclic graph produced by a rogue dynamics model
    class _RogueCyclicDynamics:
        @property
        def mechanism_provenance(self) -> MechanismProvenance:
            return MechanismProvenance(
                mechanism_id="RogueCyclicDynamics",
                version="1.0.0",
                kind=MechanismKind.KNOWN_DETERMINISTIC,
                evidence_level=EvidenceLevel.STRUCTURAL,
                source="test_rogue",
                uncertainty_contract="point_deterministic",
                invariants_affected=(),
                failure_semantics=InfeasibleResidualPolicy.REJECT,
            )

        def transition(
            self,
            state: WorldState,
            actions: Sequence[Action],
            exogenous_events: Sequence[ExogenousEvent],
            rng: RandomGenerator,
            runtime_context: ContextSnapshot | None = None,
        ) -> TransitionResult:
            _ = (actions, exogenous_events, rng, runtime_context)
            rogue_payload = {
                "repository_id": SWE_REPO_ENTITY_ID,
                "files": {
                    "x.py": CodeFileNode(file_path="x.py").model_dump(mode="python"),
                    "y.py": CodeFileNode(file_path="y.py").model_dump(mode="python"),
                },
                "dependencies": {"x.py": ["y.py"], "y.py": ["x.py"]},
                "merge_queue": [],
                "merged_prs": [],
                "build_queue": [],
                "agent_observations": {},
                "stale_observations": [],
                "max_graph_nodes": 100,
            }
            next_state = WorldState(
                step=state.step,
                timestamp=state.timestamp,
                entities={
                    SWE_REPO_ENTITY_ID: Entity(
                        id=SWE_REPO_ENTITY_ID,
                        type="swe_repository",
                        attributes={},
                    )
                },
                resources=dict(state.resources),
                memory={SWE_REPO_ENTITY_ID: rogue_payload},
            )
            return TransitionResult(
                next_state=next_state,
                applied_changes={},
                evidence_level=EvidenceLevel.STRUCTURAL,
                diagnostics={},
                model_name="RogueCyclicDynamics",
            )

    constraint = AcyclicDependencyConstraint(raise_on_violation=True)
    rogue_engine = TransitionEngine(
        router=HybridTransitionRouter(
            structural_model=_RogueCyclicDynamics(),
            ml_predictor=lambda _s, _a, _st: {"build_queue_size": 0.0},
            constraints=ConstraintRegistry([constraint]),
            pre_rules=(constraint.as_pre_rule,),
            post_rules=(constraint.as_post_rule,),
        ),
        initial_state=world0,
    )
    with pytest.raises(StructuralViolationError, match="Post-transition circular dependency"):
        rogue_engine.step(
            world0,
            build_swe_action("rogue_act", "OBSERVE_REPO", actor_id="rogue"),
        )


def test_swe_helper_edge_cases() -> None:
    """Cover enqueue_pr duplicate checks, direct file conflicts, and force_merge."""
    swe0 = create_default_swe_repository_state()
    with pytest.raises(ValueError, match="already in merge_queue"):
        swe0.enqueue_pr(swe0.merge_queue[0])

    with pytest.raises(KeyError, match="Unknown file"):
        swe0.observe_files("agent_a", ["nonexistent.py"])

    engine = build_swe_agent_engine(initial_swe_state=swe0)
    world0 = swe0.to_world_state()

    # Observe repo via action
    step_obs = engine.step(
        world0,
        build_swe_action(
            "obs_1",
            "OBSERVE_REPO",
            actor_id="agent_d",
            parameters={"file_paths": ["src/core/schema.py"]},
        ),
    )

    # Enqueue a new PR via action
    step_enq = engine.step(
        step_obs.next_state,
        build_swe_action(
            "enq_2",
            "ENQUEUE_PR",
            actor_id="agent_d",
            parameters={
                "pr_id": "PR-102",
                "modified_files": ["src/core/schema.py"],
                "added_dependencies": [["src/core/schema.py", "src/core/types.py"]],
            },
        ),
    )
    # Merge PR-102
    step_merged = engine.step(
        step_enq.next_state,
        build_swe_action(
            "merge_102",
            "MERGE_PR",
            actor_id="agent_d",
            parameters={"pr_id": "PR-102"},
        ),
    )
    swe_merged = SWEState.from_world_state(step_merged.next_state)
    assert "PR-102" in swe_merged.merged_prs
    assert "src/core/types.py" in swe_merged.files

    # Direct file conflict detection when agent_b modifies schema.py using old base_revisions
    direct_stale = detect_stale_observations_for_action(
        swe_merged,
        agent_id="agent_b",
        action_id="check_direct",
        modified_files=("src/core/schema.py",),
        base_revisions={"src/core/schema.py": 1},
    )
    assert len(direct_stale) == 1
    assert "direct_file_conflict" in direct_stale[0].reason

    # Force merge of PR-101 despite stale schema.py
    step_forced = engine.step(
        step_merged.next_state,
        build_swe_action(
            "force_merge_101",
            "MERGE_PR",
            actor_id="agent_a",
            parameters={"pr_id": "PR-101", "force_merge": True},
        ),
    )
    swe_forced = SWEState.from_world_state(step_forced.next_state)
    assert "PR-101" in swe_forced.merged_prs
    assert len(swe_forced.stale_observations) >= 1
    assert find_downstream_dependents_iterative(
        swe_forced.dependencies, ("src/core/types.py",)
    ) == (
        "src/core/types.py",
        "src/core/schema.py",
        "src/services/OrderService.py",
        "src/api/OrderController.py",
    )


def test_swe_constraint_and_dynamics_exhaustive_edge_cases() -> None:
    """Cover remaining edge branches in constraints.py, scenario.py, and state.py."""
    with pytest.raises(ValueError, match="max_graph_nodes"):
        AcyclicDependencyConstraint(max_graph_nodes=0)

    c = AcyclicDependencyConstraint(max_graph_nodes=3, raise_on_violation=False)
    assert c.severity.value == "hard"
    assert "DAG" in c.description

    empty_world = WorldState()
    with pytest.raises(KeyError, match="does not contain SWE repository"):
        SWEState.from_world_state(empty_world)

    pre_invalid = c.evaluate(empty_world, (), phase=ConstraintPhase.PRE_ACTION)
    assert not pre_invalid.satisfied
    post_invalid = c.evaluate(empty_world, (), phase=ConstraintPhase.POST_TRANSITION)
    assert not post_invalid.satisfied

    swe0 = create_default_swe_repository_state()
    w0 = swe0.to_world_state()
    assert c.as_post_rule(w0, w0, ())[0] is True

    # Empty graph cycle detection
    assert detect_cycle_iterative_o_v_plus_e({}) is None

    # Empty ADD_DEPENDENCY source/target rejected by pre-rule
    bad_add = build_swe_action(
        "bad_add",
        "ADD_DEPENDENCY",
        parameters={"source_file": "", "target_file": ""},
    )
    assert not c.as_pre_rule(w0, (bad_add,))[0]

    # Node capacity exceeded in pre-rule and post-rule
    over_cap = build_swe_action(
        "over_cap",
        "MODIFY_FILE",
        parameters={"file_path": "src/new_fourth_file.py"},
    )
    assert not c.as_pre_rule(w0, (over_cap,))[0]

    strict_c = AcyclicDependencyConstraint(
        reject_stale_observations=True,
        raise_on_violation=False,
    )
    engine_soft = build_swe_agent_engine(
        initial_swe_state=swe0,
        raise_on_violation=False,
    )
    dyn = SWERepositoryDynamics()
    assert dyn.mechanism_provenance.mechanism_id == "SWERepositoryDynamics"

    # Step 1: modify schema.py so agent_a's observation becomes stale
    s1 = engine_soft.step(
        w0,
        build_swe_action(
            "mod_schema",
            "MODIFY_FILE",
            actor_id="agent_b",
            parameters={"file_path": "src/core/schema.py"},
        ),
    )
    # Check strict_c on MODIFY_FILE with stale base_revisions
    stale_mod = build_swe_action(
        "stale_mod",
        "MODIFY_FILE",
        actor_id="agent_a",
        parameters={
            "file_path": "src/core/schema.py",
            "base_revisions": {"src/core/schema.py": 1},
        },
    )
    assert not strict_c.as_pre_rule(s1.next_state, (stale_mod,))[0]

    # Enqueue PR with removed_dependencies and explicit base_revisions, then merge creating new file
    s2 = engine_soft.step(
        s1.next_state,
        build_swe_action(
            "enq_new_file",
            "ENQUEUE_PR",
            actor_id="agent_new",
            parameters={
                "pr_id": "PR-NEW",
                "modified_files": ["src/brand_new.py"],
                "added_dependencies": [["src/brand_new.py", "src/brand_new_dep.py"]],
                "removed_dependencies": [
                    ["src/api/OrderController.py", "src/services/OrderService.py"]
                ],
                "base_revisions": {"src/core/schema.py": 2},
            },
        ),
    )
    s3 = engine_soft.step(
        s2.next_state,
        build_swe_action(
            "merge_new_file",
            "MERGE_PR",
            actor_id="agent_new",
            parameters={"pr_id": "PR-NEW"},
        ),
    )
    swe3 = SWEState.from_world_state(s3.next_state)
    assert "src/brand_new.py" in swe3.files
    assert "src/brand_new_dep.py" in swe3.files

    # Modifying a brand-new untracked file via MODIFY_FILE with added_dependencies
    s4 = engine_soft.step(
        s3.next_state,
        build_swe_action(
            "mod_untracked",
            "MODIFY_FILE",
            actor_id="agent_new",
            parameters={
                "file_path": "src/untracked_mod.py",
                "added_dependencies": [["src/untracked_mod.py", "src/untracked_dep.py"]],
            },
        ),
    )
    swe4 = SWEState.from_world_state(s4.next_state)
    assert "src/untracked_mod.py" in swe4.files
    assert "src/untracked_dep.py" in swe4.files

    # Unsupported action type and missing PR errors in SWERepositoryDynamics
    with pytest.raises(ValueError, match="No PENDING pull request"):
        dyn._apply_single_action(
            swe4,
            build_swe_action("no_pr", "MERGE_PR", parameters={"pr_id": "NON_EXISTENT"}),
        )
    with pytest.raises(ValueError, match="Unsupported SWE action_type"):
        dyn._apply_single_action(
            swe4,
            build_swe_action("bad_type", "UNKNOWN_OP"),
        )
