# SWE Domain Pack (`ewm_engine.domains.swe`)

## Overview

The **SWE Domain Pack** (`EWM-005`) models multi-agent software engineering environments—specifically directed acyclic code dependency graphs (DAGs), pull-request merge queues, downstream build/test queues, and concurrent agent observation snapshots—on top of the domain-neutral `ewm_engine.core` state fabric (`EWM-001`) and `ewm_engine.dynamics` hybrid router (`EWM-002`).

Network graph dependencies (`networkx>=3.2`) are isolated under the `[swe]` optional dependency extra in `pyproject.toml`:

```bash
pip install -e ".[swe]"
```

## Core Contracts

1. **Directed Acyclic Dependency Graph (`SWEState`)**:
   - Models repository files (`CodeFileNode`) and import/build dependencies (`dependent -> dependency`) backed by `networkx.DiGraph` (`SWEState.dependency_graph` / `SWEState.from_networkx_dag`).
   - Enforces a configurable node ceiling (`MAX_SWE_GRAPH_NODES = 2000`) to prevent memory exhaustion on large repositories.
2. **Iterative $O(V + E)$ Cycle Detection (`AcyclicDependencyConstraint`)**:
   - Uses iterative Kahn's topological sort combined with an explicit stack-based 3-color depth-first search (`detect_cycle_iterative_o_v_plus_e`) to detect and extract directed cycles in $O(V + E)$ time without recursion-limit risks (`RecursionError`).
   - Rejects any transition (`ADD_DEPENDENCY`, `ENQUEUE_PR`, `MERGE_PR`, `MODIFY_FILE`) that introduces a circular dependency 100% of the time via `HybridTransitionRouter`.
3. **Downstream Build Queue Propagation (`find_downstream_dependents_iterative`)**:
   - Modifying or merging changes to any file $Y$ automatically traverses the reverse dependency graph in $O(V + E)$ time and enqueues $Y$ along with all transitive dependents $X \to^* Y$ into `SWEState.build_queue`.
4. **Stale Observation Tracking (`detect_stale_observations_for_action`)**:
   - Tracks per-agent revision snapshots (`agent_observations`) and pull-request base revisions (`MergeQueueItem.base_revisions`).
   - Detects both direct file conflicts (`direct_file_conflict`) and upstream/downstream dependency drift (`dependency_revision_drift`) when concurrent agents modify interdependent files.

---

## Agentic Coding Concurrency Simulation Scenario

### Question

> *"What if Agent A merges PR X (`PR-101` modifying `src/services/OrderService.py`) while Agent B modifies dependent File Y (`src/core/schema.py`)?"*

### Repository Dependency Graph at Checkpoint $t = 0$

```text
src/api/OrderController.py (File Z, rev 1)
        │
        ▼  depends on
src/services/OrderService.py (File X, rev 1)  ◄── PR-101 queued by agent_a (base_revisions: X=1, Y=1)
        │
        ▼  depends on
src/core/schema.py (File Y, rev 1)            ◄── Modified concurrently by agent_b
```

### Parallel Counterfactual Branches via `engine.branch().do()`

1. **Branch 1 (`branch_agent_a_merges_first`)**:
   - **Step 1 (`do(MERGE_PR, pr_id="PR-101")`)**: `agent_a` merges `PR-101` while `src/core/schema.py` is still at `rev 1`. `PR-101` transitions to `MERGED`, `src/services/OrderService.py` advances to `rev 2`, and `('src/services/OrderService.py', 'src/api/OrderController.py')` are propagated into `build_queue`.
   - **Step 2 (`do(MODIFY_FILE, file_path="src/core/schema.py")`)**: `agent_b` modifies `src/core/schema.py` using its `t=0` observation snapshot (`OrderService.py` at `rev 1`). Structural tracking logs a `dependency_revision_drift` `StaleObservationRecord` (since dependent `OrderService.py` is now at `rev 2`), advances `src/core/schema.py` to `rev 2`, and enqueues the full transitive closure `('src/services/OrderService.py', 'src/api/OrderController.py', 'src/core/schema.py')` into `build_queue`.

2. **Branch 2 (`branch_agent_b_modifies_first`)**:
   - **Step 1 (`do(MODIFY_FILE, file_path="src/core/schema.py")`)**: `agent_b` modifies `src/core/schema.py` (`File Y`) first, advancing it from `rev 1` to `rev 2` and propagating `('src/core/schema.py', 'src/services/OrderService.py', 'src/api/OrderController.py')` into `build_queue`.
   - **Step 2 (`do(MERGE_PR, pr_id="PR-101")`)**: `agent_a` attempts to merge `PR-101`. Because `PR-101` was prepared against `src/core/schema.py` at `rev 1` (`base_revisions={"src/services/OrderService.py": 1, "src/core/schema.py": 1}`), structural stale-observation detection catches the upstream dependency drift (`expected_revision=1 != actual_revision=2`), transitions `PR-101` to `CONFLICT_STALE` (preventing a broken merge), and records a `StaleObservationRecord`.

3. **Branch 3 (`branch_cyclic_injection`)**:
   - An adversarial action attempts to add edge `src/core/schema.py -> src/api/OrderController.py`.
   - `AcyclicDependencyConstraint` detects the 3-node cycle `src/api/OrderController.py -> src/services/OrderService.py -> src/core/schema.py -> src/api/OrderController.py` in $O(V + E)$ time and raises `StructuralViolationError`.
