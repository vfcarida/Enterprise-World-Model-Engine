"""SWE Domain Pack for Enterprise World Model Engine (EWM-005).

Provides NetworkX-backed directed acyclic code dependency graphs (DAGs),
FIFO pull-request merge queues, O(V + E) iterative cycle detection,
stale observation tracking, and multi-agent concurrency interventions.
"""

from __future__ import annotations

from ewm_engine.domains.swe.constraints import (
    AcyclicDependencyConstraint,
    StructuralViolationError,
    detect_stale_observations_for_action,
)
from ewm_engine.domains.swe.scenario import (
    BranchExecutionTrace,
    ConcurrencyInterventionTrace,
    SWERepositoryDynamics,
    build_swe_agent_engine,
    create_default_swe_repository_state,
    run_concurrent_agent_intervention_scenario,
)
from ewm_engine.domains.swe.state import (
    MAX_SWE_GRAPH_NODES,
    SWE_REPO_ENTITY_ID,
    CircularDependencyError,
    CodeFileNode,
    MergeQueueItem,
    PullRequestStatus,
    StaleObservationError,
    StaleObservationRecord,
    SWEGraphCapacityError,
    SWEState,
    build_swe_action,
    detect_cycle_iterative_o_v_plus_e,
    find_downstream_dependents_iterative,
)

__all__ = [
    "MAX_SWE_GRAPH_NODES",
    "SWE_REPO_ENTITY_ID",
    "AcyclicDependencyConstraint",
    "BranchExecutionTrace",
    "CircularDependencyError",
    "CodeFileNode",
    "ConcurrencyInterventionTrace",
    "MergeQueueItem",
    "PullRequestStatus",
    "SWEGraphCapacityError",
    "SWERepositoryDynamics",
    "SWEState",
    "StaleObservationError",
    "StaleObservationRecord",
    "StructuralViolationError",
    "build_swe_action",
    "build_swe_agent_engine",
    "create_default_swe_repository_state",
    "detect_cycle_iterative_o_v_plus_e",
    "detect_stale_observations_for_action",
    "find_downstream_dependents_iterative",
    "run_concurrent_agent_intervention_scenario",
]
