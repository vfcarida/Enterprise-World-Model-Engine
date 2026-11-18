"""Software Engineering Domain Pack (`ewm_engine.domains.software_engineering`).

Exposes all SWE domain primitives alongside Dynamic Causality-Aware Root Cause Analysis
(`DynamicRCASimulator`, `DynamicRCAResult`, `DynamicRCAError`, `MAX_EIRWR_ITERATIONS`)
based on eIRWR and DynaCausal algorithms (E7).
"""

from __future__ import annotations

from ewm_engine.domains.swe import (
    MAX_EIRWR_ITERATIONS,
    MAX_SWE_GRAPH_NODES,
    SWE_REPO_ENTITY_ID,
    AcyclicDependencyConstraint,
    BranchExecutionTrace,
    CircularDependencyError,
    CodeFileNode,
    ConcurrencyInterventionTrace,
    DynamicRCAError,
    DynamicRCAResult,
    DynamicRCASimulator,
    MergeQueueItem,
    PullRequestStatus,
    StaleObservationError,
    StaleObservationRecord,
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

__all__ = [
    "MAX_EIRWR_ITERATIONS",
    "MAX_SWE_GRAPH_NODES",
    "SWE_REPO_ENTITY_ID",
    "AcyclicDependencyConstraint",
    "BranchExecutionTrace",
    "CircularDependencyError",
    "CodeFileNode",
    "ConcurrencyInterventionTrace",
    "DynamicRCAError",
    "DynamicRCAResult",
    "DynamicRCASimulator",
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
