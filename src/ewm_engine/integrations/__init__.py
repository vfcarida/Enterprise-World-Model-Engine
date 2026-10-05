"""Ecosystem adapters connecting external agents, formal solvers, OR planners, and telemetry."""

from __future__ import annotations

from ewm_engine.integrations.agents import CallableActorAdapter
from ewm_engine.integrations.gym import EnterpriseGymEnv
from ewm_engine.integrations.ortools import (
    CPSATAllocationPlanner,
    ORToolsAllocationAdapter,
)
from ewm_engine.integrations.otel import OpenTelemetryHook
from ewm_engine.integrations.protocols import (
    ActionPlanner,
    ConstraintSolver,
    SolverResult,
    SolverStatus,
)
from ewm_engine.integrations.scipy_planner import SciPyAllocationPlanner
from ewm_engine.integrations.solvers import Z3ConstraintAdapter

__all__ = [
    "ActionPlanner",
    "CPSATAllocationPlanner",
    "CallableActorAdapter",
    "ConstraintSolver",
    "EnterpriseGymEnv",
    "ORToolsAllocationAdapter",
    "OpenTelemetryHook",
    "SciPyAllocationPlanner",
    "SolverResult",
    "SolverStatus",
    "Z3ConstraintAdapter",
]
