"""Enterprise World Model Engine (EWM Engine).

A general-purpose framework for defining, simulating, evaluating,
and learning the dynamics of complex organizational and socio-technical systems.
"""

from __future__ import annotations

from ewm_engine.constraints import (
    Constraint,
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core import (
    Action,
    Entity,
    ExogenousEvent,
    Relationship,
    Resource,
    World,
    WorldState,
)
from ewm_engine.dynamics import DynamicsModel, TransitionResult
from ewm_engine.evaluation import compare_scenarios
from ewm_engine.provenance import EvidenceLevel, Provenance, TraceEdge
from ewm_engine.simulation import (
    Scenario,
    SimulationEngine,
    SimulationResult,
    Trajectory,
    TrajectoryStatus,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Constraint",
    "ConstraintPhase",
    "ConstraintResult",
    "ConstraintSeverity",
    "DynamicsModel",
    "Entity",
    "EvidenceLevel",
    "ExogenousEvent",
    "Provenance",
    "Relationship",
    "Resource",
    "Scenario",
    "SimulationEngine",
    "SimulationResult",
    "TraceEdge",
    "Trajectory",
    "TrajectoryStatus",
    "TransitionResult",
    "World",
    "WorldState",
    "__version__",
    "compare_scenarios",
]
