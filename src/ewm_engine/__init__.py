"""Enterprise World Model Engine (EWM Engine).

A general-purpose framework for defining, simulating, evaluating,
and learning the dynamics of complex organizational and socio-technical systems.
"""

from __future__ import annotations

import logging

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
    ResponsibleAIGate,
    ScenarioForecastResult,
    World,
    WorldState,
    deprecate,
    deprecated,
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

# Ensure library root logger has a NullHandler to prevent unhandled log warnings
logging.getLogger("ewm_engine").addHandler(logging.NullHandler())

try:
    from ewm_engine._version import __version__
except ImportError:
    try:
        from importlib.metadata import version as _get_version

        __version__ = _get_version("ewm-engine")
    except Exception:
        __version__ = "1.0.0"

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
    "ResponsibleAIGate",
    "Scenario",
    "ScenarioForecastResult",
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
    "deprecate",
    "deprecated",
]
