"""Enterprise World Model Engine (EWM Engine).

A general-purpose framework for defining, simulating, evaluating,
and learning the dynamics of complex organizational and socio-technical systems.
"""

from __future__ import annotations

from ewm_engine.actors import Actor, ActorContext
from ewm_engine.constraints import (
    Constraint,
    ConstraintRegistry,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core import (
    Action,
    Entity,
    ExogenousEvent,
    Intervention,
    Relationship,
    Resource,
    World,
    WorldState,
)
from ewm_engine.dynamics import (
    CompositeDynamics,
    DeterministicDynamics,
    DynamicsModel,
    StochasticDemandDynamics,
    TransitionResult,
)
from ewm_engine.evaluation import (
    ScenarioComparison,
    compare_scenarios,
    summarize_distribution,
)
from ewm_engine.provenance import EvidenceLevel, SimulationMetadata, SystemicTrace
from ewm_engine.simulation import Scenario, SimulationEngine, SimulationResult, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Action",
    "Actor",
    "ActorContext",
    "CompositeDynamics",
    "Constraint",
    "ConstraintRegistry",
    "ConstraintResult",
    "ConstraintSeverity",
    "DeterministicDynamics",
    "DynamicsModel",
    "Entity",
    "EvidenceLevel",
    "ExogenousEvent",
    "Intervention",
    "Relationship",
    "Resource",
    "Scenario",
    "ScenarioComparison",
    "SimulationEngine",
    "SimulationMetadata",
    "SimulationResult",
    "StochasticDemandDynamics",
    "SystemicTrace",
    "Trajectory",
    "TransitionResult",
    "World",
    "WorldState",
    "__version__",
    "compare_scenarios",
    "summarize_distribution",
]
