"""Deterministic simulation engine, scenario definitions, and rollout trajectories."""

from __future__ import annotations

from ewm_engine.simulation.branching import branch_scenario, branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.mpc import (
    MPCDecisionRecord,
    RecedingHorizonResult,
    RecedingHorizonSimulator,
)
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
    TrajectoryStatus,
)

__all__ = [
    "MPCDecisionRecord",
    "RecedingHorizonResult",
    "RecedingHorizonSimulator",
    "RunMetrics",
    "Scenario",
    "ScheduledAction",
    "SimulationEngine",
    "SimulationResult",
    "StepRecord",
    "Trajectory",
    "TrajectoryStatus",
    "branch_scenario",
    "branch_world",
]
