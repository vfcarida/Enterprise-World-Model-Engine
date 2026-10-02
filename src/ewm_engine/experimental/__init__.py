"""Experimental modules and algorithms for EWM Engine.

WARNING: Everything in this namespace is EXPERIMENTAL. Interfaces, signatures,
and behaviors may change or be removed across releases without deprecation cycles.
"""

from __future__ import annotations

from ewm_engine.core.actions import Intervention
from ewm_engine.dynamics.learned import (
    LearnedDynamics,
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.simulation.mpc import MPCDecisionRecord, RecedingHorizonSimulator

__all__ = [
    "Intervention",
    "LearnedDynamics",
    "LinearResidualDynamics",
    "MPCDecisionRecord",
    "RecedingHorizonSimulator",
    "TransitionDataset",
    "TransitionSample",
]
