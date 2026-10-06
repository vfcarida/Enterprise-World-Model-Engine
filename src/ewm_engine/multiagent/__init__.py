"""Multi-agent and game-theoretic module (T6).

Provides observation views, action proposals, deterministic mediation via the
constraint engine, and game-theoretic solvers (Nash equilibria, replicator dynamics).
"""

from __future__ import annotations

from ewm_engine.multiagent.adapters import (
    ConcordiaGameMasterAdapter,
    OpenSpielAdapter,
    PettingZooParallelAdapter,
    is_concordia_available,
    is_openspiel_available,
    is_pettingzoo_available,
)
from ewm_engine.multiagent.game_theory import (
    NashpyGameSolver,
    is_nashpy_available,
)
from ewm_engine.multiagent.mediator import (
    ConstraintMediator,
    Mediator,
)
from ewm_engine.multiagent.views import (
    ActorActionProposal,
    ActorObservationView,
    AdjudicationResult,
)

__all__ = [
    "ActorActionProposal",
    "ActorObservationView",
    "AdjudicationResult",
    "ConcordiaGameMasterAdapter",
    "ConstraintMediator",
    "Mediator",
    "NashpyGameSolver",
    "OpenSpielAdapter",
    "PettingZooParallelAdapter",
    "is_concordia_available",
    "is_nashpy_available",
    "is_openspiel_available",
    "is_pettingzoo_available",
]
