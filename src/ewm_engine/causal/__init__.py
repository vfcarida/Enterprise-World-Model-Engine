"""Causal fabric, graph surgery (`do(X_t = x)`), and structural causal model primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ewm_engine.causal.intervention import (
    CausalExecutionLogEntry,
    InterventionContext,
    get_active_interventions,
    normalize_node_name,
    parse_intervention_targets,
    sever_inbound_edges,
)
from ewm_engine.core.actions import Intervention
from ewm_engine.core.scm import (
    AbductionResult,
    CausalGraph,
    ExogenousVariable,
    InvertibilityType,
    NonUniqueAbductionWarning,
    StructuralAssignment,
    StructuralCausalModel,
)

if TYPE_CHECKING:
    from ewm_engine.dynamics.router import (
        CalibratedPrediction,
        CalibratedStateDistribution,
        HybridTransitionRouter,
        NumericalInstabilityError,
        TransitionEngine,
        TransitionStepOutcome,
    )

_ROUTER_EXPORTS = {
    "CalibratedPrediction",
    "CalibratedStateDistribution",
    "HybridTransitionRouter",
    "NumericalInstabilityError",
    "TransitionEngine",
    "TransitionStepOutcome",
}


def __getattr__(name: str) -> Any:
    if name in _ROUTER_EXPORTS:
        from ewm_engine.dynamics import router as _router

        return getattr(_router, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbductionResult",
    "CalibratedPrediction",
    "CalibratedStateDistribution",
    "CausalExecutionLogEntry",
    "CausalGraph",
    "ExogenousVariable",
    "HybridTransitionRouter",
    "Intervention",
    "InterventionContext",
    "InvertibilityType",
    "NonUniqueAbductionWarning",
    "NumericalInstabilityError",
    "StructuralAssignment",
    "StructuralCausalModel",
    "TransitionEngine",
    "TransitionStepOutcome",
    "get_active_interventions",
    "normalize_node_name",
    "parse_intervention_targets",
    "sever_inbound_edges",
]
