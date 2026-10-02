"""Composition architecture for hybrid and modular enterprise world dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class CompositeDynamics:
    """Orchestrates sequential or additive pipelines of modular dynamics models.

    Enables clean composition of:
        Deterministic structural flows
        + Stochastic environmental disruptions
        + Behavioral and learned residual models
    """

    def __init__(
        self,
        models: Sequence[DynamicsModel],
        name: str = "CompositeDynamics",
    ) -> None:
        self.models = list(models)
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        current_state = state
        aggregated_changes: dict[str, Any] = {}
        aggregated_diagnostics: dict[str, Any] = {}
        child_evidences: list[EvidenceLevel] = []

        for idx, model in enumerate(self.models):
            result = model.transition(
                state=current_state,
                actions=actions,
                exogenous_events=exogenous_events,
                rng=rng,
            )
            current_state = result.next_state
            aggregated_changes[f"substep_{idx}_{result.model_name or 'model'}"] = (
                result.applied_changes
            )
            aggregated_diagnostics[f"substep_{idx}_{result.model_name or 'model'}"] = (
                result.diagnostics
            )
            child_evidences.append(result.evidence_level)

        # Aggregate epistemic evidence level (lowest common denominator)
        overall_evidence = self._aggregate_evidence(child_evidences)

        return TransitionResult(
            next_state=current_state,
            applied_changes=aggregated_changes,
            evidence_level=overall_evidence,
            diagnostics=aggregated_diagnostics,
            model_name=self.name,
        )

    @staticmethod
    def _aggregate_evidence(evidences: list[EvidenceLevel]) -> EvidenceLevel:
        if not evidences:
            return EvidenceLevel.STRUCTURAL
        hierarchy = [
            EvidenceLevel.ASSUMED,
            EvidenceLevel.PREDICTIVE,
            EvidenceLevel.QUASI_CAUSAL,
            EvidenceLevel.INTERVENTIONAL,
            EvidenceLevel.STRUCTURAL,
        ]
        for level in hierarchy:
            if level in evidences:
                return level
        return EvidenceLevel.STRUCTURAL
