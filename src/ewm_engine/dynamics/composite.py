"""Composition architecture for hybrid and modular enterprise world dynamics."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import (
    DynamicsModel,
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
)
from ewm_engine.dynamics.deterministic import _build_context_diagnostics
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
        *,
        version: str = "1.0.0",
    ) -> None:
        self.models = list(models)
        self.name = name
        self.version = version

        req_keys: list[str] = []
        invariants: list[str] = []
        child_evidences: list[EvidenceLevel] = []
        for m in self.models:
            prov = getattr(m, "mechanism_provenance", None)
            if isinstance(prov, MechanismProvenance):
                for k in prov.required_context_keys:
                    if k not in req_keys:
                        req_keys.append(k)
                for inv in prov.invariants_affected:
                    if inv not in invariants:
                        invariants.append(inv)
                child_evidences.append(prov.evidence_level)

        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.HYBRID,
            required_context_keys=tuple(req_keys),
            evidence_level=self._aggregate_evidence(child_evidences),
            source="composite_pipeline",
            uncertainty_contract="composite",
            invariants_affected=tuple(invariants),
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        """Return formal mechanism provenance metadata."""
        return self._provenance

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        current_state = state
        aggregated_changes: dict[str, Any] = {}
        aggregated_diagnostics: dict[str, Any] = {}
        child_evidences: list[EvidenceLevel] = []

        for idx, model in enumerate(self.models):
            accepts_ctx = False
            if runtime_context is not None:
                try:
                    sig = inspect.signature(model.transition)
                    accepts_ctx = "runtime_context" in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                except (ValueError, TypeError):
                    accepts_ctx = False

            if accepts_ctx and runtime_context is not None:
                fn: Any = model.transition
                result = fn(
                    state=current_state,
                    actions=actions,
                    exogenous_events=exogenous_events,
                    rng=rng,
                    runtime_context=runtime_context,
                )
            else:
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
        diag = _build_context_diagnostics(
            self._provenance,
            runtime_context,
            extra=aggregated_diagnostics,
        )

        return TransitionResult(
            next_state=current_state,
            applied_changes=aggregated_changes,
            evidence_level=overall_evidence,
            diagnostics=diag,
            model_name=self.name,
        )

    @staticmethod
    def _aggregate_evidence(evidences: list[EvidenceLevel]) -> EvidenceLevel:
        if not evidences:
            return EvidenceLevel.STRUCTURAL
        hierarchy = [
            EvidenceLevel.CONFOUNDED,
            EvidenceLevel.UNIDENTIFIED,
            EvidenceLevel.HYPOTHETICAL,
            EvidenceLevel.ASSUMED,
            EvidenceLevel.DIRECTIONAL_UNCERTAIN,
            EvidenceLevel.PROVISIONAL,
            EvidenceLevel.PREDICTIVE,
            EvidenceLevel.CALIBRATED,
            EvidenceLevel.QUASI_CAUSAL,
            EvidenceLevel.INTERVENTIONAL,
            EvidenceLevel.ASSERTED,
            EvidenceLevel.STRUCTURAL,
            EvidenceLevel.COUNTERFACTUAL,
        ]
        for level in hierarchy:
            if level in evidences:
                return level
        return EvidenceLevel.STRUCTURAL
