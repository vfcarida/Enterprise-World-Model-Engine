"""Deterministic dynamics implementations based on structural rules and conservation laws."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import (
    MechanismKind,
    MechanismProvenance,
    TransitionResult,
    ValidityDomain,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def _build_context_diagnostics(
    provenance: MechanismProvenance,
    runtime_context: ContextSnapshot | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    diag: dict[str, Any] = dict(extra or {})
    diag["mechanism_provenance"] = provenance.model_dump(mode="json")
    if runtime_context is not None:
        diag["context_metadata"] = {
            "context_id": runtime_context.context_id,
            "version": runtime_context.version,
            "fingerprint": runtime_context.fingerprint,
            "domain_fingerprint": runtime_context.domain_fingerprint(
                provenance.required_context_keys
            ),
            "tenant_id": runtime_context.tenant_id,
            "environment_regime": runtime_context.environment_regime,
        }
    return diag


class DeterministicDynamics:
    """A dynamics model wrapping deterministic transformation functions.

    Transitions execute without stochasticity and carry a STRUCTURAL evidence level.
    Supports optional context-conditioned callables (3-arg or 4-arg with ContextSnapshot)
    and mechanism-scoped context projection via ``required_context_keys``.
    """

    def __init__(
        self,
        name: str = "DeterministicDynamics",
        transition_fn: Callable[..., WorldState] | None = None,
        *,
        version: str = "1.0.0",
        required_context_keys: Sequence[str] = (),
        validity_domain: ValidityDomain | None = None,
        invariants_affected: Sequence[str] = (),
        source: str = "explicit_specification",
    ) -> None:
        self.name = name
        self._transition_fn = transition_fn
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            validity_domain=validity_domain or ValidityDomain(),
            required_context_keys=tuple(required_context_keys),
            evidence_level=EvidenceLevel.STRUCTURAL,
            source=source,
            uncertainty_contract="point_deterministic",
            invariants_affected=tuple(invariants_affected),
        )
        self._accepts_context = False
        if transition_fn is not None:
            try:
                sig = inspect.signature(transition_fn)
                params = list(sig.parameters.values())
                self._accepts_context = len(params) >= 4 or any(
                    p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    for p in params
                )
            except (ValueError, TypeError):
                self._accepts_context = False

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
        """Execute deterministic state transformation conditioned on projected context."""
        projected_ctx = (
            runtime_context.project(self._provenance.required_context_keys)
            if runtime_context is not None
            else None
        )
        if self._transition_fn is not None:
            if self._accepts_context:
                next_state = self._transition_fn(state, actions, exogenous_events, projected_ctx)
            else:
                next_state = self._transition_fn(state, actions, exogenous_events)
        else:
            next_state = state

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "actions_processed": len(actions),
                "events_processed": len(exogenous_events),
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics=_build_context_diagnostics(self._provenance, runtime_context),
            model_name=self.name,
        )


class DeterministicTransferDynamics:
    """Executes conservation-preserving resource transfers between entities.

    Processes actions of type 'transfer_resource' with parameters:
      - source_resource: ResourceId
      - target_resource: ResourceId
      - quantity: float
    """

    def __init__(
        self,
        action_type: str = "transfer_resource",
        name: str = "TransferDynamics",
        *,
        version: str = "1.0.0",
        required_context_keys: Sequence[str] = (),
    ) -> None:
        self.action_type = action_type
        self.name = name
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            required_context_keys=tuple(required_context_keys),
            evidence_level=EvidenceLevel.STRUCTURAL,
            source="explicit_conservation_transfer",
            uncertainty_contract="point_deterministic",
            invariants_affected=("mass_balance", "resource_bounds"),
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
        transfers_executed = 0
        total_volume = 0.0

        for action in actions:
            if action.type != self.action_type:
                continue

            src_id = str(action.get("source_resource"))
            tgt_id = str(action.get("target_resource"))
            requested_qty = float(action.get("quantity", 0.0))

            if requested_qty <= 0.0:
                continue

            src_res = current_state.get_resource(src_id)
            tgt_res = current_state.get_resource(tgt_id)

            # Strict conservation: transfer min(requested, available_stock, target_headroom)
            actual_qty = min(
                requested_qty, src_res.current - src_res.min_value, tgt_res.available_capacity
            )

            if actual_qty > 0.0:
                current_state = current_state.update_resource(src_id, delta=-actual_qty)
                current_state = current_state.update_resource(tgt_id, delta=actual_qty)
                transfers_executed += 1
                total_volume += actual_qty

        return TransitionResult(
            next_state=current_state,
            applied_changes={
                "transfers_executed": transfers_executed,
                "total_transferred_volume": total_volume,
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics=_build_context_diagnostics(self._provenance, runtime_context),
            model_name=self.name,
        )


class DeterministicDemandDynamics:
    """Models deterministic demand consuming from a resource.

    Guarantees structural evidence and tracks served vs unserved demand in state memory.
    Optionally reads a demand multiplier from ``ContextSnapshot.parameters[context_demand_multiplier_key]``
    when ``context_demand_multiplier_key`` is configured.
    """

    def __init__(
        self,
        resource_id: str,
        demand: float,
        name: str = "DeterministicDemandDynamics",
        evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
        *,
        version: str = "1.0.0",
        context_demand_multiplier_key: str | None = None,
    ) -> None:
        self.resource_id = resource_id
        self.demand = max(0.0, float(demand))
        self.name = name
        self.evidence_level = evidence_level
        self.context_demand_multiplier_key = context_demand_multiplier_key
        req_keys = (
            (f"parameters.{context_demand_multiplier_key}",)
            if context_demand_multiplier_key is not None
            else ()
        )
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            required_context_keys=req_keys,
            evidence_level=evidence_level,
            source="explicit_demand_equation",
            uncertainty_contract="point_deterministic",
            invariants_affected=(f"resource_bounds:{resource_id}",),
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
        effective_demand = self.demand
        if self.context_demand_multiplier_key is not None and runtime_context is not None:
            projected = runtime_context.project(self._provenance.required_context_keys)
            mult = projected.parameters.get(self.context_demand_multiplier_key, 1.0)
            if isinstance(mult, (int, float)):
                effective_demand = max(0.0, self.demand * float(mult))

        current_res = state.get_resource(self.resource_id)
        available_stock = max(0.0, current_res.current - current_res.min_value)

        served = min(effective_demand, available_stock)
        unserved = effective_demand - served

        next_state = state.update_resource(self.resource_id, delta=-served)
        prior_served = float(state.memory.get("served_demand", 0.0))
        prior_unserved = float(state.memory.get("unserved_demand", 0.0))

        next_state = next_state.with_memory("served_demand", prior_served + served)
        next_state = next_state.with_memory("unserved_demand", prior_unserved + unserved)
        next_state = next_state.with_memory(f"served_{self.resource_id}", prior_served + served)
        next_state = next_state.with_memory(
            f"unserved_{self.resource_id}", prior_unserved + unserved
        )

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "resource_id": self.resource_id,
                "demand": effective_demand,
                "served_demand": served,
                "unserved_demand": unserved,
            },
            evidence_level=self.evidence_level,
            diagnostics=_build_context_diagnostics(self._provenance, runtime_context),
            model_name=self.name,
        )


class DeterministicFixedTransferDynamics:
    """Executes a fixed resource transfer from source to target resource at each step."""

    def __init__(
        self,
        source_resource: str,
        target_resource: str,
        quantity: float,
        name: str = "DeterministicFixedTransferDynamics",
        evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
        *,
        version: str = "1.0.0",
    ) -> None:
        self.source_resource = source_resource
        self.target_resource = target_resource
        self.quantity = max(0.0, float(quantity))
        self.name = name
        self.evidence_level = evidence_level
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            required_context_keys=(),
            evidence_level=evidence_level,
            source="explicit_fixed_transfer",
            uncertainty_contract="point_deterministic",
            invariants_affected=(
                f"resource_bounds:{source_resource}",
                f"resource_bounds:{target_resource}",
            ),
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
        src = state.get_resource(self.source_resource)
        tgt = state.get_resource(self.target_resource)
        actual = min(
            self.quantity,
            max(0.0, src.current - src.min_value),
            max(0.0, tgt.max_value - tgt.current),
        )
        next_state = state.update_resource(self.source_resource, delta=-actual).update_resource(
            self.target_resource, delta=actual
        )
        return TransitionResult(
            next_state=next_state,
            applied_changes={"transferred": actual},
            evidence_level=self.evidence_level,
            diagnostics=_build_context_diagnostics(self._provenance, runtime_context),
            model_name=self.name,
        )


class DeterministicDriftDynamics:
    """Applies a constant deterministic drift on a resource at each step."""

    def __init__(
        self,
        resource_id: str,
        drift: float,
        name: str = "DeterministicDriftDynamics",
        evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
        *,
        version: str = "1.0.0",
    ) -> None:
        self.resource_id = resource_id
        self.drift = float(drift)
        self.name = name
        self.evidence_level = evidence_level
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC,
            required_context_keys=(),
            evidence_level=evidence_level,
            source="explicit_drift_equation",
            uncertainty_contract="point_deterministic",
            invariants_affected=(f"resource_bounds:{resource_id}",),
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
        next_state = state.update_resource(self.resource_id, delta=self.drift)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"drift": self.drift},
            evidence_level=self.evidence_level,
            diagnostics=_build_context_diagnostics(self._provenance, runtime_context),
            model_name=self.name,
        )
