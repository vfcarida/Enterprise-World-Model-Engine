"""Deterministic dynamics implementations based on structural rules and conservation laws."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class DeterministicDynamics:
    """A dynamics model wrapping deterministic transformation functions.

    Transitions execute without stochasticity and carry a STRUCTURAL evidence level.
    """

    def __init__(
        self,
        name: str = "DeterministicDynamics",
        transition_fn: Callable[
            [WorldState, Sequence[Action], Sequence[ExogenousEvent]], WorldState
        ]
        | None = None,
    ) -> None:
        self.name = name
        self._transition_fn = transition_fn

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        """Execute deterministic state transformation."""
        if self._transition_fn is not None:
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
        self, action_type: str = "transfer_resource", name: str = "TransferDynamics"
    ) -> None:
        self.action_type = action_type
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
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
            model_name=self.name,
        )


class DeterministicDemandDynamics:
    """Models deterministic demand consuming from a resource.

    Guarantees structural evidence and tracks served vs unserved demand in state memory.
    """

    def __init__(
        self,
        resource_id: str,
        demand: float,
        name: str = "DeterministicDemandDynamics",
        evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
    ) -> None:
        self.resource_id = resource_id
        self.demand = max(0.0, float(demand))
        self.name = name
        self.evidence_level = evidence_level

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        current_res = state.get_resource(self.resource_id)
        available_stock = max(0.0, current_res.current - current_res.min_value)

        served = min(self.demand, available_stock)
        unserved = self.demand - served

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
                "demand": self.demand,
                "served_demand": served,
                "unserved_demand": unserved,
            },
            evidence_level=self.evidence_level,
            model_name=self.name,
        )
