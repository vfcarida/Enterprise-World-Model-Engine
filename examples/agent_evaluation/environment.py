"""Hospital supply logistics environment for agent evaluation.

Demonstrates the 'engine != agent' architecture:
The world model engine defines immutable state, physical constraints (corridor link capacities,
hub reserves), and consumption dynamics. Agents propose candidate interventions which are
rigorously validated and simulated by the engine.
"""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId, RandomGenerator
from ewm_engine.core.world import World
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class CorridorCapacityConstraint(Constraint):
    """Hard physical limit on the maximum units transportable along any corridor in a single step."""

    def __init__(
        self,
        max_transfer_per_corridor: float = 60.0,
        constraint_id: ConstraintId = "corridor_capacity_limit",
    ) -> None:
        self._max = max_transfer_per_corridor
        self._constraint_id = constraint_id

    @property
    def constraint_id(self) -> ConstraintId:
        return self._constraint_id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return f"Physical link limit: max {self._max} units per transfer action."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        violating: dict[str, object] = {}
        for action in actions:
            if action.type == "transfer_resource":
                qty = float(action.parameters.get("quantity", 0.0))
                if qty > self._max:
                    violating[action.id] = {
                        "quantity": qty,
                        "limit": self._max,
                        "source": action.parameters.get("source_resource"),
                        "target": action.parameters.get("target_resource"),
                    }

        satisfied = len(violating) == 0
        msg = (
            "All transfers within corridor link capacities"
            if satisfied
            else f"Actions exceed physical corridor transfer limit of {self._max} units"
        )
        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            phase=phase,
            message=msg,
            violating_values=violating,
        )


class HospitalLogisticsDynamics(DynamicsModel):
    """Simulates resource transfers followed by patient consumption demand at regional hospitals."""

    def __init__(
        self,
        patient_demands: dict[str, float] | None = None,
        name: str = "HospitalLogisticsDynamics",
    ) -> None:
        self.patient_demands = patient_demands or {
            "hospital_north": 30.0,
            "hospital_central": 45.0,
            "hospital_south": 25.0,
        }
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        # 1. Apply accepted resource transfers
        current_resources = {r.id: r.current for r in state.resources.values()}

        for action in actions:
            if action.type == "transfer_resource":
                src = str(action.parameters.get("source_resource", ""))
                dst = str(action.parameters.get("target_resource", ""))
                qty = float(action.parameters.get("quantity", 0.0))
                if src in current_resources and dst in current_resources:
                    current_resources[src] = max(0.0, current_resources[src] - qty)
                    current_resources[dst] = current_resources[dst] + qty

        # 2. Simulate patient consumption at hospitals and track unmet demand
        step_unserved: dict[str, float] = {}
        for hosp, demand in self.patient_demands.items():
            avail = current_resources.get(hosp, 0.0)
            served = min(avail, demand)
            unserved = demand - served
            current_resources[hosp] = avail - served
            step_unserved[hosp] = unserved

        total_step_unserved = sum(step_unserved.values())
        prev_cumulative = current_resources.get("cumulative_unserved", 0.0)
        current_resources["cumulative_unserved"] = prev_cumulative + total_step_unserved

        # Equity metric: disparity between maximum and minimum hospital unserved demand
        unserved_vals = list(step_unserved.values())
        equity_disparity = (max(unserved_vals) - min(unserved_vals)) if unserved_vals else 0.0
        current_resources["equity_disparity"] = equity_disparity

        # Build next state with updated resource balances
        new_resources = []
        for r_id, val in current_resources.items():
            old_r = state.get_resource(r_id)
            if old_r is not None:
                new_resources.append(
                    Resource(
                        id=r_id,
                        current=round(val, 4),
                        min_value=old_r.min_value,
                        max_value=old_r.max_value,
                        unit=old_r.unit,
                        description=old_r.description,
                    )
                )

        next_state = state.model_copy(
            update={
                "resources": {r.id: r for r in new_resources},
                "step": state.step + 1,
                "timestamp": state.timestamp + 1.0,
            }
        )

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "actions_processed": len(actions),
                "step_unserved": step_unserved,
                "total_step_unserved": total_step_unserved,
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


def create_hospital_world(
    hub_stock: float = 200.0,
    max_corridor_transfer: float = 60.0,
    patient_demands: dict[str, float] | None = None,
    actors: Sequence[object] | None = None,
) -> World:
    """Build a validated hospital supply logistics World."""
    initial_resources = [
        Resource(
            id="hub_stock",
            current=hub_stock,
            min_value=0.0,
            max_value=500.0,
            unit="units",
            description="Central medical supply depot stock",
        ),
        Resource(
            id="hospital_north",
            current=15.0,
            min_value=0.0,
            max_value=150.0,
            unit="units",
            description="North regional hospital medical stock",
        ),
        Resource(
            id="hospital_central",
            current=20.0,
            min_value=0.0,
            max_value=150.0,
            unit="units",
            description="Central regional hospital medical stock",
        ),
        Resource(
            id="hospital_south",
            current=10.0,
            min_value=0.0,
            max_value=150.0,
            unit="units",
            description="South regional hospital medical stock",
        ),
        Resource(
            id="cumulative_unserved",
            current=0.0,
            min_value=0.0,
            max_value=10000.0,
            unit="patients",
            description="Total unserved patient requests",
        ),
        Resource(
            id="equity_disparity",
            current=0.0,
            min_value=0.0,
            max_value=500.0,
            unit="disparity",
            description="Disparity between highest and lowest regional unserved demand",
        ),
    ]

    state = WorldState(resources=initial_resources)
    dynamics = HospitalLogisticsDynamics(patient_demands=patient_demands)
    constraints = [
        ActionTransferAvailabilityConstraint(),
        CorridorCapacityConstraint(max_transfer_per_corridor=max_corridor_transfer),
    ]

    from typing import cast

    from ewm_engine.actors.base import Actor

    typed_actors = cast(list[Actor], list(actors or []))

    return World(
        state=state,
        dynamics=dynamics,
        constraints=constraints,
        actors=typed_actors,
    )
