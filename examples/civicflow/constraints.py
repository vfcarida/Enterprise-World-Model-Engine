"""CivicFlow: Operational and physical constraints for disaster response logistics."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId


class RoadPassabilityConstraint:
    """Enforces that relief convoys cannot traverse flooded/closed roadways (Hard Constraint)."""

    def __init__(self, constraint_id: ConstraintId = "road_passability_check") -> None:
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
        return "Prevents relief dispatches across flooded roads (e.g. Causeway C1)."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.PRE_ACTION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        target_actions = [
            a
            for a in actions
            if a.type
            in ("dispatch_relief", "dispatch_vehicle", "reroute_delivery", "transfer_supply")
        ]
        if not target_actions:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        for action in target_actions:
            target_shelter = str(action.get("target_shelter"))
            road_c1_status = state.memory.get("road_C1_status", "open")

            # Shelter S2 access relies strictly on Causeway C1
            if target_shelter == "shelter_s2" and road_c1_status == "closed":
                return ConstraintResult(
                    satisfied=False,
                    constraint_id=self.constraint_id,
                    severity=self.severity,
                    phase=phase,
                    message="Cannot dispatch to Shelter S2: Coastal Causeway C1 is inundated and closed.",
                    entity_ids=("shelter_s2",),
                    violating_entities=("shelter_s2",),
                    violating_values={"road_C1_status": "closed"},
                    preceding_action_id=action.id,
                )

        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


class ShelterBedCapacityConstraint:
    """Enforces that evacuee occupancy does not exceed maximum shelter bed capacity."""

    def __init__(self, shelter_id: str) -> None:
        self.shelter_id = shelter_id
        self._constraint_id = f"capacity_beds_{shelter_id}"

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
        return f"Ensures occupancy in shelter '{self.shelter_id}' does not exceed bed capacity."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.POST_TRANSITION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        occupancy_res = state.get_resource(f"occupancy_{self.shelter_id}")
        satisfied = occupancy_res.current <= occupancy_res.max_value

        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
            message=(
                f"Occupancy {occupancy_res.current} within bed limit {occupancy_res.max_value}"
                if satisfied
                else f"Shelter {self.shelter_id} overcrowding: {occupancy_res.current} evacuees exceed {occupancy_res.max_value} beds."
            ),
            entity_ids=(self.shelter_id,) if not satisfied else (),
            violating_entities=(self.shelter_id,) if not satisfied else (),
            violating_values={
                "occupancy": occupancy_res.current,
                "max_beds": occupancy_res.max_value,
            }
            if not satisfied
            else {},
            penalty=max(0.0, occupancy_res.current - occupancy_res.max_value)
            if not satisfied
            else 0.0,
        )


class VehicleLoadCapacityConstraint:
    """Enforces that convoy payload does not exceed transport vehicle load capacity."""

    def __init__(
        self,
        max_load: float = 300.0,
        constraint_id: ConstraintId = "vehicle_load_capacity",
    ) -> None:
        self.max_load = max_load
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
        return f"Ensures dispatch load does not exceed vehicle limit {self.max_load} units."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.PRE_ACTION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        dispatch_actions = [
            a
            for a in actions
            if a.type
            in ("dispatch_relief", "dispatch_vehicle", "reroute_delivery", "transfer_supply")
        ]
        for a in dispatch_actions:
            load = float(a.get("rations", 0.0)) + float(a.get("water", 0.0))
            if load > self.max_load:
                return ConstraintResult(
                    satisfied=False,
                    constraint_id=self.constraint_id,
                    severity=self.severity,
                    phase=phase,
                    message=f"Dispatch load {load} exceeds vehicle capacity {self.max_load}.",
                    preceding_action_id=a.id,
                )

        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


class AllocatedSupplyAvailableConstraint:
    """Enforces that allocated supply does not exceed available depot inventory."""

    def __init__(
        self,
        constraint_id: ConstraintId = "allocated_supply_available",
    ) -> None:
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
        return "Ensures dispatched supply does not exceed available depot inventory."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.PRE_ACTION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )

        dispatch_actions = [
            a
            for a in actions
            if a.type
            in ("dispatch_relief", "dispatch_vehicle", "reroute_delivery", "transfer_supply")
        ]
        for a in dispatch_actions:
            depot = str(a.get("source_depot", "depot_valley"))
            rations = float(a.get("rations", 0.0))
            water = float(a.get("water", 0.0))

            depot_rat_id = f"rations_{depot}"
            depot_wat_id = f"water_{depot}"

            if depot_rat_id in state.resources:
                avail_rat = state.get_resource(depot_rat_id).current
                if rations > avail_rat:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        severity=self.severity,
                        phase=phase,
                        message=f"Requested rations ({rations}) exceed depot stock ({avail_rat}).",
                        preceding_action_id=a.id,
                    )

            if depot_wat_id in state.resources:
                avail_wat = state.get_resource(depot_wat_id).current
                if water > avail_wat:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        severity=self.severity,
                        phase=phase,
                        message=f"Requested water ({water}) exceeds depot stock ({avail_wat}).",
                        preceding_action_id=a.id,
                    )

        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )
