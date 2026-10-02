"""CivicFlow: Operational and physical constraints for disaster response logistics."""

from __future__ import annotations

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
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

    def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult:
        if action is None or action.type != "dispatch_relief":
            return ConstraintResult(
                satisfied=True, constraint_id=self.constraint_id, severity=self.severity
            )

        target_shelter = str(action.get("target_shelter"))
        road_c1_status = state.memory.get("road_C1_status", "open")

        # Shelter S2 access relies strictly on Causeway C1
        if target_shelter == "shelter_s2" and road_c1_status == "closed":
            return ConstraintResult(
                satisfied=False,
                constraint_id=self.constraint_id,
                severity=self.severity,
                message="Cannot dispatch to Shelter S2: Coastal Causeway C1 is inundated and closed.",
                violating_entities=("shelter_s2",),
                violating_values={"road_C1_status": "closed"},
            )

        return ConstraintResult(
            satisfied=True, constraint_id=self.constraint_id, severity=self.severity
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

    def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult:
        occupancy_res = state.get_resource(f"occupancy_{self.shelter_id}")
        satisfied = occupancy_res.current <= occupancy_res.max_value

        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            severity=self.severity,
            message=f"Occupancy {occupancy_res.current} within bed limit {occupancy_res.max_value}"
            if satisfied
            else f"Shelter {self.shelter_id} overcrowding: {occupancy_res.current} evacuees exceed {occupancy_res.max_value} beds.",
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
