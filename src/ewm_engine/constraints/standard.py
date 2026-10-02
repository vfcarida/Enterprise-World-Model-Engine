"""Standard reusable organizational, logistical, and capacity constraints."""

from __future__ import annotations

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId, ResourceId


class ResourceCapacityConstraint:
    """Enforces that a resource level does not exceed its maximum physical capacity."""

    def __init__(
        self,
        resource_id: ResourceId,
        severity: ConstraintSeverity = ConstraintSeverity.HARD,
        constraint_id: ConstraintId | None = None,
        version: str = "1.0.0",
    ) -> None:
        self.resource_id = resource_id
        self._severity = severity
        self._constraint_id = constraint_id or f"capacity_{resource_id}"
        self._version = version

    @property
    def constraint_id(self) -> ConstraintId:
        return self._constraint_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def severity(self) -> ConstraintSeverity:
        return self._severity

    @property
    def description(self) -> str:
        return f"Ensures resource '{self.resource_id}' does not exceed maximum capacity."

    def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult:
        if self.resource_id not in state.resources:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
            )

        res = state.get_resource(self.resource_id)
        satisfied = res.current <= res.max_value

        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            message=f"Resource '{self.resource_id}' level {res.current:.2f} within capacity {res.max_value:.2f}"
            if satisfied
            else f"Resource '{self.resource_id}' level {res.current:.2f} exceeded capacity {res.max_value:.2f}",
            violating_resources=(self.resource_id,) if not satisfied else (),
            violating_entities=(res.entity_id,) if (not satisfied and res.entity_id) else (),
            violating_values={"current": res.current, "max_value": res.max_value}
            if not satisfied
            else {},
            penalty=max(0.0, res.current - res.max_value) if not satisfied else 0.0,
        )


class ResourceNonNegativeConstraint:
    """Enforces that a resource level does not fall below its minimum threshold."""

    def __init__(
        self,
        resource_id: ResourceId,
        severity: ConstraintSeverity = ConstraintSeverity.HARD,
        constraint_id: ConstraintId | None = None,
        version: str = "1.0.0",
    ) -> None:
        self.resource_id = resource_id
        self._severity = severity
        self._constraint_id = constraint_id or f"non_negative_{resource_id}"
        self._version = version

    @property
    def constraint_id(self) -> ConstraintId:
        return self._constraint_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def severity(self) -> ConstraintSeverity:
        return self._severity

    @property
    def description(self) -> str:
        return f"Ensures resource '{self.resource_id}' does not drop below min_value."

    def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult:
        if self.resource_id not in state.resources:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
            )

        res = state.get_resource(self.resource_id)
        satisfied = res.current >= res.min_value

        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            message=f"Resource '{self.resource_id}' level {res.current:.2f} above lower bound {res.min_value:.2f}"
            if satisfied
            else f"Resource '{self.resource_id}' level {res.current:.2f} fell below minimum {res.min_value:.2f}",
            violating_resources=(self.resource_id,) if not satisfied else (),
            violating_entities=(res.entity_id,) if (not satisfied and res.entity_id) else (),
            violating_values={"current": res.current, "min_value": res.min_value}
            if not satisfied
            else {},
            penalty=max(0.0, res.min_value - res.current) if not satisfied else 0.0,
        )


class ActionTransferAvailabilityConstraint:
    """Pre-action validation ensuring source inventory is sufficient before attempting a transfer."""

    def __init__(
        self,
        action_type: str = "transfer_resource",
        constraint_id: ConstraintId = "transfer_stock_available",
        severity: ConstraintSeverity = ConstraintSeverity.HARD,
    ) -> None:
        self.action_type = action_type
        self._constraint_id = constraint_id
        self._severity = severity

    @property
    def constraint_id(self) -> ConstraintId:
        return self._constraint_id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return self._severity

    @property
    def description(self) -> str:
        return "Ensures requested transfer quantity does not exceed available source stock."

    def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult:
        if action is None or action.type != self.action_type:
            return ConstraintResult(
                satisfied=True, constraint_id=self.constraint_id, severity=self.severity
            )

        src_id = str(action.get("source_resource"))
        qty = float(action.get("quantity", 0.0))

        if src_id not in state.resources:
            return ConstraintResult(
                satisfied=False,
                constraint_id=self.constraint_id,
                severity=self.severity,
                message=f"Source resource '{src_id}' not found in state.",
                violating_values={"source_resource": src_id},
            )

        src_res = state.get_resource(src_id)
        available = src_res.current - src_res.min_value
        satisfied = available >= qty

        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            severity=self.severity,
            message=f"Requested {qty} units; available {available} units."
            if satisfied
            else f"Insufficient stock: requested {qty} units but only {available} units available.",
            violating_resources=(src_id,) if not satisfied else (),
            violating_values={"requested": qty, "available": available} if not satisfied else {},
        )
