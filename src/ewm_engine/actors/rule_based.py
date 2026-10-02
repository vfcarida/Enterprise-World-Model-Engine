"""Rule-based and heuristic actor implementations."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.actors.base import ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId, ResourceId


class ThresholdReplenishmentActor:
    """A policy actor triggering inventory transfer when stock breaches a lower threshold."""

    def __init__(
        self,
        actor_id: ActorId,
        source_resource: ResourceId,
        target_resource: ResourceId,
        reorder_point: float,
        order_quantity: float,
        action_type: str = "transfer_resource",
    ) -> None:
        self._actor_id = actor_id
        self.source_resource = source_resource
        self.target_resource = target_resource
        self.reorder_point = reorder_point
        self.order_quantity = order_quantity
        self.action_type = action_type

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        target_res = state.resources.get(self.target_resource)
        if target_res is None:
            return []
        if target_res.current <= self.reorder_point:
            action = Action(
                id=f"replenish_{self.target_resource}_{context.step}",
                actor_id=self.actor_id,
                type=self.action_type,
                parameters={
                    "source_resource": self.source_resource,
                    "target_resource": self.target_resource,
                    "quantity": self.order_quantity,
                },
                timestamp=context.timestamp,
            )
            return [action]
        return []
