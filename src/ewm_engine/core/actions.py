"""Action and Intervention domain primitives for enterprise decision making."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.types import ActionId, ActorId

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState


class Action(BaseModel):
    """An operation proposed or executed by an actor within the world.

    Actions represent operational steps: transferring inventory, dispatching a vehicle,
    changing a price, reallocating staff, or issuing a notification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ActionId = Field(description="Unique action instance identifier.")
    type: str = Field(description="Action category or schema type identifier.")
    actor_id: ActorId | None = Field(
        default=None,
        description="Optional identifier of the actor initiating this action.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Action payload containing target entities, quantities, and parameters.",
    )
    timestamp: float = Field(default=0.0, description="Simulation timestamp at action proposal.")
    priority: int = Field(default=0, description="Execution priority for tie-breaking.")

    def get(self, key: str, default: Any = None) -> Any:
        """Fetch a parameter value safely."""
        return self.parameters.get(key, default)


class Intervention(BaseModel):
    """A deliberate modification to the enterprise system, its policies, or initial state.

    Distinguishes operational micro-actions from strategic macro-interventions
    (e.g., implementing a new replenishment policy, increasing warehouse capacity,
    or altering regulatory thresholds).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(description="Unique intervention identifier.")
    description: str = Field(description="Human-readable rationale for the intervention.")
    target_type: str = Field(
        default="policy",
        description="Type of intervention: 'policy', 'state_override', 'rule_change', etc.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Intervention parameters or policy configurations.",
    )
    state_mutator: Callable[[Any], Any] | None = Field(
        default=None,
        description="Optional callable modifying world state when intervention is applied.",
    )

    def apply(self, state: WorldState) -> WorldState:
        """Apply this intervention to a WorldState, yielding a counterfactual initial state."""
        # 1. Update active rules with intervention parameters if applicable
        updated_rules = dict(state.active_rules)
        updated_rules[self.id] = self.parameters

        new_state = state.model_copy(update={"active_rules": updated_rules})

        # 2. Execute custom state mutator if defined
        if self.state_mutator is not None:
            new_state = self.state_mutator(new_state)

        return new_state
