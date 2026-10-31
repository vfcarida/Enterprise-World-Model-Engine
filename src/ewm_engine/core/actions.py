"""Action and Intervention domain primitives for enterprise decision making."""

from __future__ import annotations

import copy
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema

from ewm_engine.core.types import ActionId, ActorId

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState

_ACTIVE_INTERVENTIONS: ContextVar[tuple[Intervention, ...]] = ContextVar(
    "_ACTIVE_INTERVENTIONS", default=()
)
_INTERVENTION_TOKENS: dict[int, list[Token[tuple[Intervention, ...]]]] = {}


def get_active_interventions() -> tuple[Intervention, ...]:
    """Return the tuple of currently active context-managed Interventions."""
    return _ACTIVE_INTERVENTIONS.get()


class _StateMutatorAnnotation:
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.nullable_schema(
            core_schema.any_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda v: None),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        return {"type": "null", "description": "Runtime callable mutator (not serialized)"}


StateMutator = Annotated[Callable[[Any], Any] | None, _StateMutatorAnnotation]


class Action(BaseModel):
    """An operation proposed or executed by an actor within the world.

    Actions represent operational steps: transferring inventory, dispatching a vehicle,
    changing a price, reallocating staff, or issuing a notification.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Semantic schema version.")
    id: ActionId = Field(
        default_factory=lambda: f"act_{uuid.uuid4().hex[:8]}",
        description="Unique action instance identifier.",
    )
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

    def __init__(self, **kwargs: Any) -> None:
        if "parameters" in kwargs and kwargs["parameters"] is not None:
            kwargs["parameters"] = copy.deepcopy(kwargs["parameters"])
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "parameters" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

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

    def __init__(self, **kwargs: Any) -> None:
        if "parameters" in kwargs and kwargs["parameters"] is not None:
            kwargs["parameters"] = copy.deepcopy(kwargs["parameters"])
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "parameters" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

    state_mutator: StateMutator = Field(
        default=None,
        description="Optional callable modifying world state when intervention is applied.",
    )

    def forced_nodes(self) -> dict[str, Any]:
        """Return mapping of state node identifiers to forced values."""
        raw = super().__getattribute__("parameters")
        if not isinstance(raw, dict):
            return {}
        return {str(k): copy.deepcopy(v) for k, v in raw.items() if not str(k).startswith("_")}

    @property
    def is_active(self) -> bool:
        """Return True if this Intervention is currently active in a context manager block."""
        return any(iv is self or iv.id == self.id for iv in _ACTIVE_INTERVENTIONS.get())

    def __enter__(self) -> Self:
        """Enter an active Intervention context forcing specified state nodes and overriding dynamics."""
        current = _ACTIVE_INTERVENTIONS.get()
        token = _ACTIVE_INTERVENTIONS.set((*current, self))
        _INTERVENTION_TOKENS.setdefault(id(self), []).append(token)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit the active Intervention context and restore standard dynamics."""
        tokens = _INTERVENTION_TOKENS.get(id(self))
        if tokens:
            token = tokens.pop()
            if not tokens:
                _INTERVENTION_TOKENS.pop(id(self), None)
            _ACTIVE_INTERVENTIONS.reset(token)

    def apply_forced_nodes(self, state: WorldState) -> WorldState:
        """Force specific state nodes (resources, entity attributes, memory) on a WorldState."""
        new_state = state
        for key, val in self.forced_nodes().items():
            clean_key = key.split(":", 1)[1] if key.startswith("resource:") else key
            if clean_key in new_state.resources and isinstance(val, (int, float)):
                new_state = new_state.update_resource(
                    clean_key,
                    new_value=float(val),
                    clamp=False,
                    enforce_bounds=True,
                )
            elif key.startswith("entity:"):
                parts = key.split(":", 2)
                if len(parts) == 3 and parts[1] in new_state.entities:
                    ent = new_state.get_entity(parts[1])
                    updated_attrs = dict(ent.attributes)
                    updated_attrs[parts[2]] = val
                    new_state = new_state.with_entity(
                        ent.model_copy(update={"attributes": updated_attrs})
                    )
            else:
                mem_key = key.split(":", 1)[1] if key.startswith("memory:") else key
                new_state = new_state.with_memory(mem_key, val)
        return new_state

    def apply(self, state: WorldState) -> WorldState:
        """Apply this intervention to a WorldState, yielding a counterfactual initial state."""
        # 1. Update active rules with intervention parameters if applicable
        updated_rules = dict(state.active_rules)
        updated_rules[self.id] = self.parameters

        new_state = state.model_copy(update={"active_rules": updated_rules})

        # 2. If target_type indicates state/node override, force state nodes directly
        if self.target_type in ("state_override", "node_override", "do"):
            new_state = self.apply_forced_nodes(new_state)

        # 3. Execute custom state mutator if defined
        if self.state_mutator is not None:
            new_state = self.state_mutator(new_state)

        return new_state
