"""External agent framework adapters for EWM Engine.

Enables seamless integration with LangGraph, AutoGen, CrewAI, RLlib,
or custom functional policy agents.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId


class CallableActorAdapter(Actor):
    """Adapter wrapping an arbitrary callable into the EWM Engine Actor protocol.

    Allows any agent function or external LLM/RL orchestrator
    `(state: WorldState, context: ActorContext) -> Sequence[Action]`
    to participate in world model simulations.
    """

    def __init__(
        self,
        actor_id: ActorId,
        act_fn: Callable[[WorldState, ActorContext], Sequence[Action]],
    ) -> None:
        self._actor_id = actor_id
        self._act_fn = act_fn

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        """Execute the wrapped agent callable against the current world state."""
        return self._act_fn(state, context)

    def clone(self) -> CallableActorAdapter:
        """Create an independent copy of this actor adapter."""
        return CallableActorAdapter(
            actor_id=self.actor_id,
            act_fn=self._act_fn,
        )
