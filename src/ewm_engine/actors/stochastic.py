"""Stochastic actors selecting actions via probabilistic policies."""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.actors.base import ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId


class StochasticActor:
    """An actor choosing probabilistically between pre-configured action templates."""

    def __init__(
        self,
        actor_id: ActorId,
        action_candidates: Sequence[Action],
        probabilities: Sequence[float] | None = None,
    ) -> None:
        self._actor_id = actor_id
        self.action_candidates = list(action_candidates)
        if probabilities is not None:
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]
        else:
            n = len(action_candidates)
            self.probabilities = [1.0 / n] * n if n > 0 else []

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        if not self.action_candidates:
            return []

        choice_idx = int(context.rng.choice(len(self.action_candidates), p=self.probabilities))
        selected_template = self.action_candidates[choice_idx]

        instantiated = selected_template.model_copy(
            update={
                "id": f"{selected_template.id}_{context.step}",
                "timestamp": context.timestamp,
                "actor_id": self.actor_id,
            }
        )
        return [instantiated]

    def clone(self) -> StochasticActor:
        """Create an independent copy of this stochastic actor."""
        return StochasticActor(
            actor_id=self.actor_id,
            action_candidates=list(self.action_candidates),
            probabilities=list(self.probabilities),
        )
