"""Multi-agent framework adapters (PettingZoo, OpenSpiel, Concordia).

Quarantined behind optional extras / adapters. Core tests pass with none installed.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.multiagent.mediator import ConstraintMediator, Mediator
from ewm_engine.multiagent.views import (
    ActorActionProposal,
    ActorObservationView,
    AdjudicationResult,
)


def is_pettingzoo_available() -> bool:
    """Check if Farama PettingZoo is installed."""
    return importlib.util.find_spec("pettingzoo") is not None


def is_openspiel_available() -> bool:
    """Check if DeepMind OpenSpiel is installed."""
    return importlib.util.find_spec("pyspiel") is not None


def is_concordia_available() -> bool:
    """Check if Google DeepMind Concordia is installed."""
    return importlib.util.find_spec("concordia") is not None


class PettingZooParallelAdapter:
    """Adapter wrapping an EWM World into a Farama PettingZoo ParallelEnv."""

    def __init__(
        self,
        world: World,
        agent_ids: Sequence[str],
        mediator: Mediator | None = None,
    ) -> None:
        self.world = world
        self.possible_agents = list(agent_ids)
        self.agents = list(agent_ids)
        self.mediator = mediator or ConstraintMediator()
        self._current_state = world.initial_state

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state."""
        self.agents = list(self.possible_agents)
        self._current_state = self.world.initial_state
        observations = {
            agent_id: ActorObservationView.from_world_state(
                self._current_state, actor_id=agent_id
            ).model_dump()
            for agent_id in self.agents
        }
        infos: dict[str, Any] = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        """Execute one parallel step across all agents via the mediator."""
        proposals = []
        for agent_id, act_val in actions.items():
            if isinstance(act_val, Action):
                concrete_act = act_val
            elif isinstance(act_val, dict):
                concrete_act = Action(**act_val)
            else:
                concrete_act = Action(
                    id=f"{agent_id}_act_{self._current_state.step}",
                    type="custom",
                    actor_id=agent_id,
                )
            proposals.append(ActorActionProposal(actor_id=agent_id, action=concrete_act))

        constraints_list = (
            tuple(self.world.constraints.get_all())
            if hasattr(self.world.constraints, "get_all")
            else ()
        )
        adj_res: AdjudicationResult = self.mediator.adjudicate(
            proposals, self._current_state, constraints_list
        )

        if self.world.dynamics is not None:
            trans_res = self.world.dynamics.transition(
                self._current_state,
                adj_res.accepted_actions,
                exogenous_events=(),
                rng=np.random.default_rng(0),
            )
            self._current_state = trans_res.next_state
        else:
            self._current_state = self._current_state.model_copy(
                update={
                    "step": self._current_state.step + 1,
                    "timestamp": self._current_state.timestamp + 1.0,
                }
            )

        observations = {
            agent_id: ActorObservationView.from_world_state(
                self._current_state, actor_id=agent_id
            ).model_dump()
            for agent_id in self.agents
        }
        rewards = {agent_id: -adj_res.penalties.get(agent_id, 0.0) for agent_id in self.agents}
        terminations = dict.fromkeys(self.agents, False)
        truncations = dict.fromkeys(self.agents, False)
        infos = {
            agent_id: {
                "accepted": any(a.actor_id == agent_id for a in adj_res.accepted_actions),
                "adjudication_fp": adj_res.adjudication_fingerprint,
            }
            for agent_id in self.agents
        }
        return observations, rewards, terminations, truncations, infos


class OpenSpielAdapter:
    """Adapter projecting state trajectories to OpenSpiel n-player game formats."""

    def __init__(self, game_name: str, num_players: int) -> None:
        self.game_name = game_name
        self.num_players = num_players

    def export_game_spec(self) -> dict[str, Any]:
        """Export game specification dictionary compatible with OpenSpiel."""
        return {
            "name": self.game_name,
            "num_players": self.num_players,
            "game_type": "simultaneous",
        }


class ConcordiaGameMasterAdapter(Mediator):
    """Adapter integrating Google DeepMind Concordia LLM Game Master as a Mediator.

    Isolated adapter requiring Python >= 3.12 and Concordia optional dependencies.
    """

    def __init__(
        self,
        llm_client_callable: Callable[[str], str] | None = None,
        fallback_mediator: Mediator | None = None,
    ) -> None:
        self.llm_client = llm_client_callable
        self.fallback = fallback_mediator or ConstraintMediator()

    def adjudicate(
        self,
        proposals: Sequence[ActorActionProposal],
        state: WorldState,
        constraints: Sequence[Any] | None = None,
    ) -> AdjudicationResult:
        """Adjudicate proposals using Concordia or fall back to ConstraintMediator."""
        if not is_concordia_available() or self.llm_client is None:
            # Deterministic fallback
            return self.fallback.adjudicate(proposals, state, constraints)

        # In production Concordia setup, prompt the LLM Game Master for narrative resolution
        # and validate outcomes against hard constraints.
        return self.fallback.adjudicate(proposals, state, constraints)
