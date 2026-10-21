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
        seed: int = 42,
    ) -> None:
        self.world = world
        self.possible_agents = list(agent_ids)
        self.agents = list(agent_ids)
        self.mediator = mediator or ConstraintMediator()
        self._current_state = world.initial_state
        self._rng = np.random.default_rng(seed)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment to initial state."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
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
                rng=self._rng,
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

    def compute_empirical_payoff_matrix(
        self,
        outcomes: Sequence[tuple[int, int, float, float]],
        num_row_actions: int,
        num_col_actions: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct empirical normal-form payoff matrices from simulation rollouts.

        Args:
            outcomes: Sequence of (row_action_idx, col_action_idx, row_return, col_return).
            num_row_actions: Number of discrete actions for the row player.
            num_col_actions: Number of discrete actions for the column player.

        Returns:
            Tuple of (row_payoff_matrix, col_payoff_matrix) averaged across rollouts.
        """
        row_sum = np.zeros((num_row_actions, num_col_actions), dtype=np.float64)
        col_sum = np.zeros((num_row_actions, num_col_actions), dtype=np.float64)
        counts = np.zeros((num_row_actions, num_col_actions), dtype=np.float64)

        for r_idx, c_idx, r_pay, c_pay in outcomes:
            if 0 <= r_idx < num_row_actions and 0 <= c_idx < num_col_actions:
                row_sum[r_idx, c_idx] += float(r_pay)
                col_sum[r_idx, c_idx] += float(c_pay)
                counts[r_idx, c_idx] += 1.0

        mask = counts > 0
        row_mat = np.zeros_like(row_sum)
        col_mat = np.zeros_like(col_sum)
        row_mat[mask] = row_sum[mask] / counts[mask]
        col_mat[mask] = col_sum[mask] / counts[mask]
        return row_mat, col_mat


class ConcordiaGameMasterAdapter(Mediator):
    """Adapter integrating Google DeepMind Concordia LLM Game Master as a Mediator.

    Supports any callable `(prompt: str) -> str` for narrative action filtering
    followed by hard-constraint verification via the fallback `ConstraintMediator`.
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
        """Adjudicate proposals using LLM Game Master callable and hard constraint guardrails."""
        if self.llm_client is None:
            return self.fallback.adjudicate(proposals, state, constraints)

        proposal_lines = [
            f"- actor={p.actor_id} action_id={p.action.id} type={p.action.type} priority={p.priority}"
            for p in proposals
        ]
        prompt = (
            f"Concordia Game Master Adjudication at step={state.step}.\n"
            f"Resources: {dict(state.resources)}\n"
            f"Proposals:\n" + "\n".join(proposal_lines) + "\n"
            "Return comma-separated approved action_ids or actor_ids (or ALL):"
        )
        response = self.llm_client(prompt).strip()

        if not response or "ALL" in response.upper():
            return self.fallback.adjudicate(proposals, state, constraints)

        tokens = {tok.strip() for tok in response.replace("\n", ",").split(",") if tok.strip()}
        gm_approved: list[ActorActionProposal] = []
        gm_rejected: list[tuple[Action, str]] = []

        for p in proposals:
            if p.action.id in tokens or p.actor_id in tokens:
                gm_approved.append(p)
            else:
                gm_rejected.append(
                    (p.action, "Rejected by Concordia Game Master narrative adjudication")
                )

        if not gm_approved and not gm_rejected:
            return self.fallback.adjudicate(proposals, state, constraints)

        base_res = self.fallback.adjudicate(gm_approved, state, constraints)
        combined_rejected = tuple(list(base_res.rejected_actions) + gm_rejected)
        combined_penalties = dict(base_res.penalties)
        for act, _ in gm_rejected:
            if act.actor_id:
                combined_penalties[act.actor_id] = combined_penalties.get(act.actor_id, 0.0) + 1.0

        return AdjudicationResult(
            accepted_actions=base_res.accepted_actions,
            rejected_actions=combined_rejected,
            penalties=combined_penalties,
        )
