"""Canonical Action and State-Transition contract (`Action + State(t) -> State(t+1)`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ewm_engine.core.actions import (
    Action,
    Intervention,
    StateMutator,
    get_active_interventions,
)

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState

StateTransitionFn = Callable[["WorldState", Action], "WorldState"]

__all__ = [
    "Action",
    "Intervention",
    "StateMutator",
    "StateTransitionFn",
    "apply_action",
    "get_active_interventions",
]


def apply_action(
    state: WorldState,
    action: Action,
    transition_fn: StateTransitionFn | None = None,
    *,
    delta_t: float = 1.0,
) -> WorldState:
    """Execute the pure functional transition `Action + State(t) -> State(t+1)`.

    Guarantees:
    1. Input `state` ($S_t$) is never mutated in place.
    2. Output state ($S_{t+1}$) has strictly monotonically advanced discrete step ($t_{k+1} > t_k$).
    """
    if delta_t < 0.0:
        raise ValueError(f"Transition delta_t must be non-negative (got {delta_t}).")

    if transition_fn is not None:
        candidate = transition_fn(state, action)
        if candidate.step < state.step:
            raise ValueError(
                f"Retroactive transition forbidden: transition_fn returned step={candidate.step} < current step={state.step}."
            )
        if candidate.step == state.step:
            return candidate.advance_time(delta_t=delta_t, steps=1)
        return candidate

    return state.with_memory("_last_action_id", action.id).advance_time(delta_t=delta_t, steps=1)
