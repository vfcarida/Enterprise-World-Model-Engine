"""Scenario and world branching utilities for counterfactual exploration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.simulation.scenario import Scenario, ScheduledAction

if TYPE_CHECKING:
    from ewm_engine.core.world import World


def branch_world(world: World, state: WorldState | None = None) -> World:
    """Create an independent counterfactual branch from a world or specific state snapshot.

    The new branch receives an immutable state snapshot and independent container components.
    Mutations on the branch do not affect the origin world.
    """
    return world.branch(state=state)


def branch_scenario(
    base_scenario: Scenario,
    intervention: Intervention | None = None,
    name: str | None = None,
    scenario_id: str | None = None,
    scheduled_actions: Sequence[ScheduledAction] | None = None,
    seed: int | None = None,
) -> Scenario:
    """Create a new scenario branching from a baseline scenario.

    Preserves the initial_state (by fingerprint identity) while creating a new scenario_id
    and applying optional interventions or scheduled actions without mutating the parent scenario.
    """
    new_id = scenario_id or (
        f"{base_scenario.scenario_id}_{intervention.id}"
        if intervention is not None
        else f"{base_scenario.scenario_id}_branch"
    )
    return base_scenario.branch(
        scenario_id=new_id,
        name=name,
        intervention=intervention,
        scheduled_actions=scheduled_actions,
        seed=seed,
    )


__all__ = ["branch_scenario", "branch_world"]
