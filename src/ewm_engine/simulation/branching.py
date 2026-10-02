"""Scenario and world branching utilities for counterfactual exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.simulation.scenario import Scenario

if TYPE_CHECKING:
    from ewm_engine.core.world import World


def branch_world(world: World, state: WorldState | None = None) -> World:
    """Create an independent counterfactual branch from a world or specific state snapshot."""
    return world.branch(state=state)


def branch_scenario(
    base_scenario: Scenario,
    intervention: Intervention,
    name: str | None = None,
) -> Scenario:
    """Create a new scenario branching from a baseline scenario with an added intervention."""
    scenario_name = name or f"{base_scenario.name}_{intervention.id}"
    return base_scenario.model_copy(
        update={
            "name": scenario_name,
            "scenario_id": f"{base_scenario.scenario_id}_{intervention.id}",
            "intervention": intervention,
        }
    )
