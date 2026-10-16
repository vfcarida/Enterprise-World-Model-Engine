"""Reference World 3: Constraint Stress and Boundary Failure (FMI Reference-FMUs model).

Deliberate edge cases: capacity saturation, boundary threshold violations,
and fail-closed trajectory invalidation.
"""

from __future__ import annotations

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.scenario import Scenario


def create_constraint_stress_world() -> tuple[World, Scenario]:
    """Create constraint stress reference world that triggers a hard capacity violation."""
    entities = [
        Entity(id="tank_source", type="tank"),
        Entity(id="tank_target", type="tank"),
    ]
    relationships = [
        Relationship(source="tank_source", target="tank_target", type="pipe"),
    ]
    # tank_target starts at 95.0 with max capacity 100.0. Transfer of 10.0 pushes it to 105.0!
    resources = [
        Resource(
            id="liquid_src", entity_id="tank_source", current=100.0, min_value=0.0, max_value=200.0
        ),
        Resource(
            id="liquid_tgt", entity_id="tank_target", current=95.0, min_value=0.0, max_value=100.0
        ),
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    from ewm_engine.dynamics.deterministic import DeterministicDriftDynamics

    dynamics = DeterministicDriftDynamics(
        resource_id="liquid_tgt",
        drift=10.0,
        name="CapacityOverfill",
    )
    constraints = ConstraintRegistry([ResourceCapacityConstraint(resource_id="liquid_tgt")])
    world = World(state=state, dynamics=dynamics, constraints=constraints)

    scenario = Scenario(
        scenario_id="ref_constraint_stress_scen",
        name="Constraint Stress Scenario",
        horizon=3,
        samples=1,
        seed=999,
    )
    return world, scenario
