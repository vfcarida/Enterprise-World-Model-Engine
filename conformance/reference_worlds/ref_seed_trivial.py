"""Reference World 1: Seed-Only Trivial World (FMI Reference-FMUs model).

Minimal deterministic 2-node enterprise system with constant resource transfer.
Verifies baseline clock stepping, resource bounds, and canonical state hashing.
"""

from __future__ import annotations

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.scenario import Scenario


def create_seed_trivial_world() -> tuple[World, Scenario]:
    """Create minimal deterministic reference world and its standard scenario."""
    entities = [
        Entity(id="node_a", type="depot"),
        Entity(id="node_b", type="depot"),
    ]
    relationships = [
        Relationship(source="node_a", target="node_b", type="connected"),
    ]
    resources = [
        Resource(id="stock_a", entity_id="node_a", current=100.0, min_value=0.0, max_value=500.0),
        Resource(id="stock_b", entity_id="node_b", current=50.0, min_value=0.0, max_value=500.0),
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    from ewm_engine.dynamics.deterministic import DeterministicFixedTransferDynamics

    dynamics = DeterministicFixedTransferDynamics(
        source_resource="stock_a",
        target_resource="stock_b",
        quantity=10.0,
        name="TrivialTransfer",
    )
    constraints = ConstraintRegistry([ResourceCapacityConstraint(resource_id="stock_b")])
    world = World(state=state, dynamics=dynamics, constraints=constraints)

    scenario = Scenario(
        scenario_id="ref_seed_trivial_scen",
        name="Seed Trivial Scenario",
        horizon=5,
        samples=1,
        seed=42,
    )
    return world, scenario
