"""Reference World 2: Closed-Form Stochastic Process (FMI Reference-FMUs model).

Stochastic Gaussian demand process with known theoretical moments.
Verifies SeedSequence.spawn determinism and cross-backend bit-identical branch fingerprints.
"""

from __future__ import annotations

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.scenario import Scenario


def create_closed_form_stochastic_world() -> tuple[World, Scenario]:
    """Create closed-form stochastic reference world and its standard scenario."""
    entities = [
        Entity(id="hub_0", type="warehouse"),
        Entity(id="hub_1", type="retail"),
    ]
    relationships = [
        Relationship(source="hub_0", target="hub_1", type="supplies"),
    ]
    resources = [
        Resource(id="stock_0", entity_id="hub_0", current=200.0, min_value=0.0, max_value=1000.0),
        Resource(id="stock_1", entity_id="hub_1", current=100.0, min_value=0.0, max_value=1000.0),
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = StochasticDemandDynamics(resource_id="stock_0", mean_demand=5.0, std_demand=1.0)
    constraints = ConstraintRegistry([ResourceCapacityConstraint(resource_id="stock_0")])
    world = World(state=state, dynamics=dynamics, constraints=constraints)

    scenario = Scenario(
        scenario_id="ref_stochastic_scen",
        name="Stochastic Demand Scenario",
        horizon=5,
        samples=3,
        seed=1337,
    )
    return world, scenario
