"""Unit tests for MLCommons Croissant 1.0 JSON-LD dataset manifest generator (T08)."""

from __future__ import annotations

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.reporting.croissant import generate_croissant_manifest
from ewm_engine.simulation.scenario import Scenario


def _create_croissant_test_world() -> World:
    entities = [Entity(id="hub_0", type="hub"), Entity(id="hub_1", type="hub")]
    relationships = [Relationship(source="hub_0", target="hub_1", type="link")]
    resources = [
        Resource(id="res_0", entity_id="hub_0", current=100.0, min_value=0.0, max_value=500.0),
        Resource(id="res_1", entity_id="hub_1", current=50.0, min_value=0.0, max_value=500.0),
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = DeterministicTransferDynamics()
    constraints = ConstraintRegistry([ResourceCapacityConstraint(resource_id="res_1")])
    return World(state=state, dynamics=dynamics, constraints=constraints)


def test_croissant_manifest_structure_and_conformance() -> None:
    """Verify generated manifest strictly conforms to Croissant 1.0 specification."""
    world = _create_croissant_test_world()
    scenario = Scenario(scenario_id="scen_croissant", horizon=3, samples=2, seed=42)
    sim_result = world.simulate(scenario=scenario)

    manifest = generate_croissant_manifest(sim_result, dataset_name="test_sim_dataset")

    assert manifest["@type"] == "Dataset"
    assert manifest["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert manifest["name"] == "test_sim_dataset"
    assert "croissant/" in manifest["@context"]["cr"]
    assert len(manifest["recordSet"]) > 0

    trajectories_record = manifest["recordSet"][0]
    assert trajectories_record["name"] == "trajectories"
    field_names = [f["name"] for f in trajectories_record["field"]]
    assert "sample_id" in field_names
    assert "step" in field_names
    assert "state_hash" in field_names
    assert "timestamp" in field_names

    # Check to_croissant method on SimulationResult
    helper_manifest = sim_result.to_croissant(dataset_name="helper_sim_dataset")
    assert helper_manifest["name"] == "helper_sim_dataset"
    assert helper_manifest["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert helper_manifest["extra"]["fingerprint"] == sim_result.fingerprint
