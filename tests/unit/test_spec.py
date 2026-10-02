"""Unit tests for safe declarative WorldSpecification parser."""

from __future__ import annotations

import pytest

from ewm_engine.core.spec import WorldSpecification
from ewm_engine.exceptions import SimulationConfigurationError

VALID_YAML_SPEC = """
world:
  name: "WarehouseNetwork"
  step: 0
  timestamp: 0.0
  memory:
    total_inbound: 0

entities:
  - id: "hub_alpha"
    type: "distribution_center"
    attributes:
      region: "Midwest"
  - id: "depot_beta"
    type: "regional_depot"
    attributes:
      region: "East"

relationships:
  - source: "hub_alpha"
    target: "depot_beta"
    type: "corridor"
    attributes:
      distance_km: 120.0

resources:
  - id: "pallets_alpha"
    entity_id: "hub_alpha"
    current: 500.0
    min_value: 0.0
    max_value: 1000.0
    unit: "pallets"
  - id: "pallets_beta"
    entity_id: "depot_beta"
    current: 100.0
    min_value: 0.0
    max_value: 300.0
    unit: "pallets"

constraints:
  - type: "capacity"
    parameters:
      resource_id: "pallets_alpha"
      severity: "hard"
  - type: "non_negative"
    parameters:
      resource_id: "pallets_beta"
      severity: "hard"
  - type: "transfer_availability"
    parameters:
      severity: "hard"
"""


def test_world_specification_from_yaml_and_build() -> None:
    """Verify parsing valid YAML into a WorldSpecification and compiling to World."""
    spec = WorldSpecification.from_yaml(VALID_YAML_SPEC)

    assert spec.world.name == "WarehouseNetwork"
    assert len(spec.entities) == 2
    assert len(spec.relationships) == 1
    assert len(spec.resources) == 2
    assert len(spec.constraints) == 3

    # Compile to World
    world = spec.build_world()
    assert world.initial_state.get_entity("hub_alpha").attributes["region"] == "Midwest"
    assert world.initial_state.get_resource("pallets_alpha").current == 500.0
    assert len(world.constraints) == 3


def test_world_specification_from_json() -> None:
    """Verify parsing from JSON string."""
    json_str = """
    {
        "world": {"name": "JsonWorld"},
        "entities": [{"id": "e1", "type": "node"}],
        "resources": [{"id": "r1", "current": 42.0}]
    }
    """
    spec = WorldSpecification.from_json(json_str)
    assert spec.world.name == "JsonWorld"
    assert len(spec.entities) == 1
    assert spec.resources[0].current == 42.0

    state = spec.build_state()
    assert state.get_resource("r1").current == 42.0


def test_world_specification_errors() -> None:
    """Verify descriptive error handling for malformed input."""
    with pytest.raises(SimulationConfigurationError, match="Invalid JSON"):
        WorldSpecification.from_json("invalid-json-content{")

    with pytest.raises(SimulationConfigurationError, match="Invalid YAML"):
        WorldSpecification.from_yaml(":\n  invalid: yaml: [")

    # Non-dictionary root
    with pytest.raises(SimulationConfigurationError, match="top-level dictionary"):
        WorldSpecification.from_yaml("- item1\n- item2")

    # Unknown constraint type
    bad_constraint_yaml = """
    constraints:
      - type: "quantum_entanglement"
        parameters:
          resource_id: "r1"
    """
    bad_spec = WorldSpecification.from_yaml(bad_constraint_yaml)
    with pytest.raises(SimulationConfigurationError, match="Unknown constraint type"):
        bad_spec.build_world()


def test_world_specification_security_safe_load() -> None:
    """Verify that YAML parser strictly refuses unsafe Python execution."""
    unsafe_yaml = """
    world:
      name: !!python/object/apply:os.system ["echo hacked"]
    """
    with pytest.raises(SimulationConfigurationError):
        WorldSpecification.from_yaml(unsafe_yaml)
