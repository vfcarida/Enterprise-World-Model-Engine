"""Unit and security tests for World Specification Language (WSL) (P10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ewm_engine.core.world import World
from ewm_engine.exceptions import (
    SerializationError,
    SerializationSecurityError,
    SimulationConfigurationError,
)
from ewm_engine.serialization.wsl import (
    WSLDocument,
    compile_wsl,
    dump_wsl_yaml,
    export_wsl,
    parse_wsl_yaml,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

VALID_WSL_YAML = """
schema_version: "wsl/2.0.0"

metadata:
  id: "test_supply_chain"
  name: "Test Supply Chain Network"
  version: "2.0.0"
  description: "Test world specification"
  author: "Test Suite"
  tags: ["test", "supply"]

temporal:
  time_unit: "step"
  step_duration: 1.0
  default_horizon: 10

entities:
  - id: "warehouse_1"
    type: "warehouse"
    attributes:
      capacity_sqft: 10000
    tags: ["central"]
  - id: "store_1"
    type: "retail"
    attributes:
      footfall: 500
    tags: ["east"]

relationships:
  - source: "warehouse_1"
    target: "store_1"
    type: "supplies"
    attributes:
      lead_time: 1
    weight: 1.0

resources:
  - id: "wh_stock"
    current: 1000.0
    min_value: 0.0
    max_value: 5000.0
    unit: "crates"
    entity_id: "warehouse_1"
  - id: "store_stock"
    current: 200.0
    min_value: 0.0
    max_value: 1000.0
    unit: "crates"
    entity_id: "store_1"

dynamics:
  type_id: "deterministic_transfer"
  parameters:
    action_type: "transfer_stock"
  evidence_level: "STRUCTURAL"

constraints:
  - id: "non_negative_store_stock"
    type_id: "resource_non_negative"
    parameters:
      resource_id: "store_stock"
    phase: "POST_DYNAMICS"
    severity: "HARD"

scenarios:
  - id: "scen_baseline"
    name: "Baseline 10 steps"
    seed: 123
    horizon: 5
    interventions: []
"""


def test_parse_valid_wsl_yaml() -> None:
    """Assert valid WSL YAML parses into a typed WSLDocument."""
    doc = parse_wsl_yaml(VALID_WSL_YAML)
    assert isinstance(doc, WSLDocument)
    assert doc.schema_version == "wsl/2.0.0"
    assert doc.metadata.id == "test_supply_chain"
    assert len(doc.entities) == 2
    assert len(doc.relationships) == 1
    assert len(doc.resources) == 2
    assert doc.dynamics is not None
    assert doc.dynamics.type_id == "deterministic_transfer"
    assert len(doc.constraints) == 1


def test_compile_wsl_to_executable_world() -> None:
    """Assert compile_wsl builds an executable World and runs simulation."""
    doc = parse_wsl_yaml(VALID_WSL_YAML)
    world = compile_wsl(doc)

    assert isinstance(world, World)
    assert "warehouse_1" in world.current_state.entities
    assert "store_1" in world.current_state.entities
    assert world.current_state.resources["wh_stock"].current == 1000.0
    assert len(world.constraints) == 1

    # Run simulation on compiled world
    engine = SimulationEngine()
    scenario = Scenario(scenario_id="test_run", seed=42, horizon=3)
    result = engine.run(world, scenario)

    assert len(result.trajectories) == 1
    traj = result.trajectories[0]
    assert len(traj.steps) == 3


def test_wsl_roundtrip_export() -> None:
    """Assert World -> WSLDocument -> YAML -> WSLDocument roundtrip preserves fidelity."""
    doc_initial = parse_wsl_yaml(VALID_WSL_YAML)
    world = compile_wsl(doc_initial)

    doc_exported = export_wsl(world, metadata=doc_initial.metadata)
    assert len(doc_exported.entities) == len(doc_initial.entities)
    assert len(doc_exported.relationships) == len(doc_initial.relationships)
    assert len(doc_exported.resources) == len(doc_initial.resources)

    yaml_str = dump_wsl_yaml(doc_exported)
    doc_reloaded = parse_wsl_yaml(yaml_str)

    assert doc_reloaded.schema_version == doc_exported.schema_version
    assert {e.id for e in doc_reloaded.entities} == {e.id for e in doc_exported.entities}


def test_wsl_security_rejects_custom_python_tags() -> None:
    """Security Invariant: Custom YAML tags (e.g. !python/object) must be rejected immediately."""
    malicious_yaml = """
schema_version: "wsl/2.0.0"
metadata: !python/object:os.system
  id: "exploit"
"""
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        parse_wsl_yaml(malicious_yaml)


def test_wsl_security_rejects_pickle_payload() -> None:
    """Security Invariant: Raw or hex pickle sequences must be rejected."""
    pickle_yaml = "cos\nsystem\n(S'echo pwned'\ntR."
    with pytest.raises(SerializationSecurityError, match="Pickle opcode sequence detected"):
        parse_wsl_yaml(pickle_yaml)


def test_wsl_rejects_scope_creep_unregistered_keys() -> None:
    """Scope creep guard: Extra undeclared fields must be rejected (extra='forbid')."""
    invalid_yaml = """
schema_version: "wsl/2.0.0"
embedded_python_code: "import os; os.system('pwn')"
metadata:
  id: "test"
"""
    with pytest.raises((ValidationError, SerializationError)):
        parse_wsl_yaml(invalid_yaml)


def test_wsl_rejects_unregistered_component_type() -> None:
    """Security Invariant: Components not registered in trusted ComponentRegistry must fail."""
    bad_component_yaml = """
schema_version: "wsl/2.0.0"
dynamics:
  type_id: "malicious_unregistered_dynamics"
  parameters: {}
"""
    doc = parse_wsl_yaml(bad_component_yaml)
    with pytest.raises(SimulationConfigurationError, match="Unknown dynamics component type"):
        compile_wsl(doc)
