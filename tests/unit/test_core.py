"""Unit tests for core domain primitives in EWM Engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import InvalidWorldStateError, ResourceBoundsError


def test_entity_creation_and_immutability() -> None:
    """Test Entity attributes, methods, and immutability."""
    entity = Entity(
        id="factory_alpha",
        type="manufacturing_plant",
        attributes={"throughput": 120.5, "city": "Detroit"},
        tags=("automotive", "tier1"),
    )
    assert entity.id == "factory_alpha"
    assert entity.type == "manufacturing_plant"
    assert entity.get("throughput") == 120.5
    assert entity.get("city") == "Detroit"
    assert entity.get("missing", 42) == 42
    assert "automotive" in entity.tags

    with pytest.raises(ValidationError):
        # Immutability check
        entity.id = "modified"


def test_relationship_creation_and_key() -> None:
    """Test Relationship attributes, key, and immutability."""
    rel = Relationship(
        source="supplier_1",
        target="factory_alpha",
        type="supplies",
        attributes={"lead_time_days": 3},
    )
    assert rel.source == "supplier_1"
    assert rel.target == "factory_alpha"
    assert rel.type == "supplies"
    assert rel.key == ("supplier_1", "factory_alpha", "supplies")
    assert rel.get("lead_time_days") == 3


def test_resource_metrics_and_bounds() -> None:
    """Test Resource utilization, bounds checking, and deltas."""
    res = Resource(
        id="iron_ore",
        entity_id="wh_1",
        current=75.0,
        min_value=0.0,
        max_value=100.0,
        unit="tons",
    )
    assert res.utilization == 0.75
    assert res.available_capacity == 25.0
    assert not res.is_exhausted
    assert not res.is_full

    # Valid delta
    updated = res.with_delta(-25.0)
    assert updated.current == 50.0
    assert updated.utilization == 0.5

    # Violating delta with enforce_bounds raises ResourceBoundsError
    with pytest.raises(ResourceBoundsError):
        res.with_delta(-100.0, enforce_bounds=True)

    # Violating delta with clamp
    clamped = res.with_delta(-100.0, clamp=True)
    assert clamped.current == 0.0
    assert clamped.is_exhausted

    # Invalid bounds definition
    with pytest.raises(ValueError, match="exceeds max_value"):
        Resource(id="invalid", current=10.0, min_value=100.0, max_value=50.0)


def test_action_and_intervention() -> None:
    """Test Action payloads and Intervention state application."""
    action = Action(
        id="act_01",
        type="transfer_inventory",
        actor_id="logistics_agent",
        parameters={"source": "wh_1", "target": "wh_2", "quantity": 30.0},
        timestamp=1.0,
    )
    assert action.get("quantity") == 30.0
    assert action.actor_id == "logistics_agent"

    intervention = Intervention(
        id="policy_express_replenishment",
        description="Switch to expedited transfers",
        target_type="policy",
        parameters={"max_transfer_limit": 500.0, "mode": "express"},
    )

    state = WorldState(
        entities=[Entity(id="wh_1", type="warehouse")],
        resources=[Resource(id="stock", current=10.0)],
        active_rules={"safety_stock": 20.0},
    )
    counterfactual_state = intervention.apply(state)
    assert "policy_express_replenishment" in counterfactual_state.active_rules
    assert counterfactual_state.active_rules["policy_express_replenishment"]["mode"] == "express"


def test_exogenous_event() -> None:
    """Test ExogenousEvent initialization and parameters."""
    event = ExogenousEvent(
        id="storm_01",
        type="flash_flood",
        severity=4.5,
        parameters={"affected_roads": ["route_a", "route_b"]},
    )
    assert event.severity == 4.5
    assert "route_a" in event.get("affected_roads")


def test_world_state_hashing_and_reproducibility(sample_world_state: WorldState) -> None:
    """Test that WorldState hashes are deterministic and sensitive to alterations."""
    hash1 = sample_world_state.state_hash
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA-256 length

    # Identical state reproduces identical hash
    clone = sample_world_state.model_copy()
    assert clone.state_hash == hash1

    # State update alters hash
    altered = sample_world_state.update_resource("stock_wh1", delta=10.0)
    assert altered.state_hash != hash1
    assert altered.get_resource("stock_wh1").current == 110.0


def test_world_state_queries_and_errors(sample_world_state: WorldState) -> None:
    """Test querying entities, resources, and relationships, plus error handling."""
    assert sample_world_state.get_entity("wh_1").attributes["region"] == "north"
    assert sample_world_state.get_resource("stock_wh2").current == 50.0

    rels = sample_world_state.get_relationships(source="wh_1")
    assert len(rels) == 1
    assert rels[0].type == "connected_to"

    with pytest.raises(InvalidWorldStateError, match="Entity 'nonexistent' not found"):
        sample_world_state.get_entity("nonexistent")

    with pytest.raises(InvalidWorldStateError, match="Resource 'nonexistent' not found"):
        sample_world_state.get_resource("nonexistent")


def test_world_state_serialization_round_trip(sample_world_state: WorldState) -> None:
    """Test serialization to dict and reconstruction from dict."""
    state_dict = sample_world_state.to_dict()
    reconstructed = WorldState.from_dict(state_dict)

    assert reconstructed.state_hash == sample_world_state.state_hash
    assert reconstructed.step == sample_world_state.step
    assert reconstructed.timestamp == sample_world_state.timestamp
    assert len(reconstructed.entities) == len(sample_world_state.entities)
    assert len(reconstructed.resources) == len(sample_world_state.resources)


def test_world_initialization_signatures_and_simulate_alias(
    sample_world_state: WorldState,
) -> None:
    """Verify that World accepts 'initial_state' and 'exogenous_events', and engine.simulate works."""
    from ewm_engine.core.world import World
    from ewm_engine.simulation.engine import SimulationEngine
    from ewm_engine.simulation.scenario import Scenario

    # Test initial_state kwarg
    world = World(initial_state=sample_world_state)
    assert world.initial_state.state_hash == sample_world_state.state_hash
    assert len(world.event_sources) == 0

    # Test simulation via engine.simulate alias
    scenario = Scenario(name="SimulateAliasTest", horizon=2, samples=1, seed=42)
    engine = SimulationEngine()
    result = engine.simulate(world=world, scenario=scenario)
    assert len(result.trajectories) == 1
    assert len(result.trajectories[0].steps) == 2

