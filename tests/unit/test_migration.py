"""Unit and contract tests for WorldState v1 <-> v2 migrations (P10)."""

from __future__ import annotations

import pytest

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.migration import (
    migrate_v1_to_v2,
    migrate_v2_to_v1,
    validate_migration_roundtrip,
)
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


def test_migrate_v1_to_v2_upgrades_schema() -> None:
    """Assert migrate_v1_to_v2 upgrades schema_version to '2.0.0' and populates graph attributes."""
    v1_state = WorldState(
        step=5,
        timestamp=5.0,
        schema_version="1.0.0",
        entities=[Entity(id="e1", type="node", attributes={"capacity": 100})],
        resources=[Resource(id="r1", current=50.0, min_value=0.0, max_value=200.0)],
        relationships=[Relationship(source="e1", target="e1", type="self_loop")],
    )

    v2_state = migrate_v1_to_v2(v1_state)
    assert v2_state.schema_version == "2.0.0"
    assert v2_state.step == 5
    assert v2_state.timestamp == 5.0
    assert "e1" in v2_state.entities
    assert "r1" in v2_state.resources

    # v2 attributes populated with defaults
    assert "temporal_valid_from" in v2_state.entities["e1"].attributes
    assert v2_state.relationships[0].attributes.get("weight") == 1.0


def test_migrate_v2_to_v1_downgrades_schema() -> None:
    """Assert migrate_v2_to_v1 restores schema_version to '1.0.0' for legacy consumers."""
    v2_state = WorldState(
        step=3,
        timestamp=3.0,
        schema_version="2.0.0",
        entities=[
            Entity(
                id="e1",
                type="node",
                attributes={
                    "capacity": 100,
                    "temporal_valid_from": None,
                    "temporal_valid_until": None,
                },
            )
        ],
        resources=[Resource(id="r1", current=75.0, min_value=0.0, max_value=200.0)],
    )

    v1_state = migrate_v2_to_v1(v2_state)
    assert v1_state.schema_version == "1.0.0"
    assert v1_state.step == 3
    assert v1_state.resources["r1"].current == pytest.approx(75.0)


def test_migration_roundtrip_fidelity() -> None:
    """Assert validate_migration_roundtrip passes for complex WorldStates."""
    v1_state = WorldState(
        step=12,
        timestamp=12.0,
        schema_version="1.0.0",
        entities=[
            Entity(id="factory", type="production_plant", attributes={"rate": 10.0}),
            Entity(id="depot", type="distribution_center", attributes={"bays": 4}),
        ],
        resources=[
            Resource(
                id="raw_material",
                entity_id="factory",
                current=400.0,
                min_value=0.0,
                max_value=1000.0,
            ),
            Resource(
                id="finished_goods",
                entity_id="depot",
                current=150.0,
                min_value=0.0,
                max_value=500.0,
            ),
        ],
        relationships=[
            Relationship(
                source="factory", target="depot", type="supplies", attributes={"dist": 50}
            ),
        ],
    )

    assert validate_migration_roundtrip(v1_state) is True
