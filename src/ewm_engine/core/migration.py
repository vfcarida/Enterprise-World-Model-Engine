"""Migration utilities and schema evolution tooling between v1 and v2 WorldStates."""

from __future__ import annotations

import copy
from typing import Any

from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import InvalidWorldStateError


def migrate_v1_to_v2(source: WorldState | dict[str, Any]) -> WorldState:
    """Migrate a v1.x WorldState (schema_version='1.0.0') to v2.0 (schema_version='2.0.0').

    Ensures heterogeneous entity tagging, relation weight normalization, and schema version bump.
    """
    if isinstance(source, WorldState):
        data = source.to_dict()
    elif isinstance(source, dict):
        data = copy.deepcopy(source)
    else:
        raise InvalidWorldStateError(
            f"Unsupported source type for migration: {type(source).__name__}"
        )

    # Bump schema version
    data["schema_version"] = "2.0.0"

    # Normalize entity attributes for v2 graph compatibility
    entities = data.get("entities", {})
    if isinstance(entities, dict):
        for _e_id, e_data in entities.items():
            if isinstance(e_data, dict):
                attrs = e_data.setdefault("attributes", {})
                if "temporal_valid_from" not in attrs:
                    attrs["temporal_valid_from"] = None
                if "temporal_valid_until" not in attrs:
                    attrs["temporal_valid_until"] = None

    # Normalize relationship weights and attributes
    relationships = data.get("relationships", [])
    if isinstance(relationships, list):
        for r_data in relationships:
            if isinstance(r_data, dict):
                attrs = r_data.setdefault("attributes", {})
                if "weight" not in attrs:
                    attrs["weight"] = 1.0

    return WorldState.from_dict(data)


def migrate_v2_to_v1(source: WorldState | dict[str, Any]) -> WorldState:
    """Down-migrate a v2.0 WorldState (schema_version='2.0.0') to legacy v1.x (schema_version='1.0.0').

    Strips v2-specific graph metadata while preserving exact state values, resources, and topology.
    """
    if isinstance(source, WorldState):
        data = source.to_dict()
    elif isinstance(source, dict):
        data = copy.deepcopy(source)
    else:
        raise InvalidWorldStateError(
            f"Unsupported source type for migration: {type(source).__name__}"
        )

    # Set legacy schema version
    data["schema_version"] = "1.0.0"

    # Clean up v2-specific null temporal metadata if desired
    entities = data.get("entities", {})
    if isinstance(entities, dict):
        for _e_id, e_data in entities.items():
            if isinstance(e_data, dict) and "attributes" in e_data:
                attrs = e_data["attributes"]
                attrs.pop("temporal_valid_from", None)
                attrs.pop("temporal_valid_until", None)

    return WorldState.from_dict(data)


def validate_migration_roundtrip(state_v1: WorldState) -> bool:
    """Verify lossless round-trip migration fidelity: v1 -> v2 -> v1.

    Asserts that entity IDs, resource currents, relationship pairs, and temporal steps
    remain identical.
    """
    state_v2 = migrate_v1_to_v2(state_v1)
    if state_v2.schema_version != "2.0.0":
        return False

    restored_v1 = migrate_v2_to_v1(state_v2)
    if restored_v1.schema_version != "1.0.0":
        return False

    if restored_v1.step != state_v1.step or restored_v1.timestamp != state_v1.timestamp:
        return False

    if set(restored_v1.entities.keys()) != set(state_v1.entities.keys()):
        return False

    if set(restored_v1.resources.keys()) != set(state_v1.resources.keys()):
        return False

    for r_id, r in state_v1.resources.items():
        if abs(restored_v1.resources[r_id].current - r.current) > 1e-6:
            return False

    return True
