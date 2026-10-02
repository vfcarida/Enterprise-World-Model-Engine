"""Property-based tests verifying state serialization and immutability invariants."""

from __future__ import annotations

import json
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState

safe_floats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)
safe_primitives = st.one_of(
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    safe_floats,
    st.text(min_size=1, max_size=20),
)


@settings(max_examples=30)
@given(
    stock=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    step=st.integers(min_value=0, max_value=100),
    timestamp=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    memory_payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=10), values=safe_primitives, max_size=5
    ),
)
def test_state_serialization_round_trip_invariant(
    stock: float,
    step: int,
    timestamp: float,
    memory_payload: dict[str, Any],
) -> None:
    """Invariant: WorldState -> to_dict() -> from_dict() must preserve state_hash identically."""
    original_state = WorldState(
        entities=[
            Entity(id="e1", type="node", attributes={"weight": 12.5}),
        ],
        relationships=[
            Relationship(source="e1", target="e1", type="loop", attributes={"cost": 1.0}),
        ],
        resources=[
            Resource(id="r1", entity_id="e1", current=stock, min_value=0.0, max_value=1500.0),
        ],
        memory={"count": step, **memory_payload},
        active_rules={"limit": 500.0},
        context={"temp": 20.0},
        timestamp=timestamp,
        step=step,
    )

    state_dict = original_state.to_dict()
    reconstituted_state = WorldState.from_dict(state_dict)

    # Invariant: hashes and fingerprints must match bit-for-bit
    assert reconstituted_state.state_hash == original_state.state_hash
    assert reconstituted_state.fingerprint == original_state.fingerprint
    assert reconstituted_state.step == original_state.step
    assert reconstituted_state.timestamp == original_state.timestamp
    assert (
        reconstituted_state.get_resource("r1").current == original_state.get_resource("r1").current
    )


@settings(max_examples=30)
@given(
    stock=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    step=st.integers(min_value=0, max_value=100),
    timestamp=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_state_json_string_round_trip_preserves_fingerprint(
    stock: float,
    step: int,
    timestamp: float,
) -> None:
    """Invariant: WorldState -> JSON string -> WorldState must preserve canonical fingerprint."""
    state = WorldState(
        entities=[
            Entity(id="e1", type="node", attributes={"weight": 12.5}),
            Entity(id="e2", type="worker", attributes={"skill": 4}),
        ],
        relationships=[
            Relationship(source="e1", target="e2", type="assigns", attributes={"priority": 1}),
        ],
        resources=[
            Resource(id="r1", entity_id="e1", current=stock, min_value=0.0, max_value=1500.0),
        ],
        memory={"step": step},
        active_rules={"rule_a": True},
        context={"season": "summer"},
        timestamp=timestamp,
        step=step,
    )

    json_str = json.dumps(state.to_dict())
    loaded_dict = json.loads(json_str)
    reconstituted = WorldState.from_dict(loaded_dict)

    assert reconstituted.fingerprint == state.fingerprint
    assert reconstituted.state_hash == state.state_hash
