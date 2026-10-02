"""Property-based tests verifying state serialization and immutability invariants."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


@settings(max_examples=30)
@given(
    stock=st.floats(min_value=0.0, max_value=1000.0),
    step=st.integers(min_value=0, max_value=100),
    timestamp=st.floats(min_value=0.0, max_value=1000.0),
)
def test_state_serialization_round_trip_invariant(
    stock: float,
    step: int,
    timestamp: float,
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
        memory={"count": step},
        active_rules={"limit": 500.0},
        context={"temp": 20.0},
        timestamp=timestamp,
        step=step,
    )

    state_dict = original_state.to_dict()
    reconstituted_state = WorldState.from_dict(state_dict)

    # Invariant: hashes must match bit-for-bit
    assert reconstituted_state.state_hash == original_state.state_hash
    assert reconstituted_state.step == original_state.step
    assert reconstituted_state.timestamp == original_state.timestamp
    assert (
        reconstituted_state.get_resource("r1").current == original_state.get_resource("r1").current
    )
