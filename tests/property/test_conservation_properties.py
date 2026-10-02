"""Property-based tests verifying resource conservation invariants using Hypothesis."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics


@settings(max_examples=50)
@given(
    initial_stock_a=st.floats(min_value=10.0, max_value=500.0),
    initial_stock_b=st.floats(min_value=10.0, max_value=500.0),
    transfer_amount=st.floats(min_value=0.0, max_value=600.0),
)
def test_resource_conservation_under_transfer(
    initial_stock_a: float,
    initial_stock_b: float,
    transfer_amount: float,
) -> None:
    """Invariant: The total quantity of resources across closed transfer entities must be strictly conserved."""
    state = WorldState(
        entities=[
            Entity(id="e_a", type="node"),
            Entity(id="e_b", type="node"),
        ],
        resources=[
            Resource(
                id="res_a",
                entity_id="e_a",
                current=initial_stock_a,
                min_value=0.0,
                max_value=1000.0,
            ),
            Resource(
                id="res_b",
                entity_id="e_b",
                current=initial_stock_b,
                min_value=0.0,
                max_value=1000.0,
            ),
        ],
    )

    total_initial = state.get_resource("res_a").current + state.get_resource("res_b").current

    dynamics = DeterministicTransferDynamics()
    action = Action(
        id="act_xfer",
        type="transfer_resource",
        parameters={
            "source_resource": "res_a",
            "target_resource": "res_b",
            "quantity": transfer_amount,
        },
    )
    rng = np.random.default_rng(42)

    result = dynamics.transition(state, [action], [], rng)

    total_final = (
        result.next_state.get_resource("res_a").current
        + result.next_state.get_resource("res_b").current
    )

    # Resource conservation invariant: total_final must equal total_initial within floating point precision
    assert np.isclose(total_initial, total_final, atol=1e-5)
    # Individual resources must never drop below min_value
    assert result.next_state.get_resource("res_a").current >= 0.0
    assert result.next_state.get_resource("res_b").current >= 0.0
