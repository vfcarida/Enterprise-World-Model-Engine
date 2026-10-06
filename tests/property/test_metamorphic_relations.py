"""Metamorphic property-based testing for EWM Engine.

Exploits the engine's explicit rule, constraint, and fingerprinting layers
as a built-in metamorphic oracle to verify structural invariances under transformations:

MR-1: Permutation Invariance — Reordering entities or resources does not alter
      the canonical SHA-256 state fingerprint or simulation metrics.
MR-2: Linear Scaling Relation — Uniformly scaling initial resources and transfer
      quantities by scalar k > 0 scales the final resource levels by exactly k.
MR-3: Seed Determinism Relation — Executing with identical seed produces bitwise-identical
      trajectories, step hashes, and final fingerprints.
MR-4: Slack Constraint Invariance — Adding an inactive or slack-satisfied soft constraint
      never alters an already valid trajectory's state sequence.
MR-5: Monotonic Rejection Relation — If action transfer of quantity Q is rejected by a capacity
      constraint, then any transfer of quantity Q' > Q is also guaranteed to be rejected.
"""

from __future__ import annotations

import hypothesis.strategies as st
from hypothesis import given
from numpy.testing import assert_allclose

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


@given(
    permute=st.booleans(),
    stock_a=st.floats(min_value=10.0, max_value=200.0),
    stock_b=st.floats(min_value=10.0, max_value=200.0),
)
def test_mr1_permutation_invariance_on_canonical_fingerprints(
    permute: bool, stock_a: float, stock_b: float
) -> None:
    """MR-1: Permuting entity/resource insertion order produces identical canonical fingerprints."""
    e1 = Entity(id="wh_north", type="warehouse", attributes={"capacity": 500})
    e2 = Entity(id="wh_south", type="warehouse", attributes={"capacity": 500})
    r1 = Resource(id="stock_north", current=round(stock_a, 2), min_value=0.0, max_value=500.0)
    r2 = Resource(id="stock_south", current=round(stock_b, 2), min_value=0.0, max_value=500.0)

    if permute:
        state_perm = WorldState(entities=[e2, e1], resources=[r2, r1])
    else:
        state_perm = WorldState(entities=[e1, e2], resources=[r1, r2])

    state_base = WorldState(entities=[e1, e2], resources=[r1, r2])
    assert state_perm.fingerprint == state_base.fingerprint


@given(
    base_stock_a=st.floats(min_value=50.0, max_value=100.0),
    base_stock_b=st.floats(min_value=10.0, max_value=50.0),
    transfer_qty=st.floats(min_value=5.0, max_value=20.0),
    scale_factor=st.floats(min_value=0.5, max_value=5.0),
)
def test_mr2_linear_scaling_relation_on_resources(
    base_stock_a: float,
    base_stock_b: float,
    transfer_qty: float,
    scale_factor: float,
) -> None:
    """MR-2: Scaling resource inputs by k > 0 scales outputs by exactly k."""
    k = round(scale_factor, 3)

    # Base world run
    state_base = WorldState(
        resources=[
            Resource(id="r_a", current=base_stock_a, min_value=0.0, max_value=1000.0),
            Resource(id="r_b", current=base_stock_b, min_value=0.0, max_value=1000.0),
        ]
    )
    world_base = World(state=state_base, dynamics=DeterministicTransferDynamics())
    action_base = Action(
        id="act_tx",
        type="transfer_resource",
        parameters={"source_resource": "r_a", "target_resource": "r_b", "quantity": transfer_qty},
    )
    scen_base = Scenario(
        name="Base",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=action_base),),
    )
    res_base = SimulationEngine().run(world_base, scen_base)
    final_base_a = res_base.trajectories[0].final_state.get_resource("r_a").current
    final_base_b = res_base.trajectories[0].final_state.get_resource("r_b").current

    # Scaled world run
    state_scaled = WorldState(
        resources=[
            Resource(id="r_a", current=base_stock_a * k, min_value=0.0, max_value=1000.0 * k),
            Resource(id="r_b", current=base_stock_b * k, min_value=0.0, max_value=1000.0 * k),
        ]
    )
    world_scaled = World(state=state_scaled, dynamics=DeterministicTransferDynamics())
    action_scaled = Action(
        id="act_tx",
        type="transfer_resource",
        parameters={
            "source_resource": "r_a",
            "target_resource": "r_b",
            "quantity": transfer_qty * k,
        },
    )
    scen_scaled = Scenario(
        name="Scaled",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=action_scaled),),
    )
    res_scaled = SimulationEngine().run(world_scaled, scen_scaled)
    final_scaled_a = res_scaled.trajectories[0].final_state.get_resource("r_a").current
    final_scaled_b = res_scaled.trajectories[0].final_state.get_resource("r_b").current

    # Verify linear scaling with explicit atol
    assert_allclose(final_scaled_a, final_base_a * k, rtol=1e-5, atol=1e-8)
    assert_allclose(final_scaled_b, final_base_b * k, rtol=1e-5, atol=1e-8)


@given(seed=st.integers(min_value=1, max_value=10000))
def test_mr3_seed_determinism_relation(seed: int) -> None:
    """MR-3: Runs under identical seed produce bitwise-identical trajectories."""
    state = WorldState(
        resources=[
            Resource(id="r_0", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="r_1", current=50.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())
    scen = Scenario(name="DeterminismCheck", horizon=3, samples=2, seed=seed)

    engine = SimulationEngine()
    run1 = engine.run(world, scen)
    run2 = engine.run(world, scen)

    assert (
        run1.trajectories[0].final_state.fingerprint == run2.trajectories[0].final_state.fingerprint
    )
    assert (
        run1.trajectories[1].final_state.fingerprint == run2.trajectories[1].final_state.fingerprint
    )


@given(transfer_qty=st.floats(min_value=5.0, max_value=30.0))
def test_mr4_slack_constraint_invariance(transfer_qty: float) -> None:
    """MR-4: Adding an inactive soft constraint does not alter a valid trajectory."""
    state = WorldState(
        resources=[
            Resource(id="r_a", current=100.0, min_value=0.0, max_value=500.0),
            Resource(id="r_b", current=20.0, min_value=0.0, max_value=500.0),
        ]
    )
    # Baseline world without constraint
    world_base = World(state=state, dynamics=DeterministicTransferDynamics())

    # World with soft constraint that has huge slack (max 500, current < 120)
    world_with_slack = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(
                resource_id="r_b",
                severity=ConstraintSeverity.SOFT,
            )
        ],
    )

    action = Action(
        id="act_tx",
        type="transfer_resource",
        parameters={"source_resource": "r_a", "target_resource": "r_b", "quantity": transfer_qty},
    )
    scen = Scenario(
        name="SlackCheck",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=action),),
    )

    engine = SimulationEngine()
    res_base = engine.run(world_base, scen)
    res_slack = engine.run(world_with_slack, scen)

    # Final resource states must be exactly identical
    assert_allclose(
        res_base.trajectories[0].final_state.get_resource("r_a").current,
        res_slack.trajectories[0].final_state.get_resource("r_a").current,
        atol=1e-8,
    )
    assert_allclose(
        res_base.trajectories[0].final_state.get_resource("r_b").current,
        res_slack.trajectories[0].final_state.get_resource("r_b").current,
        atol=1e-8,
    )


@given(
    available_stock=st.floats(min_value=10.0, max_value=50.0),
    excess_delta=st.floats(min_value=1.0, max_value=100.0),
    further_excess_delta=st.floats(min_value=1.0, max_value=100.0),
)
def test_mr5_monotonic_action_rejection_relation(
    available_stock: float,
    excess_delta: float,
    further_excess_delta: float,
) -> None:
    """MR-5: If transfer of quantity Q > available is rejected, Q' > Q is also rejected."""
    state = WorldState(
        resources=[
            Resource(id="stock_source", current=available_stock, min_value=0.0, max_value=200.0),
            Resource(id="stock_dest", current=10.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[ActionTransferAvailabilityConstraint()],
    )

    # Q exceeds available stock
    q1 = available_stock + excess_delta
    # Q' exceeds available stock even further
    q2 = q1 + further_excess_delta

    act_q1 = Action(
        id="act_q1",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_source",
            "target_resource": "stock_dest",
            "quantity": q1,
        },
    )
    act_q2 = Action(
        id="act_q2",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_source",
            "target_resource": "stock_dest",
            "quantity": q2,
        },
    )

    engine = SimulationEngine()
    scen_q1 = Scenario(
        name="Q1",
        horizon=1,
        samples=1,
        scheduled_actions=(ScheduledAction(step=0, action=act_q1),),
    )
    scen_q2 = Scenario(
        name="Q2",
        horizon=1,
        samples=1,
        scheduled_actions=(ScheduledAction(step=0, action=act_q2),),
    )

    res_q1 = engine.run(world, scen_q1)
    res_q2 = engine.run(world, scen_q2)

    # Both actions must be rejected
    assert len(res_q1.trajectories[0].steps[0].actions_accepted) == 0
    assert len(res_q2.trajectories[0].steps[0].actions_accepted) == 0
