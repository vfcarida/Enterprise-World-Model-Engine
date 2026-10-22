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


@given(
    init_src=st.floats(min_value=10.0, max_value=80.0),
    init_dst=st.floats(min_value=10.0, max_value=80.0),
    excess=st.floats(min_value=0.5, max_value=100.0),
)
def test_mr6_rejection_semantics_invalid_action_preserves_state(
    init_src: float,
    init_dst: float,
    excess: float,
) -> None:
    """MR-6 (Rejection Semantics): A not in A_valid => S_{t+1} == S_t on all resources and entities."""
    state = WorldState(
        entities=[Entity(id="depot_1", type="depot")],
        resources=[
            Resource(
                id="src", entity_id="depot_1", current=init_src, min_value=0.0, max_value=200.0
            ),
            Resource(
                id="dst", entity_id="depot_1", current=init_dst, min_value=0.0, max_value=200.0
            ),
        ],
    )
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[ActionTransferAvailabilityConstraint()],
    )
    invalid_action = Action(
        id="act_invalid",
        type="transfer_resource",
        parameters={
            "source_resource": "src",
            "target_resource": "dst",
            "quantity": init_src + excess,
        },
    )
    scen = Scenario(
        name="RejectionSemantics",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=invalid_action),),
    )
    res = SimulationEngine().run(world, scen)
    step_rec = res.trajectories[0].steps[0]
    next_state = res.trajectories[0].final_state

    assert len(step_rec.actions_proposed) == 1
    assert len(step_rec.actions_accepted) == 0
    assert len(step_rec.constraint_violations) == 1
    assert_allclose(next_state.get_resource("src").current, init_src, rtol=0.0, atol=1e-12)
    assert_allclose(next_state.get_resource("dst").current, init_dst, rtol=0.0, atol=1e-12)
    assert next_state.entities == state.entities


@given(
    init_src=st.floats(min_value=40.0, max_value=100.0),
    init_dst=st.floats(min_value=10.0, max_value=50.0),
    transfer_qty=st.floats(min_value=5.0, max_value=30.0),
    extra_capacity=st.floats(min_value=0.0, max_value=500.0),
)
def test_mr7_monotone_capacity_increase_never_invalidates_valid_trajectory(
    init_src: float,
    init_dst: float,
    transfer_qty: float,
    extra_capacity: float,
) -> None:
    """MR-7 (Monotone Constraint Relaxation): Capacity increase alone never turns valid -> invalid."""
    from ewm_engine.simulation.trajectory import TrajectoryStatus

    base_cap = 150.0
    state_base = WorldState(
        resources=[
            Resource(id="src", current=init_src, min_value=0.0, max_value=base_cap),
            Resource(id="dst", current=init_dst, min_value=0.0, max_value=base_cap),
        ]
    )
    state_expanded = WorldState(
        resources=[
            Resource(
                id="src", current=init_src, min_value=0.0, max_value=base_cap + extra_capacity
            ),
            Resource(
                id="dst", current=init_dst, min_value=0.0, max_value=base_cap + extra_capacity
            ),
        ]
    )
    constraints = [
        ResourceCapacityConstraint(resource_id="src", severity=ConstraintSeverity.HARD),
        ResourceCapacityConstraint(resource_id="dst", severity=ConstraintSeverity.HARD),
    ]
    action = Action(
        id="act_valid",
        type="transfer_resource",
        parameters={"source_resource": "src", "target_resource": "dst", "quantity": transfer_qty},
    )
    scen = Scenario(
        name="MonotoneCapacity",
        horizon=1,
        samples=1,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=action),),
    )

    res_base = SimulationEngine().run(
        World(state=state_base, dynamics=DeterministicTransferDynamics(), constraints=constraints),
        scen,
    )
    res_exp = SimulationEngine().run(
        World(
            state=state_expanded,
            dynamics=DeterministicTransferDynamics(),
            constraints=constraints,
        ),
        scen,
    )

    if res_base.trajectories[0].status == TrajectoryStatus.COMPLETED:
        assert res_exp.trajectories[0].status == TrajectoryStatus.COMPLETED
        assert res_exp.trajectories[0].hard_violations == 0


@given(
    slope=st.floats(min_value=0.5, max_value=4.0),
    intercept=st.floats(min_value=-2.0, max_value=5.0),
    query_action_val=st.floats(min_value=1.0, max_value=10.0),
)
def test_mr8_learned_dynamics_irrelevant_id_permutation_invariance(
    slope: float,
    intercept: float,
    query_action_val: float,
) -> None:
    """MR-8 (Learned Dynamics Metamorphic Invariance): Irrelevant entity-ID permutation preserves learned dynamics."""
    import numpy as np

    from ewm_engine.dynamics.learned import (
        LinearResidualDynamics,
        TransitionDataset,
        TransitionSample,
    )

    def _make_dataset(entity_prefix: str) -> TransitionDataset:
        samples: list[TransitionSample] = []
        for idx, u_val in enumerate([1.0, 2.0, 4.0, 6.0]):
            ents = [
                Entity(id=f"{entity_prefix}_a", type="node"),
                Entity(id=f"{entity_prefix}_b", type="node"),
            ]
            s_t = WorldState(
                entities=ents,
                resources=[
                    Resource(
                        id="target_stock",
                        entity_id=f"{entity_prefix}_a",
                        current=100.0,
                        min_value=0.0,
                        max_value=500.0,
                    )
                ],
                step=idx,
                timestamp=float(idx),
            )
            delta = slope * u_val + intercept
            s_next = s_t.update_resource("target_stock", delta=delta).advance_time(1.0)
            act = Action(id=f"a_{idx}", type="replenish", parameters={"value": u_val})
            samples.append(TransitionSample(state=s_t, actions=(act,), next_state=s_next))
        return TransitionDataset(samples)

    ds_orig = _make_dataset("node_alpha")
    ds_perm = _make_dataset("permuted_entity_omega")

    model_orig = LinearResidualDynamics(target_resource="target_stock", action_type="replenish")
    model_perm = LinearResidualDynamics(target_resource="target_stock", action_type="replenish")
    model_orig.fit(ds_orig)
    model_perm.fit(ds_perm)

    assert_allclose(model_orig.action_weight, model_perm.action_weight, rtol=1e-7, atol=1e-9)
    assert_allclose(model_orig.bias, model_perm.bias, rtol=1e-7, atol=1e-9)

    q_act = Action(id="q", type="replenish", parameters={"value": query_action_val})
    out_orig = model_orig.transition(ds_orig[0].state, [q_act], [], rng=np.random.default_rng(0))
    out_perm = model_perm.transition(ds_perm[0].state, [q_act], [], rng=np.random.default_rng(0))
    assert_allclose(
        out_orig.next_state.get_resource("target_stock").current,
        out_perm.next_state.get_resource("target_stock").current,
        rtol=1e-7,
        atol=1e-9,
    )


@given(
    raw_delta_a=st.floats(min_value=-500.0, max_value=500.0),
    raw_delta_b=st.floats(min_value=-500.0, max_value=500.0),
    raw_delta_c=st.floats(min_value=-500.0, max_value=500.0),
)
def test_mr9_projected_residual_satisfies_conservation_and_bounds_invariants(
    raw_delta_a: float,
    raw_delta_b: float,
    raw_delta_c: float,
) -> None:
    """MR-9 (Projected Residual Invariance): Projected residual strictly satisfies conservation and bounds."""
    from ewm_engine.experimental.conservation import (
        ConservationProjectionLayer,
        LinearConservationLaw,
    )

    state = WorldState(
        resources=[
            Resource(id="r_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="r_b", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="r_c", current=100.0, min_value=0.0, max_value=200.0),
        ]
    )
    law = LinearConservationLaw.zero_sum(["r_a", "r_b", "r_c"])
    projector = ConservationProjectionLayer([law], enforce_capacity_bounds=True)

    projected = projector.project_deltas(
        state,
        {"r_a": raw_delta_a, "r_b": raw_delta_b, "r_c": raw_delta_c},
    )

    next_state = state
    for rid, d_val in projected.items():
        next_state = next_state.update_resource(rid, delta=d_val, clamp=False, enforce_bounds=True)
        res = next_state.get_resource(rid)
        assert res.min_value - 1e-9 <= res.current <= res.max_value + 1e-9

    # Total mass conservation: sum_i R_i(t+1) == sum_i R_i(t)
    total_before = sum(r.current for r in state.resources.values())
    total_after = sum(r.current for r in next_state.resources.values())
    assert_allclose(total_after, total_before, rtol=1e-7, atol=1e-7)
    assert law.is_satisfied(state, next_state, tolerance=1e-7)
