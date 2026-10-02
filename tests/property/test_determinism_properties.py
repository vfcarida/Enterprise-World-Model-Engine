"""Property-based tests verifying deterministic simulation reproducibility (AC-004)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


@pytest.mark.property
@settings(max_examples=25)
@given(
    seed=st.integers(min_value=1, max_value=999999),
    horizon=st.integers(min_value=2, max_value=8),
)
def test_simulation_determinism_invariant(
    seed: int,
    horizon: int,
) -> None:
    """AC-004 Invariant: Identical state + configuration + seed must yield bit-identical trajectories."""
    base_state = WorldState(
        entities=[Entity(id="wh_1", type="warehouse")],
        resources=[Resource(id="stock_wh1", current=100.0, min_value=0.0, max_value=200.0)],
    )

    world1 = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="stock_wh1", mean_demand=15.0, std_demand=5.0
        ),
    )
    world2 = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="stock_wh1", mean_demand=15.0, std_demand=5.0
        ),
    )

    scenario = Scenario(name="DeterminismTest", horizon=horizon, samples=2, seed=seed)

    res1 = world1.simulate(scenario=scenario)
    res2 = world2.simulate(scenario=scenario)

    assert res1.provenance.scenario_fingerprint == res2.provenance.scenario_fingerprint
    assert res1.provenance.initial_state_fingerprint == res2.provenance.initial_state_fingerprint

    assert len(res1.trajectories) == len(res2.trajectories) == 2
    for t1, t2 in zip(res1.trajectories, res2.trajectories, strict=True):
        assert t1.seed == t2.seed
        assert t1.status == t2.status
        assert t1.final_state.fingerprint == t2.final_state.fingerprint
        assert len(t1.steps) == len(t2.steps)
        for s1, s2 in zip(t1.steps, t2.steps, strict=True):
            assert s1.state_hash == s2.state_hash
            assert s1.step_metrics == s2.step_metrics


@pytest.mark.property
def test_simulation_engine_verify_determinism_helper() -> None:
    """AC-004: verify_determinism self-check helper must confirm logical identity."""
    base_state = WorldState(
        resources=[Resource(id="stock", current=50.0, min_value=0.0, max_value=100.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(resource_id="stock", mean_demand=10.0, std_demand=2.0),
    )
    scenario = Scenario(
        name="ScheduledDeterminism",
        horizon=5,
        samples=3,
        seed=4242,
        scheduled_actions=(
            ScheduledAction(
                step=2,
                action=Action(id="restock_act", type="restock", parameters={"amount": 25.0}),
            ),
        ),
    )

    engine = SimulationEngine()
    assert engine.verify_determinism(world, scenario) is True


@pytest.mark.property
def test_different_seeds_produce_different_rollouts() -> None:
    """AC-004: Different seeds must generate different stochastic trajectories."""
    base_state = WorldState(
        resources=[Resource(id="stock", current=500.0, min_value=0.0, max_value=1000.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(resource_id="stock", mean_demand=20.0, std_demand=10.0),
    )

    scen_a = Scenario(horizon=10, samples=1, seed=111)
    scen_b = Scenario(horizon=10, samples=1, seed=999)

    res_a = world.simulate(scen_a)
    res_b = world.simulate(scen_b)

    assert res_a.trajectories[0].final_state.get_resource("stock").current != (
        res_b.trajectories[0].final_state.get_resource("stock").current
    )
    assert res_a.trajectories[0].final_state.fingerprint != (
        res_b.trajectories[0].final_state.fingerprint
    )
