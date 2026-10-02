"""Property-based tests verifying deterministic simulation reproducibility."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.scenario import Scenario


@settings(max_examples=25)
@given(
    seed=st.integers(min_value=1, max_value=999999),
    horizon=st.integers(min_value=2, max_value=8),
)
def test_simulation_determinism_invariant(
    seed: int,
    horizon: int,
) -> None:
    """Invariant: Identical state + configuration + seed must yield bit-identical trajectories."""
    from ewm_engine.core.entities import Entity
    from ewm_engine.core.resources import Resource

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

    assert res1.metadata.fingerprint == res2.metadata.fingerprint

    for t1, t2 in zip(res1.trajectories, res2.trajectories, strict=True):
        assert t1.seed == t2.seed
        assert t1.final_state.state_hash == t2.final_state.state_hash
        for s1, s2 in zip(t1.steps, t2.steps, strict=True):
            assert s1.state_hash == s2.state_hash
            assert s1.step_metrics == s2.step_metrics
