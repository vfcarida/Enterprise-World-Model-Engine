"""Property tests verifying Monte Carlo rollout sampling and RNG stream derivation (AC-009)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


@pytest.mark.property
@pytest.mark.parametrize("samples", [1, 5, 10, 25])
def test_monte_carlo_sample_count_property(samples: int) -> None:
    """AC-009: Monte Carlo simulation returns exactly `samples` rollouts."""
    base_state = WorldState(
        resources=[Resource(id="inventory", current=500.0, min_value=0.0, max_value=1000.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory", mean_demand=20.0, std_demand=5.0
        ),
    )

    scenario = Scenario(name="SampleCountTest", horizon=5, samples=samples, seed=12345)
    result = world.simulate(scenario)

    assert len(result.trajectories) == samples
    assert result.provenance.samples == samples


@pytest.mark.property
def test_monte_carlo_rollouts_use_distinct_derived_rng_streams() -> None:
    """AC-009: Distinct rollouts within the same run receive distinct derived RNG streams."""
    base_state = WorldState(
        resources=[Resource(id="inventory", current=500.0, min_value=0.0, max_value=1000.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory", mean_demand=25.0, std_demand=8.0
        ),
    )

    num_samples = 15
    scenario = Scenario(horizon=10, samples=num_samples, seed=42)
    result = world.simulate(scenario)

    assert len(result.trajectories) == num_samples

    # 1. Distinct rollout seeds recorded on trajectories
    seeds = [t.seed for t in result.trajectories]
    assert len(set(seeds)) == num_samples, f"Rollout seeds must be unique, got: {seeds}"

    # 2. Statistically different final states across rollouts
    final_values = [t.final_state.get_resource("inventory").current for t in result.trajectories]
    assert len(set(final_values)) > 1, (
        f"Expected stochastic variance across rollouts, got: {final_values}"
    )


@pytest.mark.property
def test_monte_carlo_entire_run_is_reproducible_under_same_seed() -> None:
    """AC-009: Running multi-sample Monte Carlo with the same seed reproduces all trajectories identically."""
    base_state = WorldState(
        resources=[Resource(id="stock", current=200.0, min_value=0.0, max_value=500.0)],
    )
    world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(resource_id="stock", mean_demand=15.0, std_demand=4.0),
    )

    scenario = Scenario(horizon=8, samples=5, seed=888)

    run_1 = world.simulate(scenario)
    run_2 = world.simulate(scenario)

    assert len(run_1.trajectories) == len(run_2.trajectories) == 5
    for t1, t2 in zip(run_1.trajectories, run_2.trajectories, strict=True):
        assert t1.sample_id == t2.sample_id
        assert t1.seed == t2.seed
        assert t1.final_state.fingerprint == t2.final_state.fingerprint
        assert t1.metric_series("resource_stock") == t2.metric_series("resource_stock")


@pytest.mark.property
def test_monte_carlo_rejects_invalid_sample_count_and_horizon() -> None:
    """AC-009: Fail closed when samples < 1 or horizon < 1."""
    with pytest.raises(ValidationError):
        Scenario(samples=0, horizon=5)

    with pytest.raises(ValidationError):
        Scenario(samples=5, horizon=0)

    # Also test engine direct validation if scenario attributes bypassed validation
    engine = SimulationEngine()
    invalid_scenario_samples = Scenario(samples=1, horizon=1)
    object.__setattr__(invalid_scenario_samples, "samples", 0)
    with pytest.raises(SimulationConfigurationError):
        engine.run(world=World(state=WorldState()), scenario=invalid_scenario_samples)

    invalid_scenario_horizon = Scenario(samples=1, horizon=1)
    object.__setattr__(invalid_scenario_horizon, "horizon", 0)
    with pytest.raises(SimulationConfigurationError):
        engine.run(world=World(state=WorldState()), scenario=invalid_scenario_horizon)
