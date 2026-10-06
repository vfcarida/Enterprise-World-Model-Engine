"""Property and invariant tests for the experimentation suite.

Conforms to Track T4: Core Substrate (Zero-dep).
Verifies:
- Determinism: Identical inputs + seed produce identical sweep results and fingerprints.
- SeedSequence uniqueness: Every design point in a sweep receives an independent, distinct seed.
- ParameterSpace hash determinism and sensitivity to bounds.
- Non-mutation: Sweep and backtest harnesses do not mutate input WorldState or Scenario.
"""

from __future__ import annotations

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.experimentation.backtest import run_walk_forward_backtest
from ewm_engine.experimentation.doe import (
    generate_lhs,
    run_doe_sweep,
)
from ewm_engine.experimentation.params import (
    ParameterDef,
    ParameterSpace,
)
from ewm_engine.simulation.scenario import Scenario


def _make_test_world_and_scenario() -> tuple[World, Scenario, ParameterSpace]:
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="plant", type="factory")],
            resources=[Resource(id="capacity", current=100.0)],
            memory={"shift_hours": 8},
        )
    )
    scenario = Scenario(scenario_id="property_test_scen", horizon=3, samples=1, seed=42)
    space = ParameterSpace(
        name="PropertyTestSpace",
        parameters=(
            ParameterDef(
                name="cap",
                type="continuous",
                bounds=(50.0, 150.0),
                default=100.0,
                target="resource",
                target_id="capacity",
            ),
            ParameterDef(
                name="hours",
                type="integer",
                bounds=(4, 12),
                default=8,
                target="memory",
                target_id="shift_hours",
            ),
        ),
    )
    return world, scenario, space


def test_sweep_determinism_and_fingerprint_reproducibility() -> None:
    """Same world, scenario, space, design points, and seed produce identical results table and fingerprint."""
    world, scenario, space = _make_test_world_and_scenario()
    design = generate_lhs(space, n_samples=8, seed=123)

    res1 = run_doe_sweep(world, scenario, space, design, design_type="lhs", seed=999)
    res2 = run_doe_sweep(world, scenario, space, design, design_type="lhs", seed=999)

    assert res1.fingerprint == res2.fingerprint
    assert res1.results_table == res2.results_table

    # Different seed produces different results
    res_diff_seed = run_doe_sweep(world, scenario, space, design, design_type="lhs", seed=1000)
    assert res_diff_seed.fingerprint != res1.fingerprint


def test_seedsequence_uniqueness_across_design_points() -> None:
    """Every design point in a sweep receives an independent, distinct seed."""
    world, scenario, space = _make_test_world_and_scenario()
    design = generate_lhs(space, n_samples=16, seed=42)

    res = run_doe_sweep(world, scenario, space, design, design_type="lhs", seed=777)
    seeds = [r["seed"] for r in res.results_table]

    assert len(seeds) == 16
    assert len(set(seeds)) == 16, (
        "SeedSequence must produce unique child seeds for all design points."
    )


def test_parameter_space_hash_determinism_and_sensitivity() -> None:
    """ParameterSpace produces deterministic hashes and changes hash if bounds or names change."""
    _, _, space1 = _make_test_world_and_scenario()
    _, _, space2 = _make_test_world_and_scenario()

    assert space1.space_hash == space2.space_hash

    # Altering bounds alters space_hash
    altered_space = ParameterSpace(
        name=space1.name,
        parameters=(
            ParameterDef(
                name="cap",
                type="continuous",
                bounds=(50.0, 151.0),
                default=100.0,
                target="resource",
                target_id="capacity",
            ),
            ParameterDef(
                name="hours",
                type="integer",
                bounds=(4, 12),
                default=8,
                target="memory",
                target_id="shift_hours",
            ),
        ),
    )
    assert altered_space.space_hash != space1.space_hash


def test_sweep_and_backtest_never_mutate_inputs() -> None:
    """DoE sweep and backtest harnesses strictly read world and scenario without mutating them."""
    world, scenario, space = _make_test_world_and_scenario()

    init_state_fp = world.initial_state.fingerprint
    design = generate_lhs(space, n_samples=4, seed=42)

    # Execute sweep
    run_doe_sweep(world, scenario, space, design, seed=55)

    assert world.initial_state.fingerprint == init_state_fp

    # Execute backtest
    world_history = [world] * 6
    obs_history = [100.0] * 6
    run_walk_forward_backtest(
        world_history=world_history,
        observed_signal_history=obs_history,
        target_signal_name="capacity",
        horizon=2,
        samples=2,
        seed=10,
    )

    assert world.initial_state.fingerprint == init_state_fp
