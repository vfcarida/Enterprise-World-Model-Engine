"""Unit tests for ParameterSpace, DoE design generators, sweep harness, and tornado diagrams."""

from __future__ import annotations

import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.experimentation.doe import (
    generate_full_factorial,
    generate_halton,
    generate_lhs,
    generate_oat,
    run_doe_sweep,
)
from ewm_engine.experimentation.params import (
    ParameterDef,
    ParameterSpace,
)
from ewm_engine.experimentation.samplers import (
    HaltonSampler,
    LatinHypercubeSampler,
    RandomSampler,
)
from ewm_engine.simulation.scenario import Scenario


def _create_sample_space() -> ParameterSpace:
    return ParameterSpace(
        name="WarehouseSweepSpace",
        parameters=(
            ParameterDef(
                name="stock_capacity",
                type="continuous",
                bounds=(50.0, 200.0),
                default=100.0,
                target="resource",
                target_id="stock",
            ),
            ParameterDef(
                name="reorder_threshold",
                type="integer",
                bounds=(10, 40),
                default=20,
                target="variable",
                target_id="reorder_level",
            ),
            ParameterDef(
                name="dispatch_mode",
                type="categorical",
                categories=("standard", "priority", "bulk"),
                default="standard",
                target="variable",
                target_id="mode",
            ),
        ),
    )


def test_parameter_space_validation_and_unit_conversion() -> None:
    """Test ParameterSpace boundary validation and unit-hypercube mapping."""
    space = _create_sample_space()
    assert space.dim == 3
    assert space.names == ["stock_capacity", "reorder_threshold", "dispatch_mode"]
    assert len(space.space_hash) == 64

    # Unit hypercube transformations
    # [0, 0, 0] -> lower bounds and first category
    low_pt = space.unit_to_point([0.0, 0.0, 0.0])
    assert low_pt["stock_capacity"] == pytest.approx(50.0)
    assert low_pt["reorder_threshold"] == 10
    assert low_pt["dispatch_mode"] == "standard"

    # [1, 1, 1] -> upper bounds and last category
    high_pt = space.unit_to_point([1.0, 1.0, 1.0])
    assert high_pt["stock_capacity"] == pytest.approx(200.0)
    assert high_pt["reorder_threshold"] == 40
    assert high_pt["dispatch_mode"] == "bulk"

    # Invalid bounds check
    with pytest.raises(ValueError, match="inverted bounds"):
        ParameterDef(name="bad", type="continuous", bounds=(100.0, 50.0))

    with pytest.raises(ValueError, match="non-empty categories"):
        ParameterDef(name="bad_cat", type="categorical", categories=())


def test_design_generators() -> None:
    """Test full factorial, OAT, LHS, and Halton design generation."""
    space = _create_sample_space()

    # Full Factorial: 2 continuous levels * 2 integer levels * 3 categories = 12 points
    grid = generate_full_factorial(space, levels_per_param=2)
    assert len(grid) == 12

    # OAT: Baseline + 2 variations per parameter
    oat = generate_oat(space, steps_per_param=2)
    assert len(oat) >= 5
    assert oat[0] == space.get_baseline_point()

    # LHS: exact n_samples generated within bounds
    lhs = generate_lhs(space, n_samples=16, seed=42)
    assert len(lhs) == 16
    for pt in lhs:
        assert 50.0 <= pt["stock_capacity"] <= 200.0
        assert 10 <= pt["reorder_threshold"] <= 40
        assert pt["dispatch_mode"] in ("standard", "priority", "bulk")

    # Halton: low-discrepancy sequence
    halton = generate_halton(space, n_samples=10)
    assert len(halton) == 10


def test_samplers_protocol_implementations() -> None:
    """Verify built-in Sampler protocol classes."""
    space = _create_sample_space()

    r_sampler = RandomSampler()
    pts_rand = r_sampler.sample(space, n=5, seed=123)
    assert len(pts_rand) == 5

    lhs_sampler = LatinHypercubeSampler()
    pts_lhs = lhs_sampler.sample(space, n=8, seed=123)
    assert len(pts_lhs) == 8

    h_sampler = HaltonSampler()
    pts_h = h_sampler.sample(space, n=6)
    assert len(pts_h) == 6


def test_doe_sweep_execution_and_tornado() -> None:
    """Test run_doe_sweep running reproducible rollouts and computing tornado diagrams."""
    space = _create_sample_space()

    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            resources=[Resource(id="stock", current=100.0)],
            memory={"reorder_level": 20, "mode": "standard"},
        )
    )
    scenario = Scenario(scenario_id="sweep_test", horizon=3, samples=1, seed=42)

    oat_design = generate_oat(space, steps_per_param=2)
    sweep_res = run_doe_sweep(
        world=world,
        scenario=scenario,
        space=space,
        design_points=oat_design,
        design_type="oat",
        seed=100,
    )

    assert sweep_res.point_count == len(oat_design)
    assert len(sweep_res.results_table) == len(oat_design)
    assert len(sweep_res.fingerprint) == 64

    # Tornado data for resource 'stock'
    assert "stock" in sweep_res.tornado_data
    tornado = sweep_res.tornado_data["stock"]
    assert tornado.metric_name == "stock"
    assert len(tornado.bars) > 0
    # Bars are sorted in descending swing order
    swings = [b.swing for b in tornado.bars]
    assert swings == sorted(swings, reverse=True)
    # stock_capacity should have the highest swing on stock resource
    assert tornado.bars[0].parameter_name == "stock_capacity"


def test_parameter_space_validation_errors_and_targets() -> None:
    """Test validation errors for invalid bounds/categories and context/scenario parameter targets."""
    # Invalid bounds: lower >= upper
    with pytest.raises(ValueError, match="inverted bounds"):
        ParameterDef(name="p1", type="continuous", bounds=(10.0, 5.0), default=7.0)

    # Missing bounds
    with pytest.raises(ValueError, match="requires valid bounds"):
        ParameterDef(name="p2", type="integer", bounds=None, default=2)

    # Categorical missing categories
    with pytest.raises(ValueError, match="requires non-empty categories"):
        ParameterDef(name="p3", type="categorical", categories=(), default="a")

    # Space unit_to_point length mismatch
    space = _create_sample_space()
    with pytest.raises(ValueError, match="Expected 3 unit coordinates"):
        space.unit_to_point([0.5, 0.5])

    # Context and Scenario targets
    context_space = ParameterSpace(
        name="ContextAndScenarioSpace",
        parameters=(
            ParameterDef(
                name="region_code",
                type="categorical",
                categories=("us-east", "eu-west"),
                default="us-east",
                target="context",
            ),
            ParameterDef(
                name="simulation_horizon",
                type="integer",
                bounds=(5, 20),
                default=10,
                target="scenario_horizon",
            ),
        ),
    )
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            context={"region_code": "us-east"},
        )
    )
    scenario = Scenario(scenario_id="sc_test", horizon=10, samples=1)
    new_world, new_scenario = context_space.apply_to_world_and_scenario(
        {"region_code": "eu-west", "simulation_horizon": 15}, world, scenario
    )
    assert new_world.initial_state.context["region_code"] == "eu-west"
    assert new_scenario.horizon == 15
