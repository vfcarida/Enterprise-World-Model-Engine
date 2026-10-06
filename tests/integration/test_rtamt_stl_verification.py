"""Integration tests for RTAMT STL evaluation and cross-validation against core STL monitor.

Conforms to Track T3 (Extra [stl]):
- Cross-check core STL bounded fragment against RTAMT on overlapping formulas.
- Quantitative robustness distributions across Monte Carlo rollouts.
- Graceful isolation when rtamt is not available.
"""

from __future__ import annotations

import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.verification.rtamt_adapter import (
    RobustnessDistribution,
    RTAMTEvaluationBackend,
    is_rtamt_available,
)
from ewm_engine.verification.temporal import (
    always,
    evaluate_stl_bounded,
    eventually,
    predicate,
)

# Ensure rtamt is installed; skip cleanly if not installed
rtamt = pytest.importorskip("rtamt", reason="rtamt extra required for full STL tests")


def test_rtamt_availability_flag() -> None:
    """Verify is_rtamt_available correctly returns True when installed."""
    assert is_rtamt_available() is True


def test_cross_check_always_bounded_fragment() -> None:
    """Cross-check core STL monitor and RTAMT on bounded always formulas."""
    x_vals = [10.0, 12.0, 8.0, 15.0, 4.0, 11.0]
    threshold = 7.0

    # 1. Window [0:2] -> indices 0, 1, 2: values [10, 12, 8], all >= 7.0 (min margin = 8 - 7 = 1.0)
    core_formula_0_2 = always(predicate("x", ">=", threshold), k1=0, k2=2)
    core_verdict_0_2 = evaluate_stl_bounded(core_formula_0_2, {"x": x_vals})

    rtamt_backend_0_2 = RTAMTEvaluationBackend(
        formula=f"always[0:2](x >= {threshold})",
        variables={"x": "float"},
        spec_name="AlwaysWindow02",
    )
    rtamt_rob_0_2 = rtamt_backend_0_2.evaluate_signals({"x": x_vals})

    assert core_verdict_0_2.satisfied is True
    assert core_verdict_0_2.robustness == pytest.approx(1.0)
    assert rtamt_rob_0_2 == pytest.approx(1.0)
    assert (rtamt_rob_0_2 >= 0.0) == core_verdict_0_2.satisfied

    # 2. Window [0:4] -> indices 0..4: index 4 has x=4.0 < 7.0 (margin = 4 - 7 = -3.0)
    core_formula_0_4 = always(predicate("x", ">=", threshold), k1=0, k2=4)
    core_verdict_0_4 = evaluate_stl_bounded(core_formula_0_4, {"x": x_vals})

    rtamt_backend_0_4 = RTAMTEvaluationBackend(
        formula=f"always[0:4](x >= {threshold})",
        variables={"x": "float"},
        spec_name="AlwaysWindow04",
    )
    rtamt_rob_0_4 = rtamt_backend_0_4.evaluate_signals({"x": x_vals})

    assert core_verdict_0_4.satisfied is False
    assert core_verdict_0_4.robustness == pytest.approx(-3.0)
    assert rtamt_rob_0_4 == pytest.approx(-3.0)
    assert (rtamt_rob_0_4 >= 0.0) == core_verdict_0_4.satisfied


def test_cross_check_eventually_bounded_fragment() -> None:
    """Cross-check core STL monitor and RTAMT on bounded eventually formulas."""
    y_vals = [2.0, 4.0, 9.0, 1.0, 3.0]
    threshold = 8.0

    # 1. Window [0:2] -> values [2, 4, 9]: index 2 has 9.0 >= 8.0 (margin = 9 - 8 = +1.0)
    core_ev_0_2 = eventually(predicate("y", ">=", threshold), k1=0, k2=2)
    core_verdict_0_2 = evaluate_stl_bounded(core_ev_0_2, {"y": y_vals})

    rtamt_ev_0_2 = RTAMTEvaluationBackend(
        formula=f"eventually[0:2](y >= {threshold})",
        variables={"y": "float"},
        spec_name="EventuallyWindow02",
    )
    rtamt_rob_0_2 = rtamt_ev_0_2.evaluate_signals({"y": y_vals})

    assert core_verdict_0_2.satisfied is True
    assert core_verdict_0_2.robustness == pytest.approx(1.0)
    assert rtamt_rob_0_2 == pytest.approx(1.0)

    # 2. Window [0:1] -> values [2, 4]: max is 4.0 < 8.0 (margin = 4 - 8 = -4.0)
    core_ev_0_1 = eventually(predicate("y", ">=", threshold), k1=0, k2=1)
    core_verdict_0_1 = evaluate_stl_bounded(core_ev_0_1, {"y": y_vals})

    rtamt_ev_0_1 = RTAMTEvaluationBackend(
        formula=f"eventually[0:1](y >= {threshold})",
        variables={"y": "float"},
        spec_name="EventuallyWindow01",
    )
    rtamt_rob_0_1 = rtamt_ev_0_1.evaluate_signals({"y": y_vals})

    assert core_verdict_0_1.satisfied is False
    assert core_verdict_0_1.robustness == pytest.approx(-4.0)
    assert rtamt_rob_0_1 == pytest.approx(-4.0)


def test_rtamt_trajectory_evaluation() -> None:
    """Evaluate full STL on trajectory extracting signals automatically."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            resources=[Resource(id="r1", current=50.0)],
        )
    )
    scenario = Scenario(scenario_id="stl_traj_test", horizon=4, samples=1, seed=42)
    engine = SimulationEngine()
    result = engine.run(world, scenario)
    traj = result.trajectories[0]

    backend = RTAMTEvaluationBackend(
        formula="always(r1 >= 40.0)",
        variables={"r1": "float"},
    )
    rob = backend.evaluate_trajectory(traj)
    # r1 stays at 50.0 throughout steps, margin = 50.0 - 40.0 = +10.0
    assert rob == pytest.approx(10.0)

    # Violating formula
    backend_violation = RTAMTEvaluationBackend(
        formula="always(r1 >= 70.0)",
        variables={"r1": "float"},
    )
    rob_violation = backend_violation.evaluate_trajectory(traj)
    assert rob_violation == pytest.approx(-20.0)


def test_rtamt_monte_carlo_distribution() -> None:
    """Test evaluate_monte_carlo computes robust statistical distributions across rollouts."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            resources=[Resource(id="r1", current=50.0)],
        )
    )
    scenario = Scenario(scenario_id="stl_mc_test", horizon=5, samples=10, seed=42)
    engine = SimulationEngine()
    result = engine.run(world, scenario)

    backend = RTAMTEvaluationBackend(
        formula="always(r1 >= 35.0)",
        variables={"r1": "float"},
    )

    dist: RobustnessDistribution = backend.evaluate_monte_carlo(result.trajectories)

    assert dist.rollout_count == 10
    assert dist.formula == "always(r1 >= 35.0)"
    assert dist.satisfaction_probability == 1.0
    assert dist.mean_robustness == pytest.approx(15.0)
    assert "p05" in dist.quantiles
    assert "p50" in dist.quantiles
    assert "p95" in dist.quantiles
    assert dist.cvar_05 <= dist.quantiles["p05"]
    assert len(dist.individual_robustness) == 10


def test_rtamt_error_handling() -> None:
    """Test error cases: missing signal and empty rollout list."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            resources=[Resource(id="r1", current=50.0)],
        )
    )
    scenario = Scenario(scenario_id="stl_err_test", horizon=2, samples=1, seed=42)
    traj = SimulationEngine().run(world, scenario).trajectories[0]

    backend = RTAMTEvaluationBackend(
        formula="always(untracked_var >= 10.0)",
        variables={"untracked_var": "float"},
    )
    with pytest.raises(KeyError, match="missing required signals"):
        backend.evaluate_trajectory(traj)

    with pytest.raises(ValueError, match="empty trajectories sequence"):
        backend.evaluate_monte_carlo([])
