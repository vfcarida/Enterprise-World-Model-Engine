"""Unit tests for the pure-NumPy discrete bounded-future STL monitor."""

from __future__ import annotations

import numpy as np

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.verification.temporal import (
    AndFormula,
    NotFormula,
    OrFormula,
    always,
    evaluate_stl_bounded,
    eventually,
    extract_trajectory_signals,
    implies,
    predicate,
    until,
)


def test_atomic_predicates_and_robustness() -> None:
    signals = {"x": np.array([10.0, 5.0, 0.0, -5.0])}

    # x >= 0
    p_ge = predicate("x", ">=", 0.0)
    b, r = p_ge.evaluate(signals)
    assert list(b) == [True, True, True, False]
    assert list(r) == [10.0, 5.0, 0.0, -5.0]

    # x < 5
    p_lt = predicate("x", "<", 5.0)
    b_lt, r_lt = p_lt.evaluate(signals)
    assert list(b_lt) == [False, False, True, True]
    assert r_lt[0] == -5.0  # 5 - 10 = -5
    assert r_lt[3] == 10.0  # 5 - (-5) = 10


def test_boolean_combinators() -> None:
    signals = {"x": np.array([10.0, -5.0]), "y": np.array([2.0, 4.0])}

    p_x = predicate("x", ">=", 0.0)
    p_y = predicate("y", "<=", 3.0)

    # And: x >= 0 and y <= 3
    phi_and = AndFormula(p_x, p_y)
    b_and, r_and = phi_and.evaluate(signals)
    assert list(b_and) == [True, False]
    assert r_and[0] == 1.0  # min(10.0, 1.0) = 1.0
    assert r_and[1] == -5.0  # min(-5.0, -1.0) = -5.0

    # Or: x >= 0 or y <= 3
    phi_or = OrFormula(p_x, p_y)
    b_or, _r_or = phi_or.evaluate(signals)
    assert list(b_or) == [True, False]

    # Not: ! (x >= 0)
    phi_not = NotFormula(p_x)
    b_not, r_not = phi_not.evaluate(signals)
    assert list(b_not) == [False, True]
    assert list(r_not) == [-10.0, 5.0]

    # Implies: x >= 0 -> y <= 3
    phi_imp = implies(p_x, p_y)
    b_imp, _ = phi_imp.evaluate(signals)
    assert list(b_imp) == [True, True]  # T->T is T, F->F is T


def test_bounded_always_and_eventually() -> None:
    # Signal drops at step 3
    signals = {"inventory": np.array([100.0, 80.0, 60.0, -10.0, 50.0])}
    p_safe = predicate("inventory", ">=", 0.0)

    # always[0, 2]: holds across steps 0, 1, 2
    phi_always_early = always(p_safe, k1=0, k2=2)
    verdict_early = evaluate_stl_bounded(phi_always_early, signals)
    assert verdict_early.satisfied is True
    assert verdict_early.robustness == 60.0  # min(100, 80, 60)

    # always[0, 4]: fails because at step 3 inventory is -10
    phi_always_full = always(p_safe, k1=0, k2=4)
    verdict_full = evaluate_stl_bounded(phi_always_full, signals)
    assert verdict_full.satisfied is False
    assert verdict_full.robustness == -10.0
    assert 0 in verdict_full.violation_steps

    # eventually[1, 3]: does inventory drop < 0 between step 1 and 3?
    p_breach = predicate("inventory", "<", 0.0)
    phi_ev = eventually(p_breach, k1=1, k2=3)
    verdict_ev = evaluate_stl_bounded(phi_ev, signals)
    assert verdict_ev.satisfied is True
    assert verdict_ev.robustness == 10.0  # 0 - (-10) = 10


def test_bounded_until() -> None:
    # Phase 1 holds until step 2, where Phase 2 triggers
    signals = {
        "sla_met": np.array([1.0, 1.0, 0.0, 0.0]),
        "recovered": np.array([0.0, 0.0, 1.0, 1.0]),
    }
    p_sla = predicate("sla_met", "==", 1.0)
    p_rec = predicate("recovered", "==", 1.0)

    # sla_met U_[0, 3] recovered
    phi_until = until(p_sla, p_rec, k1=0, k2=3)
    verdict_until = evaluate_stl_bounded(phi_until, signals)
    assert verdict_until.satisfied is True


def test_stl_evaluation_on_trajectory() -> None:
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="e1", type="node")],
            resources=[Resource(id="r1", current=50.0)],
        )
    )
    scenario = Scenario(scenario_id="stl_traj_test", horizon=5, samples=1, seed=42)
    engine = SimulationEngine()
    result = engine.run(world, scenario)
    traj = result.trajectories[0]

    signals = extract_trajectory_signals(traj)
    assert "r1" in signals
    assert len(signals["r1"]) == 5

    # Test always(r1 >= 0)
    prop = always(predicate("r1", ">=", 0.0), k1=0, k2=4)
    verdict = evaluate_stl_bounded(prop, traj)
    assert verdict.satisfied is True
    assert verdict.robustness == 50.0
