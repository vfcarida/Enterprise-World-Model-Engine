"""Unit tests for scenario discovery algorithms (PRIM, CART, VulnerabilityBox)."""

from __future__ import annotations

import numpy as np

from ewm_engine.decision.discovery import (
    ScenarioDiscoveryResult,
    VulnerabilityBox,
    is_ema_workbench_available,
    run_cart,
    run_prim,
)
from ewm_engine.experimentation.params import ParameterDef, ParameterSpace


def _build_test_space() -> ParameterSpace:
    return ParameterSpace(
        name="TestSpace",
        parameters=(
            ParameterDef(name="x1", type="continuous", bounds=(0.0, 100.0), default=50.0),
            ParameterDef(name="x2", type="continuous", bounds=(0.0, 50.0), default=25.0),
        ),
    )


def test_prim_discovers_known_vulnerability_box() -> None:
    """PRIM must successfully identify a synthetic failure region."""
    rng = np.random.default_rng(42)
    space = _build_test_space()
    n_samples = 200

    points: list[dict[str, float]] = []
    targets: list[int] = []

    # Failure box: x1 >= 70.0 and x2 <= 20.0
    for _ in range(n_samples):
        x1 = float(rng.uniform(0.0, 100.0))
        x2 = float(rng.uniform(0.0, 50.0))
        is_fail = 1 if (x1 >= 70.0 and x2 <= 20.0) else 0
        points.append({"x1": x1, "x2": x2})
        targets.append(is_fail)

    result = run_prim(
        points=points,
        targets=targets,
        parameter_space=space,
        alpha=0.05,
        min_support_ratio=0.05,
        threshold=0.5,
        strategy_id="strat_test",
    )

    assert isinstance(result, ScenarioDiscoveryResult)
    assert result.strategy_id == "strat_test"
    assert result.primary_box is not None

    box = result.primary_box
    assert isinstance(box, VulnerabilityBox)
    # The box should have high density and coverage
    assert box.density > 0.6
    assert box.coverage > 0.6
    assert len(box.restricted_dimensions) > 0
    assert box.quasi_p_value < 0.05
    assert len(result.discovery_fingerprint) == 64

    # Point inside known failure box is classified as inside
    assert box.is_inside({"x1": 85.0, "x2": 10.0}) is True
    # Point outside known failure box is classified as outside
    assert box.is_inside({"x1": 20.0, "x2": 45.0}) is False


def test_prim_deterministic_fingerprint() -> None:
    """Scenario discovery must be strictly reproducible under identical inputs."""
    space = _build_test_space()
    points = [{"x1": float(i), "x2": float(i % 50)} for i in range(100)]
    targets = [1 if p["x1"] > 60 else 0 for p in points]

    res1 = run_prim(points, targets, space, strategy_id="repro")
    res2 = run_prim(points, targets, space, strategy_id="repro")

    assert res1.discovery_fingerprint == res2.discovery_fingerprint
    assert len(res1.peeling_trajectory) == len(res2.peeling_trajectory)
    assert res1.primary_box is not None and res2.primary_box is not None
    assert res1.primary_box.bounds == res2.primary_box.bounds


def test_prim_zero_failures_handling() -> None:
    """When no failures occur, PRIM returns a benign unrestricted result."""
    space = _build_test_space()
    points = [{"x1": 10.0, "x2": 20.0}, {"x1": 30.0, "x2": 40.0}]
    targets = [0, 0]

    res = run_prim(points, targets, space, strategy_id="benign")
    assert res.total_failures == 0
    assert res.primary_box is not None
    assert res.primary_box.density == 0.0
    assert res.primary_box.coverage == 0.0
    assert "No failures" in res.primary_box.description


def test_cart_scenario_discovery() -> None:
    """CART rule induction must recursively partition and find failure leaves."""
    rng = np.random.default_rng(123)
    space = _build_test_space()
    n_samples = 150

    points: list[dict[str, float]] = []
    targets: list[int] = []

    for _ in range(n_samples):
        x1 = float(rng.uniform(0.0, 100.0))
        x2 = float(rng.uniform(0.0, 50.0))
        is_fail = 1 if x1 > 75.0 else 0
        points.append({"x1": x1, "x2": x2})
        targets.append(is_fail)

    res = run_cart(points, targets, space, max_depth=2, strategy_id="cart_strat")
    assert isinstance(res, ScenarioDiscoveryResult)
    assert res.algorithm == "cart"
    assert len(res.boxes) > 0
    assert res.primary_box is not None
    assert res.primary_box.density > 0.5


def test_is_ema_workbench_available_boolean() -> None:
    """Check helper function returns boolean without exception."""
    avail = is_ema_workbench_available()
    assert isinstance(avail, bool)
