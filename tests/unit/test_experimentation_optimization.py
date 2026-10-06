"""Unit tests for ObjectiveVector, ParetoFront, and built-in optimizers."""

from __future__ import annotations

from typing import Any

import pytest

from ewm_engine.experimentation.optimizers import (
    HillClimbingOptimizer,
    RandomSearchOptimizer,
)
from ewm_engine.experimentation.params import (
    ParameterDef,
    ParameterSpace,
)
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationResult,
    ParetoFront,
)


def test_objective_vector_scalarization_and_dominance() -> None:
    """Test ObjectiveVector scalarization and Pareto dominance."""
    # Min cost, Max quality
    obj1 = ObjectiveVector(
        values={"cost": 100.0, "quality": 0.8},
        directions={"cost": "minimize", "quality": "maximize"},
    )
    # obj2 is better in both (lower cost, higher quality)
    obj2 = ObjectiveVector(
        values={"cost": 80.0, "quality": 0.9},
        directions={"cost": "minimize", "quality": "maximize"},
    )
    # obj3 is worse in both
    obj3 = ObjectiveVector(
        values={"cost": 120.0, "quality": 0.7},
        directions={"cost": "minimize", "quality": "maximize"},
    )

    assert obj2.dominates(obj1) is True
    assert obj1.dominates(obj2) is False
    assert obj1.dominates(obj3) is True

    # Scalarization: lower is always better
    score1 = obj1.scalarized({"cost": 1.0, "quality": 100.0})
    # cost * 1 - quality * 100 = 100 - 80 = 20
    assert score1 == pytest.approx(20.0)

    score2 = obj2.scalarized({"cost": 1.0, "quality": 100.0})
    # 80 - 90 = -10
    assert score2 == pytest.approx(-10.0)
    assert score2 < score1


def test_pareto_front_from_candidates() -> None:
    """Test ParetoFront filters candidates down to non-dominated frontier."""
    candidates = [
        # pt1: cost 50, quality 0.5
        (
            {"x": 1},
            ObjectiveVector(
                values={"cost": 50.0, "quality": 0.5},
                directions={"cost": "minimize", "quality": "maximize"},
            ),
        ),
        # pt2: cost 70, quality 0.9 (trade-off with pt1)
        (
            {"x": 2},
            ObjectiveVector(
                values={"cost": 70.0, "quality": 0.9},
                directions={"cost": "minimize", "quality": "maximize"},
            ),
        ),
        # pt3: cost 80, quality 0.4 (strictly dominated by both pt1 and pt2)
        (
            {"x": 3},
            ObjectiveVector(
                values={"cost": 80.0, "quality": 0.4},
                directions={"cost": "minimize", "quality": "maximize"},
            ),
        ),
    ]

    front = ParetoFront.from_candidates(candidates)
    assert front.size == 2
    # pt3 should be excluded
    candidate_params = [p.parameters["x"] for p in front.policies]
    assert 1 in candidate_params
    assert 2 in candidate_params
    assert 3 not in candidate_params


def test_random_search_optimizer() -> None:
    """Test RandomSearchOptimizer over synthetic multi-objective problem."""
    space = ParameterSpace(
        parameters=(
            ParameterDef(name="x1", type="continuous", bounds=(-5.0, 5.0), default=0.0),
            ParameterDef(name="x2", type="continuous", bounds=(-5.0, 5.0), default=0.0),
        )
    )

    # Sphere function: minimize x1^2 + x2^2
    def objective_fn(pt: dict[str, float]) -> ObjectiveVector:
        val = pt["x1"] ** 2 + pt["x2"] ** 2
        return ObjectiveVector(values={"sphere": val}, directions={"sphere": "minimize"})

    opt = RandomSearchOptimizer()
    res: OptimizationResult = opt.optimize(objective_fn, space, n_evaluations=20, seed=42)

    assert res.evaluations_count == 20
    assert len(res.history) == 20
    assert res.best_objective.values["sphere"] >= 0.0
    assert res.pareto_front is not None
    assert res.pareto_front.size >= 1


def test_hill_climbing_optimizer() -> None:
    """Test HillClimbingOptimizer locally improves the objective."""
    space = ParameterSpace(
        parameters=(
            ParameterDef(name="x1", type="continuous", bounds=(-5.0, 5.0), default=3.0),
            ParameterDef(name="x2", type="continuous", bounds=(-5.0, 5.0), default=3.0),
        )
    )

    def objective_fn(pt: dict[str, float]) -> ObjectiveVector:
        val = pt["x1"] ** 2 + pt["x2"] ** 2
        return ObjectiveVector(values={"sphere": val}, directions={"sphere": "minimize"})

    # Baseline is (3, 3) -> sphere = 18.0
    opt = HillClimbingOptimizer(step_fraction=0.1)
    res: OptimizationResult = opt.optimize(objective_fn, space, n_evaluations=30, seed=42)

    assert res.evaluations_count == 30
    # Hill climbing must improve or match baseline
    assert res.best_objective.values["sphere"] <= 18.0


def test_hill_climbing_mixed_parameters() -> None:
    """Test HillClimbingOptimizer with integer and categorical parameters."""
    space = ParameterSpace(
        parameters=(
            ParameterDef(name="count", type="integer", bounds=(1, 20), default=10),
            ParameterDef(
                name="mode",
                type="categorical",
                categories=("fast", "slow"),
                default="slow",
            ),
        )
    )

    def obj(pt: dict[str, Any]) -> ObjectiveVector:
        penalty = 0.0 if pt["mode"] == "fast" else 10.0
        return ObjectiveVector(values={"loss": float(abs(pt["count"] - 5)) + penalty})

    opt = HillClimbingOptimizer(step_fraction=0.2)
    res = opt.optimize(obj, space, n_evaluations=25, seed=42)
    assert res.evaluations_count == 25
    assert res.best_parameters is not None
