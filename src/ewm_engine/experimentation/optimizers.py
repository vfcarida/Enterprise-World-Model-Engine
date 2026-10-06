"""Built-in zero-dependency search optimizers implementing the Optimizer protocol.

Conforms to Track T4: Core Substrate (Zero-dep, pure NumPy).
Provides RandomSearchOptimizer and HillClimbingOptimizer with multi-objective
Pareto tracking.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationResult,
    Optimizer,
    ParetoFront,
)


class RandomSearchOptimizer(Optimizer):
    """Uniform random parameter search optimizer."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)

        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")

        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []

        for step in range(n_evaluations):
            point = space.sample_random(rng)
            obj = objective_fn(point)
            score = obj.scalarized(self.weights)

            candidates.append((point, obj))
            history.append(
                {"step": step, "parameters": point, "objectives": obj.values, "score": score}
            )

            if score < best_score or best_params is None:
                best_score = score
                best_params = point
                best_obj = obj

        assert best_params is not None and best_obj is not None
        pareto = ParetoFront.from_candidates(candidates)

        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=n_evaluations,
            history=history,
            pareto_front=pareto,
            metadata={"optimizer": "RandomSearchOptimizer", "seed": seed},
        )


class HillClimbingOptimizer(Optimizer):
    """Coordinate-perturbation local hill-climbing search optimizer."""

    def __init__(
        self,
        step_fraction: float = 0.1,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.step_fraction = step_fraction
        self.weights = weights

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)

        # Start from baseline or random point
        curr_point = space.get_baseline_point()
        curr_obj = objective_fn(curr_point)
        curr_score = curr_obj.scalarized(self.weights)

        history: list[dict[str, Any]] = [
            {
                "step": 0,
                "parameters": dict(curr_point),
                "objectives": curr_obj.values,
                "score": curr_score,
            }
        ]
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = [(curr_point, curr_obj)]

        best_params = curr_point
        best_obj = curr_obj
        best_score = curr_score

        for step in range(1, n_evaluations):
            # Propose candidate by perturbing one parameter
            param_to_perturb = space.parameters[rng.integers(0, space.dim)]
            cand_point = dict(curr_point)

            if param_to_perturb.type == "continuous":
                assert param_to_perturb.bounds is not None
                width = param_to_perturb.bounds[1] - param_to_perturb.bounds[0]
                delta = rng.normal(0.0, self.step_fraction * width)
                new_val = max(
                    param_to_perturb.bounds[0],
                    min(param_to_perturb.bounds[1], curr_point[param_to_perturb.name] + delta),
                )
                cand_point[param_to_perturb.name] = float(new_val)
            elif param_to_perturb.type == "integer":
                assert param_to_perturb.bounds is not None
                step_size = max(
                    1,
                    round(
                        self.step_fraction
                        * (param_to_perturb.bounds[1] - param_to_perturb.bounds[0])
                    ),
                )
                delta_int = rng.choice([-step_size, step_size])
                new_val_int = max(
                    param_to_perturb.bounds[0],
                    min(param_to_perturb.bounds[1], curr_point[param_to_perturb.name] + delta_int),
                )
                cand_point[param_to_perturb.name] = round(new_val_int)
            elif param_to_perturb.type == "categorical":
                assert param_to_perturb.categories is not None
                cand_point[param_to_perturb.name] = rng.choice(param_to_perturb.categories)

            cand_obj = objective_fn(cand_point)
            cand_score = cand_obj.scalarized(self.weights)

            candidates.append((cand_point, cand_obj))
            history.append(
                {
                    "step": step,
                    "parameters": cand_point,
                    "objectives": cand_obj.values,
                    "score": cand_score,
                }
            )

            # Accept if score improved (lower is better)
            if cand_score < curr_score:
                curr_point = cand_point
                curr_obj = cand_obj
                curr_score = cand_score

            if cand_score < best_score:
                best_score = cand_score
                best_params = cand_point
                best_obj = cand_obj

        pareto = ParetoFront.from_candidates(candidates)

        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=n_evaluations,
            history=history,
            pareto_front=pareto,
            metadata={
                "optimizer": "HillClimbingOptimizer",
                "step_fraction": self.step_fraction,
                "seed": seed,
            },
        )
