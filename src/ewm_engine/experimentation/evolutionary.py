"""Evolutionary and black-box optimizers via pycma and Nevergrad.

Conforms to Track T4: Extra [opt-evolutionary] (pycma, nevergrad).
Strictly isolated: requires optional dependencies.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationResult,
    Optimizer,
    ParetoFront,
)


def is_cma_available() -> bool:
    """Check whether cma package is installed."""
    try:
        importlib.import_module("cma")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def is_nevergrad_available() -> bool:
    """Check whether nevergrad package is installed."""
    try:
        importlib.import_module("nevergrad")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


class PyCMAOptimizer(Optimizer):
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) via pycma."""

    def __init__(
        self,
        sigma0: float = 0.2,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.sigma0 = sigma0
        self.weights = weights

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        if not is_cma_available():
            raise SimulationConfigurationError(
                "pycma is required for CMA-ES optimization. Install with: pip install 'ewm-engine[opt-evolutionary]'"
            )
        cma = importlib.import_module("cma")

        # CMA-ES operates in unit hypercube [0, 1]^D
        x0 = [0.5] * space.dim
        es = cma.CMAEvolutionStrategy(
            x0,
            self.sigma0,
            {
                "bounds": [0.0, 1.0],
                "seed": seed,
                "verbose": -9,
            },
        )

        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")

        eval_count = 0
        while not es.stop() and eval_count < n_evaluations:
            solutions = es.ask()
            fitness_values: list[float] = []

            for sol in solutions:
                point = space.unit_to_point(sol)
                obj = objective_fn(point)
                score = obj.scalarized(self.weights)

                fitness_values.append(score)
                candidates.append((point, obj))
                history.append(
                    {
                        "step": eval_count,
                        "parameters": point,
                        "objectives": obj.values,
                        "score": score,
                    }
                )
                eval_count += 1

                if score < best_score or best_params is None:
                    best_score = score
                    best_params = point
                    best_obj = obj

                if eval_count >= n_evaluations:
                    break

            es.tell(solutions[: len(fitness_values)], fitness_values)

        assert best_params is not None and best_obj is not None
        pareto = ParetoFront.from_candidates(candidates)

        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=eval_count,
            history=history,
            pareto_front=pareto,
            metadata={"optimizer": "PyCMAOptimizer", "sigma0": self.sigma0, "seed": seed},
        )


class NevergradOptimizer(Optimizer):
    """Gradient-free black-box optimization via Nevergrad."""

    def __init__(
        self,
        algorithm: str = "NGOpt",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.algorithm = algorithm
        self.weights = weights

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        if not is_nevergrad_available():
            raise SimulationConfigurationError(
                "nevergrad is required for Nevergrad optimization. Install with: pip install 'ewm-engine[opt-evolutionary]'"
            )
        ng = importlib.import_module("nevergrad")

        # Define parametrization in unit hypercube [0, 1]^D
        param = ng.p.Array(shape=(space.dim,)).set_bounds(lower=0.0, upper=1.0)
        optimizer_cls = getattr(ng.optimizers, self.algorithm, ng.optimizers.NGOpt)
        opt = optimizer_cls(parametrization=param, budget=n_evaluations)

        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")

        for step in range(n_evaluations):
            x = opt.ask()
            point = space.unit_to_point(x.value)
            obj = objective_fn(point)
            score = obj.scalarized(self.weights)

            opt.tell(x, score)
            candidates.append((point, obj))
            history.append(
                {
                    "step": step,
                    "parameters": point,
                    "objectives": obj.values,
                    "score": score,
                }
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
            metadata={"optimizer": f"Nevergrad({self.algorithm})", "seed": seed},
        )
