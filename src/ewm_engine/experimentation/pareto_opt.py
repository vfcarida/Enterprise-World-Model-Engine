"""Multi-objective Pareto optimization via pymoo (NSGA-II).

Conforms to Track T4: Extra [opt-pareto] (pymoo).
Strictly isolated: requires optional pymoo dependency.
Searches multi-objective parameter spaces returning a ParetoFront
of non-dominated policy candidates, each with its ObjectiveVector and canonical fingerprint.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationResult,
    ParetoFront,
    ParetoPolicy,
)


def is_pymoo_available() -> bool:
    """Check whether pymoo package is installed."""
    try:
        importlib.import_module("pymoo")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _load_pymoo() -> Any:
    """Lazily load pymoo or raise informative error."""
    try:
        return importlib.import_module("pymoo")
    except (ImportError, ModuleNotFoundError) as err:
        raise SimulationConfigurationError(
            "pymoo is required for multi-objective Pareto optimization. Install with: pip install 'ewm-engine[opt-pareto]'"
        ) from err


class PymooParetoOptimizer:
    """Multi-objective genetic algorithm optimizer via pymoo NSGA-II."""

    def __init__(
        self,
        population_size: int = 20,
        n_generations: int = 10,
        constraint_fn: Callable[[dict[str, Any]], Sequence[float]] | None = None,
    ) -> None:
        """Initialize PymooParetoOptimizer.

        Args:
            population_size: Population size per generation.
            n_generations: Number of evolutionary generations.
            constraint_fn: Optional function returning inequality constraints g(x) <= 0.
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.constraint_fn = constraint_fn

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        seed: int = 42,
    ) -> OptimizationResult:
        """Run NSGA-II multi-objective optimization across parameter space."""
        _load_pymoo()
        problem_mod = importlib.import_module("pymoo.core.problem")
        nsga2_mod = importlib.import_module("pymoo.algorithms.moo.nsga2")
        optimize_mod = importlib.import_module("pymoo.optimize")

        # Probe objective keys and directions
        sample_pt = space.get_baseline_point()
        sample_obj = objective_fn(sample_pt)
        obj_keys = list(sample_obj.values.keys())
        n_obj = len(obj_keys)

        # Check constraint dimension
        sample_constr = self.constraint_fn(sample_pt) if self.constraint_fn else []
        n_constr = len(sample_constr)

        history: list[dict[str, Any]] = []
        all_candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        eval_counter = 0

        ProblemBase: Any = problem_mod.Problem
        constr_fn = self.constraint_fn

        class PymooProblem(ProblemBase):  # type: ignore[misc]
            def __init__(self) -> None:
                super().__init__(
                    n_var=space.dim,
                    n_obj=n_obj,
                    n_ieq_constr=n_constr,
                    xl=np.zeros(space.dim),
                    xu=np.ones(space.dim),
                )

            def _evaluate(
                self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any
            ) -> None:
                nonlocal eval_counter
                n_points = x.shape[0]
                f_matrix = np.zeros((n_points, n_obj), dtype=np.float64)
                g_matrix = (
                    np.zeros((n_points, n_constr), dtype=np.float64) if n_constr > 0 else None
                )

                for i in range(n_points):
                    pt = space.unit_to_point(x[i, :])
                    obj = objective_fn(pt)
                    all_candidates.append((pt, obj))

                    # pymoo minimizes all objectives by default
                    for j, k_obj in enumerate(obj_keys):
                        val = obj.values[k_obj]
                        direction = obj.get_direction(k_obj)
                        f_matrix[i, j] = -val if direction == "maximize" else val

                    if self.n_ieq_constr > 0 and g_matrix is not None:
                        constr_vals = constr_fn(pt) if callable(constr_fn) else []
                        g_matrix[i, :] = np.asarray(constr_vals, dtype=np.float64)

                    history.append(
                        {
                            "step": eval_counter,
                            "parameters": pt,
                            "objectives": obj.values,
                        }
                    )
                    eval_counter += 1

                out["F"] = f_matrix
                if g_matrix is not None:
                    out["G"] = g_matrix

        problem = PymooProblem()
        algorithm = nsga2_mod.NSGA2(pop_size=self.population_size)

        res = optimize_mod.minimize(
            problem,
            algorithm,
            ("n_gen", self.n_generations),
            seed=seed,
            verbose=False,
        )

        # Extract non-dominated solutions
        pareto_policies: list[ParetoPolicy] = []
        if res.X is not None:
            sol_points = np.atleast_2d(res.X)
            for i in range(sol_points.shape[0]):
                pt = space.unit_to_point(sol_points[i, :])
                obj = objective_fn(pt)
                pareto_policies.append(
                    ParetoPolicy(
                        parameters=pt,
                        objectives=obj,
                        policy_fingerprint=f"pareto_{i}_{seed}",
                    )
                )

        pareto_front = (
            ParetoFront(policies=tuple(pareto_policies))
            if pareto_policies
            else ParetoFront.from_candidates(all_candidates)
        )

        best_policy = pareto_front.policies[0] if pareto_front.policies else None
        best_params = best_policy.parameters if best_policy else space.get_baseline_point()
        best_obj = best_policy.objectives if best_policy else sample_obj

        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=eval_counter,
            history=history,
            pareto_front=pareto_front,
            metadata={
                "optimizer": "PymooParetoOptimizer(NSGA-II)",
                "population_size": self.population_size,
                "generations": self.n_generations,
                "seed": seed,
            },
        )
