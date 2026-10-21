"""Evolutionary and black-box optimizers via pycma and Nevergrad.

Conforms to Track T4: Extra [opt-evolutionary] (pycma, nevergrad).
Strictly isolated: requires optional dependencies.
"""

from __future__ import annotations

import importlib
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
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) via pycma or pure-NumPy CMA-ES."""

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
            return self._optimize_numpy_cma(
                objective_fn=objective_fn,
                space=space,
                n_evaluations=n_evaluations,
                seed=seed,
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

    def _optimize_numpy_cma(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        """Pure-NumPy (mu/mu_w, lambda)-CMA-ES in unit hypercube [0, 1]^D (Hansen, 2016)."""
        rng = np.random.default_rng(seed)
        d = max(1, space.dim)
        lam = max(4, 4 + int(3 * np.log(d)))
        mu = max(2, lam // 2)
        raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
        w = raw_w / np.sum(raw_w)
        mu_eff = float(1.0 / np.sum(w**2))

        mean = np.full(d, 0.5, dtype=np.float64)
        sigma = float(self.sigma0)
        cov = np.eye(d, dtype=np.float64)
        p_sigma = np.zeros(d, dtype=np.float64)
        p_c = np.zeros(d, dtype=np.float64)
        c_sigma = (mu_eff + 2.0) / (d + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (d + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / d) / (d + 4.0 + 2.0 * mu_eff / d)
        c_1 = 2.0 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((d + 2.0) ** 2 + mu_eff))
        chi_n = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")
        eval_count = 0

        while eval_count < n_evaluations:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            invsqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            batch_size = min(lam, n_evaluations - eval_count)
            z_samples = rng.standard_normal((batch_size, d))
            x_samples = np.clip(mean + sigma * (z_samples @ sqrt_cov.T), 0.0, 1.0)
            scores = np.zeros(batch_size, dtype=np.float64)

            for i in range(batch_size):
                point = space.unit_to_point(x_samples[i].tolist())
                obj = objective_fn(point)
                score = obj.scalarized(self.weights)
                scores[i] = score
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

            if batch_size >= mu:
                order = np.argsort(scores)[:mu]
                x_sel = x_samples[order]
                old_mean = mean.copy()
                mean = np.sum(x_sel * w[:, None], axis=0)
                y_w = (mean - old_mean) / max(sigma, 1e-12)

                p_sigma = (1.0 - c_sigma) * p_sigma + np.sqrt(
                    c_sigma * (2.0 - c_sigma) * mu_eff
                ) * (invsqrt_cov @ y_w)
                p_c = (1.0 - c_c) * p_c + np.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w

                diffs = (x_sel - old_mean) / max(sigma, 1e-12)
                rank_mu = sum(w[k] * np.outer(diffs[k], diffs[k]) for k in range(mu))
                cov = (1.0 - c_1 - c_mu) * cov + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
                sigma *= float(
                    np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1.0))
                )
                sigma = float(np.clip(sigma, 1e-5, 0.5))

        assert best_params is not None and best_obj is not None
        pareto = ParetoFront.from_candidates(candidates)
        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=eval_count,
            history=history,
            pareto_front=pareto,
            metadata={
                "optimizer": "PyCMAOptimizer(numpy-fallback)",
                "sigma0": self.sigma0,
                "seed": seed,
            },
        )


class NevergradOptimizer(Optimizer):
    """Gradient-free black-box optimization via Nevergrad or pure-NumPy Differential Evolution."""

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
            return self._optimize_numpy_de(
                objective_fn=objective_fn,
                space=space,
                n_evaluations=n_evaluations,
                seed=seed,
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

    def _optimize_numpy_de(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        """Pure-NumPy Differential Evolution + Local Gaussian Refinement in [0, 1]^D."""
        rng = np.random.default_rng(seed)
        d = max(1, space.dim)
        pop_size = max(4, min(10, n_evaluations // 2))
        pop = rng.random((pop_size, d))
        pop[0] = np.full(d, 0.5, dtype=np.float64)
        pop_scores = np.full(pop_size, float("inf"), dtype=np.float64)

        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")
        best_u = pop[0].copy()

        for step in range(n_evaluations):
            if step < pop_size:
                u = pop[step]
            else:
                p_idx = step % pop_size
                others = [idx for idx in range(pop_size) if idx != p_idx]
                a_idx, b_idx, c_idx = rng.choice(others, size=3, replace=False)
                mutant = np.clip(
                    pop[a_idx] + 0.7 * (best_u - pop[p_idx]) + 0.5 * (pop[b_idx] - pop[c_idx]),
                    0.0,
                    1.0,
                )
                cross_mask = rng.random(d) < 0.8
                cross_mask[int(rng.integers(0, d))] = True
                u = np.where(cross_mask, mutant, pop[p_idx])

            point = space.unit_to_point(u.tolist())
            obj = objective_fn(point)
            score = obj.scalarized(self.weights)
            candidates.append((point, obj))
            history.append(
                {
                    "step": step,
                    "parameters": point,
                    "objectives": obj.values,
                    "score": score,
                }
            )

            p_idx = step % pop_size
            if score <= pop_scores[p_idx]:
                pop[p_idx] = u
                pop_scores[p_idx] = score

            if score < best_score or best_params is None:
                best_score = score
                best_params = point
                best_obj = obj
                best_u = u.copy()

        assert best_params is not None and best_obj is not None
        pareto = ParetoFront.from_candidates(candidates)
        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=n_evaluations,
            history=history,
            pareto_front=pareto,
            metadata={"optimizer": f"Nevergrad({self.algorithm}-numpy)", "seed": seed},
        )
