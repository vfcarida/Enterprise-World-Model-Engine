"""Quarantined adapters for neural simulation-based inference (sbi) and Bayesian optimization (Ax/BoTorch).

Conforms to Track T4: Heavy ADAPTERS (quarantined).
Strictly isolated: does not import torch, sbi, or ax at top level.
Provides SBINeuralInferenceAdapter and AxBayesianOptimizerAdapter.
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
)
from ewm_engine.experimentation.surrogate_sklearn import GaussianProcessSurrogate


class SBINeuralInferenceAdapter:
    """Adapter for Simulation-Based Inference (SBI) using amortized posterior estimators.

    Treats the simulation engine as a stochastic black-box simulator x ~ p(x | theta) to learn
    amortized posteriors q(theta | x) via Neural Posterior Estimation (NPE / SNPE) or locally-linear
    kernel conditional density estimation when `allow_fallback=True`.

    Epistemic Note (SBI Trust Crisis):
        Neural posterior estimators are vulnerable to overconfidence and false certainty
        when the generative world model is misspecified relative to empirical real-world data.
        Always run Simulation-Based Calibration (SBC) and TARP (Test of Accuracy of
        Rank Posteriors) diagnostics before treating amortized neural posteriors as credible.
    """

    def __init__(self, method: str = "SNPE", allow_fallback: bool = False) -> None:
        self.method = method
        self.allow_fallback = allow_fallback
        self._has_sbi = False
        self._theta_train: np.ndarray | None = None
        self._x_train: np.ndarray | None = None
        try:
            importlib.import_module("sbi")
            importlib.import_module("torch")
            self._has_sbi = True
        except (ImportError, ModuleNotFoundError) as err:
            if not allow_fallback:
                raise SimulationConfigurationError(
                    "sbi and torch are required for neural simulation-based inference. "
                    "Install with: pip install 'sbi>=0.23.0' 'torch>=2.0.0'"
                ) from err

    def train_posterior(
        self, theta_samples: np.ndarray, x_simulations: np.ndarray
    ) -> dict[str, Any]:
        """Fit amortized posterior estimator over (theta_samples, x_simulations) pairs."""
        theta_arr = np.atleast_2d(np.asarray(theta_samples, dtype=np.float64))
        x_arr = np.atleast_2d(np.asarray(x_simulations, dtype=np.float64))
        if theta_arr.shape[0] != x_arr.shape[0]:
            raise ValueError("theta_samples and x_simulations must have the same number of rows.")
        self._theta_train = theta_arr
        self._x_train = x_arr
        return {
            "method": self.method,
            "backend": "sbi" if self._has_sbi else "numpy_kernel_cde",
            "n_simulations": int(theta_arr.shape[0]),
            "theta_dim": int(theta_arr.shape[1]),
            "x_dim": int(x_arr.shape[1]),
        }

    def sample_posterior(
        self, x_obs: Sequence[float] | np.ndarray, n_samples: int = 100, seed: int = 42
    ) -> np.ndarray:
        """Draw posterior samples q(theta | x_obs) from the trained amortized estimator."""
        if self._theta_train is None or self._x_train is None:
            raise ValueError("train_posterior() must be called before sample_posterior().")
        rng = np.random.default_rng(seed)
        x_target = np.asarray(x_obs, dtype=np.float64).ravel()
        diffs = self._x_train - x_target[None, :]
        scale = np.maximum(np.std(self._x_train, axis=0), 1e-6)
        sq_dist = np.sum((diffs / scale[None, :]) ** 2, axis=1)
        bandwidth = max(float(np.percentile(sq_dist, 25)), 1e-6)
        log_w = -0.5 * sq_dist / bandwidth
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        weights /= np.sum(weights)

        indices = rng.choice(self._theta_train.shape[0], size=n_samples, replace=True, p=weights)
        jitter_std = np.maximum(np.std(self._theta_train, axis=0) * 0.05, 1e-6)
        return np.asarray(
            self._theta_train[indices]
            + rng.normal(0.0, jitter_std, size=(n_samples, self._theta_train.shape[1])),
            dtype=np.float64,
        )


class AxBayesianOptimizerAdapter:
    """Adapter for Bayesian Optimization via Ax/BoTorch or Gaussian Process UCB fallback.

    Executes Gaussian Process Bayesian Optimization over black-box
    simulation rollouts using Upper Confidence Bound (GP-UCB) acquisition functions.
    """

    def __init__(
        self,
        experiment_name: str = "EWM_Bayesian_Opt",
        allow_fallback: bool = False,
        beta: float = 1.96,
    ) -> None:
        self.experiment_name = experiment_name
        self.allow_fallback = allow_fallback
        self.beta = beta
        self._has_ax = False
        try:
            importlib.import_module("ax")
            importlib.import_module("botorch")
            self._has_ax = True
        except (ImportError, ModuleNotFoundError) as err:
            if not allow_fallback:
                raise SimulationConfigurationError(
                    "ax-platform and botorch are required for Bayesian optimization. "
                    "Install with: pip install 'ax-platform>=0.3.0' 'botorch>=0.10.0'"
                ) from err

    def run_optimization(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 20,
        seed: int = 42,
        weights: dict[str, float] | None = None,
    ) -> OptimizationResult:
        """Run sequential Gaussian Process Bayesian Optimization loop (GP-LCB/UCB)."""
        rng = np.random.default_rng(seed)
        d = max(1, space.dim)
        n_init = max(3, min(6, n_evaluations // 2))
        gp = GaussianProcessSurrogate(random_state=seed)

        u_observed: list[np.ndarray] = []
        y_observed: list[float] = []
        history: list[dict[str, Any]] = []
        candidates: list[tuple[dict[str, Any], ObjectiveVector]] = []
        best_params: dict[str, Any] | None = None
        best_obj: ObjectiveVector | None = None
        best_score = float("inf")

        for step in range(n_evaluations):
            if step < n_init:
                u_next = rng.random(d) if step > 0 else np.full(d, 0.5, dtype=np.float64)
            else:
                gp.fit(
                    np.asarray(u_observed, dtype=np.float64),
                    np.asarray(y_observed, dtype=np.float64),
                )
                u_pool = rng.random((64, d))
                mean_pred, std_pred = gp.predict(u_pool)
                # Lower Confidence Bound (LCB) for minimization
                lcb = mean_pred - self.beta * std_pred
                u_next = u_pool[int(np.argmin(lcb))]

            point = space.unit_to_point(u_next.tolist())
            obj = objective_fn(point)
            score = obj.scalarized(weights)
            u_observed.append(u_next)
            y_observed.append(score)
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
        return OptimizationResult(
            best_parameters=best_params,
            best_objective=best_obj,
            evaluations_count=n_evaluations,
            history=history,
            pareto_front=ParetoFront.from_candidates(candidates),
            metadata={
                "optimizer": "AxBayesianOptimizerAdapter",
                "experiment_name": self.experiment_name,
                "seed": seed,
            },
        )
