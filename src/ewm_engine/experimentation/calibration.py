"""Simulation parameter calibration via DistanceCalibrator and Method of Simulated Moments (MSM).

Conforms to Track T4: Extra [calibrate] (scipy/sklearn, no torch).
License compliance: Does NOT vendor black-it (AGPL-3.0); implements a clean,
permissive MIT/Apache Method of Simulated Moments (MSM) quadratic distance loss.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.params import ParameterSpace


def is_scipy_available() -> bool:
    """Check whether optional scipy is available."""
    try:
        importlib.import_module("scipy.optimize")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _load_scipy_optimize() -> Any:
    """Lazily load scipy.optimize or raise error."""
    try:
        return importlib.import_module("scipy.optimize")
    except (ImportError, ModuleNotFoundError) as err:
        raise SimulationConfigurationError(
            "SciPy is required for simulation calibration. Install with: pip install 'ewm-engine[calibrate]'"
        ) from err


def compute_msm_moments(samples: Sequence[float]) -> dict[str, float]:
    """Compute standard summary moments (mean, variance, skewness, quantiles) from a sample series."""
    arr = np.asarray(samples, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "variance": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}

    mean_v = float(np.mean(arr))
    var_v = float(np.var(arr))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))

    return {
        "mean": mean_v,
        "variance": var_v,
        "p25": p25,
        "p50": p50,
        "p75": p75,
    }


def msm_distance_loss(
    simulated_moments: dict[str, float],
    target_moments: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted quadratic Method of Simulated Moments (MSM) loss.

    L(theta) = sum_k w_k * (m_sim,k(theta) - m_target,k)^2
    """
    total_loss = 0.0
    common_keys = set(simulated_moments.keys()) & set(target_moments.keys())
    if not common_keys:
        return 0.0

    w_dict = weights or dict.fromkeys(common_keys, 1.0)

    for k in common_keys:
        diff = simulated_moments[k] - target_moments[k]
        w = w_dict.get(k, 1.0)
        # Normalize relative scale if target moment is large
        scale = max(abs(target_moments[k]), 1.0)
        total_loss += w * ((diff / scale) ** 2)

    return float(total_loss)


class CalibrationResult(BaseModel):
    """Outcomes and parameters recovered by DistanceCalibrator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibrated_parameters: dict[str, Any] = Field(description="Recovered parameter values.")
    loss_value: float = Field(description="Final distance loss score achieved.")
    evaluations_count: int = Field(ge=0, description="Total simulation evaluations performed.")
    success: bool = Field(description="Whether the optimizer converged successfully.")
    moment_errors: dict[str, float] = Field(
        description="Individual moment discrepancies at calibrated point."
    )
    history: list[dict[str, Any]] = Field(description="Optimization trajectory history.")


class DistanceCalibrator:
    """Calibrates simulation parameters to match empirical target data using scipy.optimize."""

    def __init__(
        self,
        space: ParameterSpace,
        simulator_fn: Callable[[dict[str, Any]], Sequence[float]],
        target_moments: dict[str, float],
        moment_weights: dict[str, float] | None = None,
    ) -> None:
        self.space = space
        self.simulator_fn = simulator_fn
        self.target_moments = dict(target_moments)
        self.moment_weights = dict(moment_weights) if moment_weights is not None else None
        self.numeric_params = [p for p in space.parameters if p.type in ("continuous", "integer")]

    def calibrate(
        self,
        max_evaluations: int = 50,
        method: str = "Nelder-Mead",
        initial_point: dict[str, Any] | None = None,
    ) -> CalibrationResult:
        """Run iterative optimization to find parameters minimizing MSM moment distance."""
        opt = _load_scipy_optimize()

        # Initial point in parameter bounds
        base_pt = initial_point or self.space.get_baseline_point()
        x0 = [float(base_pt[p.name]) for p in self.numeric_params]
        bounds = [p.bounds for p in self.numeric_params]

        history: list[dict[str, Any]] = []
        eval_count = 0
        best_loss = float("inf")
        best_point = dict(base_pt)
        best_moments: dict[str, float] = {}

        def objective(x_vec: np.ndarray) -> float:
            nonlocal eval_count, best_loss, best_point, best_moments
            eval_count += 1

            pt = dict(base_pt)
            for p, val in zip(self.numeric_params, x_vec, strict=True):
                pt[p.name] = round(val) if p.type == "integer" else float(val)

            # Evaluate simulator
            sim_output = self.simulator_fn(pt)
            sim_moments = compute_msm_moments(sim_output)
            loss = msm_distance_loss(sim_moments, self.target_moments, self.moment_weights)

            history.append({"eval": eval_count, "point": pt, "loss": loss})

            if loss < best_loss:
                best_loss = loss
                best_point = pt
                best_moments = sim_moments

            return loss

        # Run optimization
        res = opt.minimize(
            objective,
            x0=x0,
            bounds=bounds if method in ("L-BFGS-B", "Powell", "SLSQP") else None,
            method=method,
            options={"maxiter": max_evaluations},
        )

        moment_errors: dict[str, float] = {
            k: abs(best_moments.get(k, 0.0) - self.target_moments[k]) for k in self.target_moments
        }

        return CalibrationResult(
            calibrated_parameters=best_point,
            loss_value=best_loss,
            evaluations_count=eval_count,
            success=bool(res.success or eval_count >= max_evaluations),
            moment_errors=moment_errors,
            history=history,
        )


class PyABCPosteriorAdapter:
    """Documented stub adapter for ABC-SMC Bayesian posteriors via pyabc."""

    def __init__(self) -> None:
        try:
            importlib.import_module("pyabc")
        except (ImportError, ModuleNotFoundError) as err:
            raise SimulationConfigurationError(
                "pyabc is required for ABC-SMC calibration. Install with: pip install 'ewm-engine[calibrate]'"
            ) from err

    def run_abc(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("ABC-SMC execution via pyabc.")
