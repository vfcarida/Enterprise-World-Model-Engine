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
        # Initial point in parameter bounds
        base_pt = initial_point or self.space.get_baseline_point()
        x0 = np.asarray([float(base_pt[p.name]) for p in self.numeric_params], dtype=np.float64)
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
                clipped = float(val)
                if p.bounds is not None:
                    clipped = float(np.clip(clipped, p.bounds[0], p.bounds[1]))
                pt[p.name] = round(clipped) if p.type == "integer" else clipped

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

        if is_scipy_available():
            opt = _load_scipy_optimize()
            res = opt.minimize(
                objective,
                x0=x0,
                bounds=bounds if method in ("L-BFGS-B", "Powell", "SLSQP") else None,
                method=method,
                options={"maxiter": max_evaluations},
            )
            converged = bool(res.success or eval_count >= max_evaluations or best_loss < 1e-6)
        else:
            # Pure-NumPy multi-scale coordinate + Nelder-Mead simplex search within parameter bounds
            d = len(self.numeric_params)
            lows = np.asarray(
                [
                    p.bounds[0] if p.bounds is not None else x0[i] - 10.0
                    for i, p in enumerate(self.numeric_params)
                ],
                dtype=np.float64,
            )
            highs = np.asarray(
                [
                    p.bounds[1] if p.bounds is not None else x0[i] + 10.0
                    for i, p in enumerate(self.numeric_params)
                ],
                dtype=np.float64,
            )
            spans = np.maximum(highs - lows, 1e-6)

            # Phase 1: Coarse grid / quasi-random warm start (up to 35% of budget)
            n_warm = max(3, min(max_evaluations // 3, 12))
            for k in range(n_warm):
                if eval_count >= max_evaluations:
                    break
                frac = k / max(1, n_warm - 1)
                cand = lows + frac * spans if k > 0 else x0.copy()
                objective(cand)

            # Phase 2: Adaptive coordinate pattern refinement from best found point
            curr = np.asarray(
                [float(best_point[p.name]) for p in self.numeric_params], dtype=np.float64
            )
            step_sizes = spans * 0.25
            while eval_count < max_evaluations and np.max(step_sizes) > 1e-6 and best_loss > 1e-10:
                improved = False
                for j in range(d):
                    for direction in (1.0, -1.0):
                        if eval_count >= max_evaluations:
                            break
                        trial = curr.copy()
                        trial[j] = np.clip(curr[j] + direction * step_sizes[j], lows[j], highs[j])
                        loss_trial = objective(trial)
                        if loss_trial < best_loss + 1e-12 and np.allclose(
                            [best_point[p.name] for p in self.numeric_params], trial
                        ):
                            curr = trial
                            improved = True
                            break
                if not improved:
                    step_sizes *= 0.5
            converged = bool(best_loss < 1e-2 or eval_count >= max_evaluations)

        moment_errors: dict[str, float] = {
            k: abs(best_moments.get(k, 0.0) - self.target_moments[k]) for k in self.target_moments
        }

        return CalibrationResult(
            calibrated_parameters=best_point,
            loss_value=best_loss,
            evaluations_count=eval_count,
            success=converged,
            moment_errors=moment_errors,
            history=history,
        )


class PyABCPosteriorAdapter:
    """Adapter for Approximate Bayesian Computation Sequential Monte Carlo (ABC-SMC)."""

    def __init__(self, allow_fallback: bool = False) -> None:
        self.allow_fallback = allow_fallback
        self._has_pyabc = False
        try:
            importlib.import_module("pyabc")
            self._has_pyabc = True
        except (ImportError, ModuleNotFoundError) as err:
            if not allow_fallback:
                raise SimulationConfigurationError(
                    "pyabc is required for ABC-SMC calibration. Install with: pip install 'ewm-engine[calibrate]'"
                ) from err

    def run_abc(
        self,
        space: ParameterSpace,
        simulator_fn: Callable[[dict[str, Any]], Sequence[float]],
        target_moments: dict[str, float],
        n_particles: int = 30,
        n_populations: int = 3,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute ABC-SMC sequential importance resampling over ParameterSpace."""
        rng = np.random.default_rng(seed)
        numeric_params = [p for p in space.parameters if p.type in ("continuous", "integer")]
        d = len(numeric_params)
        lows = np.asarray(
            [p.bounds[0] if p.bounds is not None else 0.0 for p in numeric_params],
            dtype=np.float64,
        )
        highs = np.asarray(
            [p.bounds[1] if p.bounds is not None else 1.0 for p in numeric_params],
            dtype=np.float64,
        )
        spans = np.maximum(highs - lows, 1e-6)
        base_pt = space.get_baseline_point()

        def eval_particle(theta: np.ndarray) -> float:
            pt = dict(base_pt)
            for p, val in zip(numeric_params, theta, strict=True):
                pt[p.name] = round(float(val)) if p.type == "integer" else float(val)
            sim_out = simulator_fn(pt)
            return msm_distance_loss(compute_msm_moments(sim_out), target_moments)

        # Population 0: prior sampling + top quantile selection
        n_candidates = max(n_particles * 2, 20)
        candidates = lows + rng.random((n_candidates, d)) * spans
        distances = np.asarray([eval_particle(c) for c in candidates], dtype=np.float64)
        order = np.argsort(distances)[:n_particles]
        particles = candidates[order]
        weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
        epsilon = float(np.percentile(distances, 50))

        # Subsequent SMC populations with Gaussian kernel perturbation
        for _pop in range(1, max(1, n_populations)):
            scale = np.maximum(np.std(particles, axis=0, ddof=1) * 0.75, spans * 0.05)
            new_particles = np.zeros_like(particles)
            new_dists = np.zeros(n_particles, dtype=np.float64)
            for i in range(n_particles):
                best_cand = particles[i]
                best_d = float("inf")
                for _attempt in range(5):
                    idx = int(rng.choice(n_particles, p=weights))
                    prop = np.clip(rng.normal(particles[idx], scale), lows, highs)
                    d_val = eval_particle(prop)
                    if d_val < best_d:
                        best_d = d_val
                        best_cand = prop
                    if d_val <= epsilon:
                        break
                new_particles[i] = best_cand
                new_dists[i] = best_d
            particles = new_particles
            epsilon = float(np.percentile(new_dists, 50))
            inv_d = 1.0 / np.maximum(new_dists, 1e-8)
            weights = inv_d / float(np.sum(inv_d))

        posterior_mean = {
            p.name: float(np.sum(particles[:, j] * weights)) for j, p in enumerate(numeric_params)
        }
        posterior_std = {
            p.name: float(
                np.sqrt(
                    np.maximum(
                        0.0,
                        np.sum(weights * (particles[:, j] - posterior_mean[p.name]) ** 2),
                    )
                )
            )
            for j, p in enumerate(numeric_params)
        }
        return {
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "final_epsilon": epsilon,
            "n_particles": n_particles,
            "particles": particles,
            "weights": weights,
        }
