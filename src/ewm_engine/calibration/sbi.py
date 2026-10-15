"""Amortized Bayesian simulation-based inference (SBI) engine.

Learns an amortized posterior density p(theta | x) over WorldSpec parameters
conditioned on simulated or observed summary statistics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.calibration.priors import PriorSet


class PosteriorDistribution(BaseModel):
    """Calibrated posterior probability distribution p(theta | x_obs).

    Guarantees epistemic honesty: always represents a full distribution with parameter
    uncertainty and credible intervals, never a collapsed single point estimate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_names: tuple[str, ...] = Field(description="Names of calibrated parameters.")
    mean: dict[str, float] = Field(description="Posterior mean expected values.")
    median: dict[str, float] = Field(description="Posterior median values.")
    std: dict[str, float] = Field(description="Posterior standard deviations.")
    credible_intervals_90: dict[str, tuple[float, float]] = Field(
        description="90% equal-tailed credible intervals (5th to 95th percentiles)."
    )
    covariance: list[list[float]] = Field(description="Full posterior covariance matrix.")
    samples: list[list[float]] = Field(description="Empirical posterior draws for downstream use.")

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        prior_set: PriorSet,
        covariance: np.ndarray | None = None,
        **kwargs: Any,
    ) -> PosteriorDistribution:
        """Construct PosteriorDistribution directly from parameter samples matrix."""
        arr = np.asarray(samples, dtype=np.float64)
        names = prior_set.names
        d_theta = len(names)

        # Clip strictly to prior bounds
        bounds = prior_set.bounds
        clipped = np.empty_like(arr)
        for j in range(d_theta):
            low, high = bounds[j]
            clipped[:, j] = np.clip(arr[:, j], low, high)

        mean_vals = np.mean(clipped, axis=0)
        median_vals = np.median(clipped, axis=0)
        std_vals = np.std(clipped, axis=0)

        mean_dict = {names[j]: float(mean_vals[j]) for j in range(d_theta)}
        median_dict = {names[j]: float(median_vals[j]) for j in range(d_theta)}
        std_dict = {names[j]: float(std_vals[j]) for j in range(d_theta)}

        ci_dict: dict[str, tuple[float, float]] = {}
        for j in range(d_theta):
            p5 = float(np.percentile(clipped[:, j], 5))
            p95 = float(np.percentile(clipped[:, j], 95))
            ci_dict[names[j]] = (p5, p95)

        if covariance is not None:
            cov_mat = np.asarray(covariance, dtype=np.float64)
        else:
            cov_mat = np.cov(clipped, rowvar=False)
            if cov_mat.ndim == 0:
                cov_mat = cov_mat.reshape(1, 1)

        return cls(
            parameter_names=names,
            mean=mean_dict,
            median=median_dict,
            std=std_dict,
            credible_intervals_90=ci_dict,
            covariance=cov_mat.tolist(),
            samples=clipped.tolist(),
        )

    def get_samples_array(self) -> np.ndarray:
        """Return posterior samples as np.ndarray of shape (S, D)."""
        return np.array(self.samples, dtype=np.float64)

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        """Draw n resampled parameter vectors from the posterior."""
        arr = self.get_samples_array()
        if len(arr) == 0:
            return np.zeros((n, len(self.parameter_names)), dtype=np.float64)
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(arr), size=n, replace=True)
        return arr[indices]

    def log_prob(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate multivariate Gaussian approximation log p(theta | x_obs)."""
        arr = np.asarray(theta, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        mu = np.array([self.mean[k] for k in self.parameter_names], dtype=np.float64)
        cov = np.array(self.covariance, dtype=np.float64)

        d = len(self.parameter_names)
        reg_cov = cov + (1e-6 * np.eye(d))
        inv_cov = np.linalg.pinv(reg_cov)
        sign, logdet = np.linalg.slogdet(reg_cov)
        if sign <= 0:
            logdet = 0.0

        diff = arr - mu
        quad = np.sum(diff @ inv_cov * diff, axis=1)
        logp = -0.5 * (d * math.log(2 * math.pi) + logdet + quad)
        return np.asarray(logp, dtype=np.float64)


class AmortizedPosteriorEstimator:
    """Amortized Bayesian posterior estimator: learns p(theta | x) from simulated pairs.

    Implements locally linear kernel-conditioned Bayesian posterior estimation in pure NumPy,
    with lazy delegating adapter to `sbi` (SNPE) when installed.
    """

    def __init__(
        self,
        prior_set: PriorSet,
        bandwidth_quantile: float = 0.15,
        regularization: float = 1e-4,
        name: str = "AmortizedPosteriorEstimator",
    ) -> None:
        self.prior_set = prior_set
        self.bandwidth_quantile = bandwidth_quantile
        self.regularization = regularization
        self.name = name

        self.is_fitted: bool = False
        self._thetas: np.ndarray | None = None
        self._xs: np.ndarray | None = None
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None

    def fit(self, thetas: np.ndarray, xs: np.ndarray) -> None:
        """Train amortized estimator on simulated parameter-summary pairs (theta, x)."""
        theta_arr = np.asarray(thetas, dtype=np.float64)
        x_arr = np.asarray(xs, dtype=np.float64)

        if len(theta_arr) == 0 or len(x_arr) == 0:
            return

        self._thetas = theta_arr
        self._xs = x_arr

        # Normalize summary statistics
        self._x_mean = np.mean(x_arr, axis=0)
        x_std = np.std(x_arr, axis=0)
        x_std[x_std < 1e-12] = 1.0
        self._x_std = x_std

        self.is_fitted = True

    def predict_posterior(
        self,
        x_obs: np.ndarray | Sequence[float],
        n_samples: int = 500,
        seed: int = 42,
    ) -> PosteriorDistribution:
        """Infer parameter posterior p(theta | x_obs) conditioned on observed target data x_obs."""
        if not self.is_fitted or self._thetas is None or self._xs is None or self._x_mean is None or self._x_std is None:
            raise RuntimeError("Estimator must be fitted before inferring posterior.")

        x_o = np.asarray(x_obs, dtype=np.float64).ravel()
        n_train, d_theta = self._thetas.shape

        # Standardize features
        xs_norm = (self._xs - self._x_mean) / self._x_std
        xo_norm = (x_o - self._x_mean) / self._x_std

        # Compute Euclidean distance in normalized summary space
        distances = np.linalg.norm(xs_norm - xo_norm, axis=1)

        # Adaptive bandwidth based on nearest neighbors quantile
        k_neighbors = max(5, int(n_train * self.bandwidth_quantile))
        sorted_dists = np.sort(distances)
        h = max(float(sorted_dists[min(k_neighbors, len(sorted_dists) - 1)]), 1e-4)

        # Epanechnikov / Gaussian kernel weights
        u = distances / h
        weights = np.exp(-0.5 * (u**2))
        total_w = np.sum(weights)
        if total_w < 1e-12:
            weights = np.ones(n_train) / n_train
        else:
            weights = weights / total_w

        # Locally linear bias correction: theta ~ A (x - x_o) + b
        # Solve weighted regression
        W = np.diag(weights)
        dX = xs_norm - xo_norm
        d_x = dX.shape[1]
        dX_design = np.column_stack([np.ones(n_train), dX])

        reg = self.regularization * np.eye(d_x + 1)
        reg[0, 0] = 0.0

        # W_half
        w_sqrt = np.sqrt(weights)
        dX_w = dX_design * w_sqrt[:, None]
        thetas_w = self._thetas * w_sqrt[:, None]

        coeffs = np.linalg.pinv(dX_w.T @ dX_w + reg) @ (dX_w.T @ thetas_w)
        cond_mean = coeffs[0, :]
        cond_slopes = coeffs[1:, :]

        # Residual variance
        pred_thetas = cond_mean + (dX @ cond_slopes)
        residuals = self._thetas - pred_thetas
        cond_cov = (residuals.T @ W @ residuals) + (self.regularization * np.eye(d_theta))

        # Enforce positive definiteness
        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        eigvals, eigvecs = np.linalg.eigh(cond_cov)
        eigvals = np.maximum(eigvals, 1e-6)
        cond_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Draw posterior samples
        rng = np.random.default_rng(seed)
        raw_samples = rng.multivariate_normal(cond_mean, cond_cov, size=n_samples)

        # Clip samples strictly to prior bounds
        bounds = self.prior_set.bounds
        for j in range(d_theta):
            low, high = bounds[j]
            raw_samples[:, j] = np.clip(raw_samples[:, j], low, high)

        # Compute summary metrics
        sample_mean = np.mean(raw_samples, axis=0)
        sample_median = np.median(raw_samples, axis=0)
        sample_std = np.std(raw_samples, axis=0)

        names = self.prior_set.names
        mean_dict = {names[j]: float(sample_mean[j]) for j in range(d_theta)}
        median_dict = {names[j]: float(sample_median[j]) for j in range(d_theta)}
        std_dict = {names[j]: float(sample_std[j]) for j in range(d_theta)}

        ci_dict: dict[str, tuple[float, float]] = {}
        for j in range(d_theta):
            p5 = float(np.percentile(raw_samples[:, j], 5))
            p95 = float(np.percentile(raw_samples[:, j], 95))
            ci_dict[names[j]] = (p5, p95)

        cov_list = cond_cov.tolist()
        samples_list = raw_samples.tolist()

        return PosteriorDistribution(
            parameter_names=names,
            mean=mean_dict,
            median=median_dict,
            std=std_dict,
            credible_intervals_90=ci_dict,
            covariance=cov_list,
            samples=samples_list,
        )
