"""Prior distributions over WorldSpec parameters for Simulation-Based Inference (SBI).

Allows declaring continuous, bounded, and informational priors over initial resource levels,
action rates, and dynamic parameters in a WorldSpec.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class CalibrationPrior(BaseModel):
    """Prior specification for a single tunable parameter in an enterprise world model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(
        description="Parameter identifier (e.g. 'initial_stock_wh1', 'transfer_rate')."
    )
    target: Literal["resource", "parameter", "rule", "context"] = Field(
        default="resource", description="Target component type in WorldSpec."
    )
    target_id: str = Field(description="Identifier of target resource, rule, or parameter key.")
    distribution: Literal["uniform", "gaussian", "log_uniform"] = Field(
        default="uniform", description="Probability distribution family."
    )
    bounds: tuple[float, float] = Field(
        description="Support bounds (min, max). Enforced strictly for all distributions."
    )
    mean: float | None = Field(default=None, description="Prior mean (for Gaussian distribution).")
    std: float | None = Field(default=None, description="Prior standard deviation (for Gaussian).")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample n scalar values from this prior distribution."""
        low, high = self.bounds
        if low >= high:
            return np.full(n, low, dtype=np.float64)

        if self.distribution == "uniform":
            return rng.uniform(low, high, size=n)
        if self.distribution == "log_uniform":
            safe_low = max(low, 1e-8)
            safe_high = max(high, safe_low + 1e-8)
            log_vals = rng.uniform(math.log(safe_low), math.log(safe_high), size=n)
            return np.exp(log_vals)
        # gaussian
        m = self.mean if self.mean is not None else 0.5 * (low + high)
        s = self.std if self.std is not None else (high - low) / 4.0
        raw = rng.normal(m, max(s, 1e-6), size=n)
        return np.clip(raw, low, high)

    def log_prob(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        """Evaluate prior log-density log p(theta)."""
        vals = np.asarray(values, dtype=np.float64)
        low, high = self.bounds
        logp = np.full_like(vals, -np.inf, dtype=np.float64)
        in_bounds = (vals >= low) & (vals <= high)

        if not np.any(in_bounds):
            return logp

        if self.distribution == "uniform":
            logp[in_bounds] = -math.log(max(high - low, 1e-12))
        elif self.distribution == "log_uniform":
            safe_low = max(low, 1e-8)
            safe_high = max(high, safe_low + 1e-8)
            in_pos = in_bounds & (vals > 0)
            logp[in_pos] = -math.log(math.log(safe_high) - math.log(safe_low)) - np.log(
                vals[in_pos]
            )
        elif self.distribution == "gaussian":
            m = self.mean if self.mean is not None else 0.5 * (low + high)
            s = self.std if self.std is not None else (high - low) / 4.0
            logp[in_bounds] = -0.5 * math.log(2 * math.pi * (s**2)) - (
                (vals[in_bounds] - m) ** 2 / (2 * (s**2))
            )

        return logp


class PriorSet(BaseModel):
    """Ordered collection of parameter priors defining the multivariate prior support."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    priors: tuple[CalibrationPrior, ...] = Field(description="Tuple of parameter priors.")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.priors)

    @property
    def dim(self) -> int:
        return len(self.priors)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [p.bounds for p in self.priors]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample theta matrix of shape (n, dim)."""
        if self.dim == 0 or n <= 0:
            return np.zeros((n, self.dim), dtype=np.float64)

        cols = [p.sample(n, rng) for p in self.priors]
        return np.column_stack(cols)

    def log_prob(self, thetas: np.ndarray) -> np.ndarray:
        """Evaluate joint prior log-density log p(theta) assuming factorized prior components."""
        arr = np.asarray(thetas, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        total_logp = np.zeros(arr.shape[0], dtype=np.float64)
        for j, p in enumerate(self.priors):
            total_logp += p.log_prob(arr[:, j])
        return total_logp

    def to_dict_list(self, thetas: np.ndarray) -> list[dict[str, float]]:
        """Convert theta matrix to list of parameter dictionaries."""
        arr = np.asarray(thetas, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        res = []
        for i in range(arr.shape[0]):
            res.append({p.name: float(arr[i, j]) for j, p in enumerate(self.priors)})
        return res
