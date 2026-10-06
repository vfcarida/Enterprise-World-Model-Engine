"""Built-in samplers implementing the Sampler protocol.

Conforms to Track T4: Core Substrate (Zero-dep, pure NumPy).
Provides RandomSampler, LatinHypercubeSampler, and HaltonSampler.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ewm_engine.experimentation.doe import generate_halton, generate_lhs
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.experimentation.protocols import Sampler


class RandomSampler(Sampler):
    """Uniform random parameter sampler."""

    def sample(
        self, space: ParameterSpace, n: int, seed: int | None = None
    ) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed)
        return [space.sample_random(rng) for _ in range(n)]


class LatinHypercubeSampler(Sampler):
    """Latin Hypercube Sampling (LHS) parameter sampler."""

    def sample(
        self, space: ParameterSpace, n: int, seed: int | None = None
    ) -> list[dict[str, Any]]:
        return generate_lhs(space=space, n_samples=n, seed=seed if seed is not None else 42)


class HaltonSampler(Sampler):
    """Low-discrepancy Halton quasi-random sequence sampler."""

    def sample(
        self, space: ParameterSpace, n: int, seed: int | None = None
    ) -> list[dict[str, Any]]:
        return generate_halton(space=space, n_samples=n)
