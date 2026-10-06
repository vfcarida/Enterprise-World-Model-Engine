"""Protocols and abstractions for sampling, surrogates, and optimization.

Conforms to Track T4: Core Substrate (Zero-dep).
Establishes typed protocols for Sampler, Surrogate, and Optimizer abstractions,
alongside ObjectiveVector for standardized multi-objective evaluations and Pareto dominance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.experimentation.params import ParameterSpace

OptimizationDirection = Literal["minimize", "maximize"]


class ObjectiveVector(BaseModel):
    """Multi-objective outcome vector with explicit optimization directions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    values: dict[str, float] = Field(description="Objective metric values.")
    directions: dict[str, OptimizationDirection] = Field(
        default_factory=dict,
        description="Optimization direction per metric ('minimize' or 'maximize').",
    )

    def get_direction(self, metric: str) -> OptimizationDirection:
        """Get optimization direction for a metric, defaulting to 'minimize'."""
        return self.directions.get(metric, "minimize")

    def scalarized(self, weights: dict[str, float] | None = None) -> float:
        """Compute scalarized objective score where lower is always better."""
        total = 0.0
        w_dict = weights or dict.fromkeys(self.values.keys(), 1.0)
        for k, v in self.values.items():
            weight = w_dict.get(k, 1.0)
            # If maximizing, negate so that lower is better
            direction = self.get_direction(k)
            term = -v if direction == "maximize" else v
            total += weight * term
        return float(total)

    def dominates(self, other: ObjectiveVector) -> bool:
        """Check whether self Pareto-dominates other.

        Self dominates other if self is no worse than other in all objectives,
        and strictly better in at least one objective.
        """
        keys = set(self.values.keys()) & set(other.values.keys())
        if not keys:
            return False

        at_least_as_good = True
        strictly_better = False

        for k in keys:
            v_self = self.values[k]
            v_other = other.values[k]
            direction = self.get_direction(k)

            if direction == "minimize":
                if v_self > v_other:
                    at_least_as_good = False
                    break
                if v_self < v_other:
                    strictly_better = True
            else:  # maximize
                if v_self < v_other:
                    at_least_as_good = False
                    break
                if v_self > v_other:
                    strictly_better = True

        return at_least_as_good and strictly_better


class ParetoPolicy(BaseModel):
    """An individual non-dominated policy candidate in the Pareto front."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameters: dict[str, Any] = Field(description="Policy or configuration parameters.")
    objectives: ObjectiveVector = Field(description="Evaluated multi-objective outcomes.")
    policy_fingerprint: str = Field(
        description="Canonical cryptographic hash of the policy candidate."
    )


class ParetoFront(BaseModel):
    """Collection of mutually non-dominated policy candidates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policies: tuple[ParetoPolicy, ...] = Field(
        default_factory=tuple,
        description="Sequence of non-dominated Pareto-optimal policies.",
    )

    @property
    def size(self) -> int:
        return len(self.policies)

    @classmethod
    def from_candidates(
        cls, candidates: Sequence[tuple[dict[str, Any], ObjectiveVector]]
    ) -> ParetoFront:
        """Filter a list of (parameters, objectives) candidates down to the true non-dominated Pareto front."""
        non_dominated: list[tuple[dict[str, Any], ObjectiveVector]] = []

        for p_cand, obj_cand in candidates:
            # Check if any existing member dominates obj_cand
            if any(existing_obj.dominates(obj_cand) for _, existing_obj in non_dominated):
                continue

            # Remove any members that obj_cand dominates
            non_dominated = [(p, obj) for (p, obj) in non_dominated if not obj_cand.dominates(obj)]
            non_dominated.append((p_cand, obj_cand))

        policies = [
            ParetoPolicy(
                parameters=p,
                objectives=obj,
                policy_fingerprint=canonical_sha256({"parameters": p, "objectives": obj.values}),
            )
            for p, obj in non_dominated
        ]
        return cls(policies=tuple(policies))


class OptimizationResult(BaseModel):
    """Outcomes and audit history of an optimization run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    best_parameters: dict[str, Any] = Field(
        description="Best single-objective or representative parameters found."
    )
    best_objective: ObjectiveVector = Field(description="Evaluated best objective vector.")
    evaluations_count: int = Field(
        ge=0, description="Total number of objective function evaluations."
    )
    history: list[dict[str, Any]] = Field(
        description="Step-by-step optimization evaluation history."
    )
    pareto_front: ParetoFront | None = Field(
        default=None,
        description="Pareto front of non-dominated policies if multi-objective search was conducted.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optimizer algorithm details and hyperparameters."
    )


class Sampler(Protocol):
    """Protocol for generating sample parameter points from a ParameterSpace."""

    def sample(
        self, space: ParameterSpace, n: int, seed: int | None = None
    ) -> list[dict[str, Any]]:
        """Generate n sample parameter points."""
        ...


class Surrogate(Protocol):
    """Protocol for fast emulator surrogates approximating simulation rollouts."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit emulator on design matrix X and target y."""
        ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected value and standard deviation / uncertainty for design points X."""
        ...


class Optimizer(Protocol):
    """Protocol for parameter and intervention search optimizers."""

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], ObjectiveVector],
        space: ParameterSpace,
        n_evaluations: int = 50,
        seed: int = 42,
    ) -> OptimizationResult:
        """Search parameter space to optimize objective_fn."""
        ...
