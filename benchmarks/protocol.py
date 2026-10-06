"""Scientific Benchmark Protocol and typed result models for EWM Engine.

Defines the benchmarking protocol as a research instrument measuring the structural
generalization, constraint handling, and multi-step dynamics of world models.

DISCLAIMER: Benchmark results are strictly properties of artificial, synthetic
environments. They do not constitute empirical validation of real-world superiority.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine import __version__ as ENGINE_VERSION
from ewm_engine.dynamics.base import DynamicsModel


class BenchmarkFamily(StrEnum):
    """The five canonical benchmark families probing structural world-model capabilities."""

    INTERVENTION_SHIFT = "InterventionShift"
    RULE_SHIFT = "RuleShift"
    CONSTRAINT_STRESS = "ConstraintStress"
    LONG_HORIZON = "LongHorizon"
    MULTI_AGENT_CASCADE = "MultiAgentCascade"


class ModelEvaluationSummary(BaseModel):
    """Evaluation summary of an individual DynamicsModel on a benchmark family."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Name or class of the dynamics model.")
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Numeric metrics computed for this model (e.g. MAE, divergence, violation rate).",
    )
    invariants_passed: bool = Field(
        default=True,
        description="Whether declared invariants (bounds, conservation) held during rollout.",
    )
    invariant_violations: int = Field(
        default=0,
        description="Total number of invariant breaches during evaluation.",
    )
    divergence_or_gap: float = Field(
        default=0.0,
        description="Primary gap or divergence metric (e.g. interventional gap, horizon drift).",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific diagnostics and execution traces.",
    )


class BenchmarkResult(BaseModel):
    """Machine-readable, seeded, and reproducible benchmark outcome with full provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    benchmark_name: str = Field(description="Specific benchmark instance title.")
    family: BenchmarkFamily = Field(description="Canonical benchmark family category.")
    seed: int = Field(description="Random master seed used for execution.")
    provenance_fingerprint: str = Field(
        description="SHA-256 hash sealing seed, parameters, library versions, and platform."
    )
    timestamp: str = Field(description="ISO-8601 UTC timestamp of execution.")
    engine_version: str = Field(default=ENGINE_VERSION, description="EWM Engine release version.")
    python_version: str = Field(
        default_factory=lambda: sys.version.split()[0],
        description="Python runtime environment version.",
    )
    platform_info: str = Field(
        default_factory=lambda: f"{platform.system()}-{platform.machine()}",
        description="Hardware and operating system architecture.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Experimental benchmark parameters and hyperparameters.",
    )
    models_evaluated: list[ModelEvaluationSummary] = Field(
        default_factory=list,
        description="Comparative metrics across all evaluated dynamics models.",
    )
    phenomenon_surfaced: str = Field(
        description="Scientific phenomenon revealed (e.g. interventional degradation, compounding drift).",
    )
    epistemic_disclaimer: str = Field(
        default=(
            "SYNTHETIC RESEARCH INSTRUMENT: Results reflect properties of artificial, "
            "synthetic worlds only. They do NOT imply real-world causal validity or automated "
            "decision readiness. Models must not be ranked as globally 'superior' without a "
            "domain-specific utility and constraint contract."
        ),
        description="Mandatory epistemic humility banner.",
    )


class Benchmark(Protocol):
    """Protocol for an executable scientific benchmark."""

    @property
    def name(self) -> str:
        """Benchmark identifier."""
        ...

    @property
    def family(self) -> BenchmarkFamily:
        """Associated benchmark family category."""
        ...

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        """Execute the benchmark across candidate dynamics models."""
        ...


def compute_benchmark_fingerprint(
    benchmark_name: str,
    family: str,
    seed: int,
    parameters: dict[str, Any],
) -> str:
    """Compute a deterministic SHA-256 fingerprint sealing benchmark configuration."""
    raw = f"{benchmark_name}:{family}:{seed}:{ENGINE_VERSION}:{sorted(parameters.items())}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
