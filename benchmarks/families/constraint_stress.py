"""ConstraintStress benchmark family: probes violation handling and rollout invalidation under increasing boundary stress.

Tests whether world models respect hard physical and organizational boundaries under
high-load stress conditions vs fabricating phantom capacity.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np

from benchmarks.protocol import (
    Benchmark,
    BenchmarkFamily,
    BenchmarkResult,
    ModelEvaluationSummary,
    compute_benchmark_fingerprint,
)
from ewm_engine.core import Action, Resource, WorldState
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.experimental.dynamics_eval import evaluate_one_step
from ewm_engine.provenance.evidence import EvidenceLevel


class ClampedCapacityDynamics(DynamicsModel):
    """Safe dynamics model that strictly clamps state updates to declared resource limits."""

    def __init__(self, name: str = "ClampedCapacityDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        delta = sum(float(a.parameters.get("value", 0.0)) for a in actions)
        # Clamps strictly to min_value and max_value
        next_state = state.update_resource("storage", delta=delta, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"delta": delta, "clamped": True},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class UnclampedPhantomDynamics(DynamicsModel):
    """Unsafe model that ignores boundary capacity constraints and fabricates phantom resources."""

    def __init__(self, name: str = "UnclampedPhantomDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        delta = sum(float(a.parameters.get("value", 0.0)) for a in actions)
        # Bypasses clamp, creating invalid physical states
        next_state = state.update_resource("storage", delta=delta, clamp=False)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"delta": delta, "clamped": False},
            evidence_level=EvidenceLevel.PREDICTIVE,
            model_name=self.name,
        )


class ConstraintStressBenchmark(Benchmark):
    """Evaluates constraint violation rates and boundary safety under escalating stress."""

    def __init__(
        self,
        name: str = "ConstraintStress_StorageCapacity",
        seed: int = 42,
        max_capacity: float = 100.0,
        n_samples: int = 60,
    ) -> None:
        self._name = name
        self.seed = seed
        self.max_capacity = max_capacity
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> BenchmarkFamily:
        return BenchmarkFamily.CONSTRAINT_STRESS

    def _generate_stress_datasets(self) -> dict[str, Any]:
        from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample

        rng = np.random.default_rng(self.seed)

        # 1. Normal load (50% max capacity)
        normal_samples: list[TransitionSample] = []
        for i in range(self.n_samples // 2):
            curr = float(rng.uniform(20.0, 50.0))
            s1 = WorldState(
                resources={
                    "storage": Resource(
                        id="storage", current=curr, min_value=0.0, max_value=self.max_capacity
                    )
                }
            )
            act = Action(id=f"act_norm_{i}", type="store", parameters={"value": 15.0})
            s2 = s1.update_resource("storage", delta=15.0, clamp=True)
            normal_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

        # 2. Overload stress (attempts to push to 150% capacity)
        stress_samples: list[TransitionSample] = []
        for i in range(self.n_samples // 2):
            curr = float(rng.uniform(70.0, 95.0))
            s1 = WorldState(
                resources={
                    "storage": Resource(
                        id="storage", current=curr, min_value=0.0, max_value=self.max_capacity
                    )
                }
            )
            # Action demands 50 units when only 5-30 units remain
            act = Action(id=f"act_stress_{i}", type="store", parameters={"value": 50.0})
            s2 = s1.update_resource("storage", delta=50.0, clamp=True)
            stress_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

        return {
            "normal": TransitionDataset(normal_samples),
            "stress": TransitionDataset(stress_samples),
        }

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        datasets = self._generate_stress_datasets()
        normal_ds = datasets["normal"]
        stress_ds = datasets["stress"]

        eval_models: list[DynamicsModel] = list(models or [])
        if not eval_models:
            clamped = ClampedCapacityDynamics()
            unclamped = UnclampedPhantomDynamics()
            eval_models = [clamped, unclamped]

        summaries: list[ModelEvaluationSummary] = []
        for m in eval_models:
            m_name = getattr(m, "name", type(m).__name__)
            _, norm_inv = evaluate_one_step(m, normal_ds, seed=self.seed)
            _, stress_inv = evaluate_one_step(m, stress_ds, seed=self.seed)

            violation_rate = stress_inv.violation_rate
            total_violations = norm_inv.violations_count + stress_inv.violations_count

            summaries.append(
                ModelEvaluationSummary(
                    model_name=m_name,
                    metrics={
                        "normal_load_violations": float(norm_inv.violations_count),
                        "stress_load_violations": float(stress_inv.violations_count),
                        "stress_violation_rate": violation_rate,
                        "bounds_violations": float(stress_inv.bounds_violations),
                    },
                    invariants_passed=stress_inv.passed,
                    invariant_violations=total_violations,
                    divergence_or_gap=round(violation_rate, 4),
                    diagnostics={
                        "clamped_safely": stress_inv.passed,
                        "violating_resources": stress_inv.violating_resources,
                    },
                )
            )

        params = {
            "max_capacity": self.max_capacity,
            "n_samples": self.n_samples,
        }
        fp = compute_benchmark_fingerprint(self.name, self.family.value, self.seed, params)

        return BenchmarkResult(
            benchmark_name=self.name,
            family=self.family,
            seed=self.seed,
            provenance_fingerprint=fp,
            timestamp=datetime.now(UTC).isoformat(),
            parameters=params,
            models_evaluated=summaries,
            phenomenon_surfaced=(
                "Constraint Stress Vulnerability: Unconstrained dynamics create physically impossible "
                "phantom capacity under boundary load, failing invariant gates."
            ),
        )
