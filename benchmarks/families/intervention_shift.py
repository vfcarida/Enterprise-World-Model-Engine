"""InterventionShift benchmark family: probes predictive degradation under out-of-distribution interventions.

Measures whether a dynamics model can simulate structural policy interventions outside
the support of observational demonstration data.
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
from ewm_engine.dynamics.learned import LinearResidualDynamics, TransitionDataset, TransitionSample
from ewm_engine.experimental.dynamics_eval import evaluate_one_step
from ewm_engine.provenance.evidence import EvidenceLevel


class GroundTruthCongestedDynamics(DynamicsModel):
    """Ground truth dynamics with non-linear saturation / logistical congestion at high volume."""

    def __init__(self, name: str = "GroundTruthCongestedDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        restock = 0.0
        for act in actions:
            if act.type == "restock":
                restock += float(act.parameters.get("value", 0.0))

        # True physics: linear at small volume, logarithmic saturation at large volume
        if restock <= 15.0:
            effective_addition = restock
        else:
            effective_addition = 15.0 + (5.0 * float(np.log(1.0 + restock - 15.0)))

        next_state = state.update_resource("inventory", delta=effective_addition, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"added": effective_addition},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class InterventionShiftBenchmark(Benchmark):
    """Synthetic benchmark evaluating model degradation under held-out action distributions."""

    def __init__(
        self,
        name: str = "InterventionShift_SupplyRestock",
        seed: int = 42,
        n_in_samples: int = 60,
        n_out_samples: int = 60,
        include_neural: bool = False,
    ) -> None:
        self._name = name
        self.seed = seed
        self.n_in_samples = n_in_samples
        self.n_out_samples = n_out_samples
        self.include_neural = include_neural

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> BenchmarkFamily:
        return BenchmarkFamily.INTERVENTION_SHIFT

    def _generate_datasets(self) -> tuple[TransitionDataset, TransitionDataset]:
        rng = np.random.default_rng(self.seed)
        gt = GroundTruthCongestedDynamics()

        # In-distribution: restock in [2.0, 12.0]
        in_samples: list[TransitionSample] = []
        for i in range(self.n_in_samples):
            inv = float(rng.uniform(50.0, 150.0))
            s1 = WorldState(
                resources={
                    "inventory": Resource(
                        id="inventory", current=inv, min_value=0.0, max_value=500.0
                    )
                }
            )
            act_val = float(rng.uniform(2.0, 12.0))
            act = Action(id=f"in_act_{i}", type="restock", parameters={"value": act_val})
            tr = gt.transition(s1, [act], [], rng)
            in_samples.append(TransitionSample(state=s1, actions=(act,), next_state=tr.next_state))

        # Held-out out-of-distribution interventions: restock in [50.0, 100.0]
        out_samples: list[TransitionSample] = []
        for i in range(self.n_out_samples):
            inv = float(rng.uniform(50.0, 150.0))
            s1 = WorldState(
                resources={
                    "inventory": Resource(
                        id="inventory", current=inv, min_value=0.0, max_value=500.0
                    )
                }
            )
            act_val = float(rng.uniform(50.0, 100.0))
            act = Action(id=f"out_act_{i}", type="restock", parameters={"value": act_val})
            tr = gt.transition(s1, [act], [], rng)
            out_samples.append(TransitionSample(state=s1, actions=(act,), next_state=tr.next_state))

        return TransitionDataset(in_samples), TransitionDataset(out_samples)

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        in_ds, out_ds = self._generate_datasets()
        gt = GroundTruthCongestedDynamics()

        eval_models: list[DynamicsModel] = list(models or [])
        if not eval_models:
            # Default comparison set: Ground Truth + Linear Residual + Neural (if available)
            linear = LinearResidualDynamics(target_resource="inventory", action_type="restock")
            linear.fit(in_ds)
            eval_models = [gt, linear]

            if self.include_neural:
                try:
                    from ewm_engine.experimental.dynamics_torch import TorchNeuralResidualDynamics

                    torch_model = TorchNeuralResidualDynamics(
                        target_resources=["inventory"],
                        action_types=["restock"],
                        hidden_dims=(32, 32),
                        epochs=60,
                        seed=self.seed,
                    )
                    torch_model.fit(in_ds)
                    eval_models.append(torch_model)
                except Exception:
                    pass

        summaries: list[ModelEvaluationSummary] = []
        for m in eval_models:
            m_name = getattr(m, "name", type(m).__name__)
            in_err, in_inv = evaluate_one_step(m, in_ds, seed=self.seed)
            out_err, out_inv = evaluate_one_step(m, out_ds, seed=self.seed)

            gap = out_err.mae - in_err.mae
            gap_ratio = out_err.mae / max(1e-6, in_err.mae)

            summaries.append(
                ModelEvaluationSummary(
                    model_name=m_name,
                    metrics={
                        "in_distribution_mae": in_err.mae,
                        "held_out_intervention_mae": out_err.mae,
                        "interventional_gap": gap,
                        "gap_ratio": gap_ratio,
                    },
                    invariants_passed=out_inv.passed,
                    invariant_violations=out_inv.violations_count,
                    divergence_or_gap=round(gap, 6),
                    diagnostics={
                        "in_bounds_violations": in_inv.bounds_violations,
                        "out_bounds_violations": out_inv.bounds_violations,
                    },
                )
            )

        params = {
            "n_in_samples": self.n_in_samples,
            "n_out_samples": self.n_out_samples,
            "action_type": "restock",
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
                "Interventional Generalization Gap: Models fitted on in-distribution observations "
                "degrade significantly when subjected to structural out-of-distribution interventions."
            ),
        )
