"""RuleShift benchmark family: probes model adaptation when organizational rules or tax regimes shift mid-horizon.

Tests the Lucas Critique in an organizational simulation context: ungrounded empirical
models fitted under one regulatory regime fail when governance rules change.
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


class StructuralRuleAwareDynamics(DynamicsModel):
    """Ground truth dynamics that inspects active regulatory rules in world state."""

    def __init__(self, name: str = "StructuralRuleAwareDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        rev = 0.0
        for act in actions:
            if act.type == "earn_revenue":
                rev += float(act.parameters.get("value", 0.0))

        # Check whether rule 'excise_tax_25' is active in state
        tax_rate = 0.25 if state.active_rules.get("excise_tax_25", False) else 0.0
        net_cash = rev * (1.0 - tax_rate)

        next_state = state.update_resource("cash", delta=net_cash, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"revenue": rev, "tax_rate": tax_rate, "net_cash": net_cash},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class RuleShiftBenchmark(Benchmark):
    """Evaluates error spikes and adaptation failure when organizational rules shift."""

    def __init__(
        self,
        name: str = "RuleShift_RegulatoryTaxRegime",
        seed: int = 42,
        n_samples_per_regime: int = 50,
        include_neural: bool = False,
    ) -> None:
        self._name = name
        self.seed = seed
        self.n_samples_per_regime = n_samples_per_regime
        self.include_neural = include_neural

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> BenchmarkFamily:
        return BenchmarkFamily.RULE_SHIFT

    def _generate_datasets(self) -> tuple[TransitionDataset, TransitionDataset]:
        rng = np.random.default_rng(self.seed)
        gt = StructuralRuleAwareDynamics()

        # Regime 1: Pre-shift (No tax)
        pre_samples: list[TransitionSample] = []
        for i in range(self.n_samples_per_regime):
            cash = float(rng.uniform(100.0, 500.0))
            s1 = WorldState(
                resources={
                    "cash": Resource(id="cash", current=cash, min_value=0.0, max_value=2000.0)
                },
                active_rules={"excise_tax_25": False},
            )
            val = float(rng.uniform(10.0, 50.0))
            act = Action(id=f"rev_{i}", type="earn_revenue", parameters={"value": val})
            tr = gt.transition(s1, [act], [], rng)
            pre_samples.append(TransitionSample(state=s1, actions=(act,), next_state=tr.next_state))

        # Regime 2: Post-shift (25% tax rule enacted)
        post_samples: list[TransitionSample] = []
        for i in range(self.n_samples_per_regime):
            cash = float(rng.uniform(100.0, 500.0))
            s1 = WorldState(
                resources={
                    "cash": Resource(id="cash", current=cash, min_value=0.0, max_value=2000.0)
                },
                active_rules={"excise_tax_25": True},
            )
            val = float(rng.uniform(10.0, 50.0))
            act = Action(id=f"rev_post_{i}", type="earn_revenue", parameters={"value": val})
            tr = gt.transition(s1, [act], [], rng)
            post_samples.append(
                TransitionSample(state=s1, actions=(act,), next_state=tr.next_state)
            )

        return TransitionDataset(pre_samples), TransitionDataset(post_samples)

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        pre_ds, post_ds = self._generate_datasets()
        gt = StructuralRuleAwareDynamics()

        eval_models: list[DynamicsModel] = list(models or [])
        if not eval_models:
            # Fit empirical model on pre-shift data only
            linear = LinearResidualDynamics(target_resource="cash", action_type="earn_revenue")
            linear.fit(pre_ds)
            eval_models = [gt, linear]

            if self.include_neural:
                try:
                    from ewm_engine.experimental.dynamics_torch import TorchNeuralResidualDynamics

                    torch_model = TorchNeuralResidualDynamics(
                        target_resources=["cash"],
                        action_types=["earn_revenue"],
                        hidden_dims=(32, 32),
                        epochs=60,
                        seed=self.seed,
                    )
                    torch_model.fit(pre_ds)
                    eval_models.append(torch_model)
                except Exception:
                    pass

        summaries: list[ModelEvaluationSummary] = []
        for m in eval_models:
            m_name = getattr(m, "name", type(m).__name__)
            pre_err, _ = evaluate_one_step(m, pre_ds, seed=self.seed)
            post_err, post_inv = evaluate_one_step(m, post_ds, seed=self.seed)

            degradation = post_err.mae - pre_err.mae
            ratio = post_err.mae / max(1e-6, pre_err.mae)

            summaries.append(
                ModelEvaluationSummary(
                    model_name=m_name,
                    metrics={
                        "pre_shift_mae": pre_err.mae,
                        "post_shift_mae": post_err.mae,
                        "rule_shift_degradation": degradation,
                        "degradation_ratio": ratio,
                    },
                    invariants_passed=post_inv.passed,
                    invariant_violations=post_inv.violations_count,
                    divergence_or_gap=round(degradation, 6),
                    diagnostics={"pre_mae": pre_err.mae, "post_mae": post_err.mae},
                )
            )

        params = {
            "n_samples_per_regime": self.n_samples_per_regime,
            "rule_name": "excise_tax_25",
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
                "Rule Shift Vulnerability: Empirical dynamics models trained under static governance "
                "experience immediate predictive failure when organizational rules change."
            ),
        )
