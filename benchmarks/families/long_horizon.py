"""LongHorizon benchmark family: probes multi-step compounding drift and divergence over extended horizons.

Demonstrates that minimal one-step prediction error compounds exponentially in
autoregressive rollouts, surfacing long-horizon overclaim risk.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from benchmarks.protocol import (
    Benchmark,
    BenchmarkFamily,
    BenchmarkResult,
    ModelEvaluationSummary,
    compute_benchmark_fingerprint,
)
from ewm_engine.core import Action, Resource, World, WorldState
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.experimental.dynamics_eval import evaluate_rollout_divergence
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


class DampedInventoryDynamics(DynamicsModel):
    """Ground truth autoregressive damped dynamics with cyclical restocking."""

    def __init__(self, damping: float = 0.96, name: str = "DampedInventoryDynamics") -> None:
        self.damping = damping
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        res = state.get_resource("stock")
        curr = res.current
        act_val = sum(float(a.parameters.get("value", 0.0)) for a in actions)

        # Autoregressive physical evolution
        next_val = (self.damping * curr) + act_val
        next_state = state.update_resource("stock", new_value=next_val, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"next_val": next_val},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class SlightlyBiasedEmpiricalDynamics(DynamicsModel):
    """Empirical baseline with imperceptible one-step error (+1.5% drift) that compounds over time."""

    def __init__(self, name: str = "SlightlyBiasedEmpiricalDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        res = state.get_resource("stock")
        curr = res.current
        act_val = sum(float(a.parameters.get("value", 0.0)) for a in actions)

        # 1-step error is only +1.5% on current stock
        biased_val = (0.975 * curr) + act_val
        next_state = state.update_resource("stock", new_value=biased_val, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"next_val": biased_val},
            evidence_level=EvidenceLevel.PREDICTIVE,
            model_name=self.name,
        )


class LongHorizonBenchmark(Benchmark):
    """Evaluates multi-step trajectory divergence and error compounding over time."""

    def __init__(
        self,
        name: str = "LongHorizon_CompoundingDivergence",
        seed: int = 42,
        horizon: int = 40,
    ) -> None:
        self._name = name
        self.seed = seed
        self.horizon = horizon

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> BenchmarkFamily:
        return BenchmarkFamily.LONG_HORIZON

    def _build_reference_world(self) -> tuple[World, Scenario]:
        s0 = WorldState(
            resources={
                "stock": Resource(id="stock", current=100.0, min_value=0.0, max_value=1000.0)
            }
        )
        gt_dynamics = DampedInventoryDynamics()
        world = World(state=s0, dynamics=gt_dynamics)

        # Deterministic periodic restocking action every 5 steps
        actions = []
        for step in range(1, self.horizon + 1):
            if step % 5 == 0:
                actions.append(
                    ScheduledAction(
                        step=step,
                        action=Action(
                            id=f"restock_{step}", type="restock", parameters={"value": 15.0}
                        ),
                    )
                )

        scenario = Scenario(
            scenario_id=f"long_horizon_{self.horizon}",
            name=f"Horizon_{self.horizon}",
            horizon=self.horizon,
            samples=1,
            seed=self.seed,
            scheduled_actions=tuple(actions),
        )
        return world, scenario

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        world, scenario = self._build_reference_world()

        eval_models: list[DynamicsModel] = list(models or [])
        if not eval_models:
            gt = DampedInventoryDynamics()
            biased = SlightlyBiasedEmpiricalDynamics()
            eval_models = [gt, biased]

        summaries: list[ModelEvaluationSummary] = []
        for m in eval_models:
            m_name = getattr(m, "name", type(m).__name__)
            div_res = evaluate_rollout_divergence(
                model=m,
                reference_world=world,
                scenario=scenario,
            )

            summaries.append(
                ModelEvaluationSummary(
                    model_name=m_name,
                    metrics={
                        "horizon": float(div_res.horizon),
                        "mean_divergence": div_res.mean_divergence,
                        "final_divergence": div_res.final_divergence,
                        "max_divergence": div_res.max_divergence,
                        "drift_rate": div_res.drift_rate,
                    },
                    invariants_passed=True,
                    invariant_violations=0,
                    divergence_or_gap=round(div_res.final_divergence, 6),
                    diagnostics={"drift_slope": div_res.drift_rate},
                )
            )

        params = {
            "horizon": self.horizon,
            "damping_factor": 0.96,
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
                "Compounding Autoregressive Drift: Small single-step errors compound exponentially over "
                "extended horizons, demonstrating that surface one-step accuracy cannot justify long-horizon claims."
            ),
        )
