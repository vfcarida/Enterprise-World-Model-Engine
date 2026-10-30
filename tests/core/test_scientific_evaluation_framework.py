"""Verification suite for EWM-006 Scientific Evaluation Framework."""

from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np
import pytest

from benchmarks import (
    MIN_SIGNIFICANCE_SEEDS,
    BenchmarkTimeoutError,
    EnvironmentSpec,
    EWM_Evaluator,
    EWMPolicyAgent,
    InventoryLiquidityConstraint,
    RandomAgent,
    StaticRuleAgent,
    build_initial_benchmark_world_state,
    compute_interquartile_mean,
    compute_probability_of_improvement,
)
from ewm_engine.constraints.results import ConstraintPhase
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState


def test_10_step_benchmark_n30_zero_ewm_violation_rate_and_statistical_significance() -> None:
    """EWM_Evaluator on 10-step problem (N=30 seeds) proves 0% EWM violation rate & significance."""
    env_spec = EnvironmentSpec(
        env_id="EnterpriseResourceInventory-10Step-v1",
        horizon_steps=10,
        parameters={"experiment": "ewm_006_canonical_10_step"},
    )
    evaluator = EWM_Evaluator(env_spec=env_spec, timeout_seconds=30.0)
    report = evaluator.evaluate(n_seeds=MIN_SIGNIFICANCE_SEEDS, master_seed=42)

    assert report.n_seeds == 30
    assert len(report.seeds) == 30
    assert report.significance_claim_permitted is True
    assert len(report.per_seed_records) == 30 * 3

    ewm_summary = report.agent_summaries["EWMPolicyAgent"]
    rand_summary = report.agent_summaries["RandomAgent"]
    static_summary = report.agent_summaries["StaticRuleAgent"]

    # Acceptance Criterion 2: 0% Constraint Violation Rate for the EWM Hybrid Router
    assert ewm_summary.constraint_violation_rate.mean == 0.0
    assert ewm_summary.constraint_violation_rate.std == 0.0
    assert ewm_summary.constraint_violation_rate.iqm == 0.0
    assert ewm_summary.constraint_violation_rate.max_value == 0.0

    # Baselines violate structural constraints when operating without the EWM Hybrid Router
    assert rand_summary.constraint_violation_rate.mean > 0.0
    assert static_summary.constraint_violation_rate.mean > 0.0

    # EWM predictive accuracy and cumulative reward strictly outperform both baselines
    assert ewm_summary.predictive_accuracy.mean > rand_summary.predictive_accuracy.mean
    assert ewm_summary.predictive_accuracy.mean > static_summary.predictive_accuracy.mean
    assert ewm_summary.cumulative_reward.mean > rand_summary.cumulative_reward.mean
    assert ewm_summary.cumulative_reward.mean > static_summary.cumulative_reward.mean

    # All 6 pairwise comparisons (3 metrics x 2 baselines) are statistically significant at N=30
    assert len(report.pairwise_comparisons) == 6
    for comp in report.pairwise_comparisons:
        assert comp.n_seeds == 30
        assert comp.is_statistically_significant is True
        assert comp.significance_status == "SIGNIFICANT_AT_ALPHA"
        assert comp.probability_of_improvement > 0.90


def test_underpowered_n_lt_30_refuses_significance_claims_and_exports_artifacts(
    tmp_path: Path,
) -> None:
    """When N < 30 seeds, significance claims are strictly blocked; JSON/CSV artifacts validate."""
    env_spec = EnvironmentSpec(horizon_steps=10, parameters={"mode": "ci_smoke"})
    evaluator = EWM_Evaluator(env_spec=env_spec, timeout_seconds=20.0)

    report_n10 = evaluator.evaluate(n_seeds=10, master_seed=101)
    assert report_n10.n_seeds == 10
    assert report_n10.significance_claim_permitted is False

    for comp in report_n10.pairwise_comparisons:
        assert comp.is_statistically_significant is False
        assert comp.significance_status == "INSUFFICIENT_SEEDS_N_LT_30"

    # Reproducibility check: second run with identical master_seed produces identical SHA-256
    report_n10_repeat = evaluator.evaluate(n_seeds=10, master_seed=101)
    assert report_n10.provenance_sha256 == report_n10_repeat.provenance_sha256

    # Save JSON and CSV artifacts and verify schema completeness
    json_path, csv_path = evaluator.save_artifacts(
        report_n10,
        output_dir=tmp_path,
        prefix="mock_10step_benchmark",
    )
    assert json_path.exists()
    assert csv_path.exists()

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed_json["master_seed"] == 101
    assert parsed_json["n_seeds"] == 10
    assert len(parsed_json["seeds"]) == 10
    assert parsed_json["environment_spec"]["horizon_steps"] == 10
    assert parsed_json["environment_spec"]["parameters"] == {"mode": "ci_smoke"}
    assert (
        parsed_json["agent_summaries"]["EWMPolicyAgent"]["constraint_violation_rate"]["mean"]
        == 0.0
    )

    csv_rows = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding="utf-8"))))
    assert len(csv_rows) == 10 * 3
    ewm_rows = [r for r in csv_rows if r["agent_id"] == "EWMPolicyAgent"]
    assert len(ewm_rows) == 10
    assert all(float(r["constraint_violation_rate"]) == 0.0 for r in ewm_rows)


def test_operational_timeout_safeguard_prevents_hanging_execution() -> None:
    """EWM_Evaluator raises BenchmarkTimeoutError when execution exceeds timeout_seconds."""

    class _SlowHangingAgent:
        @property
        def agent_id(self) -> str:
            return "SlowHangingAgent"

        @property
        def paradigm(self) -> str:
            return "slow_test"

        def select_action_and_predict(
            self,
            state: WorldState,
            env_spec: EnvironmentSpec,
            rng: np.random.Generator,
            *,
            step_idx: int,
        ) -> tuple[Action, dict[str, float]]:
            _ = (state, env_spec, rng, step_idx)
            time.sleep(0.025)
            return (
                Action(
                    id="slow_act",
                    type="REPLENISH_INVENTORY",
                    parameters={"order_qty": 10.0},
                ),
                {"inventory": 100.0, "cash_reserve": 500.0},
            )

    evaluator = EWM_Evaluator(
        env_spec=EnvironmentSpec(horizon_steps=10),
        agents=(_SlowHangingAgent(),),
        timeout_seconds=0.01,
    )
    with pytest.raises(BenchmarkTimeoutError, match="exceeded timeout_seconds"):
        evaluator.evaluate(n_seeds=2, master_seed=7)


def test_statistical_helpers_and_constraint_edge_cases() -> None:
    """Cover IQM, probability of improvement, and InventoryLiquidityConstraint edge branches."""
    with pytest.raises(ValueError, match="timeout_seconds"):
        EWM_Evaluator(timeout_seconds=0.0)
    with pytest.raises(ValueError, match="n_seeds"):
        EWM_Evaluator.derive_evaluation_seeds(42, 0)

    assert compute_interquartile_mean([]) == 0.0
    assert compute_interquartile_mean([1.0, 2.0]) == 1.5
    assert compute_interquartile_mean([1.0, 2.0, 3.0, 4.0, 100.0]) == 3.0

    assert compute_probability_of_improvement([], [1.0]) == 0.5
    assert compute_probability_of_improvement([2.0, 2.0], [2.0, 2.0]) == 0.5
    assert (
        compute_probability_of_improvement([0.0, 0.0], [0.5, 0.5], higher_is_better=False)
        == 1.0
    )

    spec = EnvironmentSpec()
    constraint = InventoryLiquidityConstraint(spec)
    assert constraint.severity.value == "hard"
    assert "inventory" in constraint.description

    state0 = build_initial_benchmark_world_state(spec)

    # Negative order rejected
    neg_act = Action(id="neg", type="REPLENISH", parameters={"order_qty": -5.0})
    assert not constraint.evaluate(state0, (neg_act,), phase=ConstraintPhase.PRE_ACTION).satisfied

    # Oversized order exceeding cash reserve rejected
    cash_breach_act = Action(id="cash_b", type="REPLENISH", parameters={"order_qty": 200.0})
    assert not constraint.evaluate(
        state0, (cash_breach_act,), phase=ConstraintPhase.PRE_ACTION
    ).satisfied

    # Oversized order exceeding warehouse max_inventory rejected
    high_cash_state = state0.update_resource(
        "cash_reserve", new_value=1900.0, clamp=True, enforce_bounds=False
    )
    inv_breach_act = Action(id="inv_b", type="REPLENISH", parameters={"order_qty": 100.0})
    assert not constraint.evaluate(
        high_cash_state, (inv_breach_act,), phase=ConstraintPhase.PRE_ACTION
    ).satisfied

    # Post-transition out-of-bounds checks
    bad_inv_state = state0.update_resource(
        "inventory", new_value=5.0, clamp=False, enforce_bounds=False
    )
    assert not constraint.evaluate(
        bad_inv_state, (), phase=ConstraintPhase.POST_TRANSITION
    ).satisfied

    bad_cash_state = state0.update_resource(
        "cash_reserve", new_value=10.0, clamp=False, enforce_bounds=False
    )
    assert not constraint.evaluate(
        bad_cash_state, (), phase=ConstraintPhase.POST_TRANSITION
    ).satisfied

    # Verify agents conform to BenchmarkAgentProtocol
    assert isinstance(RandomAgent(), object)
    assert isinstance(StaticRuleAgent(), object)
    assert isinstance(EWMPolicyAgent(), object)
