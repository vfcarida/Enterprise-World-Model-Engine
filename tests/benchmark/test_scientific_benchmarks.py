"""Tests for the scientific benchmark families (research instrument)."""

from __future__ import annotations

import json

import pytest

from benchmarks.families.constraint_stress import (
    ClampedCapacityDynamics,
    ConstraintStressBenchmark,
    UnclampedPhantomDynamics,
)
from benchmarks.families.intervention_shift import InterventionShiftBenchmark
from benchmarks.families.long_horizon import LongHorizonBenchmark
from benchmarks.families.multi_agent_cascade import MultiAgentCascadeBenchmark
from benchmarks.families.rule_shift import RuleShiftBenchmark
from benchmarks.suite import run_scientific_benchmark_suite


@pytest.mark.benchmark
def test_intervention_shift_benchmark_reproducibility_and_gap() -> None:
    """InterventionShift benchmark must reproduce deterministically and detect interventional gap."""
    bench = InterventionShiftBenchmark(seed=42, n_in_samples=20, n_out_samples=20)
    res1 = bench.run()
    res2 = bench.run()

    # Reproducibility assertions
    assert res1.provenance_fingerprint == res2.provenance_fingerprint
    assert len(res1.models_evaluated) == len(res2.models_evaluated)
    assert len(res1.models_evaluated) >= 2

    # Phenomenon assertion: LinearResidualDynamics has positive gap
    linear_eval = next(m for m in res1.models_evaluated if m.model_name == "LinearResidualDynamics")
    assert linear_eval.divergence_or_gap > 0.0
    assert linear_eval.metrics["interventional_gap"] > 0.0

    # Ground truth has zero gap
    gt_eval = next(
        m for m in res1.models_evaluated if m.model_name == "GroundTruthCongestedDynamics"
    )
    assert gt_eval.divergence_or_gap == 0.0


@pytest.mark.benchmark
def test_rule_shift_benchmark_adaptation_degradation() -> None:
    """RuleShift benchmark must detect predictive failure when regulatory rules shift."""
    bench = RuleShiftBenchmark(seed=42, n_samples_per_regime=20)
    res = bench.run()

    assert res.family.value == "RuleShift"
    assert len(res.models_evaluated) >= 2

    # Structural model adapts (error = 0)
    struct_eval = next(
        m for m in res.models_evaluated if m.model_name == "StructuralRuleAwareDynamics"
    )
    assert struct_eval.metrics["post_shift_mae"] == 0.0

    # Linear model experiences post-shift degradation
    linear_eval = next(m for m in res.models_evaluated if m.model_name == "LinearResidualDynamics")
    assert linear_eval.metrics["post_shift_mae"] > 0.0
    assert linear_eval.metrics["rule_shift_degradation"] > 0.0


@pytest.mark.benchmark
def test_constraint_stress_benchmark_invariants_and_violation_rate() -> None:
    """ConstraintStress benchmark must flag unclamped models under boundary stress."""
    bench = ConstraintStressBenchmark(seed=42, n_samples=20)
    clamped = ClampedCapacityDynamics()
    unclamped = UnclampedPhantomDynamics()

    res = bench.run(models=[clamped, unclamped])

    clamped_eval = next(
        m for m in res.models_evaluated if m.model_name == "ClampedCapacityDynamics"
    )
    unclamped_eval = next(
        m for m in res.models_evaluated if m.model_name == "UnclampedPhantomDynamics"
    )

    assert clamped_eval.invariants_passed
    assert clamped_eval.invariant_violations == 0
    assert clamped_eval.metrics["stress_violation_rate"] == 0.0

    assert not unclamped_eval.invariants_passed
    assert unclamped_eval.invariant_violations > 0
    assert unclamped_eval.metrics["stress_violation_rate"] > 0.0


@pytest.mark.benchmark
def test_long_horizon_benchmark_compounding_drift() -> None:
    """LongHorizon benchmark must measure compounding multi-step error drift."""
    bench = LongHorizonBenchmark(seed=42, horizon=20)
    res = bench.run()

    gt_eval = next(m for m in res.models_evaluated if m.model_name == "DampedInventoryDynamics")
    biased_eval = next(
        m for m in res.models_evaluated if m.model_name == "SlightlyBiasedEmpiricalDynamics"
    )

    assert gt_eval.metrics["drift_rate"] == 0.0
    assert biased_eval.metrics["drift_rate"] > 0.0
    assert biased_eval.metrics["final_divergence"] > 0.0


@pytest.mark.benchmark
def test_multi_agent_cascade_benchmark_systemic_propagation() -> None:
    """MultiAgentCascade benchmark must capture delayed downstream feedback and stockouts."""
    bench = MultiAgentCascadeBenchmark(seed=42, horizon=6)
    res = bench.run()

    coupled_eval = next(
        m for m in res.models_evaluated if m.model_name == "CascadeSupplyChainDynamics"
    )
    decoupled_eval = next(
        m for m in res.models_evaluated if m.model_name == "DecoupledMyopicDynamics"
    )

    assert coupled_eval.metrics["final_unserved_demand"] > 0.0
    assert coupled_eval.metrics["cascade_propagation_delay_steps"] < 6.0
    # Decoupled model is blind to cascade and reports 0 unserved demand
    assert decoupled_eval.metrics["final_unserved_demand"] == 0.0


@pytest.mark.benchmark
def test_benchmark_suite_report_json_and_table_export() -> None:
    """BenchmarkSuiteReport must produce machine-readable JSON and formatted table without winner declaration."""
    report = run_scientific_benchmark_suite(seed=123, fast=True, include_neural=False)

    assert len(report.results) == 5
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["suite_name"] == "EWM_Scientific_Benchmark_Suite"
    assert len(parsed["results"]) == 5

    # Check disclaimer in JSON and table
    assert "EPISTEMIC DISCLAIMER" in report.epistemic_disclaimer
    text_table = report.to_text_table()
    assert "EPISTEMIC DISCLAIMER" in text_table
    assert "InterventionShift" in text_table
    assert "RuleShift" in text_table
    assert "ConstraintStress" in text_table
    assert "LongHorizon" in text_table
    assert "MultiAgentCascade" in text_table

    md_table = report.to_markdown_table()
    assert "| Benchmark Family | Model Name |" in md_table


@pytest.mark.benchmark
def test_scientific_benchmark_with_neural_baseline() -> None:
    """Benchmark suite can opt-in to evaluating PyTorch neural baseline when requested."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch extra not installed")

    bench = InterventionShiftBenchmark(
        seed=42, n_in_samples=15, n_out_samples=15, include_neural=True
    )
    res = bench.run()
    model_names = [m.model_name for m in res.models_evaluated]
    assert "TorchNeuralResidualDynamics" in model_names
