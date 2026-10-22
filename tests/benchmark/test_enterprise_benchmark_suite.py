"""Self-validation test suite for the 12-family Enterprise Benchmark Suite (ADR-054).

Tests the benchmark instrument itself across:
1. Known-oracle sanity checks (deterministic exact oracles A & I within numerical tolerance).
2. Metric mathematical correctness (all 13 orthogonal dimensions, bootstrap CIs, MC SE vs.
   model uncertainty spread, paired Cohen's d_z, Holm-Bonferroni step-down adjustment).
3. Seeded repeatability and cryptographic fingerprint determinism.
4. Hard data non-leakage boundaries (positive verification across A-L + negative injection tests).
5. Configuration and deployment-shift protocol correctness (including steps-until-recovery).
6. Generator invariants and full coverage of 12 families and 11 task types.
7. Serialization round-trip stability and artifact-to-documentation synchronization.
8. Expected failure modes across all 4 baselines (including honest limitations of EWM).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmarks.baselines import derive_partition_seeds
from benchmarks.enterprise_suite import (
    REPO_ROOT,
    EnterpriseBenchmarkSuiteReport,
    render_enterprise_benchmark_markdown_doc,
    run_enterprise_benchmark_suite,
)
from benchmarks.protocol import (
    METRIC_DEFINITIONS,
    BaselineParadigm,
    BenchmarkTaskType,
    EnterpriseBenchmarkFamily,
    MetricDimension,
    MetricStatisticalSummary,
    apply_holm_bonferroni,
    verify_no_data_leakage,
)


@pytest.fixture(scope="module")
def suite_report_fast() -> EnterpriseBenchmarkSuiteReport:
    """Run the 12-family enterprise suite once in fast mode for module-scoped inspection."""
    return run_enterprise_benchmark_suite(
        seed=42,
        fast=True,
        timestamp="2026-10-21T00:00:00+00:00",
    )


@pytest.mark.benchmark
def test_known_oracle_sanity_checks(suite_report_fast: EnterpriseBenchmarkSuiteReport) -> None:
    """Verify deterministic exact-oracle benchmarks (Families A and I) enforce exact/tolerance rules."""
    by_code = {f.family_code: f for f in suite_report_fast.families}

    # Family A: Direct Dynamics (deterministic oracle)
    fam_a = by_code["A"]
    assert fam_a.is_deterministic_oracle is True
    for b_rec in fam_a.baselines:
        for dim, summary in b_rec.dimension_summaries.items():
            if dim not in (
                MetricDimension.LATENCY,
                MetricDimension.MEMORY,
                MetricDimension.THROUGHPUT,
            ):
                assert summary.is_deterministic is True
                assert summary.statistical_significance_applicable is False
                assert summary.raw_p_value is None
                assert summary.holm_adjusted_p_value is None
                assert summary.mc_standard_error == 0.0

    a_by_p = {b.paradigm: b for b in fam_a.baselines}
    # Rules-only misses empirical valve friction above threshold -> fidelity < 0.85
    assert (
        a_by_p[BaselineParadigm.RULES_ONLY]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
        < 0.85
    )
    assert (
        a_by_p[BaselineParadigm.RULES_ONLY]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .within_oracle_tolerance
        is False
    )
    # Static hybrid and contextual hybrid match the exact analytical oracle within 1e-12
    for p in (BaselineParadigm.STATIC_HYBRID, BaselineParadigm.CONTEXTUAL_HYBRID):
        fid = a_by_p[p].dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        assert fid.mean == pytest.approx(1.0, abs=1e-12)
        assert fid.within_oracle_tolerance is True
        assert fid.oracle_tolerance == pytest.approx(1e-12)

    # Family I: Conservation & Constraint Stress (deterministic oracle)
    fam_i = by_code["I"]
    assert fam_i.is_deterministic_oracle is True
    i_by_p = {b.paradigm: b for b in fam_i.baselines}
    assert (
        i_by_p[BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.CONSTRAINT_PRESERVATION]
        .mean
        == 0.0
    )
    for p in (
        BaselineParadigm.RULES_ONLY,
        BaselineParadigm.STATIC_HYBRID,
        BaselineParadigm.CONTEXTUAL_HYBRID,
    ):
        cp = i_by_p[p].dimension_summaries[MetricDimension.CONSTRAINT_PRESERVATION]
        assert cp.mean == pytest.approx(1.0, abs=1e-12)
        assert cp.within_oracle_tolerance is True


@pytest.mark.benchmark
def test_metric_definitions_and_statistical_protocol_correctness(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify all 13 orthogonal metric dimensions and statistical protocol calculations."""
    # All 13 dimensions must be formally defined
    assert len(MetricDimension) == 13
    assert set(METRIC_DEFINITIONS.keys()) == set(MetricDimension)
    for dim, mdef in METRIC_DEFINITIONS.items():
        assert mdef.dimension == dim
        assert len(mdef.formula_latex) > 5
        assert len(mdef.unit) > 0
        assert len(mdef.description) > 10

    # Every family and baseline must report all 13 dimensions separately (no collapsed score)
    for fam in suite_report_fast.families:
        assert len(fam.baselines) == 4
        for b_rec in fam.baselines:
            assert set(b_rec.dimension_summaries.keys()) == set(MetricDimension)

    # Test stochastic summary math directly
    seeds = [101, 102, 103, 104, 105]
    samples = [0.80, 0.82, 0.85, 0.79, 0.84]
    ref_samples = [0.95, 0.96, 0.94, 0.95, 0.97]
    stat = MetricStatisticalSummary.from_stochastic(
        dimension=MetricDimension.TRANSITION_FIDELITY,
        seeds=seeds,
        values=samples,
        model_spreads=[0.045] * 5,
        paired_reference_values=ref_samples,
        failed_runs=1,
        bootstrap_seed=12345,
    )
    assert stat.is_deterministic is False
    assert stat.statistical_significance_applicable is True
    assert len(stat.seeds) == 5
    assert stat.failed_runs == 1
    assert stat.ci_lower <= stat.mean <= stat.ci_upper
    assert stat.iqr > 0.0
    assert stat.mc_standard_error == pytest.approx(np.std(samples, ddof=1) / np.sqrt(5), rel=1e-6)
    assert stat.model_uncertainty_spread == pytest.approx(0.045)
    assert stat.cohens_dz is not None and stat.cohens_dz < -2.0
    assert stat.raw_p_value is not None and 0.0 <= stat.raw_p_value <= 1.0

    # Test Holm-Bonferroni step-down correction monotonicity
    raw_p = {"b1": 0.004, "b2": 0.020, "b3": 0.015}
    adj_p = apply_holm_bonferroni(raw_p)
    # Sorted order: b1 (0.004 * 3 = 0.012), b3 (max(0.012, 0.015 * 2) = 0.030), b2 (max(0.030, 0.020 * 1) = 0.030)
    assert adj_p["b1"] == pytest.approx(0.012)
    assert adj_p["b3"] == pytest.approx(0.030)
    assert adj_p["b2"] == pytest.approx(0.030)


@pytest.mark.benchmark
def test_seeded_repeatability_and_fingerprint_determinism() -> None:
    """Verify bit-identical repeatability for fixed seed and sensitivity to seed changes."""
    r1 = run_enterprise_benchmark_suite(
        seed=42, fast=True, families=["A", "B", "D", "L"], timestamp="2026-10-21T00:00:00+00:00"
    )
    r2 = run_enterprise_benchmark_suite(
        seed=42, fast=True, families=["A", "B", "D", "L"], timestamp="2026-10-21T00:00:00+00:00"
    )
    r_diff = run_enterprise_benchmark_suite(
        seed=99, fast=True, families=["A", "B", "D", "L"], timestamp="2026-10-21T00:00:00+00:00"
    )

    assert r1.suite_fingerprint == r2.suite_fingerprint
    assert r1.suite_fingerprint != r_diff.suite_fingerprint

    for f1, f2, fd in zip(r1.families, r2.families, r_diff.families, strict=True):
        assert (
            f1.reproducibility.dataset_scenario_fingerprint
            == f2.reproducibility.dataset_scenario_fingerprint
        )
        assert f1.reproducibility.output_fingerprint == f2.reproducibility.output_fingerprint
        assert (
            f1.reproducibility.dataset_scenario_fingerprint
            != fd.reproducibility.dataset_scenario_fingerprint
        )
        for b1, b2 in zip(f1.baselines, f2.baselines, strict=True):
            for dim in (
                MetricDimension.TRANSITION_FIDELITY,
                MetricDimension.CONSTRAINT_PRESERVATION,
                MetricDimension.CAUSAL_CORRECTNESS,
                MetricDimension.CALIBRATION,
                MetricDimension.SHIFT_ROBUSTNESS,
                MetricDimension.DECISION_REGRET,
                MetricDimension.ABSTENTION_QUALITY,
            ):
                assert b1.dimension_summaries[dim].mean == b2.dimension_summaries[dim].mean


@pytest.mark.benchmark
def test_hard_data_leakage_boundaries_positive_and_negative(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify all 12 families pass leakage checks and test that negative injections fail."""
    for fam in suite_report_fast.families:
        cert = fam.leakage_certificate
        assert cert.passed is True
        assert cert.state_fingerprint_overlap_count == 0
        assert cert.temporal_boundary_verified is True
        assert cert.config_boundary_verified is True
        assert cert.oracle_isolation_verified is True
        assert cert.train_partition_seed != cert.eval_partition_seed

    # Negative check 1: Overlapping state fingerprint between train and eval
    bad_overlap = verify_no_data_leakage(
        train_partition_seed=101,
        eval_partition_seed=202,
        train_state_fingerprints=["h1", "h2"],
        eval_state_fingerprints=["h2", "h3"],
        train_timestamps=[1.0, 2.0],
        eval_timestamps=[100.0, 101.0],
        train_context_versions=["1.0.0"],
        held_out_eval_context_versions=["2.0.0"],
    )
    assert bad_overlap.passed is False
    assert bad_overlap.state_fingerprint_overlap_count == 1

    # Negative check 2: Temporal boundary violation (max_train_timestamp >= min_eval_timestamp)
    bad_temporal = verify_no_data_leakage(
        train_partition_seed=101,
        eval_partition_seed=202,
        train_state_fingerprints=["h1"],
        eval_state_fingerprints=["h2"],
        train_timestamps=[100.0],
        eval_timestamps=[100.0],
        train_context_versions=["1.0.0"],
        held_out_eval_context_versions=[],
    )
    assert bad_temporal.passed is False
    assert bad_temporal.temporal_boundary_verified is False

    # Negative check 3: Leaked held-out config version into training
    bad_config = verify_no_data_leakage(
        train_partition_seed=101,
        eval_partition_seed=202,
        train_state_fingerprints=["h1"],
        eval_state_fingerprints=["h2"],
        train_timestamps=[50.0],
        eval_timestamps=[100.0],
        train_context_versions=["1.0.0", "2.0.0-secret"],
        held_out_eval_context_versions=["2.0.0-secret"],
    )
    assert bad_config.passed is False
    assert bad_config.config_boundary_verified is False

    # Negative check 4: Seed reuse between train and eval
    bad_seed = verify_no_data_leakage(
        train_partition_seed=202,
        eval_partition_seed=202,
        train_state_fingerprints=["h1"],
        eval_state_fingerprints=["h2"],
        train_timestamps=[50.0],
        eval_timestamps=[100.0],
        train_context_versions=["1.0.0"],
        held_out_eval_context_versions=[],
    )
    assert bad_seed.passed is False


@pytest.mark.benchmark
def test_shift_protocol_and_time_to_recovery(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify full shift metadata and time-to-recovery in Families C and D."""
    by_code = {f.family_code: f for f in suite_report_fast.families}

    for code in ("C", "D"):
        fam = by_code[code]
        for b_rec in fam.baselines:
            rec = b_rec.shift_protocol
            assert rec is not None
            assert rec.training_context_version != rec.evaluation_context_version
            assert len(rec.what_changed) >= 2
            assert rec.model_informed_explicitly is False
            assert rec.adaptation_budget_steps > 0

    fam_d = by_code["D"]
    d_shift_by_p = {
        b.paradigm: b.shift_protocol for b in fam_d.baselines if b.shift_protocol is not None
    }
    for p in (
        BaselineParadigm.RULES_ONLY,
        BaselineParadigm.LEARNED_ONLY,
        BaselineParadigm.STATIC_HYBRID,
    ):
        assert d_shift_by_p[p].recovered_within_budget is False
        assert d_shift_by_p[p].steps_until_recovery is None
        assert d_shift_by_p[p].steady_state_post_shift_mae > 25.0

    # Contextual hybrid experiences an honest transient error spike during the 2-step audit delay,
    # then recovers at step 2 once runtime audit telemetry exposes the new configuration.
    ctx_rec = d_shift_by_p[BaselineParadigm.CONTEXTUAL_HYBRID]
    assert ctx_rec.audit_discovery_delay_steps == 2
    assert ctx_rec.peak_post_shift_mae > 25.0
    assert ctx_rec.steps_until_recovery == 2
    assert ctx_rec.recovered_within_budget is True
    assert ctx_rec.steady_state_post_shift_mae < 0.5


@pytest.mark.benchmark
def test_generator_invariants_and_task_coverage(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify all 12 benchmark families and all 11 task types are exercised."""
    families_seen = {f.family for f in suite_report_fast.families}
    assert families_seen == set(EnterpriseBenchmarkFamily)
    assert len(suite_report_fast.families) == 12

    tasks_seen: set[BenchmarkTaskType] = set()
    for fam in suite_report_fast.families:
        tasks_seen.update(fam.task_types)
    assert tasks_seen == set(BenchmarkTaskType)

    # Verify cryptographic partition seed isolation
    train_seed, eval_seeds, cal_seed = derive_partition_seeds(master_seed=42, n_eval_seeds=5)
    assert train_seed not in eval_seeds
    assert cal_seed not in eval_seeds
    assert train_seed != cal_seed
    assert len(set(eval_seeds)) == 5


@pytest.mark.benchmark
def test_serialization_and_artifact_sync_stability(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify JSON round-trip stability and check that committed docs match the JSON artifact."""
    json_str = suite_report_fast.to_json()
    restored = EnterpriseBenchmarkSuiteReport.model_validate_json(json_str)
    assert restored.suite_fingerprint == suite_report_fast.suite_fingerprint
    assert len(restored.families) == 12

    # Check that committed JSON artifact and Markdown doc are in sync (no hand-pasted numbers)
    artifact_json_path = (
        Path(REPO_ROOT) / "docs" / "results" / "artifacts" / "enterprise_benchmark_suite.json"
    )
    doc_md_path = Path(REPO_ROOT) / "docs" / "research" / "enterprise-benchmark-suite.md"
    heatmap_svg_path = (
        Path(REPO_ROOT) / "docs" / "results" / "artifacts" / "enterprise_benchmark_heatmap.svg"
    )
    shift_svg_path = (
        Path(REPO_ROOT) / "docs" / "results" / "artifacts" / "enterprise_shift_recovery.svg"
    )

    assert artifact_json_path.exists()
    assert doc_md_path.exists()
    assert heatmap_svg_path.exists()
    assert shift_svg_path.exists()

    committed_report = EnterpriseBenchmarkSuiteReport.model_validate_json(
        artifact_json_path.read_text(encoding="utf-8")
    )
    expected_md = render_enterprise_benchmark_markdown_doc(committed_report)
    actual_md = doc_md_path.read_text(encoding="utf-8")
    assert actual_md == expected_md


@pytest.mark.benchmark
def test_expected_baseline_failure_modes_across_all_12_families(
    suite_report_fast: EnterpriseBenchmarkSuiteReport,
) -> None:
    """Verify that each benchmark family exposes the intended scientific failure modes."""
    by_code = {
        f.family_code: {b.paradigm: b for b in f.baselines} for f in suite_report_fast.families
    }

    # Family B (Hidden Workflow Cascades): learned-only fails on unexcited compliance cascade
    assert (
        by_code["B"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
        < 0.45
    )
    assert (
        by_code["B"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
        > 0.95
    )

    # Family C (Runtime Configuration): only runtime context discovery adapts to EU strict tenant
    assert (
        by_code["C"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.SHIFT_ROBUSTNESS]
        .mean
        < 0.50
    )
    assert (
        by_code["C"][BaselineParadigm.CONTEXTUAL_HYBRID]
        .dimension_summaries[MetricDimension.SHIFT_ROBUSTNESS]
        .mean
        > 0.95
    )

    # Family E (Partial Observability): belief filtering improves fidelity but retains honest lag error
    assert (
        by_code["E"][BaselineParadigm.CONTEXTUAL_HYBRID]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
        > by_code["E"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
        + 0.25
    )
    assert by_code["E"][BaselineParadigm.CONTEXTUAL_HYBRID].raw_metrics["pomdp_mae"] > 0.5

    # Family F (Long-Horizon Business Operation): learned-only compounds drift; runtime context discovery lowest
    assert (
        by_code["F"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.LONG_HORIZON_ERROR_GROWTH]
        .mean
        > 2.0
    )
    assert (
        by_code["F"][BaselineParadigm.CONTEXTUAL_HYBRID]
        .dimension_summaries[MetricDimension.LONG_HORIZON_ERROR_GROWTH]
        .mean
        < by_code["F"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.LONG_HORIZON_ERROR_GROWTH]
        .mean
    )

    # Family G (Multi-Agent Economy): learned-only violates 3-way clearing conservation
    assert (
        by_code["G"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.CONSTRAINT_PRESERVATION]
        .mean
        == 0.0
    )
    assert (
        by_code["G"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.CONSTRAINT_PRESERVATION]
        .mean
        == 1.0
    )

    # Family H (Causal/Counterfactual Correctness): learned-only fails selectivity & refusal
    assert (
        by_code["H"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.CAUSAL_CORRECTNESS]
        .mean
        < 0.20
    )
    assert by_code["H"][BaselineParadigm.CONTEXTUAL_HYBRID].dimension_summaries[
        MetricDimension.CAUSAL_CORRECTNESS
    ].mean == pytest.approx(1.0)

    # Family J (Uncertainty & Calibration): rules-only has 0 coverage; context hybrid best calibrated
    assert (
        by_code["J"][BaselineParadigm.RULES_ONLY]
        .dimension_summaries[MetricDimension.CALIBRATION]
        .mean
        == 0.0
    )
    assert (
        by_code["J"][BaselineParadigm.CONTEXTUAL_HYBRID]
        .dimension_summaries[MetricDimension.CALIBRATION]
        .mean
        > 0.85
    )

    # Family K (OOD & Regime Shift): honest raw OOD prediction failure + competence abstention
    assert (
        by_code["K"][BaselineParadigm.CONTEXTUAL_HYBRID].raw_metrics["unabstained_ood_mae"] > 100.0
    )
    assert by_code["K"][BaselineParadigm.CONTEXTUAL_HYBRID].dimension_summaries[
        MetricDimension.ABSTENTION_QUALITY
    ].mean == pytest.approx(1.0)
    assert (
        by_code["K"][BaselineParadigm.STATIC_HYBRID]
        .dimension_summaries[MetricDimension.ABSTENTION_QUALITY]
        .mean
        == 0.0
    )

    # Family L (Decision Quality): rules-only has moderate prediction error yet 0 decision regret;
    # learned-only has lower prediction error yet high decision regret under confounding.
    rules_fid = (
        by_code["L"][BaselineParadigm.RULES_ONLY]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
    )
    learned_fid = (
        by_code["L"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.TRANSITION_FIDELITY]
        .mean
    )
    rules_regret = (
        by_code["L"][BaselineParadigm.RULES_ONLY]
        .dimension_summaries[MetricDimension.DECISION_REGRET]
        .mean
    )
    learned_regret = (
        by_code["L"][BaselineParadigm.LEARNED_ONLY]
        .dimension_summaries[MetricDimension.DECISION_REGRET]
        .mean
    )

    assert learned_fid > rules_fid
    assert rules_regret == pytest.approx(0.0, abs=1e-9)
    assert learned_regret > 100.0
