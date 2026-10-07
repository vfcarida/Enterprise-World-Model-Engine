"""Comprehensive test suite for R05: ML/AI scientific rigor, provenance, UQ, and causal gates.

Verifies:
    A. Reproducibility spine: RunManifest, seed_everything, multi-seed reporting.
    B. World-model evaluation metrics: rollout error curve, compounding ratio (Talvitie),
       horizon selection, calibrated-dynamics gate.
    C. Uncertainty quantification: CRPS, Gaussian log score (NLL), Brier score,
       adaptive calibration error (ACE), PIT calibration, conformal intervals (CQR),
       adaptive conformal inference (ACI), and UQ Pareto frontier.
    D. Causal rigor: identifiability gate, off-policy evaluation (OPE) with mandatory diagnostics
       and evidence downgrade (never auto-upgrade), positivity/overlap with D'Amour high-dim warning
       and trimmed mass, E-values, and causal refutation battery.
    E. ML Test Score & metamorphic: Google ML Test Score readiness report, metamorphic regression gates.
"""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.seed import seed_everything
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.evaluation.conformal import (
    AdaptiveConformalInference,
    ConformalIntervalPredictor,
    UQParetoCandidate,
    compute_uq_pareto_front,
)
from ewm_engine.evaluation.multiseed import evaluate_multiseed
from ewm_engine.evaluation.uncertainty import (
    compute_adaptive_calibration_error,
    compute_brier_score,
    compute_crps,
    compute_log_score,
    compute_mean_crps,
    compute_pit_calibration,
)
from ewm_engine.experimental.causal import (
    CausalGraph,
    InterventionalQueryResult,
    NotIdentifiableResult,
    check_positivity_overlap,
    compute_e_value,
    query_interventional,
    run_causal_refutations,
)
from ewm_engine.experimental.dynamics_eval import (
    CalibrationResult,
    RolloutDivergenceResult,
    evaluate_calibrated_dynamics_gate,
    select_reliable_horizon,
)
from ewm_engine.experimental.ope import (
    evaluate_off_policy,
)
from ewm_engine.experimental.readiness import (
    ReadinessLevel,
    compute_readiness_report,
    run_metamorphic_regression_gate,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.manifest import (
    RunManifest,
    create_run_manifest,
)

# ---------------------------------------------------------------------------
# A. Reproducibility Spine Tests
# ---------------------------------------------------------------------------


def test_run_manifest_generation_and_hashing() -> None:
    """RunManifest captures full reproducibility configuration matching NeurIPS checklist."""
    manifest = create_run_manifest(
        component_id="dynamics_v2",
        scenario_id="supply_chain_stress",
        seed=1337,
        resolved_config={"learning_rate": 0.001, "batch_size": 32},
        engine_version="1.0.0",
        component_version="2.1.0",
        deterministic=True,
    )

    assert isinstance(manifest, RunManifest)
    assert "dynamics_v2" in manifest.component_ids
    assert manifest.scenario_id == "supply_chain_stress"
    assert manifest.seed == 1337
    assert manifest.determinism_flags.get("deterministic") is True
    assert len(manifest.manifest_hash) == 64
    assert len(manifest.resolved_config_hash) == 64

    # Identical inputs produce identical canonical hash
    manifest_dup = create_run_manifest(
        component_id="dynamics_v2",
        scenario_id="supply_chain_stress",
        seed=1337,
        resolved_config={"learning_rate": 0.001, "batch_size": 32},
        engine_version="1.0.0",
        component_version="2.1.0",
        deterministic=True,
    )
    assert manifest.manifest_hash == manifest_dup.manifest_hash

    # Serializes to valid dictionary
    d = manifest.to_dict()
    assert "manifest_hash" in d
    assert "reproducibility_checklist" in d


def test_seed_everything_determinism() -> None:
    """seed_everything ensures deterministic random numbers across Python and NumPy."""
    seed_everything(42, deterministic=True, warn_cuda=False)
    rand1 = np.random.rand(5)

    seed_everything(42, deterministic=True, warn_cuda=False)
    rand2 = np.random.rand(5)

    np.testing.assert_allclose(rand1, rand2)


def test_multiseed_reporting_bootstrap_ci() -> None:
    """Multi-seed evaluation produces mean +- bootstrap CI, never a bare point estimate."""

    def dummy_runner(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        return {"throughput": float(100.0 + rng.normal(0.0, 5.0))}

    report = evaluate_multiseed(
        trial_fn=dummy_runner,
        n_seeds=10,
        master_seed=100,
        confidence_level=0.95,
    )

    assert report.n_seeds == 10
    assert "throughput" in report.metrics
    summary = report.metrics["throughput"]
    assert 90.0 < summary.mean < 110.0
    assert summary.ci_lower <= summary.mean <= summary.ci_upper
    assert summary.std > 0.0


# ---------------------------------------------------------------------------
# B. World-Model Evaluation Metrics Tests
# ---------------------------------------------------------------------------


def test_compounding_error_curve_and_horizon_selection() -> None:
    """Rollout error curve computes Talvitie compounding ratio and selects reliable horizon."""
    # Super-linear drift: err(h) = 1.0 * (h ** 1.5)
    error_curve = [1.0 * (h**1.5) for h in range(1, 11)]
    # C(h) = err(h) / (h * err(1)) = h^1.5 / h = h^0.5
    compounding_ratios = [err / (h * error_curve[0]) for h, err in enumerate(error_curve, start=1)]

    res = RolloutDivergenceResult(
        horizon=10,
        step_divergences=error_curve,
        error_curve=error_curve,
        compounding_error_ratios=compounding_ratios,
        mean_divergence=float(np.mean(error_curve)),
        max_divergence=float(np.max(error_curve)),
        final_divergence=float(error_curve[-1]),
        drift_rate=float(error_curve[-1] - error_curve[0]),
        mean_compounding_ratio=float(np.mean(compounding_ratios)),
        max_compounding_ratio=float(np.max(compounding_ratios)),
        recommended_planning_horizon=5,
    )

    assert res.error_curve[0] == 1.0
    assert res.compounding_error_ratios[0] == 1.0
    assert res.compounding_error_ratios[-1] > 1.0

    # Horizon selector cuts off where compounding exceeds tolerance
    safe_h = select_reliable_horizon(
        error_curve, max_compounding_ratio=2.0, max_error_threshold=15.0
    )
    assert 1 <= safe_h <= 10


def test_calibrated_dynamics_gate() -> None:
    """Calibrated dynamics gate fails accurate-but-overconfident models."""
    good_cal = CalibrationResult(
        crps=0.15,
        coverage=0.88,
        is_calibrated=True,
    )
    overconfident_cal = CalibrationResult(
        crps=0.85,
        coverage=0.60,
        is_calibrated=False,
    )

    assert evaluate_calibrated_dynamics_gate(good_cal, min_coverage=0.80, max_crps=0.50) is True
    assert (
        evaluate_calibrated_dynamics_gate(overconfident_cal, min_coverage=0.80, max_crps=0.50)
        is False
    )


# ---------------------------------------------------------------------------
# C. Uncertainty Quantification Tests
# ---------------------------------------------------------------------------


def test_proper_scoring_rules() -> None:
    """Verifies CRPS, Gaussian Log Score (NLL), and Brier Score."""
    # CRPS
    samples = [9.0, 10.0, 11.0]
    crps_exact = compute_crps(samples, actual=10.0)
    assert crps_exact >= 0.0

    batch_crps = compute_mean_crps([[9.0, 10.0, 11.0], [4.0, 5.0, 6.0]], [10.0, 5.0])
    assert batch_crps >= 0.0

    # Gaussian Log Score (NLL)
    nll = compute_log_score(means=[10.0, 20.0], stds=[1.0, 2.0], actuals=[10.0, 20.0])
    assert nll < 2.0

    # Brier Score (Binary)
    brier_binary = compute_brier_score(probabilities=[0.9, 0.1], labels=[1, 0])
    assert brier_binary == pytest.approx(0.01, abs=1e-3)

    # Brier Score (Multiclass)
    brier_multi = compute_brier_score(
        probabilities=[[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
        labels=[0, 1],
    )
    assert brier_multi < 0.10


def test_adaptive_calibration_error_quantile_binning() -> None:
    """Adaptive calibration error partitions by equal-mass quantiles to avoid empty bin bias."""
    # Well calibrated: 100 probabilities uniformly distributed with matching binary outcomes
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.9, 200).tolist()
    labels = [1 if rng.random() < p else 0 for p in probs]

    ace_res = compute_adaptive_calibration_error(probs, labels, n_bins=5)
    assert len(ace_res.bins) == 5
    assert ace_res.ace >= 0.0
    assert ace_res.debiased_ace <= ace_res.ace


def test_pit_calibration_continuous() -> None:
    """Continuous predictive distributions evaluated via Probability Integral Transform."""
    # Generate calibrated forecasts where each actual is a true draw from the forecast distribution
    rng = np.random.default_rng(42)
    means = rng.normal(10.0, 2.0, 100).tolist()
    actuals = [float(rng.normal(m, 1.0)) for m in means]
    forecasts = [rng.normal(m, 1.0, 100).tolist() for m in means]

    pit_res = compute_pit_calibration(forecasts, actuals, n_bins=10)
    assert len(pit_res.pit_bins) == 10
    assert pit_res.ks_statistic < 0.20
    assert pit_res.is_well_calibrated is True


def test_conformal_interval_predictor_and_aci() -> None:
    """Conformal interval predictor guarantees coverage; ACI tracks regime shift."""
    rng = np.random.default_rng(42)
    # Split conformal calibration
    y_cal = rng.normal(10.0, 1.0, 100).tolist()
    y_pred = [10.0] * 100

    predictor = ConformalIntervalPredictor(nominal_coverage=0.90)
    q_hat = predictor.calibrate(actuals=y_cal, predictions=y_pred)
    assert q_hat > 0.0

    eval_summary = predictor.evaluate(actuals=y_cal, predictions=y_pred)
    assert eval_summary.empirical_coverage >= 0.85
    assert eval_summary.guarantee_satisfied is True

    # Adaptive Conformal Inference (ACI)
    aci = AdaptiveConformalInference(nominal_coverage=0.90, gamma=0.10)
    # Consecutive severe errors trigger regime shift alert
    state = None
    for _ in range(5):
        _, state = aci.predict_and_update(point_prediction=0.0, actual=100.0)
    assert state is not None
    assert state.regime_shift_detected is True


def test_uq_pareto_front() -> None:
    """UQ Pareto analysis models sharpness, coverage, calibration, and compute cost."""
    candidates = [
        UQParetoCandidate(
            name="ensemble",
            sharpness=1.5,
            empirical_coverage=0.92,
            calibration_error=0.05,
            latency_ms=120.0,
        ),
        UQParetoCandidate(
            name="mc_dropout",
            sharpness=2.5,
            empirical_coverage=0.88,
            calibration_error=0.12,
            latency_ms=30.0,
        ),
        UQParetoCandidate(
            name="conformal_cqr",
            sharpness=1.8,
            empirical_coverage=0.94,
            calibration_error=0.04,
            latency_ms=15.0,
        ),
    ]

    front = compute_uq_pareto_front(candidates)
    assert len(front.frontier) >= 1
    assert "conformal_cqr" in front.frontier


# ---------------------------------------------------------------------------
# D. Causal Rigor Tests (Gated Identifiability, OPE, Overlap, E-Values)
# ---------------------------------------------------------------------------


def _create_synthetic_causal_dataset() -> tuple[TransitionDataset, CausalGraph]:
    graph = CausalGraph()
    graph.add_node("cost")
    graph.add_node("marketing")
    graph.add_node("revenue")
    graph.add_edge("cost", "marketing")
    graph.add_edge("cost", "revenue")
    graph.add_edge("marketing", "revenue")

    rng = np.random.default_rng(42)
    samples: list[TransitionSample] = []
    for i in range(60):
        c_val = float(rng.uniform(10.0, 50.0))
        # Treatment conditioned on cost
        treated = bool(c_val > 30.0 or rng.random() > 0.5)
        actions = [Action(id=f"act_marketing_{i}", type="marketing")] if treated else []
        # Outcome depends on cost and marketing
        rev = float(100.0 + 2.0 * c_val + (20.0 if treated else 0.0) + rng.normal(0, 2))

        st = WorldState(
            resources={"cost": Resource(id="cost", current=c_val, min_value=0.0, max_value=100.0)}
        )
        nxt = WorldState(
            resources={
                "revenue": Resource(id="revenue", current=rev, min_value=0.0, max_value=500.0)
            }
        )
        samples.append(TransitionSample(state=st, actions=tuple(actions), next_state=nxt))

    return TransitionDataset(samples=samples), graph


def test_causal_identifiability_gate() -> None:
    """Unidentifiable query returns NotIdentifiableResult with reason, NEVER an effect estimate."""
    unid_graph = CausalGraph()
    unid_graph.add_node("marketing")
    unid_graph.add_node("revenue")
    unid_graph.add_node("hidden_sentiment", is_unobserved=True)
    unid_graph.add_edge("hidden_sentiment", "marketing")
    unid_graph.add_edge("hidden_sentiment", "revenue")

    dataset, _ = _create_synthetic_causal_dataset()
    res = query_interventional(
        dataset=dataset,
        treatment_action="marketing",
        outcome_resource="revenue",
        graph=unid_graph,
        conditioning_set=(),
    )

    assert isinstance(res, NotIdentifiableResult)
    assert res.causal_level == EvidenceLevel.PREDICTIVE
    assert "NOT identifiable" in res.diagnostic_message


def test_ope_estimators_and_mandatory_diagnostics() -> None:
    """OPE computes DM, IPS, SNIPW, DR (default) with mandatory diagnostics gating confidence."""
    rng = np.random.default_rng(42)
    rewards = rng.normal(10.0, 1.0, 100).tolist()
    mu = rng.uniform(0.2, 0.8, 100).tolist()
    pi = rng.uniform(0.2, 0.8, 100).tolist()

    res = evaluate_off_policy(
        rewards=rewards,
        behavior_probs=mu,
        target_probs=pi,
        initial_evidence_level=EvidenceLevel.INTERVENTIONAL,
    )

    assert res.passed_gate is True
    assert res.evidence_level == EvidenceLevel.INTERVENTIONAL
    assert res.diagnostics.effective_sample_size > 10.0
    assert res.dr_value != 0.0

    # Pathological weights -> Gate failure down-grades confidence (NO auto-upgrade)
    pathological_mu = [0.001] * 100  # Extreme propensity explosion
    bad_res = evaluate_off_policy(
        rewards=rewards,
        behavior_probs=pathological_mu,
        target_probs=pi,
        weight_clip=10.0,
        initial_evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    assert bad_res.passed_gate is False
    assert bad_res.evidence_level == EvidenceLevel.PREDICTIVE  # Downgraded!


def test_positivity_overlap_and_e_value() -> None:
    """Positivity check computes trimmed mass and D'Amour high-dim warning; E-value computed."""
    dataset, graph = _create_synthetic_causal_dataset()

    pos_rep = check_positivity_overlap(
        dataset=dataset,
        treatment_action="marketing",
        covariate_resources=["cost"],
    )
    assert 0.0 <= pos_rep.overlap_index <= 1.0
    assert 0.0 <= pos_rep.trimmed_mass <= 1.0

    # E-Value computation
    e_res = compute_e_value(point_estimate=2.5, ci_lower=1.8, ci_upper=3.2, is_risk_ratio=True)
    assert e_res.e_value_point > 1.0
    assert e_res.e_value_ci_limit is not None and e_res.e_value_ci_limit > 1.0

    # Interventional query with E-values attached
    query_res = query_interventional(
        dataset=dataset,
        treatment_action="marketing",
        outcome_resource="revenue",
        graph=graph,
        conditioning_set=["cost"],
        compute_evalues=True,
    )
    assert isinstance(query_res, InterventionalQueryResult)
    assert query_res.e_value is not None


def test_causal_refutation_battery() -> None:
    """Causal refutation suite runs placebo, random common cause, subset, and confounder tests."""
    dataset, graph = _create_synthetic_causal_dataset()

    refutation_res = run_causal_refutations(
        dataset=dataset,
        treatment_action="marketing",
        outcome_resource="revenue",
        graph=graph,
        conditioning_set=["cost"],
        seed=42,
    )

    assert len(refutation_res.tests) == 4
    test_names = [t.test_name for t in refutation_res.tests]
    assert "placebo_treatment" in test_names
    assert "random_common_cause" in test_names
    assert "subset_stability" in test_names
    assert "unobserved_confounder_sensitivity" in test_names


# ---------------------------------------------------------------------------
# E. ML Test Score & Metamorphic Readiness Tests
# ---------------------------------------------------------------------------


def test_metamorphic_regression_gate() -> None:
    """Metamorphic regression gate verifies conservation and bounded version divergence."""
    base_preds = [{"inventory": 100.0, "cash": 500.0}, {"inventory": 80.0, "cash": 600.0}]
    cand_preds_good = [{"inventory": 102.0, "cash": 495.0}, {"inventory": 81.0, "cash": 598.0}]
    cand_preds_bad = [{"inventory": -5.0, "cash": 200.0}, {"inventory": 80.0, "cash": 600.0}]

    good_gates = run_metamorphic_regression_gate(base_preds, cand_preds_good, tolerance=0.10)
    assert all(g.passed for g in good_gates)

    bad_gates = run_metamorphic_regression_gate(base_preds, cand_preds_bad, tolerance=0.10)
    assert any(not g.passed for g in bad_gates)


def test_ml_test_score_readiness_report() -> None:
    """ML readiness report computes min across Data, Model, Infra, Monitoring categories."""
    manifest = create_run_manifest(
        component_id="world_model",
        scenario_id="s1",
        seed=42,
        resolved_config={"param": 1},
        engine_version="1.0.0",
        deterministic=True,
    )

    meta_results = run_metamorphic_regression_gate([{"res": 10.0}], [{"res": 10.0}], tolerance=0.05)

    report = compute_readiness_report(
        manifest=manifest,
        dynamics_report=None,
        metamorphic_results=meta_results,
    )

    assert 0.0 <= report.overall_ml_test_score <= 4.0
    assert report.readiness_level in list(ReadinessLevel)
    assert report.overall_ml_test_score == min(
        report.data_score.score,
        report.model_score.score,
        report.infra_score.score,
        report.monitoring_score.score,
    )
