"""Unit tests for Responsible AI, trust guarantees, anti-overclaiming enforcement, and governance renderers."""

from __future__ import annotations

import pytest

from ewm_engine.cards.annex_iv import DISCLAIMER_TEXT
from ewm_engine.cards.models import ModelCard, ScenarioCard
from ewm_engine.core.state import WorldState
from ewm_engine.core.trust import (
    BarePointEstimateWarning,
    DecisionBlockedError,
    ExtrapolationWarning,
    InsufficientCausalEvidenceWarning,
    ResponsibleAIGate,
    ScenarioForecastResult,
    UncertaintyRequiredError,
    UnqualifiedCausalClaimError,
    WideUncertaintyWarning,
    validate_causal_claim,
)
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.evaluation.uncertainty import summarize_distribution
from ewm_engine.experimental.causal import (
    IdentifiabilityResult,
    InterventionalQueryResult,
    PositivityReport,
)
from ewm_engine.integrations.rai_erroranalysis import (
    ErrorAnalysisAdapter,
    ErrorAnalysisResult,
    ErrorCohort,
)
from ewm_engine.integrations.rai_explain import (
    CounterfactualExplainerAdapter,
    CounterfactualExplanation,
    CounterfactualExplanationResult,
)
from ewm_engine.integrations.rai_fairness import (
    FairnessAuditAdapter,
    FairnessAuditResult,
    GroupMetricSummary,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
    TrajectoryStatus,
)

# ==============================================================================
# 1. Causal Anti-Overclaiming & Evidence Level Promotion Tests
# ==============================================================================


def test_validate_causal_claim_strict_blocks_unidentified() -> None:
    """Strict mode blocks promoting unidentifiable effects to INTERVENTIONAL."""
    with pytest.raises(
        UnqualifiedCausalClaimError, match="Effect is NOT mathematically identifiable"
    ):
        validate_causal_claim(
            claimed_level=EvidenceLevel.INTERVENTIONAL,
            is_identifiable=False,
            overlap_satisfied=True,
            strict=True,
        )


def test_validate_causal_claim_strict_blocks_positivity_breach() -> None:
    """Strict mode blocks causal claim when common support / positivity fails."""
    with pytest.raises(UnqualifiedCausalClaimError, match="Positivity / common support breached"):
        validate_causal_claim(
            claimed_level=EvidenceLevel.QUASI_CAUSAL,
            is_identifiable=True,
            overlap_satisfied=False,
            strict=True,
        )


def test_validate_causal_claim_non_strict_downgrades_to_predictive() -> None:
    """Non-strict mode warns and downgrades to PREDICTIVE."""
    with pytest.warns(
        InsufficientCausalEvidenceWarning, match="Automatically downgrading to PREDICTIVE"
    ):
        downgraded = validate_causal_claim(
            claimed_level=EvidenceLevel.INTERVENTIONAL,
            is_identifiable=False,
            overlap_satisfied=True,
            strict=False,
        )
    assert downgraded == EvidenceLevel.PREDICTIVE


def test_validate_causal_claim_passes_when_guarantees_met() -> None:
    """Valid causal claim passes unchanged."""
    level = validate_causal_claim(
        claimed_level=EvidenceLevel.INTERVENTIONAL,
        is_identifiable=True,
        overlap_satisfied=True,
        strict=True,
    )
    assert level == EvidenceLevel.INTERVENTIONAL


def test_identifiability_result_pydantic_validator_blocks_overclaim() -> None:
    """IdentifiabilityResult model validator prevents unidentifiable interventional claims."""
    with pytest.raises(Exception, match="Effect is NOT mathematically identifiable"):
        IdentifiabilityResult(
            is_identifiable=False,
            treatment="price_cut",
            outcome="revenue",
            conditioning_set=(),
            causal_level=EvidenceLevel.INTERVENTIONAL,
            diagnostic_message="Unblocked backdoor path exists",
        )


def test_interventional_query_result_pydantic_validator_blocks_positivity_breach() -> None:
    """InterventionalQueryResult model validator prevents causal claim when positivity fails."""
    valid_id = IdentifiabilityResult(
        is_identifiable=True,
        treatment="price_cut",
        outcome="revenue",
        conditioning_set=("competitor_action",),
        causal_level=EvidenceLevel.INTERVENTIONAL,
        diagnostic_message="Identifiable via backdoor adjustment",
    )
    failed_positivity = PositivityReport(
        treatment="price_cut",
        positivity_satisfied=False,
        overlap_index=0.12,
        violation_rate=0.45,
        treated_count=10,
        control_count=100,
        diagnostic_message="Extreme propensity scores detected",
    )

    with pytest.raises(Exception, match="Positivity / common support breached"):
        InterventionalQueryResult(
            treatment="price_cut",
            outcome="revenue",
            adjusted_effect=15.4,
            identifiability=valid_id,
            causal_level=EvidenceLevel.INTERVENTIONAL,
            assumptions_summary="Backdoor adjustment",
            positivity_report=failed_positivity,
        )


# ==============================================================================
# 2. Mandatory Uncertainty & Forecast Result Tests
# ==============================================================================


def test_scenario_forecast_result_blocks_bare_point_estimate() -> None:
    """Calling get_point_estimate() without acknowledgment raises UncertaintyRequiredError."""
    dist = summarize_distribution([10.0, 12.0, 14.0, 11.0, 13.0])
    forecast = ScenarioForecastResult(
        metric_name="throughput",
        point_estimate=12.0,
        ci_lower=10.5,
        ci_upper=13.5,
        confidence_level=0.90,
        distribution=dist,
        extrapolation_flag=False,
        sample_count=5,
        evidence_level=EvidenceLevel.PREDICTIVE,
    )

    with pytest.raises(UncertaintyRequiredError, match="Responsible AI Policy Violation"):
        forecast.get_point_estimate()

    # With explicit opt-in, warns and returns value
    with pytest.warns(BarePointEstimateWarning, match="without uncertainty"):
        val = forecast.get_point_estimate(acknowledge_no_uncertainty=True)
    assert val == 12.0


def test_simulation_result_get_forecast_and_extrapolation_flag() -> None:
    """SimulationResult produces a valid ScenarioForecastResult and detects OOD steps."""
    init_state = WorldState(step=0, timestamp=0.0)
    step1 = StepRecord(
        step=0,
        timestamp=0.0,
        state_hash=init_state.fingerprint,
        transition_result=TransitionResult(next_state=init_state, model_name="test"),
        step_metrics={"kpi": 100.0, "is_ood": 0.0},
    )
    traj1 = Trajectory(
        sample_id=0,
        seed=1,
        initial_state=init_state,
        steps=[step1],
        status=TrajectoryStatus.COMPLETED,
    )

    # Second trajectory leaves grounded space
    step2 = StepRecord(
        step=0,
        timestamp=0.0,
        state_hash=init_state.fingerprint,
        transition_result=TransitionResult(next_state=init_state, model_name="test"),
        step_metrics={"kpi": 120.0, "is_ood": 1.0},
    )
    traj2 = Trajectory(
        sample_id=1,
        seed=2,
        initial_state=init_state,
        steps=[step2],
        status=TrajectoryStatus.COMPLETED,
    )

    sim = SimulationResult(
        scenario=Scenario(scenario_id="test_scen", name="test"),
        trajectories=[traj1, traj2],
        provenance=Provenance(),
    )

    assert sim.has_extrapolation is True
    forecast = sim.get_forecast("kpi", confidence_level=0.90)
    assert forecast.metric_name == "kpi"
    assert forecast.extrapolation_flag is True
    assert forecast.ci_lower <= forecast.point_estimate <= forecast.ci_upper
    assert "EXTRAPOLATING (OOD)" in str(forecast)
    assert "kpi" in sim.uncertainty_by_metric


# ==============================================================================
# 3. MANAGE Gating Hook Tests
# ==============================================================================


def test_responsible_ai_gate_passes_grounded_narrow_interval() -> None:
    """ResponsibleAIGate validates and passes when criteria are met."""
    dist = summarize_distribution([100.0, 102.0, 98.0, 101.0, 99.0])
    forecast = ScenarioForecastResult(
        metric_name="revenue",
        point_estimate=100.0,
        ci_lower=98.0,
        ci_upper=102.0,
        distribution=dist,
        extrapolation_flag=False,
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )

    gate = ResponsibleAIGate(
        max_relative_ci_width=0.5,
        require_causal_identifiability=True,
        block_on_extrapolation=True,
        strict=True,
    )
    report = gate.validate_forecast(forecast)
    assert report.passed is True
    assert len(report.reasons) == 0


def test_responsible_ai_gate_blocks_on_wide_uncertainty() -> None:
    """ResponsibleAIGate blocks pipeline when CI relative width exceeds threshold in strict mode."""
    dist = summarize_distribution([10.0, 50.0, 100.0, 20.0, 80.0])
    forecast = ScenarioForecastResult(
        metric_name="revenue",
        point_estimate=20.0,
        ci_lower=5.0,
        ci_upper=95.0,  # relative width (95-5)/20 = 4.5
        distribution=dist,
        extrapolation_flag=False,
    )

    gate = ResponsibleAIGate(max_relative_ci_width=1.0, strict=True)
    with pytest.raises(
        DecisionBlockedError, match="Uncertainty interval for 'revenue' is too wide"
    ):
        gate.validate_forecast(forecast)


def test_responsible_ai_gate_warns_on_wide_uncertainty_non_strict() -> None:
    """ResponsibleAIGate emits WideUncertaintyWarning when strict=False."""
    dist = summarize_distribution([10.0, 50.0, 100.0, 20.0, 80.0])
    forecast = ScenarioForecastResult(
        metric_name="revenue",
        point_estimate=20.0,
        ci_lower=5.0,
        ci_upper=95.0,
        distribution=dist,
        extrapolation_flag=False,
    )

    gate = ResponsibleAIGate(max_relative_ci_width=1.0, strict=False)
    with pytest.warns(WideUncertaintyWarning, match="relative width 4.50 exceeds threshold"):
        report = gate.validate_forecast(forecast)
    assert report.passed is False
    assert len(report.reasons) == 1


def test_responsible_ai_gate_blocks_on_extrapolation() -> None:
    """ResponsibleAIGate blocks pipeline when rollout entered OOD regime in strict mode."""
    dist = summarize_distribution([10.0, 11.0, 12.0])
    forecast = ScenarioForecastResult(
        metric_name="cash",
        point_estimate=11.0,
        ci_lower=10.0,
        ci_upper=12.0,
        distribution=dist,
        extrapolation_flag=True,  # OOD
    )

    gate = ResponsibleAIGate(block_on_extrapolation=True, strict=True)
    with pytest.raises(DecisionBlockedError, match="EXTRAPOLATING beyond grounded support"):
        gate.validate_forecast(forecast)


def test_responsible_ai_gate_warns_on_extrapolation_non_strict() -> None:
    """ResponsibleAIGate emits ExtrapolationWarning when strict=False."""
    dist = summarize_distribution([10.0, 11.0, 12.0])
    forecast = ScenarioForecastResult(
        metric_name="cash",
        point_estimate=11.0,
        ci_lower=10.0,
        ci_upper=12.0,
        distribution=dist,
        extrapolation_flag=True,
    )

    gate = ResponsibleAIGate(block_on_extrapolation=True, strict=False)
    with pytest.warns(ExtrapolationWarning, match="EXTRAPOLATING beyond grounded support"):
        report = gate.validate_forecast(forecast)
    assert report.passed is False


# ==============================================================================
# 4. ModelCard & ScenarioCard Mitchell Polish & Annex IV Export Tests
# ==============================================================================


def test_model_card_mitchell_validation_and_annex_iv() -> None:
    """ModelCard enforces Mitchell completeness, auto-populates metrics, and renders Annex IV."""
    card = ModelCard(
        model_id="net_res_v1",
        model_name="Neural Residual Predictor",
        artifact_fingerprint="sha256_mock_hash",
        description="Predicts residual transitions",
        intended_use=["Scenario planning within +-15% variance"],
        out_of_scope=["Unchecked high-frequency automated algorithmic trading"],
        assumptions=["Linear demand elasticity"],
        limitations=["Uncalibrated outside historical bands"],
        metrics={"mae": 0.05},
        factors=["Market volatility", "Inventory tier"],
        ethical_considerations=["Workforce scheduling fairness"],
        caveats_and_recommendations=["Requires quarterly recalibration"],
    )

    # Validates cleanly under Mitchell checklist
    assert card.validate_mitchell_completeness(strict=True) == []

    # Annex IV technical documentation rendering
    annex_iv = card.to_annex_iv()
    assert annex_iv["regulation"] == "Regulation (EU) 2024/1689 (EU AI Act)"
    assert DISCLAIMER_TEXT in annex_iv["disclaimer"]
    assert annex_iv["section_1_general_description"]["system_identifier"] == "net_res_v1"
    assert annex_iv["section_1_general_description"]["out_of_scope_use_cases"] == [
        "Unchecked high-frequency automated algorithmic trading"
    ]
    assert "ethical_considerations" in annex_iv["section_4_risk_management"]


def test_scenario_card_assumptions_and_scope_validation() -> None:
    """ScenarioCard requires non-empty assumptions and out_of_scope sections."""
    # Incomplete card
    incomplete = ScenarioCard(
        scenario_id="scen_empty",
        artifact_fingerprint="sha256_mock_hash",
        description="Missing scope",
        horizon=10,
        samples=50,
        seed=42,
    )
    with pytest.raises(ValueError, match="failed Responsible AI validation"):
        incomplete.validate_assumptions_and_scope(strict=True)

    # Complete card
    complete = ScenarioCard(
        scenario_id="scen_supply_shock",
        artifact_fingerprint="sha256_mock_hash",
        description="Supply shock simulation",
        horizon=10,
        samples=50,
        seed=42,
        assumptions=["Supplier lead times follow Gamma(2, 3)", "Price elasticity is constant"],
        out_of_scope=["Catastrophic global multi-year trade embargoes"],
    )
    assert complete.validate_assumptions_and_scope(strict=True) == []

    annex_iv = complete.to_annex_iv()
    assert annex_iv["section_1_test_scope"]["scenario_identifier"] == "scen_supply_shock"
    assert annex_iv["section_2_assumptions_and_boundaries"]["assumptions"] == [
        "Supplier lead times follow Gamma(2, 3)",
        "Price elasticity is constant",
    ]


def test_cards_populate_from_simulation_result() -> None:
    """Both cards auto-populate metrics and quantitative analyses from SimulationResult."""
    init_state = WorldState(step=0, timestamp=0.0)
    step = StepRecord(
        step=0,
        timestamp=0.0,
        state_hash=init_state.fingerprint,
        transition_result=TransitionResult(next_state=init_state, model_name="test"),
        step_metrics={"inventory": 50.0},
    )
    traj = Trajectory(
        sample_id=0,
        seed=1,
        initial_state=init_state,
        steps=[step],
        status=TrajectoryStatus.COMPLETED,
    )
    sim = SimulationResult(
        scenario=Scenario(scenario_id="s1", name="test"),
        trajectories=[traj],
        provenance=Provenance(),
    )

    model_card = ModelCard(model_id="m1", model_name="M1", artifact_fingerprint="sha256_m1")
    model_card = model_card.populate_from_simulation(sim)
    assert "inventory_mean" in model_card.metrics
    assert "inventory" in model_card.quantitative_analyses

    scen_card = ScenarioCard(
        scenario_id="s1",
        artifact_fingerprint="sha256_s1",
        description="S1",
        horizon=1,
        samples=1,
        seed=1,
    )
    scen_card = scen_card.populate_from_simulation(sim)
    assert "inventory_mean" in scen_card.metrics
    assert "inventory" in scen_card.quantitative_analyses


# ==============================================================================
# 5. Optional RAI Integration Adapters (Extras)
# ==============================================================================


def test_rai_adapters_informative_importerror_when_uninstalled() -> None:
    """Adapters provide clean error messages explaining pip install requirements if missing."""
    # Test FairnessAuditAdapter instantiation or call
    try:
        import fairlearn  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install ewm-engine\\[fairness\\]"):
            FairnessAuditAdapter()

    # Test CounterfactualExplainerAdapter instantiation or call
    try:
        import dice_ml  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install ewm-engine\\[explain\\]"):
            CounterfactualExplainerAdapter()

    # Test ErrorAnalysisAdapter instantiation or call
    try:
        import erroranalysis  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pip install ewm-engine\\[erroranalysis\\]"):
            ErrorAnalysisAdapter()


def test_rai_audit_result_models_and_formatting() -> None:
    """Structured RAI result models instantiate and render cleanly."""
    fairness = FairnessAuditResult(
        sensitive_feature="region",
        demographic_parity_difference=0.045,
        equalized_odds_difference=0.021,
        by_group=[
            GroupMetricSummary(
                group_value="North", sample_count=100, selection_rate=0.75, accuracy_or_metric=0.92
            ),
            GroupMetricSummary(
                group_value="South", sample_count=100, selection_rate=0.71, accuracy_or_metric=0.90
            ),
        ],
    )
    summary_txt = fairness.summary()
    assert "Demographic Parity Difference: 0.0450" in summary_txt
    assert "[North]" in summary_txt

    cf = CounterfactualExplanationResult(
        target_outcome=1.0,
        total_counterfactuals=1,
        explanations=[
            CounterfactualExplanation(
                original_features={"buffer": 10.0},
                counterfactual_features={"buffer": 18.0},
                predicted_outcome=1.0,
                features_changed=["buffer"],
                distance=8.0,
            )
        ],
    )
    assert "Path 1" in cf.summary()

    err = ErrorAnalysisResult(
        total_samples=250,
        overall_error_rate=0.12,
        top_error_cohorts=[
            ErrorCohort(
                cohort_name="HighLoad_Cohort",
                sample_size=35,
                error_rate=0.38,
                feature_conditions=["load > 0.85"],
            )
        ],
    )
    assert "Overall Error Rate: 0.1200" in err.summary()
    assert "HighLoad_Cohort" in err.summary()
