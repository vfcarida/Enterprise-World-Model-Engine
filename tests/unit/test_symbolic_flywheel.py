"""Unit tests for the neuro-symbolic flywheel: SINDy regression, advisory priors, and audited promotion."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.exceptions import DynamicsError
from ewm_engine.experimental.flywheel import (
    AdvisoryEquationPrior,
    ResidualPromotionWorkflow,
    SINDyRegressor,
)
from ewm_engine.experimental.grey_box import CallableResidualEstimator, GreyBoxDynamics
from ewm_engine.provenance.evidence import EvidenceLevel


def test_sindy_sparse_regression_polynomial_discovery() -> None:
    """Test SINDy STLSQ regression discovers exact quadratic term with high parsimony."""
    rng = np.random.default_rng(42)
    n = 100
    x = rng.uniform(10.0, 100.0, size=(n, 1))
    feature_names = ["stock_wh1"]

    # Ground truth residual: y = -0.05 * x + 0.001 * x^2 + tiny_noise
    y = (-0.05 * x[:, 0]) + (0.001 * (x[:, 0] ** 2)) + rng.normal(0, 1e-4, size=n)

    regressor = SINDyRegressor(threshold=0.005, degree=2)
    fit_res = regressor.fit(X=x, y=y, feature_names=feature_names, target_resource="stock_wh1")

    # High goodness-of-fit
    assert fit_res.r2_score >= 0.99
    assert fit_res.mse < 1e-4

    # Discovered active terms: linear and quadratic
    assert "stock_wh1" in fit_res.terms
    assert "stock_wh1^2" in fit_res.terms
    assert pytest.approx(fit_res.terms["stock_wh1"], abs=0.01) == -0.05
    assert pytest.approx(fit_res.terms["stock_wh1^2"], abs=0.001) == 0.001

    # Parsimony: exactly 2 non-zero terms
    assert fit_res.parsimony_score == 2
    assert "stock_wh1" in fit_res.formula


def test_advisory_equation_prior_is_strictly_advisory() -> None:
    """Test LLM-SR advisory priors carry mandatory advisory flags and do not execute automatically."""
    prior = AdvisoryEquationPrior(
        name="SpoilageDepreciationHypothesis",
        expression_template="c0 * stock_wh1",
        proposed_by="LLM-SR-Advisory",
        rationale="Hypothesized linear inventory degradation proportional to volume.",
    )

    assert prior.is_advisory_only is True
    assert prior.proposed_by == "LLM-SR-Advisory"


def test_promotion_workflow_approval_gate_and_execution(sample_world_state: WorldState) -> None:
    """Test that rule promotion requires human approval and produces executable deterministic dynamics."""
    # Create synthetic fit result
    regressor = SINDyRegressor(threshold=0.01, degree=1)
    X = np.array([[50.0], [100.0], [150.0], [200.0]])
    y = np.array([-2.5, -5.0, -7.5, -10.0])  # y = -0.05 * stock_wh1
    fit_res = regressor.fit(X=X, y=y, feature_names=["stock_wh1"], target_resource="stock_wh1")

    # Generate promotion candidate
    candidate = ResidualPromotionWorkflow.generate_candidate(
        fit_result=fit_res,
        candidate_id="wh1_spoilage_rule",
        spec_reference="ADR-042",
        rationale="Empirically discovered warehouse 1 spoilage rate of 5% per cycle.",
    )

    # Initial state: unapproved
    assert candidate.is_approved is False
    assert candidate.evidence_level == EvidenceLevel.PREDICTIVE
    assert "stock_wh1" in candidate.wsl_rule_template
    assert "def rule_wh1_spoilage_rule" in candidate.python_rule_code

    # Promoting unapproved candidate must fail
    with pytest.raises(DynamicsError, match="cannot be promoted without formal human approval"):
        ResidualPromotionWorkflow.promote_to_deterministic_rule(candidate)

    # Human approval step
    approved_candidate = ResidualPromotionWorkflow.approve_candidate(
        candidate=candidate,
        reviewer="lead_engineer@enterprise.ai",
        review_notes="Audited against physical warehouse logs. Goodness-of-fit R2 > 0.99.",
    )
    assert approved_candidate.is_approved is True
    assert approved_candidate.reviewed_by == "lead_engineer@enterprise.ai"

    # Now promotion succeeds
    promoted_model = ResidualPromotionWorkflow.promote_to_deterministic_rule(approved_candidate)
    rng = np.random.default_rng(42)

    # Execute transition with sample state (stock_wh1 = 100.0)
    res = promoted_model.transition(sample_world_state, [], [], rng)

    # 100.0 - (0.05 * 100.0) = 95.0
    assert pytest.approx(res.next_state.get_resource("stock_wh1").current) == 95.0


def test_neuro_symbolic_flywheel_self_whitening(sample_world_state: WorldState) -> None:
    """Test self-whitening grey-box: discovered rule moves mass from LEARNED -> KNOWN, zeroing residual."""
    known_model = DeterministicTransferDynamics()

    # Reality has an unmodeled leakage of 4.0 boxes in stock_wh1
    leakage_estimator = CallableResidualEstimator(
        fn=lambda s, a, e, next_s: {"stock_wh1": -4.0}
    )

    grey_box = GreyBoxDynamics(
        known_model=known_model,
        residual_estimator=leakage_estimator,
        name="InitialGreyBox",
    )

    # Collect transition dataset under initial grey-box
    dataset = TransitionDataset()
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Sample step with leakage
        step_res = grey_box.transition(sample_world_state, [], [], rng)
        dataset.add(
            TransitionSample(
                state=sample_world_state,
                actions=(),
                events=(),
                next_state=step_res.next_state,
            )
        )

    # Step 1: Extract residual dataset against initial known model
    X, y, feats = ResidualPromotionWorkflow.extract_residual_dataset(
        dataset=dataset,
        known_model=known_model,
        target_resource="stock_wh1",
    )
    # y represents unexplained residual (-4.0 everywhere)
    assert np.allclose(y, -4.0)

    # Step 2: Fit SINDy
    regressor = SINDyRegressor(threshold=0.01, degree=1, include_bias=True)
    fit_res = regressor.fit(X=X, y=y, feature_names=feats, target_resource="stock_wh1")
    assert fit_res.r2_score >= 0.99

    # Step 3: Propose and approve candidate
    candidate = ResidualPromotionWorkflow.generate_candidate(
        fit_result=fit_res,
        candidate_id="stock_wh1_fixed_drain",
        spec_reference="ADR-042",
        rationale="Discovered fixed drain of 4 units per step.",
    )
    approved = ResidualPromotionWorkflow.approve_candidate(
        candidate=candidate,
        reviewer="audit_officer",
        review_notes="Verified persistent inventory leakage.",
    )

    # Step 4: Promote into grey-box
    whitened_grey_box = ResidualPromotionWorkflow.incorporate_into_grey_box(
        grey_box=grey_box,
        candidate=approved,
    )

    # Step 5: Verify self-whitening: Evaluate residual against the newly whitened known model
    _X_new, y_new, _ = ResidualPromotionWorkflow.extract_residual_dataset(
        dataset=dataset,
        known_model=whitened_grey_box.known_model,
        target_resource="stock_wh1",
    )

    # The unexplained residual is now zeroed out because the discovered term was absorbed into KNOWN!
    assert np.allclose(y_new, 0.0, atol=1e-8)
