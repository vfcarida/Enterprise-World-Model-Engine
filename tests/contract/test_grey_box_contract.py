"""Contract tests for Grey-Box dynamics, architectural conservation, and neuro-symbolic flywheel."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.exceptions import DynamicsError
from ewm_engine.experimental.conservation import (
    ConservationProjectionLayer,
    LinearConservationLaw,
    evaluate_conservation_honesty,
)
from ewm_engine.experimental.flywheel import (
    AdvisoryEquationPrior,
    CandidateRulePromotion,
    ResidualPromotionWorkflow,
    SINDyRegressor,
)
from ewm_engine.experimental.grey_box import (
    CallableResidualEstimator,
    GreyBoxDynamics,
    ZeroResidualEstimator,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_grey_box_zero_residual_invariant_5_contract(sample_world_state: WorldState) -> None:
    """Contract: Grey-box with zero residual reproduces symbolic kernel bit-for-bit (Invariant 5)."""
    known_model = DeterministicTransferDynamics()
    grey_box = GreyBoxDynamics(known_model=known_model, residual_estimator=ZeroResidualEstimator())

    action = Action(
        id="act_contract_xfer",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "target_resource": "stock_wh2",
            "quantity": 18.5,
        },
    )

    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)

    res_known = known_model.transition(sample_world_state, [action], [], rng1)
    res_grey = grey_box.transition(sample_world_state, [action], [], rng2)

    # Identical state fingerprint and hash
    assert res_known.next_state.fingerprint == res_grey.next_state.fingerprint
    assert res_known.next_state.state_hash == res_grey.next_state.state_hash
    assert res_grey.evidence_level == EvidenceLevel.STRUCTURAL


def test_conservation_layer_structural_zero_violation_contract(
    sample_world_state: WorldState,
) -> None:
    """Contract: Architectural projection guarantees 0.0% violation rate across adversarial stress tests."""
    law = LinearConservationLaw.zero_sum(resource_ids=["stock_wh1", "stock_wh2"])
    layer = ConservationProjectionLayer(laws=[law], enforce_capacity_bounds=True)

    known_model = DeterministicTransferDynamics()
    rng = np.random.default_rng(42)

    # Adversarial estimator emitting arbitrary random perturbations
    def adversarial_fn(
        s: WorldState, a: Sequence[Action], e: Sequence[ExogenousEvent], next_s: WorldState
    ) -> dict[str, float]:
        return {
            "stock_wh1": float(rng.uniform(-100.0, 100.0)),
            "stock_wh2": float(rng.uniform(-100.0, 100.0)),
        }

    grey_box = GreyBoxDynamics(
        known_model=known_model,
        residual_estimator=CallableResidualEstimator(fn=adversarial_fn),
        conservation_layer=layer,
        name="AdversarialGreyBox",
    )

    dataset = TransitionDataset(
        [
            TransitionSample(
                state=sample_world_state,
                actions=(),
                events=(),
                next_state=sample_world_state,
            )
            for _ in range(50)
        ]
    )

    honesty_report = evaluate_conservation_honesty(
        model=grey_box,
        dataset=dataset,
        laws=[law],
        tolerance=1e-8,
    )

    assert honesty_report.is_structurally_conserved is True
    assert honesty_report.violation_rate == 0.0
    assert honesty_report.violations_detected == 0
    assert honesty_report.max_absolute_violation <= 1e-10


def test_discovered_dynamics_evidence_level_contract() -> None:
    """Contract: Discovered dynamic terms default strictly to observational PREDICTIVE evidence."""
    regressor = SINDyRegressor(threshold=0.01)
    X = np.array([[10.0], [20.0], [30.0]])
    y = np.array([-1.0, -2.0, -3.0])
    fit_res = regressor.fit(X=X, y=y, feature_names=["cash"], target_resource="cash")

    candidate = ResidualPromotionWorkflow.generate_candidate(
        fit_result=fit_res,
        candidate_id="cash_burn_rule",
        spec_reference="FEAT-010",
        rationale="Empirical burn rate regression.",
    )

    # Mandatory epistemic level is PREDICTIVE (observational)
    assert candidate.evidence_level == EvidenceLevel.PREDICTIVE


def test_promotion_workflow_human_in_the_loop_audit_contract() -> None:
    """Contract: Automated unreviewed promotion is blocked; human audit is mandatory."""
    candidate = CandidateRulePromotion(
        candidate_id="unreviewed_rule_01",
        target_resource="stock_wh1",
        discovered_expression="-0.02 * stock_wh1",
        terms={"stock_wh1": -0.02},
        r2_score=0.98,
        mse=0.01,
        bic=-12.5,
        parsimony_score=1,
        evidence_level=EvidenceLevel.PREDICTIVE,
        spec_reference="ADR-042",
        rationale="Inventory shrinkage candidate.",
        wsl_rule_template="rule unreviewed: ...",
        python_rule_code="def rule(): pass",
        is_approved=False,
    )

    # Attempting to promote without human approval raises DynamicsError
    with pytest.raises(DynamicsError, match="cannot be promoted without formal human approval"):
        ResidualPromotionWorkflow.promote_to_deterministic_rule(candidate)


def test_advisory_prior_never_in_simulation_core_contract() -> None:
    """Contract: Advisory equation priors are strictly advisory and never injected automatically."""
    prior = AdvisoryEquationPrior(
        name="ExternalLLMPrior",
        expression_template="c0 * inventory**2",
        proposed_by="LLM-SR-Advisory",
        rationale="Quadratic congestion hypothesis.",
    )

    assert prior.is_advisory_only is True
    # Verify no transition or simulation methods exist on the advisory prior
    assert not hasattr(prior, "transition")
    assert not hasattr(prior, "evaluate")
