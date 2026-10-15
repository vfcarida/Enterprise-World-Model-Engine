"""Unit tests for UDE Grey-Box dynamics and residual-only learned transitions."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.experimental.grey_box import (
    CallableResidualEstimator,
    GreyBoxDynamics,
    LinearResidualEstimator,
    ZeroResidualEstimator,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_grey_box_zero_residual_fingerprint_identical(sample_world_state: WorldState) -> None:
    """Test S00 Invariant 5: Zero-residual grey-box reduces bit-for-bit to symbolic kernel."""
    known_model = DeterministicTransferDynamics()
    grey_box_none = GreyBoxDynamics(known_model=known_model, residual_estimator=None)
    grey_box_zero = GreyBoxDynamics(
        known_model=known_model, residual_estimator=ZeroResidualEstimator()
    )

    action = Action(
        id="act_transfer",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "target_resource": "stock_wh2",
            "quantity": 25.0,
        },
    )

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    rng3 = np.random.default_rng(42)

    known_result = known_model.transition(sample_world_state, [action], [], rng1)
    none_result = grey_box_none.transition(sample_world_state, [action], [], rng2)
    zero_result = grey_box_zero.transition(sample_world_state, [action], [], rng3)

    # Invariant 5: Bit-for-bit identical state fingerprints
    assert known_result.next_state.fingerprint == none_result.next_state.fingerprint
    assert known_result.next_state.fingerprint == zero_result.next_state.fingerprint
    assert known_result.next_state.state_hash == zero_result.next_state.state_hash

    # Epistemic status remains STRUCTURAL when no residual is enacted
    assert none_result.evidence_level == EvidenceLevel.STRUCTURAL
    assert zero_result.evidence_level == EvidenceLevel.STRUCTURAL


def test_grey_box_learned_residual_application(sample_world_state: WorldState) -> None:
    """Test that a non-zero learned residual correctly updates next state and sets PREDICTIVE evidence."""
    known_model = DeterministicTransferDynamics()

    # Residual estimator modeling friction/loss of 2 boxes in warehouse 2
    estimator = CallableResidualEstimator(
        fn=lambda s, a, e, next_s: {"stock_wh2": -2.0}
    )

    grey_box = GreyBoxDynamics(known_model=known_model, residual_estimator=estimator)

    action = Action(
        id="act_transfer",
        type="transfer_resource",
        parameters={
            "source_resource": "stock_wh1",
            "target_resource": "stock_wh2",
            "quantity": 30.0,
        },
    )

    rng = np.random.default_rng(42)
    result = grey_box.transition(sample_world_state, [action], [], rng)

    # Initial: wh1=100, wh2=50
    # Known transfer: wh1=70, wh2=80
    # Learned residual: stock_wh2 - 2.0 = 78.0
    assert result.next_state.get_resource("stock_wh1").current == 70.0
    assert result.next_state.get_resource("stock_wh2").current == 78.0

    # Evidence shifts to PREDICTIVE because empirical residual was applied
    assert result.evidence_level == EvidenceLevel.PREDICTIVE
    assert "learned_residual" in result.applied_changes
    assert result.applied_changes["learned_residual"]["stock_wh2"] == -2.0


def test_linear_residual_estimator_fit_on_discrepancy(sample_world_state: WorldState) -> None:
    """Test LinearResidualEstimator fits on the difference between reality and known model."""
    known_model = DeterministicTransferDynamics()

    # Generate synthetic training data where actual environment has 10% inventory evaporation on wh1
    dataset = TransitionDataset()
    for stock in [60.0, 80.0, 100.0, 120.0]:
        init_s = sample_world_state.update_resource("stock_wh1", new_value=stock)
        # Reality: wh1 drops by an unmodeled 5.0 units
        actual_next = init_s.update_resource("stock_wh1", delta=-5.0)
        sample = TransitionSample(
            state=init_s,
            actions=(),
            events=(),
            next_state=actual_next,
        )
        dataset.add(sample)

    estimator = LinearResidualEstimator(target_resources=["stock_wh1"])
    estimator.fit(dataset=dataset, known_model=known_model)

    assert estimator.is_fitted

    pred = estimator.predict_residual(
        state=sample_world_state,
        actions=(),
        events=(),
        known_next_state=sample_world_state,
    )

    # Should predict approximately -5.0 delta
    assert "stock_wh1" in pred
    assert pytest.approx(pred["stock_wh1"], abs=0.1) == -5.0
