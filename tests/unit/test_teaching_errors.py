"""Unit tests verifying structured TeachingError hierarchy and refusal mechanisms."""

from __future__ import annotations

import pytest

from ewm_engine.calibration.certification import CoverageCertificate
from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.exceptions import (
    ConstraintViolationError,
    ConstraintViolationTeachingError,
    EWMError,
    NonIdentifiableCausalQueryError,
    RefusalToRankError,
    TeachingError,
    UncertifiedCalibrationError,
)
from ewm_engine.experimental.causal import IdentifiabilityResult, NotIdentifiableResult
from ewm_engine.provenance.evidence import EvidenceLevel


def test_teaching_error_formatting() -> None:
    err = TeachingError(
        reason="Model invariant breached.",
        remediation="Check resource capacities.",
        doc_url="https://example.com/docs",
        title="Custom Teaching Refusal",
    )
    msg = str(err)
    assert "[EWM Refusal] Custom Teaching Refusal" in msg
    assert "Why I refused: Model invariant breached." in msg
    assert "How to resolve: Check resource capacities." in msg
    assert "Documentation: https://example.com/docs" in msg
    assert isinstance(err, EWMError)


def test_constraint_violation_teaching_error() -> None:
    res = ConstraintResult(
        satisfied=False,
        constraint_id="mass_balance",
        severity=ConstraintSeverity.HARD,
        step=5,
        violating_entities=("hub_east", "hub_west"),
    )
    with pytest.raises(ConstraintViolationError) as exc_info:
        res.raise_for_status()

    err = exc_info.value
    assert isinstance(err, ConstraintViolationTeachingError)
    assert isinstance(err, TeachingError)
    assert err.constraint_id == "mass_balance"
    assert err.step == 5
    assert "hub_east" in err.violating_entities
    assert "Why I refused:" in str(err)
    assert "How to resolve:" in str(err)


def test_uncertified_calibration_teaching_error() -> None:
    cert = CoverageCertificate(
        is_certified=False,
        status="UNCERTIFIED_OVERCONFIDENT",
        decision_usable=False,
        tarp_max_discrepancy=0.40,
        tarp_ks_pvalue=0.0001,
        sbc_rank_ks_pvalue=0.0001,
        misspecification_score=0.2,
        is_misspecified=False,
        refusal_reason="Empirical coverage significantly below nominal credibility.",
        nominal_alphas=(0.1, 0.5, 0.9),
        empirical_coverages=(0.01, 0.20, 0.50),
    )
    with pytest.raises(UncertifiedCalibrationError) as exc_info:
        cert.raise_for_status()

    err = exc_info.value
    assert isinstance(err, TeachingError)
    assert "Uncertified Calibration Refusal" in str(err)
    assert "Why I refused:" in str(err)
    assert "How to resolve:" in str(err)


def test_non_identifiable_causal_query_teaching_error() -> None:
    id_res = IdentifiabilityResult(
        is_identifiable=False,
        treatment="interest_rate",
        outcome="default_rate",
        conditioning_set=(),
        open_backdoor_paths=(("interest_rate", "macro_index", "default_rate"),),
        causal_level=EvidenceLevel.PREDICTIVE,
        diagnostic_message="Open backdoor path through macro_index.",
    )
    not_id = NotIdentifiableResult(
        treatment="interest_rate",
        outcome="default_rate",
        identifiability=id_res,
        diagnostic_message="Refused due to confounding.",
    )
    with pytest.raises(NonIdentifiableCausalQueryError) as exc_info:
        not_id.raise_for_status()

    err = exc_info.value
    assert isinstance(err, TeachingError)
    assert "interest_rate" in str(err)
    assert "default_rate" in str(err)
    assert "macro_index" in str(err)
    assert "Why I refused:" in str(err)
    assert "How to resolve:" in str(err)


def test_refusal_to_rank_error() -> None:
    err = RefusalToRankError(disagreements={"MaxRegret": "Policy A", "Satisficing": "Policy B"})
    assert isinstance(err, TeachingError)
    assert "Epistemic Refusal to Rank Decisions" in str(err)
    assert "Policy A" in str(err)
    assert "Policy B" in str(err)
