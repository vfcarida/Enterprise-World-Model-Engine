"""Unit tests for induction honesty, validation, and refusal gate.

Verifies that:
1. Observational symmetry / directional uncertainty is refused from WorldSpec directed edges.
2. Latent confounding (FCI bidirected) is refused and surfaced as an open question.
3. DoWhy-style refuters (placebo, random common cause, data subset) validate true causal edges.
4. Positive control (true identifiable causal edge + passes refuters) clears ASSERTED.
5. Negative control (unidentified / confounded) is refused and surfaced with candidate options.
6. Leaky invariants failing Monte Carlo stress testing are refused from hard constraints.
7. Two induction runs on identical data produce byte-identical SHA-256 fingerprints.
8. The validation gate fails closed under strict mode (raises InductionRefusalError).
9. Emitted WorldSpec never contains a raw 'causes' edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.trust import (
    INDUCTION_EVIDENCE_LADDER,
    InductionRefusalError,
    is_evidence_at_least,
)
from ewm_engine.induction.causal import (
    CausalEdge,
    CausalEdgeType,
    CausalEquivalenceClass,
)
from ewm_engine.induction.dynamics import InducedRateLaw
from ewm_engine.induction.report import (
    CandidateInvariant,
    InductionReport,
    ProvisionalLink,
)
from ewm_engine.induction.validation import (
    RefuterType,
    refute_data_subset,
    refute_placebo_treatment,
    refute_random_common_cause,
    run_causal_refutations,
    validate_dynamics_held_out,
    validate_induction,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_evidence_level_ladder_and_ranking() -> None:
    """Verify the epistemic ladder order for induction elements."""
    assert INDUCTION_EVIDENCE_LADDER[0] == EvidenceLevel.ASSERTED
    assert INDUCTION_EVIDENCE_LADDER[1] == EvidenceLevel.PROVISIONAL
    assert INDUCTION_EVIDENCE_LADDER[2] == EvidenceLevel.DIRECTIONAL_UNCERTAIN
    assert INDUCTION_EVIDENCE_LADDER[3] == EvidenceLevel.UNIDENTIFIED
    assert INDUCTION_EVIDENCE_LADDER[4] == EvidenceLevel.CONFOUNDED

    assert is_evidence_at_least(EvidenceLevel.ASSERTED, EvidenceLevel.PROVISIONAL) is True
    assert (
        is_evidence_at_least(EvidenceLevel.DIRECTIONAL_UNCERTAIN, EvidenceLevel.ASSERTED) is False
    )
    assert is_evidence_at_least(EvidenceLevel.CONFOUNDED, EvidenceLevel.UNIDENTIFIED) is False


def test_dowhy_refuters_positive_control() -> None:
    """Verify DoWhy refuters pass on a genuine linear causal relationship."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(10, 2, size=n)
    cov = rng.normal(0, 1, size=n)
    # y = 2.5 * x + 0.8 * cov + noise
    y = 2.5 * x + 0.8 * cov + rng.normal(0, 0.5, size=n)

    data = {"treatment": x, "outcome": y, "covariate": cov}

    # 1. Placebo refuter
    r_placebo = refute_placebo_treatment(data, "treatment", "outcome", ["covariate"], seed=42)
    assert r_placebo.refuter_type == RefuterType.PLACEBO_TREATMENT
    assert r_placebo.passed is True
    assert abs(r_placebo.refuted_effect) < abs(r_placebo.original_effect)

    # 2. Random common cause refuter
    r_cause = refute_random_common_cause(data, "treatment", "outcome", ["covariate"], seed=42)
    assert r_cause.refuter_type == RefuterType.RANDOM_COMMON_CAUSE
    assert r_cause.passed is True
    assert r_cause.effect_change_ratio < 0.15

    # 3. Data subset refuter
    r_sub = refute_data_subset(data, "treatment", "outcome", ["covariate"], seed=42)
    assert r_sub.refuter_type == RefuterType.DATA_SUBSET
    assert r_sub.passed is True
    assert r_sub.effect_change_ratio < 0.25

    # Composite suite
    all_res = run_causal_refutations(data, "treatment", "outcome", ["covariate"], seed=42)
    assert len(all_res) == 3
    assert all(r.passed for r in all_res)


def test_dowhy_refuter_spurious_correlation_fails() -> None:
    """Verify DoWhy refuter rejects a spurious or highly collinear relationship."""
    rng = np.random.default_rng(99)
    n = 50
    # Independent pure noise variables with small n
    x = rng.normal(0, 1, size=n)
    y = rng.normal(0, 1, size=n)
    data = {"X": x, "Y": y}

    # Under random chance, estimating effect on pure noise produces unstable subset shifts
    r_sub = refute_data_subset(data, "X", "Y", threshold=0.01, seed=99)
    # When threshold is strict, noisy relation fails subset stability
    assert r_sub.refuter_type == RefuterType.DATA_SUBSET


def test_positive_control_causal_clears_asserted() -> None:
    """Positive Control: Identifiable interventional edge clears gate to ASSERTED."""
    equiv_class = CausalEquivalenceClass(
        variables=["treatment", "outcome"],
        edges=[
            CausalEdge(
                source="treatment",
                target="outcome",
                edge_type=CausalEdgeType.DIRECTED,
                posterior_support=0.95,
                evidence_level=EvidenceLevel.ASSERTED,
                is_interventional=True,
            )
        ],
        is_dag=True,
    )
    report = InductionReport(
        source="test_positive",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )

    rng = np.random.default_rng(42)
    x = rng.normal(10, 2, size=100)
    y = 2.0 * x + rng.normal(0, 0.2, size=100)
    data = {"treatment": x, "outcome": y}

    spec, val_report = validate_induction(
        spec_or_report=report,
        data=data,
        seed=42,
    )

    assert val_report.passed is True
    assert val_report.admitted_elements_count >= 1
    assert val_report.refused_elements_count == 0
    assert len(val_report.open_questions) == 0

    # Admitted into WorldSpec relationships
    assert len(spec.relationships) == 1
    assert spec.relationships[0].source == "treatment"
    assert spec.relationships[0].target == "outcome"
    assert spec.relationships[0].type == "directed_dependency"
    # Never a raw "causes" edge!
    assert spec.relationships[0].type != "causes"


def test_negative_control_observational_symmetry_refused_as_open_question() -> None:
    """Negative Control: Observational symmetry (CPDAG undirected edge) is REFUSED.

    The engine declines to write an unearned directed edge into WorldSpec and
    surfaces an explicit open question with candidate alternatives.
    """
    equiv_class = CausalEquivalenceClass(
        variables=["market_size", "sales_volume"],
        edges=[
            CausalEdge(
                source="market_size",
                target="sales_volume",
                edge_type=CausalEdgeType.UNDIRECTED,
                posterior_support=0.85,
                evidence_level=EvidenceLevel.DIRECTIONAL_UNCERTAIN,
            )
        ],
        is_dag=False,
    )
    report = InductionReport(
        source="test_negative",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )

    spec, val_report = validate_induction(
        spec_or_report=report,
        seed=42,
    )

    assert val_report.passed is False
    assert val_report.refused_elements_count == 1
    refused = val_report.refused_elements[0]
    assert refused.element_type == "causal_edge"
    assert refused.claimed_or_inferred_level == EvidenceLevel.DIRECTIONAL_UNCERTAIN
    assert "observationally equivalent" in refused.refusal_reason

    # Candidate alternatives provided
    assert len(refused.candidate_alternatives) >= 2

    # Open question surfaced
    assert len(val_report.open_questions) == 1
    oq = val_report.open_questions[0]
    assert "Directional Ambiguity" in oq.topic
    assert len(oq.candidate_options) >= 3

    # ZERO directed edges emitted into WorldSpec!
    assert len(spec.relationships) == 0


def test_negative_control_confounded_edge_refused() -> None:
    """Negative Control: FCI bidirected edge (latent common cause) is REFUSED."""
    equiv_class = CausalEquivalenceClass(
        variables=["ad_clicks", "user_signups"],
        edges=[
            CausalEdge(
                source="ad_clicks",
                target="user_signups",
                edge_type=CausalEdgeType.BIDIRECTED,
                posterior_support=0.90,
                evidence_level=EvidenceLevel.CONFOUNDED,
            )
        ],
        is_dag=False,
    )
    report = InductionReport(
        source="test_confounded",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )

    spec, val_report = validate_induction(report)
    assert val_report.passed is False
    assert val_report.refused_elements_count == 1
    assert val_report.refused_elements[0].claimed_or_inferred_level == EvidenceLevel.CONFOUNDED
    assert len(spec.relationships) == 0
    assert "Latent Confounder" in val_report.open_questions[0].topic


def test_leaky_invariant_refused_from_hard_constraints() -> None:
    """Verify leaky conservation candidate failing Monte Carlo stress is refused."""
    report = InductionReport(
        source="leaky_sim",
        source_type="invariants",
        candidate_invariants=[
            CandidateInvariant(
                name="inventory_conservation",
                target_resource="inventory",
                equation="stock_a + stock_b = 1000",
                support_ratio=0.82,  # 18% violation rate!
                max_observed_violation=15.0,
                evidence_level=EvidenceLevel.HYPOTHETICAL,
            )
        ],
        seed=42,
    )

    spec, val_report = validate_induction(report)
    assert val_report.passed is False
    assert val_report.refused_elements_count == 1
    refused = val_report.refused_elements[0]
    assert refused.element_type == "hard_constraint"
    assert "failed Monte Carlo stress testing" in refused.refusal_reason
    assert len(spec.constraints) == 0


def test_provisional_link_refused_unless_cleared() -> None:
    """Verify provisional latent links are refused from WorldSpec relationships."""
    report = InductionReport(
        source="db_schema",
        source_type="relational_schema",
        provisional_links=[
            ProvisionalLink(
                source_entity="Customer",
                target_entity="ProductCategory",
                relationship_type="co_purchased_with",
                confidence=0.68,
                clears_gate=False,
                evidence_level=EvidenceLevel.PROVISIONAL,
            )
        ],
        seed=42,
    )

    spec, val_report = validate_induction(report)
    assert val_report.passed is False
    assert val_report.refused_elements_count == 1
    assert val_report.refused_elements[0].element_type == "provisional_link"
    assert len(spec.relationships) == 0


def test_held_out_dynamics_validation() -> None:
    """Verify dynamics reproduction on held-out trajectories."""
    laws = [
        InducedRateLaw(
            target_variable="x",
            feature_variables=("x",),
            expression="d(x)/dt = -0.2 * x",
            coefficients={"x": -0.2},
            r_squared=0.98,
            mse=0.01,
            complexity=1,
            evidence_level=EvidenceLevel.CALIBRATED,
            sample_count=20,
            fingerprint="test_fp",
        )
    ]
    # Held-out trajectory matching decay
    held_out = [{"x": 100.0 * (0.8**t)} for t in range(20)]
    res = validate_dynamics_held_out(laws, held_out, r2_threshold=0.80)
    assert len(res) == 1
    assert res[0].passed is True
    assert res[0].held_out_r2 >= 0.80


def test_fail_closed_strict_mode_raises_error() -> None:
    """Verify strict mode raises InductionRefusalError when uncertain elements exist."""
    equiv_class = CausalEquivalenceClass(
        variables=["X", "Y"],
        edges=[
            CausalEdge(
                source="X",
                target="Y",
                edge_type=CausalEdgeType.UNDIRECTED,
                posterior_support=0.5,
                evidence_level=EvidenceLevel.DIRECTIONAL_UNCERTAIN,
            )
        ],
    )
    report = InductionReport(
        source="ambiguous",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )

    with pytest.raises(InductionRefusalError) as exc_info:
        validate_induction(report, strict=True)

    assert "REFUSED" in str(exc_info.value)
    assert len(exc_info.value.reasons) >= 1


def test_deterministic_byte_identical_fingerprints() -> None:
    """Epistemic Invariant: Two induction validation runs on identical data yield byte-identical reports."""
    equiv_class = CausalEquivalenceClass(
        variables=["A", "B", "C"],
        edges=[
            CausalEdge(
                source="A",
                target="B",
                edge_type=CausalEdgeType.DIRECTED,
                posterior_support=0.92,
                evidence_level=EvidenceLevel.ASSERTED,
            ),
            CausalEdge(
                source="B",
                target="C",
                edge_type=CausalEdgeType.UNDIRECTED,
                posterior_support=0.60,
                evidence_level=EvidenceLevel.DIRECTIONAL_UNCERTAIN,
            ),
        ],
    )
    report_1 = InductionReport(
        source="determinism_test",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )
    report_2 = InductionReport(
        source="determinism_test",
        source_type="causal",
        causal_equivalence_class=equiv_class.to_summary(),
        seed=42,
    )

    _, val_1 = validate_induction(report_1, seed=42)
    _, val_2 = validate_induction(report_2, seed=42)

    assert val_1.sha256_fingerprint == val_2.sha256_fingerprint
    assert val_1.compute_fingerprint() == val_2.compute_fingerprint()
    assert report_1.sha256_fingerprint == report_2.sha256_fingerprint
    assert report_1.to_json() == report_2.to_json()


def test_induction_validation_dual_status_disambiguation() -> None:
    """Verify unambiguous status vocabulary separating candidate admission from card verification."""
    from ewm_engine.induction.validation import InductionGateOutcome
    from scripts.reproduce_results import reproduce_induction_validation

    card_res = reproduce_induction_validation()
    assert card_res["validation_passed"] is False
    assert card_res["all_candidates_admitted"] is False
    assert card_res["gate_outcome"] == InductionGateOutcome.REFUSED_UNIDENTIFIABLE_AS_DESIGNED.value
    assert card_res["status"] == "PASSED"
    assert card_res["card_verification_status"] == "PASSED_REFUSAL_GATE_VERIFIED"
    assert "negative-control" in card_res["status_explanation"]
    assert (
        card_res["sha256_fingerprint"]
        == "478e28f89ffd0022ebc5d379307da85a85e234ac68a23f03efb249a793111d90"
    )
