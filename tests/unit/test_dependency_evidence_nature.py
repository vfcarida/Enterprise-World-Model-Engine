"""Unit, contract, and property tests for DependencyEvidenceNature and query gating."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ewm_engine.core import (
    EVIDENCE_LEVEL_TO_NATURE,
    EVIDENCE_NATURE_TO_DEFAULT_LEVEL,
    PERMISSIBLE_QUERIES_BY_NATURE,
    CausalGraph,
    DependencyEvidenceNature,
    DependencyQueryKind,
    EvidenceNature,
    InsufficientCausalEvidenceWarning,
    Relationship,
    StructuralAssignment,
    StructuralCausalModel,
    UnqualifiedCausalClaimError,
    evidence_level_to_nature,
    evidence_nature_to_level,
    validate_causal_claim,
    validate_dependency_for_query,
)
from ewm_engine.provenance import EvidenceLevel


@pytest.mark.contract
def test_dependency_evidence_nature_taxonomy_completeness() -> None:
    """Verify the required evidence-nature taxonomy values and alias."""
    assert EvidenceNature is DependencyEvidenceNature
    required = {
        "KNOWN_FUNCTIONAL",
        "STATISTICAL_ASSOCIATION",
        "CAUSAL_HYPOTHESIS",
        "EXPERIMENTALLY_SUPPORTED",
        "BUSINESS_RULE_MECHANISM",
    }
    assert required.issubset(set(DependencyEvidenceNature.__members__.keys()))
    assert set(EVIDENCE_NATURE_TO_DEFAULT_LEVEL.keys()) == set(DependencyEvidenceNature)
    assert set(EVIDENCE_LEVEL_TO_NATURE.keys()) == set(EvidenceLevel)
    assert set(PERMISSIBLE_QUERIES_BY_NATURE.keys()) == set(DependencyEvidenceNature)


@pytest.mark.contract
def test_bidirectional_reconciliation_with_evidence_level() -> None:
    """Verify documented mapping between DependencyEvidenceNature and EvidenceLevel."""
    for level in EvidenceLevel:
        mapped_nature = evidence_level_to_nature(level)
        assert isinstance(mapped_nature, DependencyEvidenceNature)

    # STATISTICAL_ASSOCIATION must never map above PREDICTIVE even with identification flag
    assert (
        evidence_nature_to_level(
            DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
            has_identification_evidence=False,
        )
        == EvidenceLevel.PREDICTIVE
    )
    assert (
        evidence_nature_to_level(
            DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
            has_identification_evidence=True,
        )
        == EvidenceLevel.PREDICTIVE
    )

    # CAUSAL_HYPOTHESIS maps to ASSUMED without identification, QUASI_CAUSAL with identification
    assert (
        evidence_nature_to_level(
            DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
            has_identification_evidence=False,
        )
        == EvidenceLevel.ASSUMED
    )
    assert (
        evidence_nature_to_level(
            DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
            has_identification_evidence=True,
        )
        == EvidenceLevel.QUASI_CAUSAL
    )

    assert (
        evidence_nature_to_level(DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED)
        == EvidenceLevel.INTERVENTIONAL
    )
    assert (
        evidence_nature_to_level(DependencyEvidenceNature.BUSINESS_RULE_MECHANISM)
        == EvidenceLevel.ASSERTED
    )
    assert (
        evidence_nature_to_level(DependencyEvidenceNature.KNOWN_FUNCTIONAL)
        == EvidenceLevel.STRUCTURAL
    )


def test_statistical_association_refused_for_interventional_and_counterfactual() -> None:
    """STATISTICAL_ASSOCIATION must never be silently treated as causal."""
    for q_obs in (
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
        DependencyQueryKind.STATE_EVOLUTION,
    ):
        lvl = validate_dependency_for_query(
            DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
            q_obs,
        )
        assert lvl == EvidenceLevel.PREDICTIVE

    for q_causal in (
        DependencyQueryKind.INTERVENTIONAL_DO,
        DependencyQueryKind.COUNTERFACTUAL_TWIN,
    ):
        with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
            validate_dependency_for_query(
                DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
                q_causal,
                is_identifiable=True,
                overlap_satisfied=True,
                identification_provenance="Attempted override",
            )

        with pytest.warns(InsufficientCausalEvidenceWarning):
            downgraded = validate_dependency_for_query(
                DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
                q_causal,
                is_identifiable=True,
                overlap_satisfied=True,
                identification_provenance="Attempted override",
                strict=False,
            )
            assert downgraded == EvidenceLevel.PREDICTIVE

    # Also through validate_causal_claim
    with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
        validate_causal_claim(
            EvidenceLevel.INTERVENTIONAL,
            is_identifiable=True,
            overlap_satisfied=True,
            evidence_nature=DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
            identification_provenance="DAG adjustment",
        )


def test_causal_hypothesis_requires_identification_evidence_for_intervention() -> None:
    """CAUSAL_HYPOTHESIS cannot be used for interventional claims without identification evidence."""
    # Refused when unidentifiable
    with pytest.raises(UnqualifiedCausalClaimError, match="CAUSAL_HYPOTHESIS"):
        validate_dependency_for_query(
            DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
            DependencyQueryKind.INTERVENTIONAL_DO,
            is_identifiable=False,
            overlap_satisfied=True,
            identification_provenance="Backdoor set {Z}",
        )

    # Refused when provenance is missing
    with pytest.raises(UnqualifiedCausalClaimError, match="identification_provenance"):
        validate_dependency_for_query(
            DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
            DependencyQueryKind.INTERVENTIONAL_DO,
            is_identifiable=True,
            overlap_satisfied=True,
            identification_provenance=None,
        )

    # Admitted as QUASI_CAUSAL when identifiable + overlap + provenance are all present
    lvl = validate_dependency_for_query(
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        DependencyQueryKind.INTERVENTIONAL_DO,
        is_identifiable=True,
        overlap_satisfied=True,
        identification_provenance="Backdoor adjustment set {region, season} + DoWhy refuters passed",
    )
    assert lvl == EvidenceLevel.QUASI_CAUSAL

    # Still refused for unit-level Rung-3 counterfactual twin without structural functional equation
    with pytest.raises(UnqualifiedCausalClaimError, match="CAUSAL_HYPOTHESIS"):
        validate_dependency_for_query(
            DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
            DependencyQueryKind.COUNTERFACTUAL_TWIN,
            is_identifiable=True,
            overlap_satisfied=True,
            identification_provenance="Backdoor adjustment set {region}",
        )


def test_relationship_evidence_nature_and_query_gate() -> None:
    """Untagged Relationship defaults fail-closed to STATISTICAL_ASSOCIATION and gates queries."""
    rel = Relationship(source="price", target="demand", type="correlates_with")
    assert rel.evidence_nature == DependencyEvidenceNature.STATISTICAL_ASSOCIATION
    assert rel.effective_evidence_level == EvidenceLevel.PREDICTIVE

    assert (
        rel.validate_for_query(DependencyQueryKind.OBSERVATIONAL_PREDICTION)
        == EvidenceLevel.PREDICTIVE
    )
    with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
        rel.validate_for_query(DependencyQueryKind.INTERVENTIONAL_DO)

    # Upgrade to CAUSAL_HYPOTHESIS without identification -> still blocked for intervention
    rel_hyp = rel.with_evidence_nature(
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        is_identifiable=False,
    )
    assert rel_hyp.evidence_nature == DependencyEvidenceNature.CAUSAL_HYPOTHESIS
    with pytest.raises(UnqualifiedCausalClaimError, match="CAUSAL_HYPOTHESIS"):
        rel_hyp.validate_for_query(DependencyQueryKind.INTERVENTIONAL_DO)

    # Upgrade to CAUSAL_HYPOTHESIS with identification -> permitted as QUASI_CAUSAL
    rel_id = rel.with_evidence_nature(
        DependencyEvidenceNature.CAUSAL_HYPOTHESIS,
        is_identifiable=True,
        overlap_satisfied=True,
        identification_provenance="IV: supplier_cost_shock",
    )
    assert (
        rel_id.validate_for_query(DependencyQueryKind.INTERVENTIONAL_DO)
        == EvidenceLevel.QUASI_CAUSAL
    )

    # Upgrade to BUSINESS_RULE_MECHANISM -> permitted as ASSERTED
    rel_rule = rel.with_evidence_nature(DependencyEvidenceNature.BUSINESS_RULE_MECHANISM)
    assert (
        rel_rule.validate_for_query(DependencyQueryKind.INTERVENTIONAL_DO) == EvidenceLevel.ASSERTED
    )


def test_scm_intervene_enforces_dependency_evidence_nature() -> None:
    """StructuralCausalModel.intervene refuses surgery when downstream relies on STATISTICAL_ASSOCIATION."""
    graph = CausalGraph(
        nodes={"resource:price", "resource:demand"},
        edges={("resource:price", "resource:demand")},
    )
    asgn_price = StructuralAssignment(
        target="resource:price",
        parents=(),
        mechanism_id="pricing_policy",
        evidence_nature=DependencyEvidenceNature.BUSINESS_RULE_MECHANISM,
    )
    asgn_demand_assoc = StructuralAssignment(
        target="resource:demand",
        parents=("resource:price",),
        mechanism_id="ols_regression",
        evidence_nature=DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
    )
    scm_assoc = StructuralCausalModel(
        name=" observational_scm",
        endogenous_variables={"resource:price": "resource", "resource:demand": "resource"},
        assignments={"resource:price": asgn_price, "resource:demand": asgn_demand_assoc},
        causal_graph=graph,
    )

    with pytest.raises(UnqualifiedCausalClaimError, match="STATISTICAL_ASSOCIATION"):
        scm_assoc.intervene({"resource:price": 120.0})

    # When downstream mechanism is EXPERIMENTALLY_SUPPORTED, intervention succeeds
    asgn_demand_exp = StructuralAssignment(
        target="resource:demand",
        parents=("resource:price",),
        mechanism_id="rct_price_elasticity",
        evidence_nature=DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED,
        overlap_satisfied=True,
    )
    scm_exp = StructuralCausalModel(
        name="experimental_scm",
        endogenous_variables={"resource:price": "resource", "resource:demand": "resource"},
        assignments={"resource:price": asgn_price, "resource:demand": asgn_demand_exp},
        causal_graph=graph,
    )
    mutilated = scm_exp.intervene({"resource:price": 120.0})
    assert mutilated.assignments["resource:price"].is_intervened is True


@pytest.mark.property
@given(
    nature=st.sampled_from(list(DependencyEvidenceNature)),
    query_kind=st.sampled_from(list(DependencyQueryKind)),
    is_identifiable=st.booleans(),
    overlap_satisfied=st.booleans(),
    has_provenance=st.booleans(),
)
def test_property_dependency_evidence_nature_constrains_permissible_queries(
    nature: DependencyEvidenceNature,
    query_kind: DependencyQueryKind,
    is_identifiable: bool,
    overlap_satisfied: bool,
    has_provenance: bool,
) -> None:
    """Property: a dependency's evidence-nature strictly constrains which queries may use it."""
    prov = "Explicit identification certificate" if has_provenance else None

    if query_kind in (
        DependencyQueryKind.OBSERVATIONAL_PREDICTION,
        DependencyQueryKind.STATE_EVOLUTION,
    ):
        level = validate_dependency_for_query(
            nature,
            query_kind,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=prov,
        )
        if nature == DependencyEvidenceNature.STATISTICAL_ASSOCIATION:
            assert level == EvidenceLevel.PREDICTIVE
        return

    # Interventional or Counterfactual query
    if nature == DependencyEvidenceNature.STATISTICAL_ASSOCIATION:
        with pytest.raises(UnqualifiedCausalClaimError):
            validate_dependency_for_query(
                nature,
                query_kind,
                is_identifiable=is_identifiable,
                overlap_satisfied=overlap_satisfied,
                identification_provenance=prov,
            )
    elif nature == DependencyEvidenceNature.CAUSAL_HYPOTHESIS:
        if query_kind == DependencyQueryKind.COUNTERFACTUAL_TWIN or not (
            is_identifiable and overlap_satisfied and has_provenance
        ):
            with pytest.raises(UnqualifiedCausalClaimError):
                validate_dependency_for_query(
                    nature,
                    query_kind,
                    is_identifiable=is_identifiable,
                    overlap_satisfied=overlap_satisfied,
                    identification_provenance=prov,
                )
        else:
            lvl = validate_dependency_for_query(
                nature,
                query_kind,
                is_identifiable=is_identifiable,
                overlap_satisfied=overlap_satisfied,
                identification_provenance=prov,
            )
            assert lvl == EvidenceLevel.QUASI_CAUSAL
    elif nature == DependencyEvidenceNature.EXPERIMENTALLY_SUPPORTED:
        if not overlap_satisfied or (
            query_kind == DependencyQueryKind.COUNTERFACTUAL_TWIN
            and not (is_identifiable and has_provenance)
        ):
            with pytest.raises(UnqualifiedCausalClaimError):
                validate_dependency_for_query(
                    nature,
                    query_kind,
                    is_identifiable=is_identifiable,
                    overlap_satisfied=overlap_satisfied,
                    identification_provenance=prov,
                )
        else:
            lvl = validate_dependency_for_query(
                nature,
                query_kind,
                is_identifiable=is_identifiable,
                overlap_satisfied=overlap_satisfied,
                identification_provenance=prov,
            )
            assert lvl == EvidenceLevel.INTERVENTIONAL
    else:
        # BUSINESS_RULE_MECHANISM or KNOWN_FUNCTIONAL
        if not overlap_satisfied:
            with pytest.raises(UnqualifiedCausalClaimError):
                validate_dependency_for_query(
                    nature,
                    query_kind,
                    is_identifiable=is_identifiable,
                    overlap_satisfied=overlap_satisfied,
                    identification_provenance=prov,
                )
        else:
            lvl = validate_dependency_for_query(
                nature,
                query_kind,
                is_identifiable=is_identifiable,
                overlap_satisfied=overlap_satisfied,
                identification_provenance=prov,
            )
            assert lvl in (EvidenceLevel.ASSERTED, EvidenceLevel.STRUCTURAL)
