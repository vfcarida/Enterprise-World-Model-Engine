"""Contract tests for Induction Honesty, Validation, Refusal Gate, and MCP integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewm_engine.cli.main import main
from ewm_engine.core.trust import (
    INDUCTION_EVIDENCE_LADDER,
    EvidenceLevel,
    InductionRefusalError,
    is_evidence_at_least,
)
from ewm_engine.induction.causal import (
    CausalEdge,
    CausalEdgeType,
    CausalEquivalenceClass,
    induce_causal,
)
from ewm_engine.induction.report import InductionReport
from ewm_engine.induction.validation import (
    InductionValidationGate,
    InductionValidationReport,
    validate_induction,
)
from ewm_engine.mcp.server import create_mcp_server


@pytest.mark.contract
def test_evidence_ladder_contract() -> None:
    """Verify the formal EvidenceLevel ladder order and ordering helper."""
    assert list(INDUCTION_EVIDENCE_LADDER) == [
        EvidenceLevel.ASSERTED,
        EvidenceLevel.PROVISIONAL,
        EvidenceLevel.DIRECTIONAL_UNCERTAIN,
        EvidenceLevel.UNIDENTIFIED,
        EvidenceLevel.CONFOUNDED,
    ]
    # ASSERTED is highest rank
    assert is_evidence_at_least(EvidenceLevel.ASSERTED, EvidenceLevel.PROVISIONAL)
    assert is_evidence_at_least(EvidenceLevel.ASSERTED, EvidenceLevel.ASSERTED)
    assert not is_evidence_at_least(EvidenceLevel.UNIDENTIFIED, EvidenceLevel.PROVISIONAL)
    assert not is_evidence_at_least(EvidenceLevel.CONFOUNDED, EvidenceLevel.ASSERTED)


@pytest.mark.contract
def test_validation_report_serialization_and_roundtrip() -> None:
    """Verify InductionValidationReport serializes to valid JSON, maintains fingerprint, and roundtrips."""
    report = InductionReport(
        source="test_contract",
        source_type="causal",
        entities_count=3,
        relationships_count=2,
    )
    _validated_spec, val_rep = validate_induction(report, seed=42)

    assert isinstance(val_rep, InductionValidationReport)
    assert val_rep.sha256_fingerprint != ""
    assert len(val_rep.sha256_fingerprint) == 64

    # JSON round-trip
    val_json = val_rep.to_json()
    val_dict = json.loads(val_json)
    reloaded = InductionValidationReport.model_validate(val_dict)

    assert reloaded.sha256_fingerprint == val_rep.sha256_fingerprint
    assert reloaded.passed == val_rep.passed
    assert reloaded.refused_elements_count == val_rep.refused_elements_count
    assert len(reloaded.open_questions) == len(val_rep.open_questions)


@pytest.mark.contract
def test_positive_control_and_negative_control_refusal() -> None:
    """Verify positive control passes and negative control fails closed with open questions."""
    # Positive control: identifiable relationship with strong directional certainty
    edge_pos = CausalEdge(
        source="driver",
        target="outcome",
        edge_type=CausalEdgeType.DIRECTED,
        evidence_level=EvidenceLevel.ASSERTED,
        posterior_support=0.98,
        direction_support=0.95,
    )
    eq_pos = CausalEquivalenceClass(
        variables=["driver", "outcome"],
        edges=[edge_pos],
        markov_equivalent_dags_count=1,
    )
    report_pos = InductionReport(
        source="test_pos",
        source_type="causal",
        entities_count=2,
        relationships_count=1,
        causal_equivalence_class=eq_pos.to_summary(),
    )

    gate = InductionValidationGate(strict=True, seed=42)
    val_pos = gate.validate(report_pos)
    assert val_pos.passed is True
    assert val_pos.refused_elements_count == 0
    assert len(val_pos.open_questions) == 0

    # Negative control: unoriented edge with equivalence-class ambiguity
    edge_neg = CausalEdge(
        source="var_a",
        target="var_b",
        edge_type=CausalEdgeType.UNDIRECTED,
        evidence_level=EvidenceLevel.DIRECTIONAL_UNCERTAIN,
        posterior_support=0.55,
        direction_support=0.50,
    )
    eq_neg = CausalEquivalenceClass(
        variables=["var_a", "var_b"],
        edges=[edge_neg],
        markov_equivalent_dags_count=2,
    )
    report_neg = InductionReport(
        source="test_neg",
        source_type="causal",
        entities_count=2,
        relationships_count=1,
        causal_equivalence_class=eq_neg.to_summary(),
    )

    # In strict mode, gate must fail closed and raise InductionRefusalError
    with pytest.raises(InductionRefusalError) as exc_info:
        gate.validate(report_neg)
    assert "refused 1 candidate element(s)" in str(exc_info.value)

    # In non-strict mode, returns report marking refused items and open questions
    gate_lenient = InductionValidationGate(strict=False, seed=42)
    val_neg = gate_lenient.validate(report_neg)
    assert val_neg.passed is False
    assert val_neg.refused_elements_count == 1
    assert len(val_neg.open_questions) == 1
    question = val_neg.open_questions[0]
    assert "Directional Ambiguity" in question.topic
    assert len(question.candidate_options) >= 2


@pytest.mark.contract
def test_cli_induce_validate_contract(tmp_path: Path) -> None:
    """Verify `ewm induce validate` CLI command for report validation."""
    # Create an initial induction report
    data = [
        {"x": 1.0, "y": 2.0},
        {"x": 2.0, "y": 4.0},
        {"x": 3.0, "y": 6.0},
        {"x": 4.0, "y": 8.0},
    ]
    _, initial_rep = induce_causal(data, seed=42)
    rep_file = tmp_path / "initial_rep.json"
    rep_file.write_text(initial_rep.to_json(), encoding="utf-8")

    out_spec = tmp_path / "validated_world.yaml"
    out_val_rep = tmp_path / "validation_rep.json"

    # Non-strict run
    exit_code = main(
        [
            "induce",
            "validate",
            "--report",
            str(rep_file),
            "--out",
            str(out_spec),
            "--report-out",
            str(out_val_rep),
            "--seed",
            "42",
        ]
    )
    assert exit_code == 0
    assert out_spec.exists()
    assert out_val_rep.exists()
    val_data = json.loads(out_val_rep.read_text(encoding="utf-8"))
    assert "sha256_fingerprint" in val_data
    assert "refused_elements_count" in val_data
    assert "open_questions" in val_data

    # Strict run when elements are refused should return exit code 1
    if val_data["refused_elements_count"] > 0:
        exit_code_strict = main(
            [
                "induce",
                "validate",
                "--report",
                str(rep_file),
                "--strict",
                "--seed",
                "42",
            ]
        )
        assert exit_code_strict == 1


@pytest.mark.contract
def test_mcp_induction_resource_and_explain_contract() -> None:
    """Verify MCP registration, URI resource lookup ewm://induction/{fingerprint}, and explain tool."""
    report = InductionReport(
        source="test_mcp",
        source_type="causal",
        entities_count=4,
        relationships_count=2,
    )
    _, val_rep = validate_induction(report, seed=42)

    server = create_mcp_server()

    # Register in MCP storage
    registered_fp = server.register_induction_report(report)
    assert registered_fp == report.sha256_fingerprint

    # Read resource via resource handler get_induction_report
    resource_payload = server.get_induction_report(report.sha256_fingerprint)
    assert resource_payload is not None
    data = json.loads(resource_payload)
    assert data["sha256_fingerprint"] == report.sha256_fingerprint
    assert data["validation_fingerprint"] == val_rep.sha256_fingerprint

    # Query explain tool with fingerprint
    explanation = server.explain(identifier=report.sha256_fingerprint)
    assert explanation["sha256_fingerprint"] == report.sha256_fingerprint
    assert explanation["validation_fingerprint"] == val_rep.sha256_fingerprint

    # Query explain tool with validation_fingerprint
    explanation_val = server.explain(identifier=val_rep.sha256_fingerprint)
    assert explanation_val["sha256_fingerprint"] == report.sha256_fingerprint

    # Unknown fingerprint returns not found payload
    not_found_res = json.loads(server.get_induction_report("0" * 64))
    assert "error" in not_found_res
    not_found_exp = server.explain(identifier="0" * 64)
    assert not_found_exp["status"] == "NOT_FOUND"
