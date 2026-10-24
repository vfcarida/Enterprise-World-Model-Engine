"""Unit tests for EWM Engine Model Context Protocol (MCP) server.

Tests tool registration, deterministic execution, safe boundary enforcement,
Pearl counterfactual refusal for non-identifiable queries, dynamic resources,
prompts, MCP protocol invocation, and provenance-strict audit logging.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from ewm_engine.mcp.safety import (
    SimulateInput,
)
from ewm_engine.mcp.server import (
    create_mcp_server,
    is_mcp_available,
)
from ewm_engine.provenance.evidence import EvidenceLevel

SAMPLE_WORLD_YAML = """
schema_version: "1.0.0"
world:
  name: "MCP_Test_World"
  description: "Test world for MCP server suite"
entities:
  - id: "warehouse_alpha"
    type: "facility"
    attributes:
      capacity: 500
resources:
  - id: "inventory_widgets"
    current: 120.0
    min_value: 0.0
    max_value: 1000.0
constraints: []
"""

SAMPLE_WORLD_WITH_SCM_YAML = """
schema_version: "1.0.0"
world:
  name: "MCP_SCM_World"
entities:
  - id: "factory_1"
    type: "factory"
resources:
  - id: "production"
    current: 50.0
    min_value: 0.0
    max_value: 500.0
scm:
  structural_equations:
    production: "base_demand * 1.2"
constraints: []
"""

SAMPLE_WORLD_UNIDENTIFIED_YAML = """
schema_version: "1.0.0"
world:
  name: "MCP_Unidentified_World"
entities:
  - id: "market"
    type: "market"
resources:
  - id: "unobserved_latent"
    current: 10.0
    min_value: 0.0
    max_value: 100.0
scm:
  structural_equations: {}
  unobserved_confounders:
    - "latent_bias"
constraints: []
"""


def test_mcp_availability_and_creation() -> None:
    """Verify MCP SDK detection and server creation."""
    assert is_mcp_available() is True
    server = create_mcp_server(server_name="test-mcp", provenance_strict=False)
    assert server.config.server_name == "test-mcp"
    assert server.config.provenance_strict is False
    assert server.server is not None


def test_mcp_safety_schema_bounds() -> None:
    """Verify that untrusted input exceeding safe resource caps is rejected."""
    # Valid input passes
    valid = SimulateInput(world_yaml=SAMPLE_WORLD_YAML, horizon=10, samples=5, seed=42)
    assert valid.horizon == 10

    # Horizon cap (> 100)
    with pytest.raises(ValidationError):
        SimulateInput(world_yaml=SAMPLE_WORLD_YAML, horizon=150)

    # Samples cap (> 1000)
    with pytest.raises(ValidationError):
        SimulateInput(world_yaml=SAMPLE_WORLD_YAML, samples=2000)

    # Negative seed
    with pytest.raises(ValidationError):
        SimulateInput(world_yaml=SAMPLE_WORLD_YAML, seed=-5)

    # Unknown field injection blocked by extra="forbid"
    with pytest.raises(ValidationError):
        SimulateInput(
            world_yaml=SAMPLE_WORLD_YAML,  # type: ignore[call-arg]
            malicious_payload="import os; os.system('calc')",
        )


def test_tool_simulate() -> None:
    """Verify ewm.simulate executes deterministically and stores artifacts."""
    server = create_mcp_server(provenance_strict=False)
    res = server.simulate(world_yaml=SAMPLE_WORLD_YAML, horizon=3, samples=1, seed=123)

    assert res["status"] == "COMPLETED"
    assert len(res["state_fingerprint"]) == 64
    assert res["evidence_level"] == EvidenceLevel.STRUCTURAL.value
    assert res["horizon"] == 3
    assert res["samples"] == 1
    assert "ewm://runs/" in res["resource_uri"]
    assert "ewm://traces/" in res["trace_uri"]

    # Verify run manifest is queryable via explain
    run_info = server.explain(identifier=res["run_id"])
    assert run_info["run_id"] == res["run_id"]
    assert run_info["fingerprint"] == res["state_fingerprint"]


def test_mcp_protocol_call_tool() -> None:
    """Verify that tools are invokable through the official MCPServer.call_tool protocol."""
    server = create_mcp_server()
    res = asyncio.run(
        server.server.call_tool(
            "ewm.simulate",
            {"world_yaml": SAMPLE_WORLD_YAML, "horizon": 2, "samples": 1, "seed": 42},
        )
    )
    assert res.is_error is False
    assert len(res.content) > 0
    # Also verify alias "simulate" works
    alias_res = asyncio.run(
        server.server.call_tool(
            "simulate",
            {"world_yaml": SAMPLE_WORLD_YAML, "horizon": 2, "samples": 1, "seed": 42},
        )
    )
    assert alias_res.is_error is False


def test_tool_counterfactual_identifiable() -> None:
    """Verify ewm.counterfactual produces coupled twin trajectories for identifiable model."""
    server = create_mcp_server(provenance_strict=False)
    intervention = {
        "type": "boost_capacity",
        "step": 1,
        "target": "production",
        "parameters": {"delta": 25.0},
    }
    res = server.counterfactual(
        world_yaml=SAMPLE_WORLD_WITH_SCM_YAML,
        intervention=intervention,
        seed=999,
        horizon=3,
    )
    assert res["status"] == "COMPLETED"
    assert len(res["factual_fingerprint"]) == 64
    assert len(res["counterfactual_fingerprint"]) == 64
    assert "epistemic_disclaimer" in res
    assert "arXiv:2402.10877" in res["epistemic_disclaimer"]


def test_tool_counterfactual_refuses_non_identifiable() -> None:
    """Verify ewm.counterfactual refuses to emit point estimate for non-identifiable query."""
    server = create_mcp_server(provenance_strict=True)
    intervention = {
        "type": "shift_latent",
        "step": 1,
        "target": "unobserved_latent",
        "parameters": {"value": 50.0},
    }
    res = server.counterfactual(
        world_yaml=SAMPLE_WORLD_UNIDENTIFIED_YAML,
        intervention=intervention,
        seed=42,
        horizon=3,
    )
    assert res["status"] == "NOT_IDENTIFIED"
    assert res["refusal"] is True
    assert "unobserved confounding" in res["reason"]
    assert res["e_value_sensitivity"] == 1.0
    assert "Refusing to emit counterfactual point estimate" in res["message"]


def test_tool_check_constraints() -> None:
    """Verify ewm.check_constraints validates action against world invariants."""
    server = create_mcp_server()
    action = {
        "id": "act_test",
        "type": "noop",
        "parameters": {},
    }
    res = server.check_constraints(world_yaml=SAMPLE_WORLD_YAML, action=action)
    assert res["is_valid"] is True
    assert res["status"] == "APPROVED"
    assert res["hard_violations_count"] == 0
    assert len(res["state_fingerprint"]) == 64


def test_tool_decide() -> None:
    """Verify ewm.decide ranks candidate policies under parametric uncertainty."""
    server = create_mcp_server()
    candidates = {
        "strategy_a": {"type": "invest", "parameters": {"budget": 100}},
        "strategy_b": {"type": "reallocate", "parameters": {"ratio": 0.5}},
    }
    res = server.decide(
        world_yaml=SAMPLE_WORLD_YAML,
        candidates=candidates,
        satisficing_threshold=0.1,
        robustness_metric="regret",
        horizon=2,
        samples=5,
        seed=42,
    )
    assert res["status"] == "COMPLETED"
    assert res["robustness_metric"] == "regret"
    assert len(res["rankings"]) == 2
    assert res["best_candidate"] == "strategy_a"
    assert "abstention_decision" in res
    assert res["abstention_decision"]["status_banner"] == "WITHIN_DECLARED_VALIDITY_LIMITS"
    for r in res["rankings"]:
        assert "robustness_score" in r
        assert "max_regret" in r
        assert "cvar_regret_90" in r
        assert "satisficing_rate" in r
        assert "mean_outcome" in r


def test_tool_verify_explanation_and_check_abstention() -> None:
    """Verify ewm.verify_explanation and ewm.check_abstention boundary tools."""
    server = create_mcp_server()
    sim_res = server.simulate(world_yaml=SAMPLE_WORLD_YAML, horizon=2, samples=1, seed=42)
    run_id = sim_res["run_id"]
    trace_info = server.explain(identifier=run_id)
    assert trace_info["run_id"] == run_id

    # Verify check_abstention within limits vs triggered
    ok_abs = server.check_abstention(parameter_std=0.05, max_parameter_std=0.25)
    assert ok_abs["abstain"] is False
    assert ok_abs["status_banner"] == "WITHIN_DECLARED_VALIDITY_LIMITS"

    trig_abs = server.check_abstention(
        parameter_std=0.45,
        max_parameter_std=0.25,
        performative_invalidation=True,
        insufficient_causal_evidence=True,
    )
    assert trig_abs["abstain"] is True
    assert trig_abs["do_not_use_me_here"] is True
    assert trig_abs["status_banner"] == "DO_NOT_USE_ME_HERE"
    assert "parameter" in trig_abs["triggered_uncertainty_sources"]

    # Verify verify_explanation against a recorded VerifiableMechanisticExplanation
    from ewm_engine.provenance.evidence import DependencyEvidenceNature
    from ewm_engine.provenance.explainability import (
        MechanismAttributionRecord,
        VerifiableMechanisticExplanation,
    )

    gt = VerifiableMechanisticExplanation(
        explanation_id="exp_mcp_test",
        source_epistemic_fingerprint="a" * 64,
        effective_evidence_level=EvidenceLevel.STRUCTURAL,
        effective_evidence_nature=DependencyEvidenceNature.KNOWN_FUNCTIONAL,
        participating_mechanisms=(
            MechanismAttributionRecord(
                mechanism_id="inventory_transfer_rule",
                mechanism_version="1.0.0",
            ),
        ),
        modified_resources=("inventory_widgets",),
        explanation_fingerprint="b" * 64,
    )
    faithful_res = server.verify_explanation(
        narrative_text="Transition governed by mech:inventory_transfer_rule modifying res:inventory_widgets.",
        explanation_dict=gt.model_dump(mode="json"),
        known_universe_mechanisms=["inventory_transfer_rule", "unexecuted_pricing_model"],
    )
    assert faithful_res["is_faithful"] is True
    assert faithful_res["fabricated_mechanisms"] == []

    fabricated_res = server.verify_explanation(
        narrative_text="Transition governed by mech:unexecuted_pricing_model.",
        explanation_dict=gt.model_dump(mode="json"),
        known_universe_mechanisms=["inventory_transfer_rule", "unexecuted_pricing_model"],
    )
    assert fabricated_res["is_faithful"] is False
    assert "unexecuted_pricing_model" in fabricated_res["fabricated_mechanisms"]


def test_cli_example_fraud_cascade(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify `ewm example fraud_cascade` executes the flagship scenario cleanly."""
    from ewm_engine.cli.main import main

    rc = main(["example", "fraud_cascade"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Scenario Fingerprint:" in captured.out
    assert "Full EWM Regret:" in captured.out


def test_tool_explain_not_found() -> None:
    """Verify ewm.explain handles missing identifiers gracefully."""
    server = create_mcp_server()
    res = server.explain(identifier="non-existent-id")
    assert res["status"] == "NOT_FOUND"


def test_resources_and_prompts() -> None:
    """Verify dynamic resources and prompts are accessible on the server."""
    server = create_mcp_server()
    sim_res = server.simulate(world_yaml=SAMPLE_WORLD_YAML, horizon=2, samples=1, seed=42)

    # Test dynamic resources via server get_resource
    run_id = sim_res["run_id"]
    trace_id = sim_res["trace_uri"].replace("ewm://traces/", "")

    run_content = server.get_run(run_id)
    assert run_content is not None
    run_obj = json.loads(run_content)
    assert run_obj["run_id"] == run_id

    trace_content = server.get_trace(trace_id)
    assert trace_content is not None
    trace_obj = json.loads(trace_content)
    assert trace_obj["trace_id"] == trace_id

    # Test MCP protocol read_resource
    res_content = asyncio.run(server.server.read_resource(f"ewm://runs/{run_id}"))
    assert len(res_content) > 0
    assert run_id in res_content[0].content

    # Test prompt generation
    prompt_text = server.author_what_if(topic="supply_chain_shock")
    assert "supply_chain_shock" in prompt_text
    assert "ewm.check_constraints" in prompt_text

    exp_text = server.explain_result(run_id="run-12345")
    assert "run-12345" in exp_text
    assert "SHA-256" in exp_text

    # Test MCP protocol get_prompt
    proto_prompt = asyncio.run(
        server.server.get_prompt("author_what_if", {"topic": "inventory_drift"})
    )
    assert proto_prompt.messages[0].content.text is not None
    assert "inventory_drift" in proto_prompt.messages[0].content.text


def test_provenance_strict_mode() -> None:
    """Verify --provenance-strict forces EvidenceLevel.ASSUMED and records audit trail."""
    server = create_mcp_server(provenance_strict=True)
    assert server.policy.enabled is True

    res = server.simulate(world_yaml=SAMPLE_WORLD_YAML, horizon=2, samples=1, seed=42)

    # Under provenance-strict, LLM/external specs default to ASSUMED
    assert res["evidence_level"] == EvidenceLevel.ASSUMED.value

    # Verify audit log recorded event with hash
    events = server.policy.audit_events
    assert len(events) >= 1
    ev = events[-1]
    assert ev.tool_name == "ewm.simulate"
    assert ev.evidence_level == EvidenceLevel.ASSUMED
    assert len(ev.event_hash) == 64
