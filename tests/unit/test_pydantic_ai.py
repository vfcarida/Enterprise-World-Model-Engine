"""Unit tests for EWM Engine Pydantic-AI agent integration (AGENTS extra).

Tests toolset registration, callable execution, error handling without pydantic_ai installed,
and mocked agent construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.integrations.pydantic_ai import (
    EwmAgentToolset,
    create_ewm_agent,
    get_ewm_toolset,
    is_pydantic_ai_available,
)
from ewm_engine.provenance.evidence import EvidenceLevel

SAMPLE_WORLD_YAML = """
schema_version: "1.0.0"
world:
  name: "Agent_Test_World"
entities:
  - id: "agent_node"
    type: "agent"
resources:
  - id: "capital"
    current: 1000.0
    min_value: 0.0
    max_value: 10000.0
constraints: []
"""

SAMPLE_WORLD_WITH_SCM_YAML = """
schema_version: "1.0.0"
world:
  name: "Agent_SCM_World"
entities:
  - id: "agent_node"
    type: "agent"
resources:
  - id: "capital"
    current: 1000.0
    min_value: 0.0
    max_value: 10000.0
scm:
  structural_equations:
    capital: "interest_rate * 1000"
constraints: []
"""


def test_toolset_instantiation_and_callables() -> None:
    """Verify toolset methods are callable and return structured dictionaries."""
    toolset = EwmAgentToolset(provenance_strict=False)
    tools = toolset.as_tools()
    assert len(tools) == 5

    # 1. ewm_simulate
    sim_res = toolset.ewm_simulate(world_yaml=SAMPLE_WORLD_YAML, horizon=2, samples=1, seed=42)
    assert sim_res["status"] == "COMPLETED"
    assert len(sim_res["state_fingerprint"]) == 64
    assert sim_res["evidence_level"] == EvidenceLevel.STRUCTURAL.value

    # 2. ewm_counterfactual
    cf_res = toolset.ewm_counterfactual(
        world_yaml=SAMPLE_WORLD_WITH_SCM_YAML,
        intervention={
            "type": "inject",
            "step": 1,
            "target": "capital",
            "parameters": {"amount": 500},
        },
        seed=42,
        horizon=2,
    )
    assert cf_res["status"] == "COMPLETED"
    assert len(cf_res["counterfactual_fingerprint"]) == 64

    # 3. ewm_check_constraints
    chk_res = toolset.ewm_check_constraints(
        world_yaml=SAMPLE_WORLD_YAML,
        action={"type": "noop", "parameters": {}},
    )
    assert chk_res["is_valid"] is True
    assert chk_res["status"] == "APPROVED"

    # 4. ewm_decide
    dec_res = toolset.ewm_decide(
        world_yaml=SAMPLE_WORLD_YAML,
        candidates={"opt1": {"type": "noop"}, "opt2": {"type": "noop"}},
        horizon=2,
        samples=2,
        seed=42,
    )
    assert dec_res["status"] == "COMPLETED"
    assert len(dec_res["rankings"]) == 2

    # 5. ewm_explain
    exp_res = toolset.ewm_explain(identifier=sim_res["run_id"])
    assert exp_res["run_id"] == sim_res["run_id"]


def test_get_ewm_toolset_helper() -> None:
    """Verify get_ewm_toolset returns ready-to-use tool functions."""
    tools = get_ewm_toolset(provenance_strict=True)
    assert len(tools) == 5
    tool_names = [f.__name__ for f in tools]
    assert "ewm_simulate" in tool_names
    assert "ewm_counterfactual" in tool_names
    assert "ewm_check_constraints" in tool_names
    assert "ewm_decide" in tool_names
    assert "ewm_explain" in tool_names


def test_create_ewm_agent_without_pydantic_ai() -> None:
    """Verify informative exception when pydantic_ai is not installed."""
    if not is_pydantic_ai_available():
        with pytest.raises(SimulationConfigurationError) as exc_info:
            create_ewm_agent()
        assert "pydantic-ai is required" in str(exc_info.value)
        assert "pip install ewm-engine[agents]" in str(exc_info.value)


def test_create_ewm_agent_with_mocked_pydantic_ai() -> None:
    """Verify create_ewm_agent wiring when pydantic_ai is present."""
    mock_agent_class = MagicMock()
    mock_module = MagicMock()
    mock_module.Agent = mock_agent_class

    with patch("ewm_engine.integrations.pydantic_ai.is_pydantic_ai_available", return_value=True):
        with patch.dict("sys.modules", {"pydantic_ai": mock_module}):
            create_ewm_agent(model="test-model", provenance_strict=True)

            mock_agent_class.assert_called_once()
            _, kwargs = mock_agent_class.call_args
            assert kwargs["model"] == "test-model"
            assert len(kwargs["tools"]) == 5
            assert "Enterprise World Model Engine" in kwargs["system_prompt"]
