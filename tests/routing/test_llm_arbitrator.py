"""Unit tests for the Frontier LLM Arbitrator with mocked LLM responses."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from ewm_engine.routing.llm_arbitrator import FrontierLLMArbitrator


def _make_mock_llm_response(content: Any) -> Any:
    """Create a mock LiteLLM completion response object."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.unit
def test_extract_causal_vector_financial_semiconductor_disruption() -> None:
    """Verify financial causal vector extraction on semiconductor supply chain disruption."""
    arbitrator = FrontierLLMArbitrator(model="gpt-4-turbo", temperature=0.0)
    mock_payload = {
        "primary_target_entities": ["NVDA", "AMD"],
        "directional_sentiment": -0.82,
        "magnitude": 0.88,
        "contagion_pathways": [
            {"source": "NVDA", "target": "AUTO_SECTOR", "weight": 0.65},
            {"source": "AMD", "target": "AUTO_SECTOR", "weight": 0.55},
        ],
        "confidence": 0.91,
        "extraction_metadata": {
            "source_type": "newswire_api",
            "timestamp": "2026-06-01T12:00:00Z",
            "reasoning": "Fab outage halts wafer shipments to GPU designers and auto OEMs.",
        },
    }

    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response(json.dumps(mock_payload)),
    ) as mock_comp:
        text = "Semiconductor supply chain disruption affects NVDA, AMD, and downstream auto manufacturers."
        vector = arbitrator.extract_causal_vector(text, domain="financial")

        assert vector.primary_target_entities == ["NVDA", "AMD"]
        assert vector.directional_sentiment < 0.0
        assert vector.directional_sentiment == pytest.approx(-0.82)
        assert vector.magnitude == pytest.approx(0.88)
        assert vector.confidence == pytest.approx(0.91)
        assert any(edge["target"] == "AUTO_SECTOR" for edge in vector.contagion_pathways)

        mock_comp.assert_called_once()
        call_kwargs = mock_comp.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4-turbo"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.unit
def test_extract_causal_vector_aiops_domain() -> None:
    """Verify AIOps causal vector extraction for cascading service degradation."""
    arbitrator = FrontierLLMArbitrator()
    mock_payload = {
        "primary_target_entities": ["payment-service", "auth-api"],
        "directional_sentiment": -0.75,
        "magnitude": 0.80,
        "contagion_pathways": [{"source": "payment-service", "target": "auth-api", "weight": 0.72}],
        "confidence": 0.87,
        "extraction_metadata": {
            "source_type": "post_mortem",
            "timestamp": "2026-06-01T14:00:00Z",
            "reasoning": "Connection pool exhaustion in payment-service saturates auth-api.",
        },
    }

    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response(json.dumps(mock_payload)),
    ):
        vector = arbitrator.extract_causal_vector(
            "Connection pool exhaustion in payment-service cascades to auth-api.",
            domain="aiops",
        )
        assert vector.primary_target_entities == ["payment-service", "auth-api"]
        assert vector.directional_sentiment == pytest.approx(-0.75)
        assert vector.confidence == pytest.approx(0.87)


@pytest.mark.unit
def test_extract_causal_vector_unknown_domain_raises_value_error() -> None:
    """Unknown domain raises ValueError before invoking LLM completion."""
    arbitrator = FrontierLLMArbitrator()
    with pytest.raises(ValueError, match="Unknown domain: healthcare"):
        arbitrator.extract_causal_vector("Some event text", domain="healthcare")


@pytest.mark.unit
def test_extract_causal_vector_malformed_outputs_raise_value_error() -> None:
    """Malformed JSON, missing required fields, or non-string outputs raise ValueError."""
    arbitrator = FrontierLLMArbitrator()

    # 1. Invalid JSON syntax
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response("{not valid json"),
    ):
        with pytest.raises(ValueError, match="LLM extraction failed"):
            arbitrator.extract_causal_vector("Crisis event", domain="financial")

    # 2. Missing required fields (missing directional_sentiment, magnitude, confidence)
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response(json.dumps({"primary_target_entities": ["AAPL"]})),
    ):
        with pytest.raises(ValueError, match="LLM extraction failed"):
            arbitrator.extract_causal_vector("Crisis event", domain="financial")

    # 3. Top-level JSON array instead of object
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response(json.dumps([1, 2, 3])),
    ):
        with pytest.raises(ValueError, match="Top-level JSON output must be an object"):
            arbitrator.extract_causal_vector("Crisis event", domain="financial")

    # 4. Non-string content returned by provider
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_make_mock_llm_response(None),
    ):
        with pytest.raises(ValueError, match="Expected string JSON content"):
            arbitrator.extract_causal_vector("Crisis event", domain="financial")
