"""Mechanistic Router layer for multimodal compute orchestration.

Provides deterministic rule filtering, ModernBERT semantic vector routing,
Frontier LLM JSON-mode causal intervention extraction, and the asynchronous
FastAPI gateway for Financial and AIOps World Models.
"""

from __future__ import annotations

from ewm_engine.routing.deterministic_engine import DeterministicRuleEngine
from ewm_engine.routing.llm_arbitrator import FrontierLLMArbitrator
from ewm_engine.routing.schemas import (
    AIOpsEventInput,
    CausalInterventionVector,
    EventModality,
    FinancialEventInput,
    RoutingDecision,
)

try:
    from ewm_engine.routing.gateway import app, ingest_aiops_event, ingest_financial_event
except ImportError:  # pragma: no cover - optional [serve] extra (fastapi)
    app = None
    ingest_aiops_event = None  # type: ignore[assignment]
    ingest_financial_event = None  # type: ignore[assignment]

try:
    from ewm_engine.routing.semantic_router import SemanticVectorRouter, compute_rankme
except ImportError:  # pragma: no cover - optional [ml] extra (torch)
    SemanticVectorRouter = None  # type: ignore[assignment,misc]
    compute_rankme = None  # type: ignore[assignment]

__all__ = [
    "AIOpsEventInput",
    "CausalInterventionVector",
    "DeterministicRuleEngine",
    "EventModality",
    "FinancialEventInput",
    "FrontierLLMArbitrator",
    "RoutingDecision",
    "SemanticVectorRouter",
    "app",
    "compute_rankme",
    "ingest_aiops_event",
    "ingest_financial_event",
]
