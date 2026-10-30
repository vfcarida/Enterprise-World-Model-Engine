"""Mechanistic Router FastAPI Gateway.

Asynchronous compute orchestration gateway that routes financial and AIOps events
across deterministic rule engines, semantic vector routers, numerical SLMs, and
frontier LLM arbitrators in <100ms with structured JSON audit logging and
OpenTelemetry span instrumentation.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any, cast

from fastapi import FastAPI, HTTPException

from ewm_engine.routing.deterministic_engine import DeterministicRuleEngine
from ewm_engine.routing.llm_arbitrator import FrontierLLMArbitrator
from ewm_engine.routing.schemas import (
    AIOpsEventInput,
    CausalInterventionVector,
    FinancialEventInput,
    RoutingDecision,
)
from ewm_engine.routing.semantic_router import SemanticVectorRouter

trace: Any = None
StatusCode: Any = None
_tracer: Any = None
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import StatusCode as _otel_status_code

    trace = _otel_trace
    StatusCode = _otel_status_code
    _HAS_OTEL = True
    _tracer = trace.get_tracer("ewm_engine.routing")
except ImportError:  # pragma: no cover
    _HAS_OTEL = False

logger = logging.getLogger("ewm_engine.routing.gateway")

app = FastAPI(title="EWM Mechanistic Router Gateway", version="1.0.0")

# Initialize routing components
deterministic_engine = DeterministicRuleEngine()
semantic_router = SemanticVectorRouter()
llm_arbitrator = FrontierLLMArbitrator()

# In-memory store of extracted causal vectors keyed by event_id for downstream consumption
extracted_causal_vectors: dict[str, CausalInterventionVector] = {}


def _start_span(name: str, attributes: dict[str, Any] | None = None) -> AbstractContextManager[Any]:
    """Start an OpenTelemetry span if opentelemetry-api is available."""
    if _HAS_OTEL and _tracer is not None:
        return cast(
            AbstractContextManager[Any],
            _tracer.start_as_current_span(name, attributes=attributes),
        )
    return nullcontext(None)


def _log_ingested_event(domain: str, event: FinancialEventInput | AIOpsEventInput) -> None:
    """Emit structured JSON log entry for every validated input event."""
    payload = {
        "log_type": "event_ingested",
        "domain": domain,
        "event_id": event.event_id,
        "event_time": event.event_time.isoformat(),
        "knowledge_time": event.knowledge_time.isoformat(),
        "ingestion_time": event.ingestion_time.isoformat(),
        "source": event.source,
        "modality": event.modality.value,
    }
    logger.info(json.dumps(payload, sort_keys=True))


def _log_routing_decision(decision: RoutingDecision) -> None:
    """Emit structured JSON log entry for every routing decision."""
    payload = {
        "log_type": "routing_decision",
        "event_id": decision.event_id,
        "route_destination": decision.route_destination,
        "confidence": decision.confidence,
        "processing_time_ms": round(decision.processing_time_ms, 4),
        "reasoning": decision.reasoning,
    }
    logger.info(json.dumps(payload, sort_keys=True))


def _finalize_decision(decision: RoutingDecision, span: Any) -> RoutingDecision:
    """Record decision attributes on OpenTelemetry span and emit structured JSON log."""
    if span is not None and _HAS_OTEL:
        span.set_attribute("ewm.route_destination", decision.route_destination)
        span.set_attribute("ewm.confidence", decision.confidence)
        span.set_attribute("ewm.processing_time_ms", decision.processing_time_ms)
        if StatusCode is not None:
            span.set_status(StatusCode.OK)
    _log_routing_decision(decision)
    return decision


async def health_check() -> dict[str, str]:
    """Health check endpoint for gateway readiness/liveness probes."""
    return {"status": "healthy", "service": "ewm-mechanistic-router"}


async def ingest_financial_event(event: FinancialEventInput) -> RoutingDecision:
    """Route financial event to optimal processor."""
    start_time = time.perf_counter()
    _log_ingested_event("financial", event)

    with _start_span(
        "ewm.routing.financial_ingest",
        attributes={
            "ewm.domain": "financial",
            "ewm.event_id": event.event_id,
            "ewm.modality": event.modality.value,
        },
    ) as root_span:
        # Layer 1: Deterministic Rule Engine
        with _start_span("ewm.routing.deterministic_engine") as det_span:
            det_start = time.perf_counter()
            is_routine, reasoning = deterministic_engine.evaluate_financial(event)
            det_ms = (time.perf_counter() - det_start) * 1000.0
            if det_span is not None and _HAS_OTEL:
                det_span.set_attribute("ewm.is_routine", is_routine)
                det_span.set_attribute("ewm.layer_latency_ms", det_ms)

        if is_routine:
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            decision = RoutingDecision(
                event_id=event.event_id,
                route_destination="numerical_slm",
                confidence=1.0,
                reasoning=f"Deterministic fast-path: {reasoning}",
                processing_time_ms=processing_time_ms,
            )
            return _finalize_decision(decision, root_span)

        # Layer 2: Semantic Vector Router (requires text content)
        if not event.text_content:
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            decision = RoutingDecision(
                event_id=event.event_id,
                route_destination="numerical_slm",
                confidence=0.8,
                reasoning="Quantitative event bypassed deterministic rules but has no text",
                processing_time_ms=processing_time_ms,
            )
            return _finalize_decision(decision, root_span)

        with _start_span("ewm.routing.semantic_router") as sem_span:
            sem_start = time.perf_counter()
            destination, confidence, semantic_reasoning = semantic_router.route(event.text_content)
            sem_ms = (time.perf_counter() - sem_start) * 1000.0
            if sem_span is not None and _HAS_OTEL:
                sem_span.set_attribute("ewm.semantic_destination", destination)
                sem_span.set_attribute("ewm.semantic_confidence", confidence)
                sem_span.set_attribute("ewm.layer_latency_ms", sem_ms)

        # Layer 3: Frontier LLM Arbitrator (if high complexity)
        if destination == "frontier_llm":
            with _start_span("ewm.routing.llm_arbitrator") as llm_span:
                llm_start = time.perf_counter()
                try:
                    causal_vector = llm_arbitrator.extract_causal_vector(
                        event.text_content, domain="financial"
                    )
                    extracted_causal_vectors[event.event_id] = causal_vector
                    llm_ms = (time.perf_counter() - llm_start) * 1000.0
                    if llm_span is not None and _HAS_OTEL:
                        llm_span.set_attribute("ewm.layer_latency_ms", llm_ms)
                        llm_span.set_attribute(
                            "ewm.entities_count", len(causal_vector.primary_target_entities)
                        )
                    processing_time_ms = (time.perf_counter() - start_time) * 1000.0
                    decision = RoutingDecision(
                        event_id=event.event_id,
                        route_destination="frontier_llm",
                        confidence=causal_vector.confidence,
                        reasoning=(
                            f"LLM extraction: {semantic_reasoning}. "
                            f"Entities: {causal_vector.primary_target_entities}"
                        ),
                        processing_time_ms=processing_time_ms,
                    )
                    return _finalize_decision(decision, root_span)
                except Exception as e:
                    if root_span is not None and _HAS_OTEL and StatusCode is not None:
                        root_span.set_status(StatusCode.ERROR, str(e))
                    raise HTTPException(
                        status_code=500, detail=f"LLM extraction failed: {e!s}"
                    ) from e

        # Default: Semantic router routed to numerical SLM
        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        decision = RoutingDecision(
            event_id=event.event_id,
            route_destination=destination,
            confidence=confidence,
            reasoning=semantic_reasoning,
            processing_time_ms=processing_time_ms,
        )
        return _finalize_decision(decision, root_span)


async def ingest_aiops_event(event: AIOpsEventInput) -> RoutingDecision:
    """Route AIOps event to optimal processor."""
    start_time = time.perf_counter()
    _log_ingested_event("aiops", event)

    with _start_span(
        "ewm.routing.aiops_ingest",
        attributes={
            "ewm.domain": "aiops",
            "ewm.event_id": event.event_id,
            "ewm.modality": event.modality.value,
        },
    ) as root_span:
        # Layer 1: Deterministic Rule Engine
        with _start_span("ewm.routing.deterministic_engine") as det_span:
            det_start = time.perf_counter()
            is_routine, reasoning = deterministic_engine.evaluate_aiops(event)
            det_ms = (time.perf_counter() - det_start) * 1000.0
            if det_span is not None and _HAS_OTEL:
                det_span.set_attribute("ewm.is_routine", is_routine)
                det_span.set_attribute("ewm.layer_latency_ms", det_ms)

        if is_routine:
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            decision = RoutingDecision(
                event_id=event.event_id,
                route_destination="numerical_slm",
                confidence=1.0,
                reasoning=f"Deterministic fast-path: {reasoning}",
                processing_time_ms=processing_time_ms,
            )
            return _finalize_decision(decision, root_span)

        # Layer 2: Semantic Vector Router (requires text content)
        if not event.text_content:
            processing_time_ms = (time.perf_counter() - start_time) * 1000.0
            decision = RoutingDecision(
                event_id=event.event_id,
                route_destination="numerical_slm",
                confidence=0.8,
                reasoning="Quantitative event, no text content",
                processing_time_ms=processing_time_ms,
            )
            return _finalize_decision(decision, root_span)

        with _start_span("ewm.routing.semantic_router") as sem_span:
            sem_start = time.perf_counter()
            destination, confidence, semantic_reasoning = semantic_router.route(event.text_content)
            sem_ms = (time.perf_counter() - sem_start) * 1000.0
            if sem_span is not None and _HAS_OTEL:
                sem_span.set_attribute("ewm.semantic_destination", destination)
                sem_span.set_attribute("ewm.semantic_confidence", confidence)
                sem_span.set_attribute("ewm.layer_latency_ms", sem_ms)

        # Layer 3: Frontier LLM Arbitrator (if high complexity)
        if destination == "frontier_llm":
            with _start_span("ewm.routing.llm_arbitrator") as llm_span:
                llm_start = time.perf_counter()
                try:
                    causal_vector = llm_arbitrator.extract_causal_vector(
                        event.text_content, domain="aiops"
                    )
                    extracted_causal_vectors[event.event_id] = causal_vector
                    llm_ms = (time.perf_counter() - llm_start) * 1000.0
                    if llm_span is not None and _HAS_OTEL:
                        llm_span.set_attribute("ewm.layer_latency_ms", llm_ms)
                        llm_span.set_attribute(
                            "ewm.entities_count", len(causal_vector.primary_target_entities)
                        )
                    processing_time_ms = (time.perf_counter() - start_time) * 1000.0
                    decision = RoutingDecision(
                        event_id=event.event_id,
                        route_destination="frontier_llm",
                        confidence=causal_vector.confidence,
                        reasoning=(
                            f"LLM extraction: {semantic_reasoning}. "
                            f"Services: {causal_vector.primary_target_entities}"
                        ),
                        processing_time_ms=processing_time_ms,
                    )
                    return _finalize_decision(decision, root_span)
                except Exception as e:
                    if root_span is not None and _HAS_OTEL and StatusCode is not None:
                        root_span.set_status(StatusCode.ERROR, str(e))
                    raise HTTPException(
                        status_code=500, detail=f"LLM extraction failed: {e!s}"
                    ) from e

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        decision = RoutingDecision(
            event_id=event.event_id,
            route_destination=destination,
            confidence=confidence,
            reasoning=semantic_reasoning,
            processing_time_ms=processing_time_ms,
        )
        return _finalize_decision(decision, root_span)


app.add_api_route("/health", health_check, methods=["GET"])
app.add_api_route(
    "/financial/ingest",
    ingest_financial_event,
    methods=["POST"],
    response_model=RoutingDecision,
)
app.add_api_route(
    "/aiops/ingest",
    ingest_aiops_event,
    methods=["POST"],
    response_model=RoutingDecision,
)
