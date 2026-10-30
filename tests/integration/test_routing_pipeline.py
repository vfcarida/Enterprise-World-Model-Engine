"""End-to-end integration and benchmark tests for the Mechanistic Router FastAPI gateway."""

from __future__ import annotations

import json
import logging
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")
from fastapi.testclient import TestClient

import ewm_engine.routing.gateway as gateway_mod
from ewm_engine.routing.gateway import app, extracted_causal_vectors


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient fixture for the Mechanistic Router gateway."""
    return TestClient(app)


def _mock_llm_response(payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


@pytest.mark.integration
def test_gateway_health_endpoint(client: TestClient) -> None:
    """GET /health returns 200 healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "service": "ewm-mechanistic-router"}


@pytest.mark.integration
def test_financial_ingest_all_routing_paths(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify deterministic fast-path, semantic SLM path, and Frontier LLM path on /financial/ingest."""
    caplog.set_level(logging.INFO, logger="ewm_engine.routing.gateway")

    # Path 1: Deterministic fast-path (routine CPI release within 2% consensus)
    resp_det = client.post(
        "/financial/ingest",
        json={
            "event_id": "fin_det_01",
            "event_time": "2026-06-01T12:30:00",
            "knowledge_time": "2026-06-01T12:30:01",
            "source": "bls_macro",
            "modality": "quantitative",
            "metric_name": "cpi",
            "metric_value": 2.02,
            "consensus_value": 2.00,
        },
    )
    assert resp_det.status_code == 200
    data_det = resp_det.json()
    assert data_det["event_id"] == "fin_det_01"
    assert data_det["route_destination"] == "numerical_slm"
    assert data_det["confidence"] == 1.0
    assert "Deterministic fast-path" in data_det["reasoning"]
    assert data_det["processing_time_ms"] < 100.0

    # Path 2: Quantitative event bypassing deterministic rules without text content
    resp_no_text = client.post(
        "/financial/ingest",
        json={
            "event_id": "fin_notext_01",
            "event_time": "2026-06-01T12:35:00",
            "knowledge_time": "2026-06-01T12:35:01",
            "source": "bls_macro",
            "modality": "quantitative",
            "metric_name": "cpi",
            "metric_value": 2.25,
            "consensus_value": 2.00,
        },
    )
    assert resp_no_text.status_code == 200
    data_no_text = resp_no_text.json()
    assert data_no_text["route_destination"] == "numerical_slm"
    assert data_no_text["confidence"] == pytest.approx(0.8)
    assert "has no text" in data_no_text["reasoning"]

    # Path 3: Semantic Vector Router -> numerical_slm (low/standard complexity text)
    resp_sem_slm = client.post(
        "/financial/ingest",
        json={
            "event_id": "fin_sem_slm_01",
            "event_time": "2026-06-01T13:00:00",
            "knowledge_time": "2026-06-01T13:00:02",
            "source": "newswire_api",
            "modality": "qualitative",
            "text_content": "Standard Q3 earnings beat consensus by 2%",
            "affected_tickers": ["MSFT"],
        },
    )
    assert resp_sem_slm.status_code == 200
    data_sem_slm = resp_sem_slm.json()
    assert data_sem_slm["route_destination"] == "numerical_slm"
    assert data_sem_slm["confidence"] == pytest.approx(0.7)
    assert "Standard processing sufficient" in data_sem_slm["reasoning"]

    # Path 4: Semantic Vector Router -> Frontier LLM Arbitrator (high complexity)
    llm_payload = {
        "primary_target_entities": ["NVDA", "AMD"],
        "directional_sentiment": -0.85,
        "magnitude": 0.90,
        "contagion_pathways": [{"source": "NVDA", "target": "TSM", "weight": 0.78}],
        "confidence": 0.93,
        "extraction_metadata": {
            "source_type": "sec_edgar_8k",
            "timestamp": "2026-06-01T14:00:00Z",
            "reasoning": "Export restriction shock",
        },
    }
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_mock_llm_response(llm_payload),
    ):
        resp_llm = client.post(
            "/financial/ingest",
            json={
                "event_id": "fin_llm_01",
                "event_time": "2026-06-01T14:00:00",
                "knowledge_time": "2026-06-01T14:00:05",
                "source": "sec_edgar_8k",
                "modality": "qualitative",
                "text_content": "Unexpected regulatory action and geopolitical export crisis affecting NVDA and AMD",
                "affected_tickers": ["NVDA", "AMD"],
            },
        )
    assert resp_llm.status_code == 200
    data_llm = resp_llm.json()
    assert data_llm["route_destination"] == "frontier_llm"
    assert data_llm["confidence"] == pytest.approx(0.93)
    assert "NVDA" in data_llm["reasoning"]
    assert "fin_llm_01" in extracted_causal_vectors
    assert extracted_causal_vectors["fin_llm_01"].primary_target_entities == ["NVDA", "AMD"]

    # Verify structured JSON logs contain both event_ingested and routing_decision records
    parsed_logs = [json.loads(rec.message) for rec in caplog.records]
    ingested_logs = [log for log in parsed_logs if log.get("log_type") == "event_ingested"]
    decision_logs = [log for log in parsed_logs if log.get("log_type") == "routing_decision"]
    assert len(ingested_logs) >= 4
    assert len(decision_logs) >= 4
    assert all(
        "event_time" in log and "knowledge_time" in log and "ingestion_time" in log
        for log in ingested_logs
    )


@pytest.mark.integration
def test_aiops_ingest_all_routing_paths(client: TestClient) -> None:
    """Verify deterministic fast-path, semantic SLM path, and Frontier LLM path on /aiops/ingest."""
    # Path 1: Deterministic fast-path (standard CPU metric, no alert)
    resp_det = client.post(
        "/aiops/ingest",
        json={
            "event_id": "ops_det_01",
            "event_time": "2026-06-01T15:00:00",
            "knowledge_time": "2026-06-01T15:00:01",
            "source": "prometheus_alert",
            "modality": "quantitative",
            "service_name": "payment-service",
            "metric_name": "cpu_utilization",
            "metric_value": 68.5,
            "threshold_breached": False,
        },
    )
    assert resp_det.status_code == 200
    assert resp_det.json()["route_destination"] == "numerical_slm"
    assert resp_det.json()["confidence"] == 1.0

    # Path 2: Quantitative threshold breach with no text content
    resp_notext = client.post(
        "/aiops/ingest",
        json={
            "event_id": "ops_notext_01",
            "event_time": "2026-06-01T15:05:00",
            "knowledge_time": "2026-06-01T15:05:01",
            "source": "prometheus_alert",
            "modality": "quantitative",
            "service_name": "payment-service",
            "metric_name": "memory_bytes",
            "metric_value": 92.0,
            "threshold_breached": True,
        },
    )
    assert resp_notext.status_code == 200
    assert resp_notext.json()["route_destination"] == "numerical_slm"
    assert resp_notext.json()["confidence"] == pytest.approx(0.8)

    # Path 3: Semantic Vector Router -> numerical_slm
    resp_sem = client.post(
        "/aiops/ingest",
        json={
            "event_id": "ops_sem_01",
            "event_time": "2026-06-01T15:10:00",
            "knowledge_time": "2026-06-01T15:10:01",
            "source": "kubernetes_watch",
            "modality": "qualitative",
            "service_name": "payment-service",
            "text_content": "Scaled replica count from 3 to 5 for scheduled load test",
        },
    )
    assert resp_sem.status_code == 200
    assert resp_sem.json()["route_destination"] == "numerical_slm"

    # Path 4: Semantic Vector Router -> Frontier LLM Arbitrator
    ops_llm_payload = {
        "primary_target_entities": ["payment-service", "auth-api"],
        "directional_sentiment": -0.90,
        "magnitude": 0.95,
        "contagion_pathways": [{"source": "payment-service", "target": "auth-api", "weight": 0.85}],
        "confidence": 0.89,
        "extraction_metadata": {"source_type": "log_parser"},
    }
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        return_value=_mock_llm_response(ops_llm_payload),
    ):
        resp_llm = client.post(
            "/aiops/ingest",
            json={
                "event_id": "ops_llm_01",
                "event_time": "2026-06-01T15:15:00",
                "knowledge_time": "2026-06-01T15:15:01",
                "source": "log_parser",
                "modality": "qualitative",
                "service_name": "payment-service",
                "text_content": "Unexpected emergency outage and cascading thread crisis in payment-service",
            },
        )
    assert resp_llm.status_code == 200
    assert resp_llm.json()["route_destination"] == "frontier_llm"
    assert "payment-service" in resp_llm.json()["reasoning"]


@pytest.mark.integration
def test_gateway_schema_violation_422_and_llm_failure_500(client: TestClient) -> None:
    """Verify HTTP 422 on causality violation and HTTP 500 on LLM failure."""
    # Causality violation: knowledge_time < event_time -> HTTP 422
    resp_422 = client.post(
        "/financial/ingest",
        json={
            "event_id": "fin_422",
            "event_time": "2026-06-01T12:00:10",
            "knowledge_time": "2026-06-01T12:00:00",
            "source": "newswire_api",
            "modality": "quantitative",
        },
    )
    assert resp_422.status_code == 422
    assert "knowledge_time cannot precede event_time" in json.dumps(resp_422.json())

    # LLM failure on financial and AIOps -> HTTP 500
    with patch(
        "ewm_engine.routing.llm_arbitrator.completion",
        side_effect=RuntimeError("Upstream LLM timeout"),
    ):
        resp_fin_500 = client.post(
            "/financial/ingest",
            json={
                "event_id": "fin_500",
                "event_time": "2026-06-01T12:00:00",
                "knowledge_time": "2026-06-01T12:00:05",
                "source": "newswire_api",
                "modality": "qualitative",
                "text_content": "Unexpected regulatory crisis in banking sector",
            },
        )
        assert resp_fin_500.status_code == 500
        assert "LLM extraction failed" in resp_fin_500.json()["detail"]

        resp_ops_500 = client.post(
            "/aiops/ingest",
            json={
                "event_id": "ops_500",
                "event_time": "2026-06-01T12:00:00",
                "knowledge_time": "2026-06-01T12:00:05",
                "source": "log_parser",
                "modality": "qualitative",
                "text_content": "Unexpected emergency crisis in cluster ingress",
            },
        )
        assert resp_ops_500.status_code == 500
        assert "LLM extraction failed" in resp_ops_500.json()["detail"]


@pytest.mark.integration
def test_gateway_opentelemetry_span_hierarchy(client: TestClient) -> None:
    """Verify OpenTelemetry spans capture deterministic -> semantic -> LLM latency breakdown."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    test_tracer = provider.get_tracer("ewm_engine.routing.test")

    llm_payload = {
        "primary_target_entities": ["AAPL"],
        "directional_sentiment": -0.5,
        "magnitude": 0.6,
        "contagion_pathways": [],
        "confidence": 0.85,
        "extraction_metadata": {},
    }

    with (
        patch.object(gateway_mod, "_tracer", test_tracer),
        patch(
            "ewm_engine.routing.llm_arbitrator.completion",
            return_value=_mock_llm_response(llm_payload),
        ),
    ):
        resp = client.post(
            "/financial/ingest",
            json={
                "event_id": "otel_evt_01",
                "event_time": "2026-06-01T12:00:00",
                "knowledge_time": "2026-06-01T12:00:01",
                "source": "sec_edgar_8k",
                "modality": "qualitative",
                "text_content": "Unexpected regulatory investigation announced for AAPL",
            },
        )
        assert resp.status_code == 200

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "ewm.routing.deterministic_engine" in span_names
    assert "ewm.routing.semantic_router" in span_names
    assert "ewm.routing.llm_arbitrator" in span_names
    assert "ewm.routing.financial_ingest" in span_names


@pytest.mark.benchmark
def test_gateway_throughput_and_p95_latency(client: TestClient) -> None:
    """Performance: Gateway processes >100 req/s with p95 latency <100ms (excluding LLM)."""
    latencies_ms: list[float] = []
    n_requests = 120

    batch_start = time.perf_counter()
    for i in range(n_requests):
        if i % 2 == 0:
            payload = {
                "event_id": f"perf_fin_{i}",
                "event_time": "2026-06-01T12:00:00",
                "knowledge_time": "2026-06-01T12:00:01",
                "source": "market_1t",
                "modality": "quantitative",
                "metric_name": "cpi",
                "metric_value": 2.01,
                "consensus_value": 2.00,
            }
            t0 = time.perf_counter()
            resp = client.post("/financial/ingest", json=payload)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200
        else:
            payload_sem = {
                "event_id": f"perf_ops_{i}",
                "event_time": "2026-06-01T12:00:00",
                "knowledge_time": "2026-06-01T12:00:01",
                "source": "kubernetes_watch",
                "modality": "qualitative",
                "text_content": "Standard Q3 deployment update with routine replica adjustment",
            }
            t0 = time.perf_counter()
            resp = client.post("/aiops/ingest", json=payload_sem)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200

    total_time_s = time.perf_counter() - batch_start
    throughput_rps = n_requests / total_time_s
    p95_ms = float(np.percentile(latencies_ms, 95))

    assert throughput_rps > 100.0, f"Throughput {throughput_rps:.1f} req/s <= 100 req/s"
    assert p95_ms < 100.0, f"p95 latency {p95_ms:.2f}ms >= 100ms"
