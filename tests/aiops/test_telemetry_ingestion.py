"""Unit and property-based tests for AIOps telemetry ingestion pipelines."""
# ruff: noqa: E402

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pd = pytest.importorskip("pandas")

from ewm_engine.aiops.data.jaeger_collector import JaegerCollector
from ewm_engine.aiops.data.log_parser import LogParser
from ewm_engine.aiops.data.prometheus_collector import (
    STANDARD_SRE_METRICS,
    PrometheusCollector,
)
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry
from ewm_engine.routing.schemas import CausalInterventionVector


class _StubPromClient:
    """Deterministic stub for Prometheus custom_query_range."""

    def __init__(self, empty: bool = False) -> None:
        self.empty = empty
        self.queries: list[str] = []

    def custom_query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str,
    ) -> list[dict[str, Any]]:
        self.queries.append(query)
        if self.empty:
            return []
        base_ts = start_time.timestamp()
        return [
            {
                "metric": {
                    "service": "payment-service",
                    "namespace": "prod",
                },
                "values": [
                    [base_ts, "0.42"],
                    [base_ts + 60.0, "0.55"],
                    [base_ts + 120.0, "0.68"],
                ],
            }
        ]


@pytest.mark.unit
def test_prometheus_collector_range_and_service_metrics() -> None:
    """Verify PrometheusCollector queries 1-minute metrics and enforces bitemporal timestamps."""
    stub = _StubPromClient()
    collector = PrometheusCollector(prometheus_url="http://localhost:9090", client=stub)

    start = datetime(2026, 5, 1, 10, 0, 0)
    end = start + timedelta(minutes=2)

    df = collector.query_range_metric(
        metric_name="container_cpu_usage_seconds_total",
        service_labels={"service": "payment-service", "namespace": "prod"},
        start_time=start,
        end_time=end,
        step="1m",
    )
    assert len(df) == 3
    assert list(df["value"]) == pytest.approx([0.42, 0.55, 0.68])
    assert (df["service"] == "payment-service").all()
    assert (df["namespace"] == "prod").all()
    assert (df["knowledge_time"] >= df["event_time"]).all()

    service_metrics = collector.collect_service_metrics(
        service_name="payment-service",
        namespace="prod",
        start_time=start,
        end_time=end,
    )
    assert set(service_metrics.keys()) == set(STANDARD_SRE_METRICS)
    for m_df in service_metrics.values():
        assert len(m_df) == 3

    # Build 3D telemetry tensor (T, N, 5)
    tensor = PrometheusCollector.build_telemetry_tensor(
        {"payment-service": service_metrics},
        service_order=["payment-service", "checkout-service"],
    )
    assert tensor.shape == (3, 2, 5)
    assert tensor[0, 0, 0] == pytest.approx(0.42)
    assert tensor[2, 0, 0] == pytest.approx(0.68)

    # Empty response and invalid range checks
    empty_collector = PrometheusCollector(client=_StubPromClient(empty=True))
    empty_df = empty_collector.query_range_metric("up", {"service": "payment-service"}, start, end)
    assert empty_df.empty
    assert PrometheusCollector.build_telemetry_tensor({}, []).shape == (0, 0, 5)

    with pytest.raises(ValueError, match="end_time"):
        collector.query_range_metric("up", {}, end, start)


@pytest.mark.unit
def test_jaeger_collector_dependencies_and_graph_sync() -> None:
    """Verify JaegerCollector extracts call graph edges and populates ContinuousGraphRegistry."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"parent": "api-gateway", "child": "checkout-service", "avgLatencyMs": 42.5},
            {"parent": "checkout-service", "child": "payment-service", "callCount": 600},
            {"parent": "", "child": "ignored-service"},
        ]
    }
    mock_session.get.return_value = mock_response

    jaeger = JaegerCollector(jaeger_query_url="http://localhost:16686", session=mock_session)
    edges = jaeger.get_service_dependencies("api-gateway", lookback_hours=12)

    assert len(edges) == 2
    assert edges[0][0] == "api-gateway"
    assert edges[0][1] == "checkout-service"
    assert edges[0][2] == pytest.approx(42.5)
    assert edges[1][0] == "checkout-service"
    assert edges[1][1] == "payment-service"
    assert edges[1][2] > 0.0

    registry = ContinuousGraphRegistry()
    JaegerCollector.sync_to_graph_registry(registry, edges)
    assert "api-gateway" in registry.nodes
    assert "checkout-service" in registry.nodes
    assert "payment-service" in registry.nodes
    assert ("api-gateway", "checkout-service") in registry.edges
    assert ("checkout-service", "payment-service") in registry.edges

    with pytest.raises(ValueError, match="lookback_hours"):
        jaeger.get_service_dependencies("api-gateway", lookback_hours=0)


@pytest.mark.unit
def test_log_parser_regex_prefilter_and_llm_extraction() -> None:
    """Verify LogParser filters benign logs and routes FATAL/OOM errors to LLM arbitrator."""
    mock_llm = MagicMock()
    mock_llm.extract_causal_vector.return_value = CausalInterventionVector(
        primary_target_entities=["payment-service"],
        directional_sentiment=-0.92,
        magnitude=0.95,
        contagion_pathways=[
            {"source": "payment-service", "target": "checkout-service", "weight": 0.85}
        ],
        confidence=0.94,
    )

    parser = LogParser(llm_arbitrator=mock_llm)
    logs = [
        "2026-05-01T10:00:00Z INFO [auth-service] Health check succeeded in 2ms",
        "2026-05-01T10:01:00Z ERROR [checkout-service] Retrying transient upstream request",
        "2026-05-01T10:02:00Z FATAL [payment-service] java.lang.OutOfMemoryError: Java heap space",
    ]

    anomalies = parser.parse_logs(logs)
    assert len(anomalies) == 2

    # First matched line is medium-severity ERROR (handled without LLM call)
    assert anomalies[0]["service"] == "checkout-service"
    assert anomalies[0]["severity"] == "medium"
    assert anomalies[0]["anomaly_signal"] < 0.0

    # Second matched line is FATAL OutOfMemoryError (routed to LLM arbitrator)
    assert anomalies[1]["service"] == "payment-service"
    assert anomalies[1]["severity"] == "high"
    assert anomalies[1]["anomaly_signal"] == pytest.approx(-0.92)
    mock_llm.extract_causal_vector.assert_called_once()


@pytest.mark.property
@settings(max_examples=20, deadline=None)
@given(
    num_steps=st.integers(min_value=2, max_value=12),
    num_services=st.integers(min_value=1, max_value=5),
)
def test_property_telemetry_tensor_assembly(num_steps: int, num_services: int) -> None:
    """Property: build_telemetry_tensor always outputs finite (T, N, 5) float32 arrays."""
    services = [f"svc-{i}" for i in range(num_services)]
    metrics_by_node: dict[str, dict[str, Any]] = {}
    for svc in services:
        metrics_by_node[svc] = {
            "cpu": pd.DataFrame({"value": np.linspace(0.2, 0.7, num_steps)}),
            "memory": pd.DataFrame({"value": np.linspace(0.4, 0.8, num_steps)}),
        }

    tensor = PrometheusCollector.build_telemetry_tensor(metrics_by_node, services)
    assert tensor.shape == (num_steps, num_services, 5)
    assert tensor.dtype == np.float32
    assert np.all(np.isfinite(tensor))
