"""Jaeger distributed tracing collector for microservice call-graph extraction.

Extracts directed service dependency edges and latency characteristics from Jaeger
distributed traces and synchronizes them into the ContinuousGraphRegistry.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import requests

from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node

try:
    from jaeger_client import Config as JaegerConfig
except ImportError:  # pragma: no cover
    JaegerConfig = None

logger = logging.getLogger(__name__)


def _utc_now_ms() -> int:
    """Return current UTC epoch time in milliseconds."""
    return int(datetime.now(UTC).timestamp() * 1000)


class JaegerCollector:
    """Extracts microservice call-graph topology and edge latencies from Jaeger."""

    def __init__(
        self,
        jaeger_query_url: str = "http://localhost:16686",
        session: Any | None = None,
    ) -> None:
        self.jaeger_url = jaeger_query_url.rstrip("/")
        self.session = session

    def get_service_dependencies(
        self,
        service_name: str,
        lookback_hours: int = 24,
    ) -> list[tuple[str, str, float]]:
        """Extract service dependencies (call-graph edges) from Jaeger traces.

        Args:
            service_name: Target microservice name.
            lookback_hours: Trace lookback window in hours.

        Returns:
            List of `(source_service, target_service, avg_latency_ms)` tuples.
        """
        if lookback_hours <= 0:
            raise ValueError(f"lookback_hours must be > 0, got {lookback_hours}")

        http_getter = self.session.get if self.session is not None else requests.get
        response = http_getter(
            f"{self.jaeger_url}/api/dependencies",
            params={
                "service": service_name,
                "endTs": _utc_now_ms(),
                "lookback": lookback_hours * 3600 * 1000,
            },
        )
        payload = response.json() if hasattr(response, "json") else {}
        dependencies = payload.get("data", [])

        edges: list[tuple[str, str, float]] = []
        for dep in dependencies:
            parent = str(dep.get("parent", "")).strip()
            child = str(dep.get("child", "")).strip()
            if not parent or not child:
                continue

            if "avgLatencyMs" in dep:
                avg_latency_ms = float(dep["avgLatencyMs"])
            elif "latency_ms" in dep:
                avg_latency_ms = float(dep["latency_ms"])
            else:
                call_count = float(dep.get("callCount", 0))
                # Deterministic latency estimate when only callCount is returned by /api/dependencies
                avg_latency_ms = 100.0 if call_count <= 0 else min(500.0, 25.0 + 0.05 * call_count)

            edges.append((parent, child, avg_latency_ms))

        return edges

    @staticmethod
    def sync_to_graph_registry(
        registry: ContinuousGraphRegistry,
        dependencies: Sequence[tuple[str, str, float]],
        node_features: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        """Populate a ContinuousGraphRegistry from Jaeger call-graph edges."""
        now = datetime.now(UTC).replace(tzinfo=None)
        features_map = node_features or {}

        for parent, child, latency_ms in dependencies:
            for svc in (parent, child):
                if svc not in registry.nodes:
                    svc_feats = dict(
                        features_map.get(
                            svc,
                            {
                                "cpu_utilization": 0.25,
                                "memory_usage": 0.40,
                                "error_rate": 0.001,
                                "p95_latency_ms": float(latency_ms),
                                "active_connections": 50.0,
                            },
                        )
                    )
                    registry.add_node(
                        Node(
                            node_id=svc,
                            node_type="service",
                            features=svc_feats,
                            last_updated=now,
                        )
                    )

            # Convert latency into coupling weight in (0, 1]
            coupling_weight = float(min(1.0, max(0.1, latency_ms / 200.0)))
            registry.add_edge(
                Edge(
                    source_id=parent,
                    target_id=child,
                    edge_type="api_call",
                    weight=coupling_weight,
                    last_updated=now,
                )
            )
