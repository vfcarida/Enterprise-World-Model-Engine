"""Prometheus time-series metrics collector for the AIOps World Model.

Ingests pod and service telemetry (CPU, memory, HTTP 5xx error rate, p95 latency,
and active connections) at 1-minute resolution with strict bitemporal timestamps
(`event_time` vs. `knowledge_time`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import requests

try:
    from prometheus_api_client import PrometheusConnect
except ImportError:  # pragma: no cover
    PrometheusConnect = None

logger = logging.getLogger(__name__)

STANDARD_SRE_METRICS: tuple[str, ...] = (
    "cpu",
    "memory",
    "error_rate",
    "latency_p95",
)


def _utc_now_naive() -> datetime:
    """Return current UTC timestamp as a timezone-naive datetime."""
    return datetime.now(UTC).replace(tzinfo=None)


class _FallbackPrometheusConnect:
    """Lightweight HTTP fallback matching PrometheusConnect.custom_query_range."""

    def __init__(self, url: str, disable_ssl: bool = True) -> None:
        self.url = url.rstrip("/")
        self.disable_ssl = disable_ssl

    def custom_query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str,
    ) -> list[dict[str, Any]]:
        """Query Prometheus HTTP API `/api/v1/query_range`."""
        response = requests.get(
            f"{self.url}/api/v1/query_range",
            params={
                "query": query,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": step,
            },
            verify=not self.disable_ssl,
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {})
        result: list[dict[str, Any]] = list(data.get("result", []))
        return result


class PrometheusCollector:
    """Collects time-series SRE metrics from Prometheus."""

    def __init__(
        self,
        prometheus_url: str = "http://localhost:9090",
        client: Any | None = None,
    ) -> None:
        self.prometheus_url = prometheus_url
        if client is not None:
            self.prom = client
        elif PrometheusConnect is not None:
            self.prom = PrometheusConnect(url=prometheus_url, disable_ssl=True)
        else:
            self.prom = _FallbackPrometheusConnect(url=prometheus_url, disable_ssl=True)

    def query_range_metric(
        self,
        metric_name: str,
        service_labels: Mapping[str, str],
        start_time: datetime,
        end_time: datetime,
        step: str = "1m",
    ) -> pd.DataFrame:
        """Query Prometheus metric over a time range at 1-minute default resolution.

        Args:
            metric_name: Prometheus metric name or selector expression.
            service_labels: Label selector mapping (e.g. `{'service': 'payment-service', 'namespace': 'prod'}`).
            start_time: Range query start time.
            end_time: Range query end time.
            step: Resolution step (default `'1m'`).

        Returns:
            DataFrame with columns: `timestamp`, `value`, `service`, `namespace`,
            `metric_name`, `event_time`, `knowledge_time`.
        """
        if end_time < start_time:
            raise ValueError(f"end_time ({end_time}) must be >= start_time ({start_time})")

        label_selector = ",".join([f'{k}="{v}"' for k, v in service_labels.items()])
        if "{" in metric_name and metric_name.endswith("}"):
            prefix = metric_name[:-1]
            query = f"{prefix},{label_selector}}}" if label_selector else metric_name
        else:
            query = f"{metric_name}{{{label_selector}}}" if label_selector else metric_name

        result = self.prom.custom_query_range(
            query=query,
            start_time=start_time,
            end_time=end_time,
            step=step,
        )

        query_knowledge_time = _utc_now_naive()
        records: list[dict[str, Any]] = []
        for metric_data in result:
            metric_labels = metric_data.get("metric", {})
            values = metric_data.get("values", [])

            for ts_raw, val_raw in values:
                ts_dt = (
                    datetime.fromtimestamp(float(ts_raw), tz=UTC).replace(tzinfo=None)
                    if isinstance(ts_raw, (int, float, str))
                    else ts_raw
                )
                records.append(
                    {
                        "timestamp": ts_dt,
                        "value": float(val_raw),
                        "service": str(
                            metric_labels.get(
                                "service",
                                service_labels.get("service", "unknown"),
                            )
                        ),
                        "namespace": str(
                            metric_labels.get(
                                "namespace",
                                service_labels.get("namespace", "default"),
                            )
                        ),
                        "metric_name": metric_name,
                    }
                )

        if not records:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "value",
                    "service",
                    "namespace",
                    "metric_name",
                    "event_time",
                    "knowledge_time",
                ]
            )

        df = pd.DataFrame(records)
        df["event_time"] = df["timestamp"]
        df["knowledge_time"] = df["event_time"].apply(
            lambda et: (
                max(et, query_knowledge_time) if isinstance(et, datetime) else query_knowledge_time
            )
        )
        return df

    def collect_service_metrics(
        self,
        service_name: str,
        namespace: str,
        start_time: datetime,
        end_time: datetime,
        step: str = "1m",
    ) -> dict[str, pd.DataFrame]:
        """Collect standard SRE metrics (CPU, memory, error rate, p95 latency) for a service.

        Returns:
            Dict with keys: `'cpu'`, `'memory'`, `'error_rate'`, `'latency_p95'`.
        """
        labels = {"service": service_name, "namespace": namespace}
        return {
            "cpu": self.query_range_metric(
                "container_cpu_usage_seconds_total",
                labels,
                start_time,
                end_time,
                step=step,
            ),
            "memory": self.query_range_metric(
                "container_memory_working_set_bytes",
                labels,
                start_time,
                end_time,
                step=step,
            ),
            "error_rate": self.query_range_metric(
                'http_requests_total{status=~"5.."}',
                labels,
                start_time,
                end_time,
                step=step,
            ),
            "latency_p95": self.query_range_metric(
                'http_request_duration_seconds{quantile="0.95"}',
                labels,
                start_time,
                end_time,
                step=step,
            ),
        }

    @staticmethod
    def build_telemetry_tensor(
        service_metrics_by_node: Mapping[str, Mapping[str, pd.DataFrame]],
        service_order: Sequence[str],
        num_steps: int | None = None,
    ) -> np.ndarray:
        """Assemble per-service metric DataFrames into a `(T, N, 5)` telemetry tensor.

        The 5 feature channels are `[cpu, memory, error_rate, latency_p95, active_connections]`.
        """
        n_nodes = len(service_order)
        if n_nodes == 0:
            return np.zeros((0, 0, 5), dtype=np.float32)

        inferred_t = num_steps or 0
        if inferred_t == 0:
            for svc in service_order:
                m_dict = service_metrics_by_node.get(svc, {})
                for df in m_dict.values():
                    inferred_t = max(inferred_t, len(df))
        if inferred_t == 0:
            return np.zeros((0, n_nodes, 5), dtype=np.float32)

        channel_names = ("cpu", "memory", "error_rate", "latency_p95", "active_connections")
        tensor = np.zeros((inferred_t, n_nodes, 5), dtype=np.float32)

        for n_idx, svc in enumerate(service_order):
            m_dict = service_metrics_by_node.get(svc, {})
            for c_idx, ch in enumerate(channel_names):
                df = m_dict.get(ch)
                if df is not None and not df.empty and "value" in df.columns:
                    vals = df["value"].to_numpy(dtype=np.float32)
                    length = min(len(vals), inferred_t)
                    tensor[:length, n_idx, c_idx] = vals[:length]
                    if length < inferred_t and length > 0:
                        tensor[length:, n_idx, c_idx] = vals[length - 1]

        return tensor
