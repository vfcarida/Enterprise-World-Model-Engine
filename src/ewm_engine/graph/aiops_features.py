"""Domain-specific node feature schema for AIOps World Model microservices."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass
class AIOpsNodeFeatures:
    """Standardized feature schema for microservices."""

    cpu_utilization_pct: float  # CPU usage percentage (0-100)
    memory_bytes: float  # Memory consumption in bytes
    error_rate_per_sec: float  # Errors per second
    latency_p95_ms: float  # 95th percentile latency in milliseconds
    active_connections: float  # Number of active TCP connections

    def to_dict(self) -> dict[str, float]:
        """Serialize features to a dictionary with float values."""
        return {
            "cpu_utilization_pct": float(self.cpu_utilization_pct),
            "memory_bytes": float(self.memory_bytes),
            "error_rate_per_sec": float(self.error_rate_per_sec),
            "latency_p95_ms": float(self.latency_p95_ms),
            "active_connections": float(self.active_connections),
        }

    def to_vector(self) -> np.ndarray:
        """Convert features to a 1D float32 numpy array in canonical feature order."""
        return np.array(
            [
                self.cpu_utilization_pct,
                self.memory_bytes,
                self.error_rate_per_sec,
                self.latency_p95_ms,
                self.active_connections,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> AIOpsNodeFeatures:
        """Construct AIOpsNodeFeatures from a feature dictionary."""
        return cls(
            cpu_utilization_pct=float(data["cpu_utilization_pct"]),
            memory_bytes=float(data["memory_bytes"]),
            error_rate_per_sec=float(data["error_rate_per_sec"]),
            latency_p95_ms=float(data["latency_p95_ms"]),
            active_connections=float(data["active_connections"]),
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return canonical feature names."""
        return [
            "cpu_utilization_pct",
            "memory_bytes",
            "error_rate_per_sec",
            "latency_p95_ms",
            "active_connections",
        ]
