"""Stub interfaces for external distributed result stores (Redis, S3).

Documented stub interfaces conforming to Track T2:
External cache backends require the optional `[cache]` installation extra
(e.g., redis-py, boto3). Core remains zero-dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ewm_engine.durability.protocol import ResultStore
from ewm_engine.simulation.trajectory import SimulationResult


class RedisResultStore(ResultStore):
    """Stub for Redis-backed distributed ResultStore.

    Requires `pip install ewm-engine[cache]` (redis>=5.0).
    """

    def __init__(
        self, url: str = "redis://localhost:6379/0", ttl_seconds: int | None = None
    ) -> None:
        self.url = url
        self.ttl_seconds = ttl_seconds
        raise NotImplementedError(
            "RedisResultStore is available with the '[cache]' extra. "
            "Install with: pip install ewm-engine[cache]"
        )

    def get(self, fingerprint: str) -> SimulationResult | None:
        raise NotImplementedError("RedisResultStore requires [cache] extra.")

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("RedisResultStore requires [cache] extra.")

    def contains(self, fingerprint: str) -> bool:
        raise NotImplementedError("RedisResultStore requires [cache] extra.")

    def delete(self, fingerprint: str) -> bool:
        raise NotImplementedError("RedisResultStore requires [cache] extra.")

    def list_fingerprints(self) -> list[str]:
        raise NotImplementedError("RedisResultStore requires [cache] extra.")


class S3ResultStore(ResultStore):
    """Stub for AWS S3-backed object ResultStore.

    Requires `pip install ewm-engine[cache]` (boto3>=1.34).
    """

    def __init__(self, bucket: str, prefix: str = "ewm-cache/") -> None:
        self.bucket = bucket
        self.prefix = prefix
        raise NotImplementedError(
            "S3ResultStore is available with the '[cache]' extra. "
            "Install with: pip install ewm-engine[cache]"
        )

    def get(self, fingerprint: str) -> SimulationResult | None:
        raise NotImplementedError("S3ResultStore requires [cache] extra.")

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError("S3ResultStore requires [cache] extra.")

    def contains(self, fingerprint: str) -> bool:
        raise NotImplementedError("S3ResultStore requires [cache] extra.")

    def delete(self, fingerprint: str) -> bool:
        raise NotImplementedError("S3ResultStore requires [cache] extra.")

    def list_fingerprints(self) -> list[str]:
        raise NotImplementedError("S3ResultStore requires [cache] extra.")
