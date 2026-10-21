"""Distributed result stores for Redis and AWS S3.

Conforms to Track T2:
External cache backends support optional `[cache]` installation extras
(`redis>=5.0`, `boto3>=1.34`) or injected client instances while keeping core zero-dependency.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Mapping
from typing import Any

from ewm_engine.durability.protocol import ResultStore
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult, Trajectory


def _serialize_result(
    fingerprint: str,
    result: SimulationResult,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    payload = {
        "schema_version": "1.0.0",
        "fingerprint": fingerprint,
        "metadata": dict(metadata or {}),
        "scenario": result.scenario.model_dump(mode="json"),
        "provenance": result.provenance.model_dump(mode="json"),
        "run_metrics": (
            result.run_metrics.model_dump(mode="json") if result.run_metrics is not None else None
        ),
        "trajectories": [t.model_dump(mode="json") for t in result.trajectories],
    }
    return json.dumps(payload, sort_keys=True)


def _deserialize_result(raw_json: str | bytes) -> SimulationResult | None:
    try:
        text = raw_json.decode("utf-8") if isinstance(raw_json, bytes) else raw_json
        data = json.loads(text)
        scenario = Scenario.model_validate(data["scenario"])
        provenance = Provenance.model_validate(data["provenance"])
        run_metrics = (
            RunMetrics.model_validate(data["run_metrics"])
            if data.get("run_metrics") is not None
            else None
        )
        trajectories = [
            Trajectory.model_validate(t_data) for t_data in data.get("trajectories", [])
        ]
        for t in trajectories:
            t.finalize()
        return SimulationResult(
            scenario=scenario,
            provenance=provenance,
            trajectories=trajectories,
            run_metrics=run_metrics,
        )
    except Exception:
        return None


class RedisResultStore(ResultStore):
    """Redis-backed distributed ResultStore.

    Requires `pip install ewm-engine[cache]` (`redis>=5.0`) or an injected Redis client.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        ttl_seconds: int | None = None,
        prefix: str = "ewm:result:",
        client: Any | None = None,
    ) -> None:
        self.url = url
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        if client is not None:
            self._client = client
        elif importlib.util.find_spec("redis") is not None:
            redis_mod = importlib.import_module("redis")
            self._client = redis_mod.Redis.from_url(url)
        else:
            raise NotImplementedError(
                "RedisResultStore is available with the '[cache]' extra. "
                "Install with: pip install ewm-engine[cache]"
            )

    def _key(self, fingerprint: str) -> str:
        return f"{self.prefix}{fingerprint}"

    def get(self, fingerprint: str) -> SimulationResult | None:
        raw = self._client.get(self._key(fingerprint))
        if raw is None:
            return None
        return _deserialize_result(raw)

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = _serialize_result(fingerprint, result, metadata)
        if self.ttl_seconds is not None:
            self._client.set(self._key(fingerprint), payload, ex=self.ttl_seconds)
        else:
            self._client.set(self._key(fingerprint), payload)

    def contains(self, fingerprint: str) -> bool:
        return bool(self._client.exists(self._key(fingerprint)))

    def delete(self, fingerprint: str) -> bool:
        deleted = int(self._client.delete(self._key(fingerprint)))
        return deleted > 0

    def list_fingerprints(self) -> list[str]:
        keys = self._client.keys(f"{self.prefix}*")
        fps: list[str] = []
        for k in keys:
            k_str = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            if k_str.startswith(self.prefix):
                fps.append(k_str[len(self.prefix) :])
        return sorted(fps)


class S3ResultStore(ResultStore):
    """AWS S3-backed object ResultStore.

    Requires `pip install ewm-engine[cache]` (`boto3>=1.34`) or an injected S3 client.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "ewm-cache/",
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix
        if client is not None:
            self._client = client
        elif importlib.util.find_spec("boto3") is not None:
            boto3 = importlib.import_module("boto3")
            self._client = boto3.client("s3")
        else:
            raise NotImplementedError(
                "S3ResultStore is available with the '[cache]' extra. "
                "Install with: pip install ewm-engine[cache]"
            )

    def _key(self, fingerprint: str) -> str:
        return f"{self.prefix}{fingerprint}/result.json"

    def get(self, fingerprint: str) -> SimulationResult | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=self._key(fingerprint))
            body = resp["Body"]
            raw = body.read() if hasattr(body, "read") else body
            return _deserialize_result(raw)
        except Exception:
            return None

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload = _serialize_result(fingerprint, result, metadata)
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._key(fingerprint),
            Body=payload.encode("utf-8"),
        )

    def contains(self, fingerprint: str) -> bool:
        return self.get(fingerprint) is not None

    def delete(self, fingerprint: str) -> bool:
        if not self.contains(fingerprint):
            return False
        self._client.delete_object(Bucket=self.bucket, Key=self._key(fingerprint))
        return True

    def list_fingerprints(self) -> list[str]:
        resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix)
        contents = resp.get("Contents", [])
        fps: list[str] = []
        suffix = "/result.json"
        for obj in contents:
            key = str(obj.get("Key", ""))
            if key.startswith(self.prefix) and key.endswith(suffix):
                fp = key[len(self.prefix) : -len(suffix)]
                if fp:
                    fps.append(fp)
        return sorted(fps)
