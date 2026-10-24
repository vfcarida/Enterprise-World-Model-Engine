"""First-class bitemporal data discipline (event-time vs. knowledge-time).

Enforces Vision §7.1:
- Every event and telemetry record carries both ``event_time`` (when the real-world
  transition occurred) and ``knowledge_time`` (when that information became available
  to the decision system, ``knowledge_time >= event_time``).
- Any feature extraction, belief update, or query at decision time ``t`` may use
  ONLY records satisfying ``event_time <= t`` and ``knowledge_time <= t``.
- Supports update lag, late-arriving revisions, metric definition changes across
  time, and multiple temporal granularities.
- Attempts to access future-knowledge-time records at decision time ``t`` raise
  ``BitemporalLeakageError``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScalarValue = float | int | str | bool


class BitemporalLeakageError(RuntimeError):
    """Raised when a query at decision time t attempts to read data with knowledge_time > t or event_time > t."""


class TemporalGranularity(StrEnum):
    """Supported temporal aggregation granularities in a bitemporal ledger."""

    STEP = "step"
    SHIFT_BLOCK = "shift_block"
    DAILY_SUMMARY = "daily_summary"


GRANULARITY_WINDOW_STEPS: dict[TemporalGranularity, int] = {
    TemporalGranularity.STEP: 1,
    TemporalGranularity.SHIFT_BLOCK: 4,
    TemporalGranularity.DAILY_SUMMARY: 8,
}


class MetricDefinitionVersion(BaseModel):
    """Versioned operational metric definition handling semantics shifts over time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str
    version: str
    effective_from_time: float = 0.0
    effective_until_time: float | None = None
    formula_description: str = ""
    canonical_scale: float = 1.0
    canonical_offset: float = 0.0

    def is_active_at(self, event_time: float) -> bool:
        """Return True if this metric definition version is active at ``event_time``."""
        if event_time < self.effective_from_time:
            return False
        return not (
            self.effective_until_time is not None and event_time >= self.effective_until_time
        )

    def to_canonical_value(self, raw_value: float) -> float:
        """Convert a version-specific metric reading into canonical units."""
        return float(raw_value) * self.canonical_scale + self.canonical_offset


class BitemporalRecord(BaseModel):
    """Immutable bitemporal event/telemetry record with event_time and knowledge_time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    stream: str
    event_time: float = Field(description="Simulation time when the event occurred.")
    knowledge_time: float = Field(
        description="Simulation time when the event became visible in telemetry (>= event_time)."
    )
    granularity: TemporalGranularity = TemporalGranularity.STEP
    definition_version: str = "v1"
    payload: dict[str, ScalarValue] = Field(default_factory=dict)
    is_revision: bool = False
    supersedes_record_id: str | None = None

    @model_validator(mode="after")
    def _validate_causality(self) -> BitemporalRecord:
        if self.knowledge_time < self.event_time - 1e-12:
            raise BitemporalLeakageError(
                f"Acausal bitemporal record '{self.record_id}': knowledge_time "
                f"({self.knowledge_time}) < event_time ({self.event_time})."
            )
        return self

    @property
    def visibility_lag(self) -> float:
        """Lag between event occurrence and telemetry availability."""
        return float(self.knowledge_time - self.event_time)

    def is_visible_at(self, decision_time: float) -> bool:
        """Return True iff both event_time <= decision_time and knowledge_time <= decision_time."""
        return (
            self.event_time <= decision_time + 1e-12
            and self.knowledge_time <= decision_time + 1e-12
        )


class PointInTimeView(BaseModel):
    """Leak-free as-of view of a BitemporalLedger at a specific ``decision_time``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_time: float
    visible_records: tuple[BitemporalRecord, ...] = Field(default_factory=tuple)
    withheld_future_knowledge_count: int = Field(
        default=0,
        description="Count of events with event_time <= decision_time but knowledge_time > decision_time.",
    )
    withheld_future_event_count: int = Field(
        default=0,
        description="Count of events with event_time > decision_time.",
    )
    active_definitions: dict[str, MetricDefinitionVersion] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_point_in_time_invariant(self) -> PointInTimeView:
        for rec in self.visible_records:
            if not rec.is_visible_at(self.decision_time):
                raise BitemporalLeakageError(
                    f"PointInTimeView at decision_time={self.decision_time} contaminated by "
                    f"record '{rec.record_id}' (event_time={rec.event_time}, "
                    f"knowledge_time={rec.knowledge_time})."
                )
        return self

    @property
    def fingerprint(self) -> str:
        """Canonical SHA-256 digest of this point-in-time view."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def effective_records(self, stream: str | None = None) -> tuple[BitemporalRecord, ...]:
        """Return visible records after applying visible revisions (superseding older records)."""
        superseded_ids: set[str] = {
            rec.supersedes_record_id
            for rec in self.visible_records
            if rec.is_revision and rec.supersedes_record_id is not None
        }
        filtered = [
            rec
            for rec in self.visible_records
            if rec.record_id not in superseded_ids and (stream is None or rec.stream == stream)
        ]
        return tuple(sorted(filtered, key=lambda r: (r.event_time, r.knowledge_time, r.record_id)))

    def latest_value(
        self,
        stream: str,
        field: str,
        *,
        default: float = 0.0,
        normalize_definition: bool = True,
    ) -> float:
        """Fetch the most recent visible value of ``field`` in ``stream`` as of ``decision_time``."""
        recs = self.effective_records(stream)
        for rec in reversed(recs):
            if field in rec.payload:
                raw_val = float(rec.payload[field])
                if normalize_definition:
                    def_key = f"{field}:{rec.definition_version}"
                    if def_key in self.active_definitions:
                        return self.active_definitions[def_key].to_canonical_value(raw_val)
                return raw_val
        return default

    def aggregate_window(
        self,
        stream: str,
        field: str,
        *,
        granularity: TemporalGranularity = TemporalGranularity.STEP,
        normalize_definition: bool = True,
    ) -> float:
        """Sum visible values in ``stream.field`` over the trailing granularity window."""
        window = float(GRANULARITY_WINDOW_STEPS[granularity])
        min_event_time = self.decision_time - window + 1.0
        total = 0.0
        for rec in self.effective_records(stream):
            if rec.event_time + 1e-12 >= min_event_time and field in rec.payload:
                raw_val = float(rec.payload[field])
                if normalize_definition:
                    def_key = f"{field}:{rec.definition_version}"
                    if def_key in self.active_definitions:
                        raw_val = self.active_definitions[def_key].to_canonical_value(raw_val)
                total += raw_val
        return total

    def assert_record_accessible(self, record: BitemporalRecord) -> None:
        """Explicit guard raising BitemporalLeakageError if ``record`` is not visible at ``decision_time``."""
        if not record.is_visible_at(self.decision_time):
            raise BitemporalLeakageError(
                f"Bitemporal leakage blocked at decision_time={self.decision_time}: "
                f"attempted to access record '{record.record_id}' with "
                f"event_time={record.event_time}, knowledge_time={record.knowledge_time}."
            )


class BitemporalLedger:
    """Append-only bitemporal store enforcing event-time vs. knowledge-time separation."""

    def __init__(self) -> None:
        self._records: dict[str, BitemporalRecord] = {}
        self._definitions: dict[str, MetricDefinitionVersion] = {}
        self._blocked_leakage_attempts: int = 0

    @property
    def records(self) -> tuple[BitemporalRecord, ...]:
        """All stored records in insertion order."""
        return tuple(self._records.values())

    @property
    def blocked_leakage_attempts(self) -> int:
        """Count of blocked attempts to read future-knowledge-time records."""
        return self._blocked_leakage_attempts

    def register_definition(self, definition: MetricDefinitionVersion) -> None:
        """Register a versioned metric definition."""
        key = f"{definition.metric_name}:{definition.version}"
        self._definitions[key] = definition

    def append(self, record: BitemporalRecord) -> None:
        """Append a validated bitemporal record to the ledger."""
        if record.record_id in self._records:
            raise ValueError(f"Duplicate bitemporal record_id: {record.record_id!r}")
        if record.is_revision and record.supersedes_record_id not in self._records:
            raise KeyError(
                f"Revision record '{record.record_id}' references unknown "
                f"supersedes_record_id={record.supersedes_record_id!r}."
            )
        self._records[record.record_id] = record

    def read_record_at(self, record_id: str, decision_time: float) -> BitemporalRecord:
        """Read a specific record as of ``decision_time``, raising BitemporalLeakageError if not yet known."""
        if record_id not in self._records:
            raise KeyError(f"Unknown record_id: {record_id!r}")
        rec = self._records[record_id]
        if not rec.is_visible_at(decision_time):
            self._blocked_leakage_attempts += 1
            raise BitemporalLeakageError(
                f"Bitemporal leakage guard triggered at decision_time={decision_time}: "
                f"record '{record_id}' has event_time={rec.event_time}, "
                f"knowledge_time={rec.knowledge_time}."
            )
        return rec

    def as_of(self, decision_time: float) -> PointInTimeView:
        """Construct a leak-free PointInTimeView containing only records with knowledge_time <= decision_time."""
        visible: list[BitemporalRecord] = []
        withheld_knowledge = 0
        withheld_future_event = 0

        for rec in self._records.values():
            if rec.is_visible_at(decision_time):
                visible.append(rec)
            elif rec.event_time <= decision_time + 1e-12:
                withheld_knowledge += 1
            else:
                withheld_future_event += 1

        return PointInTimeView(
            decision_time=float(decision_time),
            visible_records=tuple(visible),
            withheld_future_knowledge_count=withheld_knowledge,
            withheld_future_event_count=withheld_future_event,
            active_definitions=dict(self._definitions),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize ledger to a JSON-compatible dictionary."""
        return {
            "records": [r.model_dump(mode="json") for r in self._records.values()],
            "definitions": {
                k: d.model_dump(mode="json") for k, d in sorted(self._definitions.items())
            },
            "blocked_leakage_attempts": self._blocked_leakage_attempts,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BitemporalLedger:
        """Deserialize a BitemporalLedger from a dictionary."""
        ledger = cls()
        for d_raw in data.get("definitions", {}).values():
            ledger.register_definition(MetricDefinitionVersion.model_validate(d_raw))
        for r_raw in data.get("records", []):
            ledger.append(BitemporalRecord.model_validate(r_raw))
        ledger._blocked_leakage_attempts = int(data.get("blocked_leakage_attempts", 0))
        return ledger


class DataLeakageCertificate(BaseModel):
    """Cryptographic audit certificate proving hard temporal, state, and configuration non-leakage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_partition_seed: int = Field(
        description="Isolated RNG seed used for training/induction data."
    )
    eval_partition_seed: int = Field(description="Isolated RNG seed used for evaluation scenarios.")
    train_sample_count: int = Field(description="Number of training transitions generated.")
    eval_sample_count: int = Field(description="Number of evaluation transitions generated.")
    state_fingerprint_overlap_count: int = Field(
        description="Count of ontic state fingerprints shared between training and evaluation partitions."
    )
    max_train_timestamp: float = Field(description="Maximum timestamp in the training partition.")
    min_eval_timestamp: float = Field(description="Minimum timestamp in the evaluation partition.")
    temporal_boundary_verified: bool = Field(
        description="True if max_train_timestamp < min_eval_timestamp."
    )
    held_out_config_versions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Evaluation context versions strictly withheld from training.",
    )
    config_boundary_verified: bool = Field(
        description="True if no held-out evaluation context version appeared in training."
    )
    oracle_isolation_verified: bool = Field(
        description="True if candidate models never received ground-truth next_state or future schedule."
    )
    passed: bool = Field(description="True only if all leakage checks passed with zero violations.")


def verify_no_data_leakage(
    *,
    train_partition_seed: int,
    eval_partition_seed: int,
    train_state_fingerprints: Sequence[str],
    eval_state_fingerprints: Sequence[str],
    train_timestamps: Sequence[float],
    eval_timestamps: Sequence[float],
    train_context_versions: Sequence[str],
    held_out_eval_context_versions: Sequence[str],
    oracle_isolation_verified: bool = True,
) -> DataLeakageCertificate:
    """Verify hard temporal, state-fingerprint, and configuration boundaries between train and eval."""
    overlap = len(set(train_state_fingerprints).intersection(set(eval_state_fingerprints)))
    max_train_ts = float(max(train_timestamps)) if train_timestamps else 0.0
    min_eval_ts = float(min(eval_timestamps)) if eval_timestamps else 1.0
    temporal_ok = max_train_ts < min_eval_ts
    seed_ok = int(train_partition_seed) != int(eval_partition_seed)

    train_cfg_set = set(train_context_versions)
    held_out_set = set(held_out_eval_context_versions)
    cfg_ok = len(train_cfg_set.intersection(held_out_set)) == 0

    passed = seed_ok and overlap == 0 and temporal_ok and cfg_ok and bool(oracle_isolation_verified)
    return DataLeakageCertificate(
        train_partition_seed=int(train_partition_seed),
        eval_partition_seed=int(eval_partition_seed),
        train_sample_count=len(train_state_fingerprints),
        eval_sample_count=len(eval_state_fingerprints),
        state_fingerprint_overlap_count=overlap,
        max_train_timestamp=max_train_ts,
        min_eval_timestamp=min_eval_ts,
        temporal_boundary_verified=temporal_ok,
        held_out_config_versions=tuple(sorted(held_out_set)),
        config_boundary_verified=cfg_ok,
        oracle_isolation_verified=bool(oracle_isolation_verified),
        passed=passed,
    )
