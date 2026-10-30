"""Pydantic schemas for the Mechanistic Router gateway.

Enforces bitemporal discipline (event_time <= knowledge_time <= ingestion_time),
modality typing, and strict JSON-mode causal intervention vector validation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

VALID_ROUTE_DESTINATIONS: frozenset[str] = frozenset(
    {
        "deterministic_engine",
        "semantic_classifier",
        "frontier_llm",
        "numerical_slm",
    }
)


def _utcnow_naive() -> datetime:
    """Return current UTC time as a naive datetime for consistent default comparisons."""
    return datetime.now(UTC).replace(tzinfo=None)


def _to_comparable_utc(dt: datetime) -> datetime:
    """Normalize a naive or timezone-aware datetime to naive UTC for safe ordering comparison."""
    if dt.tzinfo is not None:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


class EventModality(StrEnum):
    """Modality classification for incoming financial and operational events."""

    QUANTITATIVE = "quantitative"
    QUALITATIVE = "qualitative"
    MIXED = "mixed"


class FinancialEventInput(BaseModel):
    """Schema for financial events (news, filings, macro indicators)."""

    event_id: str = Field(..., min_length=1, description="Unique event identifier")
    event_time: datetime = Field(..., description="When the event occurred (event-time)")
    knowledge_time: datetime = Field(..., description="When we learned about it (knowledge-time)")
    ingestion_time: datetime = Field(default_factory=_utcnow_naive)

    source: str = Field(
        ..., min_length=1, description="Data source (e.g., 'newswire_api', 'sec_edgar_8k')"
    )
    modality: EventModality

    # For quantitative events (e.g., CPI release matching consensus)
    metric_name: str | None = None
    metric_value: float | None = None
    consensus_value: float | None = None

    # For qualitative events (e.g., CEO resignation announcement)
    text_content: str | None = Field(default=None, max_length=100000)
    affected_tickers: list[str] | None = Field(default_factory=list)

    @field_validator("knowledge_time")
    @classmethod
    def knowledge_after_event(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Ensure knowledge_time does not precede event_time."""
        event_time = info.data.get("event_time")
        if isinstance(event_time, datetime) and _to_comparable_utc(v) < _to_comparable_utc(
            event_time
        ):
            raise ValueError("knowledge_time cannot precede event_time (violates causality)")
        return v

    @model_validator(mode="after")
    def enforce_bitemporal_order(self) -> FinancialEventInput:
        """Enforce full bitemporal invariant: event_time <= knowledge_time <= ingestion_time."""
        ev_utc = _to_comparable_utc(self.event_time)
        kn_utc = _to_comparable_utc(self.knowledge_time)
        in_utc = _to_comparable_utc(self.ingestion_time)

        if kn_utc < ev_utc:
            raise ValueError("knowledge_time cannot precede event_time (violates causality)")

        if "ingestion_time" in self.model_fields_set:
            if in_utc < kn_utc:
                raise ValueError(
                    "ingestion_time cannot precede knowledge_time (violates bitemporal discipline)"
                )
        elif in_utc < kn_utc:
            self.ingestion_time = self.knowledge_time

        return self


class AIOpsEventInput(BaseModel):
    """Schema for operational events (deployments, config, anomalies)."""

    event_id: str = Field(..., min_length=1, description="Unique event identifier")
    event_time: datetime = Field(..., description="When the event occurred (event-time)")
    knowledge_time: datetime = Field(..., description="When we learned about it (knowledge-time)")
    ingestion_time: datetime = Field(default_factory=_utcnow_naive)

    source: str = Field(
        ...,
        min_length=1,
        description="Event source ('kubernetes_watch', 'prometheus_alert', 'jaeger_trace', 'log_parser')",
    )
    modality: EventModality

    # For quantitative telemetry
    service_name: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    threshold_breached: bool | None = None

    # For qualitative ops events (deployment commit messages, log errors)
    text_content: str | None = Field(default=None, max_length=50000)
    git_commit_sha: str | None = None
    kubernetes_manifest_diff: dict[str, Any] | None = None

    @field_validator("knowledge_time")
    @classmethod
    def knowledge_after_event(cls, v: datetime, info: ValidationInfo) -> datetime:
        """Ensure knowledge_time does not precede event_time."""
        event_time = info.data.get("event_time")
        if isinstance(event_time, datetime) and _to_comparable_utc(v) < _to_comparable_utc(
            event_time
        ):
            raise ValueError("knowledge_time cannot precede event_time")
        return v

    @model_validator(mode="after")
    def enforce_bitemporal_order(self) -> AIOpsEventInput:
        """Enforce full bitemporal invariant: event_time <= knowledge_time <= ingestion_time."""
        ev_utc = _to_comparable_utc(self.event_time)
        kn_utc = _to_comparable_utc(self.knowledge_time)
        in_utc = _to_comparable_utc(self.ingestion_time)

        if kn_utc < ev_utc:
            raise ValueError("knowledge_time cannot precede event_time")

        if "ingestion_time" in self.model_fields_set:
            if in_utc < kn_utc:
                raise ValueError(
                    "ingestion_time cannot precede knowledge_time (violates bitemporal discipline)"
                )
        elif in_utc < kn_utc:
            self.ingestion_time = self.knowledge_time

        return self


class RoutingDecision(BaseModel):
    """Structured routing decision returned by the gateway."""

    event_id: str = Field(..., min_length=1)
    route_destination: str = Field(
        ...,
        description="Target processor ('deterministic_engine', 'semantic_classifier', 'frontier_llm', 'numerical_slm')",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(..., min_length=1)
    processing_time_ms: float = Field(..., ge=0.0)

    @field_validator("route_destination")
    @classmethod
    def validate_destination(cls, v: str) -> str:
        """Ensure route_destination is one of the valid architectural targets."""
        if v not in VALID_ROUTE_DESTINATIONS:
            raise ValueError(
                f"Invalid route_destination '{v}'. Must be one of {sorted(VALID_ROUTE_DESTINATIONS)}"
            )
        return v


class CausalInterventionVector(BaseModel):
    """Structured output from Frontier LLM (JSON-mode enforced)."""

    primary_target_entities: list[str] = Field(..., description="Ticker symbols or service names")
    directional_sentiment: float = Field(
        ..., ge=-1.0, le=1.0, description="Bearish(-1) to Bullish(+1)"
    )
    magnitude: float = Field(..., ge=0.0, le=1.0, description="Severity of shock")
    contagion_pathways: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Anticipated propagation edges [(source, target, weight)]",
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_metadata: dict[str, Any] = Field(default_factory=dict)
