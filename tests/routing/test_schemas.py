"""Unit and property tests for Mechanistic Router Pydantic schemas."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ewm_engine.routing.schemas import (
    AIOpsEventInput,
    CausalInterventionVector,
    EventModality,
    FinancialEventInput,
    RoutingDecision,
)


@pytest.mark.unit
def test_financial_event_valid_bitemporal_order() -> None:
    """Valid financial event with event_time <= knowledge_time <= ingestion_time succeeds."""
    t0 = datetime(2026, 5, 1, 12, 0, 0)
    t1 = datetime(2026, 5, 1, 12, 0, 5)
    t2 = datetime(2026, 5, 1, 12, 0, 10)

    event = FinancialEventInput(
        event_id="fin_001",
        event_time=t0,
        knowledge_time=t1,
        ingestion_time=t2,
        source="sec_edgar_8k",
        modality=EventModality.QUALITATIVE,
        text_content="Executive transition announced.",
        affected_tickers=["AAPL"],
    )
    assert event.event_id == "fin_001"
    assert event.event_time <= event.knowledge_time <= event.ingestion_time
    assert event.modality == EventModality.QUALITATIVE


@pytest.mark.unit
def test_financial_event_rejects_knowledge_before_event() -> None:
    """FinancialEventInput rejects knowledge_time < event_time (causality violation)."""
    t0 = datetime(2026, 5, 1, 12, 0, 10)
    t1 = datetime(2026, 5, 1, 12, 0, 0)

    with pytest.raises(ValidationError, match="knowledge_time cannot precede event_time"):
        FinancialEventInput(
            event_id="fin_bad_1",
            event_time=t0,
            knowledge_time=t1,
            source="newswire_api",
            modality=EventModality.QUANTITATIVE,
        )


@pytest.mark.unit
def test_financial_event_rejects_ingestion_before_knowledge() -> None:
    """FinancialEventInput rejects explicit ingestion_time < knowledge_time."""
    t0 = datetime(2026, 5, 1, 12, 0, 0)
    t1 = datetime(2026, 5, 1, 12, 0, 10)
    t2 = datetime(2026, 5, 1, 12, 0, 5)

    with pytest.raises(ValidationError, match="ingestion_time cannot precede knowledge_time"):
        FinancialEventInput(
            event_id="fin_bad_2",
            event_time=t0,
            knowledge_time=t1,
            ingestion_time=t2,
            source="newswire_api",
            modality=EventModality.QUANTITATIVE,
        )


@pytest.mark.unit
def test_aiops_event_bitemporal_validation_and_timezone_normalization() -> None:
    """AIOpsEventInput enforces bitemporal ordering across naive and timezone-aware datetimes."""
    t0 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 5, 1, 11, 59, 0, tzinfo=UTC)

    with pytest.raises(ValidationError, match="knowledge_time cannot precede event_time"):
        AIOpsEventInput(
            event_id="ops_bad_1",
            event_time=t0,
            knowledge_time=t1,
            source="prometheus_alert",
            modality=EventModality.QUANTITATIVE,
        )

    t2 = datetime(2026, 5, 1, 12, 1, 0, tzinfo=UTC)
    t3 = datetime(2026, 5, 1, 12, 0, 30, tzinfo=UTC)
    with pytest.raises(ValidationError, match="ingestion_time cannot precede knowledge_time"):
        AIOpsEventInput(
            event_id="ops_bad_2",
            event_time=t0,
            knowledge_time=t2,
            ingestion_time=t3,
            source="prometheus_alert",
            modality=EventModality.QUANTITATIVE,
        )

    # Future knowledge_time without explicit ingestion_time auto-aligns ingestion_time
    future_t0 = datetime(2035, 1, 1, 0, 0, 0)
    future_t1 = datetime(2035, 1, 1, 0, 1, 0)
    valid_future = AIOpsEventInput(
        event_id="ops_future_ok",
        event_time=future_t0,
        knowledge_time=future_t1,
        source="kubernetes_watch",
        modality=EventModality.MIXED,
    )
    assert valid_future.ingestion_time >= valid_future.knowledge_time


@pytest.mark.unit
def test_routing_decision_and_causal_vector_validation() -> None:
    """RoutingDecision and CausalInterventionVector validate bounds and destinations."""
    decision = RoutingDecision(
        event_id="evt_1",
        route_destination="numerical_slm",
        confidence=0.95,
        reasoning="Fast path",
        processing_time_ms=1.2,
    )
    assert decision.route_destination == "numerical_slm"

    with pytest.raises(ValidationError, match="Invalid route_destination"):
        RoutingDecision(
            event_id="evt_1",
            route_destination="unknown_target",
            confidence=0.5,
            reasoning="Invalid target",
            processing_time_ms=1.0,
        )

    with pytest.raises(ValidationError):
        CausalInterventionVector(
            primary_target_entities=["NVDA"],
            directional_sentiment=-1.5,  # Out of [-1, 1]
            magnitude=0.8,
            confidence=0.9,
        )


@pytest.mark.property
@given(
    base_dt=st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2030, 1, 1),
    ),
    delta_secs=st.integers(min_value=-3600, max_value=3600),
)
def test_property_bitemporal_causality_enforcement(base_dt: datetime, delta_secs: int) -> None:
    """Property: FinancialEventInput accepts iff knowledge_time >= event_time."""
    kn_dt = base_dt + timedelta(seconds=delta_secs)
    if delta_secs < 0:
        with pytest.raises(ValidationError):
            FinancialEventInput(
                event_id="prop_evt",
                event_time=base_dt,
                knowledge_time=kn_dt,
                source="market_1t",
                modality=EventModality.QUANTITATIVE,
            )
    else:
        evt = FinancialEventInput(
            event_id="prop_evt",
            event_time=base_dt,
            knowledge_time=kn_dt,
            source="market_1t",
            modality=EventModality.QUANTITATIVE,
        )
        assert evt.event_time <= evt.knowledge_time <= evt.ingestion_time
