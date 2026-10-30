"""Unit and benchmark tests for the Deterministic Rule Engine."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import pytest

from ewm_engine.routing.deterministic_engine import DeterministicRuleEngine
from ewm_engine.routing.schemas import AIOpsEventInput, EventModality, FinancialEventInput


@pytest.fixture
def engine() -> DeterministicRuleEngine:
    """Fixture providing a DeterministicRuleEngine instance."""
    return DeterministicRuleEngine()


@pytest.mark.unit
def test_financial_routine_cpi_release(engine: DeterministicRuleEngine) -> None:
    """Verify routine CPI release (1.5% deviation < 2% tolerance) -> is_routine=True."""
    now = datetime(2026, 6, 1, 12, 30, 0)
    event = FinancialEventInput(
        event_id="cpi_routine_01",
        event_time=now,
        knowledge_time=now,
        source="bls_macro",
        modality=EventModality.QUANTITATIVE,
        metric_name="cpi",
        metric_value=2.03,
        consensus_value=2.00,  # 1.5% deviation
    )
    is_routine, reasoning = engine.evaluate_financial(event)
    assert is_routine is True
    assert reasoning is not None
    assert "Routine cpi release" in reasoning
    assert "1.50%" in reasoning


@pytest.mark.unit
def test_financial_unexpected_cpi_spike(engine: DeterministicRuleEngine) -> None:
    """Verify unexpected CPI spike (5% deviation >= 2% tolerance) -> is_routine=False."""
    now = datetime(2026, 6, 1, 12, 30, 0)
    event = FinancialEventInput(
        event_id="cpi_spike_01",
        event_time=now,
        knowledge_time=now,
        source="bls_macro",
        modality=EventModality.QUANTITATIVE,
        metric_name="cpi",
        metric_value=2.10,
        consensus_value=2.00,  # 5.0% deviation
    )
    is_routine, reasoning = engine.evaluate_financial(event)
    assert is_routine is False
    assert reasoning == "Event requires semantic classification"


@pytest.mark.unit
def test_financial_macro_zero_and_missing_consensus_edges(
    engine: DeterministicRuleEngine,
) -> None:
    """Verify zero consensus and missing consensus values are handled safely."""
    now = datetime(2026, 6, 1, 12, 30, 0)

    # Zero consensus with tiny metric value (< 0.02)
    evt_zero_ok = FinancialEventInput(
        event_id="ppi_zero_ok",
        event_time=now,
        knowledge_time=now,
        source="bls_macro",
        modality=EventModality.QUANTITATIVE,
        metric_name="ppi",
        metric_value=0.01,
        consensus_value=0.0,
    )
    is_routine, _ = engine.evaluate_financial(evt_zero_ok)
    assert is_routine is True

    # Missing consensus value on macro indicator -> not routine
    evt_missing_consensus = FinancialEventInput(
        event_id="cpi_missing_cons",
        event_time=now,
        knowledge_time=now,
        source="bls_macro",
        modality=EventModality.QUANTITATIVE,
        metric_name="cpi",
        metric_value=3.5,
        consensus_value=None,
    )
    is_routine_missing, _ = engine.evaluate_financial(evt_missing_consensus)
    assert is_routine_missing is False


@pytest.mark.unit
def test_financial_tick_data_and_non_quantitative_rules(
    engine: DeterministicRuleEngine,
) -> None:
    """Verify standard tick data fast-paths while multi-ticker or qualitative events do not."""
    now = datetime(2026, 6, 1, 14, 0, 0)

    # Standard tick data (< 3 tickers, no text)
    tick_evt = FinancialEventInput(
        event_id="tick_01",
        event_time=now,
        knowledge_time=now,
        source="market_1t",
        modality=EventModality.QUANTITATIVE,
        metric_name="vwap",
        metric_value=189.50,
        affected_tickers=["AAPL", "MSFT"],
    )
    is_routine, reason = engine.evaluate_financial(tick_evt)
    assert is_routine is True
    assert reason == "Standard tick data (no qualitative component)"

    # >= 3 affected tickers -> requires semantic classification
    multi_ticker_evt = FinancialEventInput(
        event_id="tick_multi",
        event_time=now,
        knowledge_time=now,
        source="market_1t",
        modality=EventModality.QUANTITATIVE,
        affected_tickers=["AAPL", "MSFT", "NVDA"],
    )
    is_routine_multi, reason_multi = engine.evaluate_financial(multi_ticker_evt)
    assert is_routine_multi is False
    assert reason_multi == "Event requires semantic classification"

    # Qualitative event -> requires semantic routing
    qual_evt = FinancialEventInput(
        event_id="qual_01",
        event_time=now,
        knowledge_time=now,
        source="sec_edgar_8k",
        modality=EventModality.QUALITATIVE,
        text_content="CEO stepped down unexpectedly.",
    )
    is_routine_qual, reason_qual = engine.evaluate_financial(qual_evt)
    assert is_routine_qual is False
    assert reason_qual == "Non-quantitative event requires semantic routing"


@pytest.mark.unit
def test_aiops_standard_cpu_metric_and_health_checks(
    engine: DeterministicRuleEngine,
) -> None:
    """Verify standard CPU metric (70%, no alert) and routine health checks -> is_routine=True."""
    now = datetime(2026, 6, 1, 15, 0, 0)

    cpu_evt = AIOpsEventInput(
        event_id="cpu_normal_01",
        event_time=now,
        knowledge_time=now,
        source="prometheus_alert",
        modality=EventModality.QUANTITATIVE,
        service_name="checkout-service",
        metric_name="cpu_utilization",
        metric_value=70.0,
        threshold_breached=False,
    )
    is_routine, reason = engine.evaluate_aiops(cpu_evt)
    assert is_routine is True
    assert reason == "Standard cpu_utilization metric update (no alert)"

    probe_evt = AIOpsEventInput(
        event_id="probe_01",
        event_time=now,
        knowledge_time=now,
        source="liveness_probe",
        modality=EventModality.QUANTITATIVE,
        service_name="payment-service",
    )
    is_routine_probe, reason_probe = engine.evaluate_aiops(probe_evt)
    assert is_routine_probe is True
    assert reason_probe == "Routine health check: liveness_probe"


@pytest.mark.unit
def test_aiops_memory_leak_alert_and_qualitative_events(
    engine: DeterministicRuleEngine,
) -> None:
    """Verify memory leak alert (85%, threshold breached) -> is_routine=False."""
    now = datetime(2026, 6, 1, 15, 5, 0)

    mem_alert = AIOpsEventInput(
        event_id="mem_leak_01",
        event_time=now,
        knowledge_time=now,
        source="prometheus_alert",
        modality=EventModality.QUANTITATIVE,
        service_name="auth-service",
        metric_name="memory_bytes",
        metric_value=85.0,
        threshold_breached=True,
    )
    is_routine, reason = engine.evaluate_aiops(mem_alert)
    assert is_routine is False
    assert reason == "Event requires semantic classification"

    qual_ops = AIOpsEventInput(
        event_id="deploy_diff_01",
        event_time=now,
        knowledge_time=now,
        source="kubernetes_watch",
        modality=EventModality.QUALITATIVE,
        text_content="Rollback deployment due to crashloopbackoff",
    )
    is_routine_qual, reason_qual = engine.evaluate_aiops(qual_ops)
    assert is_routine_qual is False
    assert reason_qual == "Non-quantitative event requires semantic routing"


@pytest.mark.benchmark
def test_deterministic_engine_10k_events_under_50ms(
    engine: DeterministicRuleEngine, benchmark: Any
) -> None:
    """Performance: Evaluate 10,000 events in <50ms."""
    now = datetime(2026, 6, 1, 12, 0, 0)
    fin_evt = FinancialEventInput(
        event_id="bench_fin",
        event_time=now,
        knowledge_time=now,
        source="bls",
        modality=EventModality.QUANTITATIVE,
        metric_name="cpi",
        metric_value=2.01,
        consensus_value=2.00,
    )
    ops_evt = AIOpsEventInput(
        event_id="bench_ops",
        event_time=now,
        knowledge_time=now,
        source="prometheus",
        modality=EventModality.QUANTITATIVE,
        metric_name="cpu_utilization",
        metric_value=65.0,
        threshold_breached=False,
    )

    def _run_10k() -> int:
        routine_count = 0
        for _ in range(5000):
            r1, _ = engine.evaluate_financial(fin_evt)
            r2, _ = engine.evaluate_aiops(ops_evt)
            if r1 and r2:
                routine_count += 2
        return routine_count

    count = benchmark(_run_10k)
    assert count == 10000
    # Verify single 10k batch latency is strictly under 50ms
    t0 = time.perf_counter()
    _run_10k()
    single_batch_ms = (time.perf_counter() - t0) * 1000.0
    assert single_batch_ms < 50.0, f"10k evaluations took {single_batch_ms:.2f}ms (>= 50ms)"
