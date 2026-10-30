"""Deterministic Rule Engine for fast-path event filtering.

Intercepts high-volume, low-complexity boilerplate data (routine macro indicators
within consensus tolerance, standard tick data, routine health checks, and
normal metric updates) to bypass neural classification in <5ms.
"""

from __future__ import annotations

from ewm_engine.routing.schemas import AIOpsEventInput, EventModality, FinancialEventInput


class DeterministicRuleEngine:
    """Fast-path filter for routine, low-complexity events."""

    def __init__(self, consensus_tolerance: float = 0.02) -> None:
        # Financial rules
        self.routine_macro_indicators: set[str] = {
            "cpi",
            "ppi",
            "unemployment_rate",
            "non_farm_payrolls",
            "retail_sales",
        }
        self.consensus_tolerance: float = consensus_tolerance  # 2% deviation from consensus

        # AIOps rules
        self.routine_health_checks: set[str] = {
            "liveness_probe",
            "readiness_probe",
            "heartbeat",
        }
        self.standard_metrics: set[str] = {
            "cpu_utilization",
            "memory_bytes",
            "disk_io_bytes",
            "network_rx_bytes",
        }

    def evaluate_financial(self, event: FinancialEventInput) -> tuple[bool, str | None]:
        """Evaluate whether a financial event is routine and can take the deterministic fast-path.

        Returns:
            Tuple of (is_routine, reasoning).
        """
        if event.modality != EventModality.QUANTITATIVE:
            return False, "Non-quantitative event requires semantic routing"

        # Rule 1: Routine macro indicator matching consensus
        if event.metric_name and event.metric_name.lower() in self.routine_macro_indicators:
            if event.consensus_value is not None and event.metric_value is not None:
                denom = abs(event.consensus_value)
                deviation = (
                    abs(event.metric_value)
                    if denom == 0.0
                    else abs(event.metric_value - event.consensus_value) / denom
                )
                if deviation < self.consensus_tolerance:
                    return (
                        True,
                        f"Routine {event.metric_name} release (deviation: {deviation:.2%} < {self.consensus_tolerance:.0%})",
                    )
            return False, "Event requires semantic classification"

        # Rule 2: Standard tick data (no text content, < 3 affected tickers)
        if not event.text_content and len(event.affected_tickers or []) < 3:
            return True, "Standard tick data (no qualitative component)"

        return False, "Event requires semantic classification"

    def evaluate_aiops(self, event: AIOpsEventInput) -> tuple[bool, str | None]:
        """Evaluate whether an AIOps event is routine and can take the deterministic fast-path.

        Returns:
            Tuple of (is_routine, reasoning).
        """
        if event.modality != EventModality.QUANTITATIVE:
            return False, "Non-quantitative event requires semantic routing"

        # Rule 1: Routine health checks
        if event.source in self.routine_health_checks:
            return True, f"Routine health check: {event.source}"

        # Rule 2: Standard metric updates (no threshold breach, no text)
        if (
            event.metric_name in self.standard_metrics
            and not event.threshold_breached
            and not event.text_content
        ):
            return True, f"Standard {event.metric_name} metric update (no alert)"

        return False, "Event requires semantic classification"
