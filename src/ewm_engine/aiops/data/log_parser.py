"""Unstructured log parser with semantic anomaly extraction via FrontierLLMArbitrator.

Pre-filters high-volume log streams using deterministic regular expressions and routes
critical operational failures (`FATAL`, `OutOfMemoryError`, `TimeoutException`,
`CrashLoopBackOff`) to the Mechanistic Router's LLM arbitrator to extract structured
CausalInterventionVectors.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ewm_engine.routing.llm_arbitrator import FrontierLLMArbitrator

logger = logging.getLogger(__name__)


class LogParser:
    """Parses unstructured microservice logs and extracts semantic anomaly signals."""

    def __init__(self, llm_arbitrator: FrontierLLMArbitrator | None = None) -> None:
        self.llm = llm_arbitrator if llm_arbitrator is not None else FrontierLLMArbitrator()
        self.error_patterns: list[str] = [
            r"ERROR",
            r"FATAL",
            r"Exception",
            r"TimeoutException",
            r"OutOfMemoryError",
            r"CrashLoopBackOff",
        ]
        self.high_severity_tokens: tuple[str, ...] = (
            "FATAL",
            "OutOfMemoryError",
            "TimeoutException",
            "CrashLoopBackOff",
        )

    @staticmethod
    def _extract_service_hint(line: str) -> str:
        """Extract service identifier from common structured/semi-structured log prefixes."""
        match = re.search(r"service[=:\s]+([a-zA-Z0-9_-]+)", line, re.IGNORECASE)
        if match:
            return match.group(1)
        bracket_match = re.search(r"\[([a-zA-Z0-9_-]+-service)\]", line, re.IGNORECASE)
        if bracket_match:
            return bracket_match.group(1)
        token_match = re.search(r"\b([a-zA-Z0-9_]+-service)\b", line)
        if token_match:
            return token_match.group(1)
        return "unknown"

    def parse_logs(self, log_lines: Sequence[str]) -> list[dict[str, Any]]:
        """Parse log lines and extract structured semantic anomaly signals.

        Args:
            log_lines: Sequence of raw log strings.

        Returns:
            List of dicts with keys: `timestamp`, `service`, `severity`, `message`,
            `anomaly_signal`, and `causal_vector`.
        """
        anomalies: list[dict[str, Any]] = []

        for line in log_lines:
            if not any(re.search(pattern, line, re.IGNORECASE) for pattern in self.error_patterns):
                continue

            now = datetime.now(UTC).replace(tzinfo=None)
            is_high_severity = any(tok in line for tok in self.high_severity_tokens)

            if is_high_severity:
                causal_vector = self.llm.extract_causal_vector(line, domain="aiops")
                inferred_service = (
                    causal_vector.primary_target_entities[0]
                    if causal_vector.primary_target_entities
                    and causal_vector.primary_target_entities[0] != "unknown"
                    else self._extract_service_hint(line)
                )
                anomalies.append(
                    {
                        "timestamp": now,
                        "service": inferred_service,
                        "severity": "high",
                        "message": line,
                        "anomaly_signal": float(causal_vector.directional_sentiment),
                        "causal_vector": causal_vector,
                    }
                )
            else:
                anomalies.append(
                    {
                        "timestamp": now,
                        "service": self._extract_service_hint(line),
                        "severity": "medium",
                        "message": line,
                        "anomaly_signal": -0.45,
                        "causal_vector": None,
                    }
                )

        return anomalies
