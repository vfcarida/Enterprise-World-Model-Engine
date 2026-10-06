"""Vega-Lite and Altair chart specification generator.

Emits deterministic, fingerprintable Vega-Lite JSON specifications for fan charts
and quantile ribbons.
"""

from __future__ import annotations

from typing import Any

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.reporting.model import ReportModel


def generate_vega_fan_chart_spec(
    report: ReportModel,
    signal_name: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Generate a canonical, fingerprintable Vega-Lite v5 specification dictionary for a fan chart."""
    if signal_name not in report.signals:
        raise KeyError(f"Signal '{signal_name}' not found in report signals.")

    q = report.signals[signal_name]
    data_values = []
    for i, step in enumerate(q.steps):
        data_values.append(
            {
                "step": step,
                "p10": q.p10[i],
                "p25": q.p25[i],
                "p50": q.p50[i],
                "p75": q.p75[i],
                "p90": q.p90[i],
                "mean": q.mean[i],
            }
        )

    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title or f"Fan Chart: {signal_name}",
        "width": 600,
        "height": 350,
        "data": {"values": data_values},
        "layer": [
            {
                "mark": {"type": "area", "opacity": 0.2, "color": "#1f77b4"},
                "encoding": {
                    "x": {"field": "step", "type": "quantitative", "title": "Step"},
                    "y": {"field": "p10", "type": "quantitative", "title": signal_name},
                    "y2": {"field": "p90"},
                },
            },
            {
                "mark": {"type": "area", "opacity": 0.35, "color": "#1f77b4"},
                "encoding": {
                    "x": {"field": "step", "type": "quantitative"},
                    "y": {"field": "p25", "type": "quantitative"},
                    "y2": {"field": "p75"},
                },
            },
            {
                "mark": {"type": "line", "color": "#1f77b4", "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "step", "type": "quantitative"},
                    "y": {"field": "p50", "type": "quantitative"},
                },
            },
            {
                "mark": {
                    "type": "line",
                    "color": "#ff7f0e",
                    "strokeDash": [4, 4],
                    "strokeWidth": 1.5,
                },
                "encoding": {
                    "x": {"field": "step", "type": "quantitative"},
                    "y": {"field": "mean", "type": "quantitative"},
                },
            },
        ],
    }
    return spec


def compute_vega_spec_fingerprint(spec: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint over Vega-Lite specification."""
    return canonical_sha256(spec)
