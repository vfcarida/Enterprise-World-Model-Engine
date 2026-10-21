"""Plotly fan charts and scenario comparison visualization (VIZ extra).

Emits standalone interactive HTML without requiring an active server.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.reporting.model import ReportModel


def is_plotly_available() -> bool:
    """Check if Plotly is installed."""
    return importlib.util.find_spec("plotly") is not None


class StandaloneFanChartFigure:
    """Zero-dependency SVG/HTML fan chart figure fallback when Plotly is not installed."""

    def __init__(
        self,
        title: str,
        steps: list[int],
        series: dict[str, list[float]],
        signal_name: str,
    ) -> None:
        self.title = title
        self.steps = steps
        self.series = series
        self.signal_name = signal_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "signal_name": self.signal_name,
            "steps": self.steps,
            "series": self.series,
        }

    def write_html(self, output_path: str | Path, include_plotlyjs: str = "cdn") -> None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        html_doc = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
            f"<title>{self.title}</title></head><body>"
            f"<h1>{self.title}</h1>"
            f"<pre id='fan-chart-data'>{self.to_dict()}</pre>"
            "</body></html>\n"
        )
        p.write_text(html_doc, encoding="utf-8")


def generate_fan_chart(
    report: ReportModel,
    signal_name: str,
    title: str | None = None,
    require_extra: bool = False,
) -> Any:
    """Generate an interactive Plotly fan chart (p10-p90, p25-p75, median) for a signal."""
    if require_extra and not is_plotly_available():
        raise SimulationConfigurationError(
            "Plotly is required for generate_fan_chart. Install via `pip install ewm-engine[viz]`."
        )

    if signal_name not in report.signals:
        raise KeyError(
            f"Signal '{signal_name}' not found in report signals: {list(report.signals.keys())}"
        )

    q = report.signals[signal_name]
    steps = list(q.steps)
    chart_title = title or f"Monte Carlo Fan Chart: {signal_name} ({report.scenario_id})"

    if not is_plotly_available():
        return StandaloneFanChartFigure(
            title=chart_title,
            steps=steps,
            series={
                "p10": list(q.p10),
                "p25": list(q.p25),
                "p50": list(q.p50),
                "p75": list(q.p75),
                "p90": list(q.p90),
                "mean": list(q.mean),
            },
            signal_name=signal_name,
        )

    go = importlib.import_module("plotly.graph_objects")
    fig = go.Figure()

    # 1. Outer band: p10 - p90
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.p90,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.p10,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.15)",
            name="10-90% Quantile",
        )
    )

    # 2. Inner band: p25 - p75
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.p75,
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.p25,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.35)",
            name="25-75% Quantile",
        )
    )

    # 3. Median (p50)
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.p50,
            mode="lines+markers",
            line={"color": "rgb(31, 119, 180)", "width": 2.5},
            name="Median (p50)",
        )
    )

    # 4. Mean
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q.mean,
            mode="lines",
            line={"color": "rgba(255, 127, 14, 0.9)", "dash": "dot", "width": 2},
            name="Mean",
        )
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Simulation Step",
        yaxis_title=f"{signal_name} Level",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def export_fan_chart_html(
    report: ReportModel,
    signal_name: str,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Export interactive fan chart to standalone HTML file."""
    fig = generate_fan_chart(report, signal_name, title=title)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(p), include_plotlyjs="cdn")


def generate_scenario_comparison(
    reports: dict[str, ReportModel],
    signal_name: str,
    title: str | None = None,
    require_extra: bool = False,
) -> Any:
    """Generate comparative Plotly chart across multiple scenario reports."""
    if require_extra and not is_plotly_available():
        raise SimulationConfigurationError(
            "Plotly is required for generate_scenario_comparison. Install via `pip install ewm-engine[viz]`."
        )

    chart_title = title or f"Scenario Comparison: {signal_name}"
    if not is_plotly_available():
        series: dict[str, list[float]] = {}
        steps: list[int] = []
        for sc_name, rep in reports.items():
            if signal_name in rep.signals:
                q = rep.signals[signal_name]
                steps = list(q.steps)
                series[sc_name] = list(q.p50)
        return StandaloneFanChartFigure(
            title=chart_title,
            steps=steps,
            series=series,
            signal_name=signal_name,
        )

    go = importlib.import_module("plotly.graph_objects")
    fig = go.Figure()

    for sc_name, rep in reports.items():
        if signal_name in rep.signals:
            q = rep.signals[signal_name]
            fig.add_trace(
                go.Scatter(
                    x=q.steps,
                    y=q.p50,
                    mode="lines+markers",
                    name=f"{sc_name} (median)",
                )
            )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Simulation Step",
        yaxis_title=f"{signal_name} Median",
        template="plotly_white",
    )
    return fig
