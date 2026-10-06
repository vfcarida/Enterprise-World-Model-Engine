"""Rich terminal table and tree renderer for ReportModel (CLI extra).

Quarantined behind the [cli] extra (Rich).
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.reporting.model import ReportModel


def is_rich_available() -> bool:
    """Check if Rich terminal rendering library is installed."""
    return importlib.util.find_spec("rich") is not None


def render_rich_report(report: ReportModel) -> Any:
    """Render ReportModel as a collection of Rich Tables and Trees."""
    if not is_rich_available():
        raise SimulationConfigurationError(
            "Rich is required for render_rich_report. Install via `pip install ewm-engine[cli]`."
        )

    rich_table = importlib.import_module("rich.table")

    table = rich_table.Table(title=f"Simulation Report: {report.scenario_id}")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Simulation ID", report.simulation_id)
    table.add_row("Fingerprint", report.fingerprint)
    table.add_row("Horizon", str(report.horizon))
    table.add_row("Samples", str(report.sample_count))
    table.add_row("Signals Tracked", ", ".join(report.signals.keys()) or "None")
    table.add_row("Violations Recorded", str(len(report.violations_summary)))
    table.add_row("Verification Verdicts", str(len(report.verification_verdicts)))

    return table


def format_rich_summary_str(report: ReportModel) -> str:
    """Render ReportModel summary to string using Rich Console."""
    if not is_rich_available():
        return f"Report {report.scenario_id} (Fingerprint: {report.fingerprint[:16]}...)"

    rich_console = importlib.import_module("rich.console")
    console = rich_console.Console(record=True, width=80)
    table = render_rich_report(report)
    console.print(table)
    return str(console.export_text())
