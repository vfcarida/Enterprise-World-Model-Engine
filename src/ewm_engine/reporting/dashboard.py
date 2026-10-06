"""Interactive dashboard front-ends for Streamlit and Dash (DASHBOARD adapter).

Quarantined adapters that only call the public EWM Engine API.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.reporting.model import ReportModel


def is_streamlit_available() -> bool:
    """Check if Streamlit is installed."""
    return importlib.util.find_spec("streamlit") is not None


def is_dash_available() -> bool:
    """Check if Dash is installed."""
    return importlib.util.find_spec("dash") is not None


def build_streamlit_dashboard_app(report: ReportModel) -> Any:
    """Construct Streamlit layout components for an EWM ReportModel."""
    if not is_streamlit_available():
        raise SimulationConfigurationError(
            "Streamlit is required for interactive dashboard. Install via `pip install ewm-engine[dashboard]`."
        )

    st = importlib.import_module("streamlit")
    st.title(f"EWM Simulation Report: {report.scenario_id}")
    st.write(f"**Fingerprint:** `{report.fingerprint}`")
    st.metric("Horizon Steps", report.horizon)
    st.metric("Samples", report.sample_count)
    return st
