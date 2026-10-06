"""Visualization and reporting module (T7).

Provides renderer-neutral ReportModel, Rich terminal outputs, Plotly fan charts,
Vega-Lite specs, and dashboard adapters.
"""

from __future__ import annotations

from ewm_engine.reporting.dashboard import (
    build_streamlit_dashboard_app,
    is_dash_available,
    is_streamlit_available,
)
from ewm_engine.reporting.model import (
    BranchTreeNode,
    ReportModel,
    SignalQuantiles,
    create_report_from_simulation,
)
from ewm_engine.reporting.rich_renderer import (
    format_rich_summary_str,
    is_rich_available,
    render_rich_report,
)
from ewm_engine.reporting.viz_altair import (
    compute_vega_spec_fingerprint,
    generate_vega_fan_chart_spec,
)
from ewm_engine.reporting.viz_export import (
    export_static_image,
    is_kaleido_available,
)
from ewm_engine.reporting.viz_plotly import (
    export_fan_chart_html,
    generate_fan_chart,
    generate_scenario_comparison,
    is_plotly_available,
)

__all__ = [
    "BranchTreeNode",
    "ReportModel",
    "SignalQuantiles",
    "build_streamlit_dashboard_app",
    "compute_vega_spec_fingerprint",
    "create_report_from_simulation",
    "export_fan_chart_html",
    "export_static_image",
    "format_rich_summary_str",
    "generate_fan_chart",
    "generate_scenario_comparison",
    "generate_vega_fan_chart_spec",
    "is_dash_available",
    "is_kaleido_available",
    "is_plotly_available",
    "is_rich_available",
    "is_streamlit_available",
    "render_rich_report",
]
