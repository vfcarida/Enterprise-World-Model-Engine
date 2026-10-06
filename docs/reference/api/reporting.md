# Reporting & Visualization API Reference

This module defines renderer-neutral simulation reporting models, Plotly interactive fan charts, and Rich terminal tables (Track T7).

---

## Canonical Report Models

::: ewm_engine.reporting.model
    options:
      show_root_heading: true
      show_source: false
      members:
        - ReportModel
        - SignalQuantiles
        - BranchTreeNode
        - create_report_from_simulation

---

## Rich Terminal Renderers

::: ewm_engine.reporting.rich_renderer
    options:
      show_root_heading: true
      show_source: false
      members:
        - render_rich_report
        - format_rich_summary_str

---

## Plotly & Vega Interactive Visualizations

::: ewm_engine.reporting.viz_plotly
    options:
      show_root_heading: true
      show_source: false
      members:
        - generate_fan_chart
        - export_fan_chart_html

::: ewm_engine.reporting.viz_altair
    options:
      show_root_heading: true
      show_source: false
      members:
        - generate_vega_fan_chart_spec
        - compute_vega_spec_fingerprint
