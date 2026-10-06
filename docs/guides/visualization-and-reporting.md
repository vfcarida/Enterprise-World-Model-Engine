# Visualization & Reporting: Renderer-Neutral Reports, Rich & Fan Charts

This guide covers extracting renderer-neutral simulation reports (`ReportModel`) and rendering them across CLI terminals (Rich), interactive web charts (Plotly), and deterministic specifications (Vega-Lite).

---

## 1. The Renderer-Neutral ReportModel

Instead of coupling simulation code directly to graphics engines, `ewm_engine.reporting` extracts a pure Pydantic data model containing quantiles, branch trees, and violation summaries:

```python
from ewm_engine.reporting import ReportModel, create_report_from_simulation

# sim_result is a completed SimulationResult
report: ReportModel = create_report_from_simulation(sim_result)

# Access empirical quantiles
inventory_q = report.signals["inventory"]
print(f"Median timeseries: {inventory_q.p50}")
print(f"90th percentile: {inventory_q.p90}")

# Save report atomically as JSON
report.save_json("artifacts/reports/run_report.json")
```

---

## 2. Rich Terminal CLI Renderer

For developer ergonomics and headless pipelines, render summaries to the terminal using `[cli]` (Rich):

```python
from ewm_engine.reporting import render_rich_report, format_rich_summary_str

# Print a formatted table to stdout
from rich.console import Console
console = Console()
console.print(render_rich_report(report))

# Or retrieve as formatted string
summary_text = format_rich_summary_str(report)
print(summary_text)
```

---

## 3. Interactive Plotly Fan Charts

Export interactive web fan charts showing median paths with 50% ($p_{25}-p_{75}$) and 80% ($p_{10}-p_{90}$) uncertainty bands using `[viz]` (Plotly):

```python
from ewm_engine.reporting import generate_fan_chart, export_fan_chart_html

# Generate a Plotly Figure
fig = generate_fan_chart(report, signal_name="inventory", title="Warehouse Stock Fan Chart")

# Export to standalone self-contained HTML
export_fan_chart_html(report, signal_name="inventory", output_path="reports/stock_fan.html")
```

---

## 4. Vega-Lite Fingerprinted Specifications

For reproducible, declarative document embedding, generate Vega-Lite v5 JSON specifications with canonical SHA-256 fingerprinting:

```python
from ewm_engine.reporting import generate_vega_fan_chart_spec, compute_vega_spec_fingerprint

vega_spec = generate_vega_fan_chart_spec(report, signal_name="inventory")
spec_hash = compute_vega_spec_fingerprint(vega_spec)
print(f"Deterministic spec hash: {spec_hash}")
```
