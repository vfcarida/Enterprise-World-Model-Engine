"""Unit tests for SystemicTrace.to_html interactive visualizer."""

from __future__ import annotations

from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace


def test_systemic_trace_to_html_renders_complete_document() -> None:
    """Ensure to_html returns an offline-ready HTML5 document with all nodes and edges."""
    trace = SystemicTrace()
    trace.add_node(
        node_id="act_transfer_1",
        step=1,
        category="action",
        label="Transfer 30 North -> South",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
        details={"quantity": 30.0, "source": "stock_north"},
    )
    trace.add_node(
        node_id="trans_step_1",
        step=1,
        category="transition",
        label="Deterministic Transfer Dynamics",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    trace.add_edge(
        source="act_transfer_1",
        target="trans_step_1",
        step=1,
        relation="drives",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )

    html_content = trace.to_html(title="Warehouse Logistics Trace")

    # Document structure assertions
    assert "<!DOCTYPE html>" in html_content
    assert "<title>Warehouse Logistics Trace - EWM Engine</title>" in html_content
    assert '<svg id="trace-svg"' in html_content
    assert "act_transfer_1" in html_content
    assert "trans_step_1" in html_content
    assert "drives" in html_content
    assert "node-" in html_content
    assert "Trace Node Inspector" in html_content
    assert "application/json" in html_content
    assert "interventional" in html_content
    assert "structural" in html_content


def test_systemic_trace_to_html_empty_trace() -> None:
    """Ensure to_html handles an empty systemic trace gracefully."""
    trace = SystemicTrace()
    html_content = trace.to_html(title="Empty Trace")

    assert "<!DOCTYPE html>" in html_content
    assert "<title>Empty Trace - EWM Engine</title>" in html_content
    assert "Nodes:</strong> 0" in html_content
    assert "Edges:</strong> 0" in html_content
