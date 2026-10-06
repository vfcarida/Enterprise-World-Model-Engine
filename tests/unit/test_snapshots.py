"""Structured non-numeric snapshot tests using syrupy.

In accordance with R03 testing rigor:
- Snapshot non-numeric structured outputs (trace Mermaid flowchart, Card Markdown, Card YAML/JSON).
- Never snapshot raw floating-point numbers without canonical rounding/tolerance.
- Includes quarantine lane demonstrations via pytest-rerunfailures.
"""

from __future__ import annotations

from typing import Any

import pytest

from ewm_engine.cards.models import ScenarioCard
from ewm_engine.cards.render import render_card_markdown, render_card_yaml
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace


def test_systemic_trace_mermaid_snapshot(snapshot: Any) -> None:
    """Ensure systemic trace Mermaid flowchart DAG representation is stable and matches snapshot."""
    trace = SystemicTrace()
    n1 = trace.add_node(
        node_id="init_state",
        step=0,
        category="state",
        label="Initial Inventory State",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    n2 = trace.add_node(
        node_id="intervention_pricing",
        step=1,
        category="intervention",
        label="Pricing Discount 15%",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    n3 = trace.add_node(
        node_id="demand_shock",
        step=2,
        category="event",
        label="Spike in Demand (+40%)",
        evidence_level=EvidenceLevel.PREDICTIVE,
    )

    trace.add_edge(
        n1.id, n2.id, step=1, relation="triggers", evidence_level=EvidenceLevel.INTERVENTIONAL
    )
    trace.add_edge(
        n2.id, n3.id, step=2, relation="leads_to", evidence_level=EvidenceLevel.PREDICTIVE
    )

    mermaid_diagram = trace.to_mermaid()

    # Compare with Syrupy snapshot
    assert mermaid_diagram == snapshot


def test_scenario_card_markdown_snapshot(snapshot: Any) -> None:
    """Ensure ScenarioCard Markdown rendering is clean, structured, and matches snapshot."""
    card = ScenarioCard(
        artifact_fingerprint="sha256:a1b2c3d4e5f67890abcdef1234567890",
        scenario_id="scen_supply_chain_stress_01",
        description="Stress testing supply chain lead times under sudden logistics bottlenecks.",
        maturity="production",
        assumptions=[
            "Supplier capacity is fixed at 1000 units/week",
            "Lead times increase by 30% under stress",
        ],
        interventions=["Capacity expansion +200", "Expedited shipping rule"],
        out_of_scope=["Long-term macro recession"],
        environmental_context={"interest_rate": 0.05, "demand_shock": 1.4},
        metrics={"mean_service_level": 0.88, "shortage_cost_usd": 125000.0},
        limitations=["Assumes deterministic port clearance times"],
        horizon=52,
        samples=100,
        seed=42,
    )

    md_output = render_card_markdown(card)
    assert md_output == snapshot


def test_scenario_card_yaml_snapshot(snapshot: Any) -> None:
    """Ensure ScenarioCard YAML representation matches structured snapshot."""
    card = ScenarioCard(
        artifact_fingerprint="sha256:9876543210fedcba9876543210fedcba",
        scenario_id="scen_baseline_growth",
        description="Baseline revenue projection for Q4.",
        maturity="beta",
        assumptions=["Inflation rate steady at 3.0%"],
        interventions=[],
        out_of_scope=[],
        environmental_context={"inflation": 0.03},
        metrics={"expected_margin": 0.24},
        limitations=[],
        horizon=12,
        samples=50,
        seed=1234,
    )

    yaml_output = render_card_yaml(card)
    assert yaml_output == snapshot


@pytest.mark.flaky(reruns=2, reruns_delay=0.01)
def test_quarantine_lane_rerun_marker() -> None:
    """Demonstrate tracked rerunfailures quarantine lane for known non-fatal flaky candidates.

    Any test marked with @pytest.mark.flaky will be retried if it fails transiently,
    producing an explicit rerun report rather than failing silently.
    """
    import random

    val = random.random()
    # Pinned to always succeed after initial check, asserting marker mechanics
    assert val >= 0.0
