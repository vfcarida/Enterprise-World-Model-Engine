"""Unit tests for native reproducibility cards (ModelCard, ScenarioCard, DatasetCard) and Croissant."""

from __future__ import annotations

import json

import yaml

from ewm_engine.cards.croissant import to_croissant_json
from ewm_engine.cards.models import DatasetCard, ModelCard, ScenarioCard


def test_model_card_rendering_and_fields() -> None:
    card = ModelCard(
        model_id="linear-transfer-v1",
        model_name="Linear Resource Transfer Dynamics",
        version="1.2.0",
        artifact_fingerprint="a" * 64,
        maturity="production",
        description="Deterministic linear stock and flow resource transfer model.",
        intended_use=[
            "Simulating supply-chain buffer transfers",
            "Evaluating capacity reallocations",
        ],
        assumptions=[
            "Conservation of total mass in closed network",
            "Zero latency in transit buffers",
        ],
        out_of_scope=[
            "High-frequency algorithmic trading regimes",
            "Stochastic breakdown events",
        ],
        limitations=[
            "First-order Euler discretization may accumulate drift under large dt",
        ],
        constraints_exercised=["capacity_limit", "non_negative_balance"],
        metrics={"mae": 0.012, "r2": 0.994},
        calibration_score=0.985,
    )

    # 1. JSON
    json_str = card.to_json()
    data = json.loads(json_str)
    assert data["card_type"] == "model"
    assert data["model_id"] == "linear-transfer-v1"
    assert data["calibration_score"] == 0.985
    assert data["schema_version"] == "1.0.0"

    # 2. YAML
    yaml_str = card.to_yaml()
    yaml_data = yaml.safe_load(yaml_str)
    assert yaml_data["artifact_fingerprint"] == "a" * 64

    # 3. Markdown
    md = card.to_markdown()
    assert "# Model Card: Linear Resource Transfer Dynamics" in md
    assert "**Maturity Level:** `PRODUCTION`" in md
    assert f"`{'a' * 64}`" in md
    assert "### Intended Use" in md
    assert "### Out-of-Scope Regimes" in md
    assert "> [!WARNING]" in md
    assert "| `r2` | **0.994** |" in md
    assert "| `population_calibration_score` | **0.9850** |" in md


def test_scenario_card_rendering_and_fields() -> None:
    card = ScenarioCard(
        scenario_id="supply_shock_2026",
        description="Simulates 50% tariff and demand contraction shock.",
        horizon=12,
        samples=100,
        seed=42,
        artifact_fingerprint="b" * 64,
        maturity="production",
        interventions=["tariff_hike_50pct"],
        assumptions=["Fixed competitor pricing for 6 months"],
        environmental_context={"interest_rate": 0.05, "inflation": 0.03},
        metrics={"median_margin": 0.14, "p05_cvar": -0.08},
    )

    md = card.to_markdown()
    assert "# Scenario Card: `supply_shock_2026`" in md
    assert "**Horizon:** `12 steps`" in md
    assert "**Rollouts:** `100`" in md
    assert "tariff_hike_50pct" in md
    assert "| `interest_rate` | `0.05` |" in md


def test_dataset_card_and_croissant_emission() -> None:
    card = DatasetCard(
        dataset_id="ewm-traces-v1",
        name="Enterprise Trajectory Benchmark Traces",
        version="1.0.0",
        artifact_fingerprint="c" * 64,
        description="10,000 Monte Carlo rollouts across 5 benchmark families.",
        num_trajectories=10000,
        num_steps=120000,
        features=["inventory_level", "cash_balance", "backlog_count"],
        intended_use=["Offline RL policy training", "Dynamics model validation"],
        license="Apache-2.0",
        provenance_fingerprints=["d" * 64],
    )

    # Markdown
    md = card.to_markdown()
    assert "# Dataset Card: Enterprise Trajectory Benchmark Traces" in md
    assert "**Trajectories:** `10000`" in md
    assert "Apache-2.0" in md

    # Croissant 1.1 JSON-LD
    croissant_dict = card.to_croissant()
    assert croissant_dict["@type"] == "sc:Dataset"
    assert croissant_dict["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert croissant_dict["name"] == "Enterprise Trajectory Benchmark Traces"
    assert croissant_dict["license"] == "Apache-2.0"

    # Verify distributions & recordSets
    dist = croissant_dict["distribution"]
    assert len(dist) == 1
    assert dist[0]["sha256"] == "c" * 64

    record_set = croissant_dict["recordSet"]
    assert len(record_set) == 1
    fields = record_set[0]["field"]
    field_names = [f["name"] for f in fields]
    assert "sample_id" in field_names
    assert "step" in field_names
    assert "state_hash" in field_names
    assert "inventory_level" in field_names
    assert "cash_balance" in field_names

    # JSON serialization
    json_ld_str = to_croissant_json(card)
    assert "http://mlcommons.org/croissant/1.0" in json_ld_str
