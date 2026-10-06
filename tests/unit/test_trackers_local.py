"""Unit tests for LocalJsonTracker and experiment provenance (T9)."""

from __future__ import annotations

import json
from pathlib import Path

from ewm_engine.cards.models import ScenarioCard
from ewm_engine.trackers.local_json import LocalJsonTracker


def test_local_json_tracker_lifecycle(tmp_path: Path) -> None:
    """Test LocalJsonTracker recording parameters, metrics, cards, and writing to disk."""
    tracker = LocalJsonTracker(base_dir=tmp_path)
    fp = "abcd1234ef567890abcd1234ef567890abcd1234ef567890abcd1234ef567890"

    tracker.log_fingerprint(fp)
    tracker.log_params({"learning_rate": 0.01, "horizon": 10})
    tracker.log_metrics({"rmse": 0.45, "coverage": 0.95}, step=1)
    tracker.log_metrics({"rmse": 0.38, "coverage": 0.97}, step=2)

    card = ScenarioCard(
        artifact_fingerprint=fp,
        scenario_id="warehouse_v1",
        description="Testing provenance logging",
        horizon=10,
        samples=5,
        seed=42,
    )
    tracker.log_card(card)
    tracker.log_artifact("artifacts/model.pt", artifact_type="weights")

    run_file = tmp_path / fp / "run.json"
    assert run_file.exists()

    with open(run_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["fingerprint"] == fp
    assert data["params"]["learning_rate"] == 0.01
    assert len(data["metrics_history"]) == 2
    assert data["latest_metrics"]["rmse"] == 0.38
    assert len(data["cards"]) == 1
    assert data["cards"][0]["scenario_id"] == "warehouse_v1"
    assert len(data["artifacts"]) == 1
