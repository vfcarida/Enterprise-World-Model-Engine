"""Unit tests for ReportModel, quantile extraction, and Vega-Lite spec generation (T7)."""

from __future__ import annotations

import json
from pathlib import Path

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.reporting.model import (
    ReportModel,
    create_report_from_simulation,
)
from ewm_engine.reporting.viz_altair import (
    compute_vega_spec_fingerprint,
    generate_vega_fan_chart_spec,
)
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
)


def _create_mock_simulation_result() -> SimulationResult:
    scenario = Scenario(scenario_id="viz_test_sc", horizon=3, samples=2, seed=42)
    init_state = WorldState(
        step=0,
        resources={"inventory": Resource(id="inventory", current=0.0)},
        memory={"demand": 0.0},
    )
    trajs = []
    for s_id in range(2):
        steps = []
        for step_idx in range(1, 4):
            val = float(10 * step_idx + s_id * 5)
            st = WorldState(
                step=step_idx,
                resources={"inventory": Resource(id="inventory", current=val)},
                memory={"demand": val * 0.5},
            )
            step_rec = StepRecord(
                step=step_idx,
                timestamp=float(step_idx),
                state_hash=st.fingerprint,
                transition_result=TransitionResult(next_state=st),
            )
            steps.append(step_rec)
        traj = Trajectory(
            sample_id=s_id,
            seed=42 + s_id,
            initial_state=init_state,
            steps=steps,
            final_state=steps[-1].transition_result.next_state,
        )
        trajs.append(traj)

    return SimulationResult(
        scenario=scenario,
        provenance=Provenance(seed=42, horizon=3, samples=2),
        trajectories=trajs,
    )


def test_create_report_from_simulation() -> None:
    """Test building ReportModel from SimulationResult, extracting quantiles and tree nodes."""
    sim_res = _create_mock_simulation_result()
    report: ReportModel = create_report_from_simulation(sim_res)

    assert report.scenario_id == "viz_test_sc"
    assert report.sample_count == 2
    assert report.horizon == 3
    assert len(report.fingerprint) == 64

    # Signals
    assert "inventory" in report.signals
    inv_q = report.signals["inventory"]
    assert inv_q.steps == [1, 2, 3]
    assert len(inv_q.p50) == 3
    assert len(inv_q.mean) == 3
    # Step 1: trajs have 10.0 and 15.0 -> median is 12.5
    assert inv_q.p50[0] == 12.5
    assert inv_q.mean[0] == 12.5

    # Branch tree nodes
    assert len(report.branch_tree) > 0


def test_report_json_serialization_and_save(tmp_path: Path) -> None:
    """Test ReportModel JSON roundtrip and atomic disk persistence."""
    sim_res = _create_mock_simulation_result()
    report: ReportModel = create_report_from_simulation(sim_res)

    json_str = report.to_json()
    assert '"scenario_id": "viz_test_sc"' in json_str

    save_path = tmp_path / "reports" / "report.json"
    report.save_json(save_path)
    assert save_path.exists()

    with open(save_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["simulation_id"] == "viz_test_sc"
    assert "inventory" in data["signals"]


def test_vega_fan_chart_specification() -> None:
    """Test Vega-Lite fan chart spec generation and deterministic fingerprinting."""
    sim_res = _create_mock_simulation_result()
    report: ReportModel = create_report_from_simulation(sim_res)

    spec = generate_vega_fan_chart_spec(report, "inventory", title="Warehouse Stock")
    assert spec["$schema"] == "https://vega.github.io/schema/vega-lite/v5.json"
    assert len(spec["layer"]) == 4

    fp1 = compute_vega_spec_fingerprint(spec)
    fp2 = compute_vega_spec_fingerprint(spec)
    assert fp1 == fp2
    assert len(fp1) == 64
