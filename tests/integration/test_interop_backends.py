"""Integration tests for optional interop backends (SimPy, Nashpy, Rich, Plotly, FastAPI, OmegaConf)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.cosim.adapters import (
    SimPySubModel,
    is_simpy_available,
)
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.multiagent.adapters import PettingZooParallelAdapter
from ewm_engine.multiagent.game_theory import (
    NashpyGameSolver,
    is_nashpy_available,
)
from ewm_engine.multiagent.mediator import ConstraintMediator
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.reporting.model import create_report_from_simulation
from ewm_engine.reporting.rich_renderer import (
    format_rich_summary_str,
    is_rich_available,
    render_rich_report,
)
from ewm_engine.reporting.viz_plotly import (
    export_fan_chart_html,
    generate_fan_chart,
    is_plotly_available,
)
from ewm_engine.service.fastapi_app import (
    create_simulation_app,
    is_fastapi_available,
)
from ewm_engine.service.runner import LocalJobRunner
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
)
from ewm_engine.trackers.omegaconf_adapter import (
    OmegaConfConfigLoader,
    is_omegaconf_available,
)


def test_simpy_submodel_execution() -> None:
    """Test SimPySubModel stepping discrete-event processes."""
    if not is_simpy_available():
        pytest.skip("simpy not installed")

    import simpy

    def env_factory() -> simpy.Environment:
        return simpy.Environment()

    def step_runner(
        env: simpy.Environment, t_step: float, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        # Advance environment by t_step
        env.run(until=env.now + t_step)
        return {"processed_count": env.now * 2.0}

    sub = SimPySubModel(
        model_id="simpy_queue",
        env_factory=env_factory,
        step_runner=step_runner,
        input_ports=["arrival_rate"],
        output_ports=["processed_count"],
    )

    out1 = sub.step(0.0, 1.0, {"arrival_rate": 5.0})
    assert out1["processed_count"] == 2.0

    out2 = sub.step(1.0, 2.0, {"arrival_rate": 5.0})
    assert out2["processed_count"] == 6.0


def test_nashpy_game_solver() -> None:
    """Test NashpyGameSolver computing 2-player bi-matrix equilibria and replicator dynamics."""
    if not is_nashpy_available():
        pytest.skip("nashpy not installed")

    solver = NashpyGameSolver()

    # Matching Pennies game
    # Row wins on match, Col wins on mismatch
    payoff_row = [[1.0, -1.0], [-1.0, 1.0]]
    payoff_col = [[-1.0, 1.0], [1.0, -1.0]]

    eqs = solver.solve_nash_equilibria(payoff_row, payoff_col)
    assert len(eqs) >= 1
    # Mixed strategy equilibrium for matching pennies is (0.5, 0.5), (0.5, 0.5)
    s_row, s_col = eqs[0]
    np.testing.assert_allclose(s_row, [0.5, 0.5], atol=1e-3)
    np.testing.assert_allclose(s_col, [0.5, 0.5], atol=1e-3)

    # Replicator dynamics on Hawk-Dove / symmetric game
    sym_payoff = [[0.0, 3.0], [1.0, 2.0]]
    time_points = np.linspace(0.0, 1.0, 5)
    traj = solver.compute_replicator_dynamics(sym_payoff, y0=[0.6, 0.4], time_points=time_points)
    assert traj.shape == (5, 2)
    # Strategy distributions must sum to 1.0 at each step
    np.testing.assert_allclose(np.sum(traj, axis=1), np.ones(5), atol=1e-3)


def test_rich_report_renderer() -> None:
    """Test Rich terminal rendering of ReportModel."""
    if not is_rich_available():
        pytest.skip("rich not installed")

    sc = Scenario(scenario_id="rich_test", horizon=2, samples=1)
    init_st = WorldState(
        step=0,
        resources={"energy": Resource(id="energy", current=50.0)},
    )
    sim_res = SimulationResult(
        scenario=sc,
        provenance=Provenance(seed=42, horizon=2, samples=1),
        trajectories=[
            Trajectory(
                sample_id=0,
                seed=42,
                initial_state=init_st,
                steps=[
                    StepRecord(
                        step=1,
                        timestamp=1.0,
                        state_hash=init_st.fingerprint,
                        transition_result=TransitionResult(
                            next_state=WorldState(
                                step=1,
                                resources={"energy": Resource(id="energy", current=50.0)},
                            ),
                        ),
                    )
                ],
                final_state=WorldState(
                    step=1,
                    resources={"energy": Resource(id="energy", current=50.0)},
                ),
            )
        ],
    )
    report = create_report_from_simulation(sim_res)

    table = render_rich_report(report)
    assert table is not None

    summary_str = format_rich_summary_str(report)
    assert "rich_test" in summary_str
    assert "energy" in summary_str


def test_plotly_fan_chart_and_html_export(tmp_path: Path) -> None:
    """Test Plotly interactive fan chart generation and export."""
    if not is_plotly_available():
        pytest.skip("plotly not installed")

    sc = Scenario(scenario_id="plotly_test", horizon=2, samples=2)
    init_st = WorldState(
        step=0,
        resources={"stock": Resource(id="stock", current=10.0)},
    )
    sim_res = SimulationResult(
        scenario=sc,
        provenance=Provenance(seed=42, horizon=2, samples=2),
        trajectories=[
            Trajectory(
                sample_id=s,
                seed=42 + s,
                initial_state=init_st,
                steps=[
                    StepRecord(
                        step=1,
                        timestamp=1.0,
                        state_hash=init_st.fingerprint,
                        transition_result=TransitionResult(
                            next_state=WorldState(
                                step=1,
                                resources={
                                    "stock": Resource(id="stock", current=float(10 + s * 5))
                                },
                            ),
                        ),
                    )
                ],
                final_state=WorldState(
                    step=1,
                    resources={"stock": Resource(id="stock", current=float(10 + s * 5))},
                ),
            )
            for s in range(2)
        ],
    )
    report = create_report_from_simulation(sim_res)

    fig = generate_fan_chart(report, "stock")
    assert fig is not None

    html_out = tmp_path / "chart.html"
    export_fan_chart_html(report, "stock", html_out)
    assert html_out.exists()
    assert html_out.stat().st_size > 0


def test_fastapi_simulation_service() -> None:
    """Test FastAPI simulation REST endpoints."""
    if not is_fastapi_available():
        pytest.skip("fastapi not installed")

    from starlette.testclient import TestClient

    world = World(
        initial_state=WorldState(
            step=0,
            entities={"e1": Entity(id="e1", type="node")},
            resources={"items": Resource(id="items", current=100.0)},
        )
    )
    runner = LocalJobRunner(max_workers=1)
    app = create_simulation_app(runner=runner, world=world)
    client = TestClient(app)

    # Health check
    h_res = client.get("/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "healthy"

    # Simulate submission
    sc_payload = {
        "scenario": {
            "scenario_id": "api_test_scenario",
            "horizon": 2,
            "samples": 1,
            "seed": 42,
        }
    }
    sim_res = client.post("/simulate", json=sc_payload)
    assert sim_res.status_code == 200
    data = sim_res.json()
    assert "job_id" in data
    assert "fingerprint" in data

    # Check job endpoint
    job_res = client.get(f"/jobs/{data['job_id']}")
    assert job_res.status_code == 200
    assert job_res.json()["job_id"] == data["job_id"]

    runner.shutdown(wait=True)


def test_omegaconf_config_loader() -> None:
    """Test OmegaConf loader with ${...} variable interpolation."""
    if not is_omegaconf_available():
        pytest.skip("omegaconf not installed")

    loader = OmegaConfConfigLoader()
    raw_config = {
        "base_url": "https://api.simulation.io",
        "endpoints": {
            "health": "${base_url}/health",
            "simulate": "${base_url}/simulate",
        },
        "params": {
            "batch_size": 32,
            "horizon": 10,
        },
    }

    resolved = loader.load_config_dict(raw_config)
    assert resolved["endpoints"]["health"] == "https://api.simulation.io/health"
    assert resolved["endpoints"]["simulate"] == "https://api.simulation.io/simulate"

    cfg_hash = loader.compute_config_hash(resolved)
    assert len(cfg_hash) == 64


def test_pettingzoo_parallel_adapter() -> None:
    """Test PettingZooParallelAdapter environment interaction."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities={
                "agent_0": Entity(id="agent_0", type="agent"),
                "agent_1": Entity(id="agent_1", type="agent"),
            },
            resources={"token": Resource(id="token", current=1.0)},
        )
    )

    env = PettingZooParallelAdapter(
        world=world,
        agent_ids=["agent_0", "agent_1"],
        mediator=ConstraintMediator(resource_locks=True),
    )

    obs, _infos = env.reset()
    assert "agent_0" in obs
    assert "agent_1" in obs

    actions = {
        "agent_0": Action(
            id="claim_token_0",
            type="custom",
            actor_id="agent_0",
            parameters={"cost": {"token": 1.0}},
        ),
        "agent_1": Action(
            id="claim_token_1",
            type="custom",
            actor_id="agent_1",
            parameters={"cost": {"token": 1.0}},
        ),
    }
    obs, _rewards, _terms, _truncs, step_infos = env.step(actions)
    assert "agent_0" in obs
    assert "agent_1" in obs
    # Since agent_0 and agent_1 both attempt to claim the single token,
    # one is accepted and the other rejected deterministically
    assert step_infos["agent_0"]["accepted"] or step_infos["agent_1"]["accepted"]
