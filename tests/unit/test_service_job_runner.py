"""Unit tests for LocalJobRunner and simulation job management (T8)."""

from __future__ import annotations

import time

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.durability.backends.in_memory import InMemoryResultStore
from ewm_engine.service.runner import JobRecord, JobStatus, LocalJobRunner
from ewm_engine.simulation.scenario import Scenario


def _create_mock_world_and_scenario() -> tuple[World, Scenario]:
    world = World(
        initial_state=WorldState(
            step=0,
            entities={"e1": Entity(id="e1", type="node")},
            resources={"res": Resource(id="res", current=100.0)},
        )
    )
    scenario = Scenario(scenario_id="runner_sc", horizon=2, samples=1, seed=42)
    return world, scenario


def test_local_job_runner_lifecycle() -> None:
    """Test LocalJobRunner submitting, executing, and retrieving results."""
    runner = LocalJobRunner(max_workers=2)
    world, scenario = _create_mock_world_and_scenario()

    job_id = runner.submit_job(world, scenario)
    assert job_id.startswith("job_")

    # Poll status briefly until completed
    for _ in range(50):
        rec: JobRecord = runner.get_job(job_id)
        if rec.status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)

    rec = runner.get_job(job_id)
    assert rec.status == JobStatus.COMPLETED
    assert rec.finished_at is not None
    assert rec.result_summary is not None
    assert rec.result_summary["trajectories_count"] == 1

    # Retrieve result by fingerprint
    res = runner.get_result(rec.fingerprint)
    assert res is not None
    assert len(res.trajectories) == 1

    runner.shutdown(wait=True)


def test_local_job_runner_result_store_cache_hit() -> None:
    """Test LocalJobRunner returns instant completed status when ResultStore has cached result."""
    store = InMemoryResultStore()
    runner = LocalJobRunner(max_workers=2, result_store=store)
    world, scenario = _create_mock_world_and_scenario()

    # First run: executes and caches
    job_id_1 = runner.submit_job(world, scenario)
    for _ in range(50):
        if runner.get_job(job_id_1).status == JobStatus.COMPLETED:
            break
        time.sleep(0.05)

    rec1 = runner.get_job(job_id_1)
    assert rec1.status == JobStatus.COMPLETED

    # Second run with identical world and scenario: instant cache hit!
    job_id_2 = runner.submit_job(world, scenario)
    rec2 = runner.get_job(job_id_2)
    assert rec2.status == JobStatus.COMPLETED
    assert rec2.result_summary is not None
    assert rec2.result_summary["cached"] is True
    assert rec2.fingerprint == rec1.fingerprint

    runner.shutdown(wait=True)
