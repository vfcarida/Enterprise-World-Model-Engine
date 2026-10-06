"""Unit tests for ResultStore backends and memoization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.durability.backends.filesystem_result_store import FilesystemResultStore
from ewm_engine.durability.backends.in_memory import InMemoryResultStore
from ewm_engine.durability.backends.stubs import RedisResultStore, S3ResultStore
from ewm_engine.durability.memoize import (
    MemoizedSimulationRunner,
    compute_simulation_fingerprint,
    run_memoized,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _make_world() -> World:
    state = WorldState(
        step=0,
        entities=[Entity(id="e1", type="node")],
        resources=[Resource(id="r1", current=100.0)],
    )
    return World(initial_state=state)


def test_filesystem_result_store_put_get_delete(tmp_path: Path) -> None:
    store = FilesystemResultStore(root_dir=tmp_path / "cache")
    world = _make_world()
    scenario = Scenario(scenario_id="cache_test", horizon=3, samples=2, seed=123)

    engine = SimulationEngine()
    result = engine.run(world, scenario)
    fp = compute_simulation_fingerprint(world, scenario)

    # 1. Contains before put
    assert not store.contains(fp)
    assert store.get(fp) is None

    # 2. Put
    store.put(fp, result, metadata={"author": "platform-team"})
    assert store.contains(fp)

    # 3. Check disk sidecars exist
    entry_dir = tmp_path / "cache" / fp
    assert (entry_dir / "result.json").is_file()
    assert (entry_dir / "arrays.npz").is_file()

    # Verify npz sidecar contents
    with np.load(entry_dir / "arrays.npz") as data:
        assert "seeds" in data
        assert len(data["seeds"]) == 2
        assert "step_counts" in data
        assert list(data["step_counts"]) == [3, 3]

    # 4. Get and verify reconstructed result
    retrieved = store.get(fp)
    assert retrieved is not None
    assert retrieved.scenario.scenario_id == scenario.scenario_id
    assert len(retrieved.trajectories) == 2
    assert retrieved.trajectories[0].initial_state.fingerprint == world.initial_state.fingerprint
    assert (
        retrieved.trajectories[0].final_state.fingerprint
        == result.trajectories[0].final_state.fingerprint
    )
    assert retrieved.run_metrics.rollout_count == 2

    # 5. List fingerprints
    fps = store.list_fingerprints()
    assert fp in fps

    # 6. Delete
    deleted = store.delete(fp)
    assert deleted is True
    assert not store.contains(fp)
    assert store.get(fp) is None


def test_memoized_simulation_runner_cache_hit_identical(tmp_path: Path) -> None:
    store = FilesystemResultStore(root_dir=tmp_path / "memo_cache")
    engine = SimulationEngine()
    runner = MemoizedSimulationRunner(engine=engine, store=store)

    world = _make_world()
    scenario = Scenario(scenario_id="memo_test", horizon=4, samples=2, seed=777)

    # First run: cache miss
    res1 = runner.run(world, scenario)
    assert bool(runner.last_cache_hit) is False

    # Second run: cache hit
    res2 = runner.run(world, scenario)
    assert bool(runner.last_cache_hit) is True

    # Validate byte-identical and logically identical trajectories
    assert len(res1.trajectories) == len(res2.trajectories)
    for t1, t2 in zip(res1.trajectories, res2.trajectories, strict=True):
        assert t1.seed == t2.seed
        assert t1.initial_state.fingerprint == t2.initial_state.fingerprint
        assert t1.final_state.fingerprint == t2.final_state.fingerprint
        assert len(t1.steps) == len(t2.steps)
        for s1, s2 in zip(t1.steps, t2.steps, strict=True):
            assert s1.step == s2.step
            assert s1.state_hash == s2.state_hash
            assert (
                s1.transition_result.next_state.fingerprint
                == s2.transition_result.next_state.fingerprint
            )

    # Force refresh should bypass cache
    res3 = runner.run(world, scenario, force_refresh=True)
    assert runner.last_cache_hit is False
    assert (
        res3.trajectories[0].final_state.fingerprint == res1.trajectories[0].final_state.fingerprint
    )


def test_run_memoized_helper_function() -> None:
    store = InMemoryResultStore()
    engine = SimulationEngine()
    world = _make_world()
    scenario = Scenario(scenario_id="helper_test", horizon=2, samples=1, seed=99)

    res1, hit1 = run_memoized(engine, world, scenario, store)
    assert hit1 is False

    res2, hit2 = run_memoized(engine, world, scenario, store)
    assert hit2 is True
    assert (
        res1.trajectories[0].final_state.fingerprint == res2.trajectories[0].final_state.fingerprint
    )


def test_redis_and_s3_stubs_raise_informative_error() -> None:
    with pytest.raises(NotImplementedError, match="available with the '\\[cache\\]' extra"):
        RedisResultStore()

    with pytest.raises(NotImplementedError, match="available with the '\\[cache\\]' extra"):
        S3ResultStore(bucket="my-bucket")
