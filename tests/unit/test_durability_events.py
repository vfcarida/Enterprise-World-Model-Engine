"""Unit tests for TransitionEvent and TraceLog DAG representation."""

from __future__ import annotations

import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.durability.events import TraceLog, TransitionEvent
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def _make_state(step: int = 0, current: float = 100.0) -> WorldState:
    return WorldState(
        step=step,
        entities=[Entity(id="e1", type="node")],
        resources=[Resource(id="r1", current=current)],
    )


def test_transition_event_creation_and_schema() -> None:
    s0 = _make_state(0, 100.0)
    s1 = _make_state(1, 90.0)

    event = TransitionEvent(
        stream_id="rollout-1",
        step=0,
        parent_fingerprint=s0.fingerprint,
        child_fingerprint=s1.fingerprint,
        timestamp=0.0,
        seed=42,
        actions_proposed=(Action(id="act1", type="consume", parameters={"amount": 10.0}),),
        actions_accepted=(Action(id="act1", type="consume", parameters={"amount": 10.0}),),
        applied_changes={"r1": -10.0},
        state_snapshot=s1,
    )

    assert event.schema_version == "1.0.0"
    assert event.stream_id == "rollout-1"
    assert event.step == 0
    assert event.parent_fingerprint == s0.fingerprint
    assert event.child_fingerprint == s1.fingerprint
    assert event.applied_changes == {"r1": -10.0}
    assert event.state_snapshot is not None
    assert event.state_snapshot.fingerprint == s1.fingerprint


def test_tracelog_append_monotonicity_and_continuity_checks() -> None:
    s0 = _make_state(0, 100.0)
    s1 = _make_state(1, 90.0)
    s2 = _make_state(2, 80.0)

    log = TraceLog(stream_id="stream-A", initial_state=s0)

    ev0 = TransitionEvent(
        stream_id="stream-A",
        step=0,
        parent_fingerprint=s0.fingerprint,
        child_fingerprint=s1.fingerprint,
        timestamp=0.0,
        state_snapshot=s1,
    )
    log.append(ev0)
    assert len(log) == 1

    # Non-monotonic step check
    bad_step_ev = TransitionEvent(
        stream_id="stream-A",
        step=0,
        parent_fingerprint=s1.fingerprint,
        child_fingerprint=s2.fingerprint,
        timestamp=1.0,
    )
    with pytest.raises(ValueError, match="Non-monotonic step index"):
        log.append(bad_step_ev)

    # Discontinuous parent fingerprint check
    discontinuous_ev = TransitionEvent(
        stream_id="stream-A",
        step=1,
        parent_fingerprint="invalid_parent_hash",
        child_fingerprint=s2.fingerprint,
        timestamp=1.0,
    )
    with pytest.raises(ValueError, match="Discontinuous state chain"):
        log.append(discontinuous_ev)

    # Valid step 1 append
    ev1 = TransitionEvent(
        stream_id="stream-A",
        step=1,
        parent_fingerprint=s1.fingerprint,
        child_fingerprint=s2.fingerprint,
        timestamp=1.0,
        state_snapshot=s2,
    )
    log.append(ev1)
    assert len(log) == 2


def test_tracelog_branching_and_dag_isolation() -> None:
    s0 = _make_state(0, 100.0)
    s1 = _make_state(1, 90.0)
    s2_a = _make_state(2, 80.0)
    s2_b = _make_state(2, 50.0)

    log_a = TraceLog(stream_id="main", initial_state=s0)
    log_a.append(
        TransitionEvent(
            stream_id="main",
            step=0,
            parent_fingerprint=s0.fingerprint,
            child_fingerprint=s1.fingerprint,
            timestamp=0.0,
            state_snapshot=s1,
        )
    )

    # Branch at step 0 (state s1)
    log_b = log_a.branch(new_stream_id="branch-exp", at_step=0)
    assert log_b.stream_id == "branch-exp"
    assert log_b.parent_stream_id == "main"
    assert len(log_b) == 1
    assert log_b.events[0].child_fingerprint == s1.fingerprint

    # Continue main branch with s2_a
    log_a.append(
        TransitionEvent(
            stream_id="main",
            step=1,
            parent_fingerprint=s1.fingerprint,
            child_fingerprint=s2_a.fingerprint,
            timestamp=1.0,
            state_snapshot=s2_a,
        )
    )

    # Continue forked branch with s2_b
    log_b.append(
        TransitionEvent(
            stream_id="branch-exp",
            step=1,
            parent_fingerprint=s1.fingerprint,
            child_fingerprint=s2_b.fingerprint,
            timestamp=1.0,
            state_snapshot=s2_b,
        )
    )

    # Verify branch isolation
    assert len(log_a) == 2
    assert len(log_b) == 2
    assert log_a.events[1].child_fingerprint == s2_a.fingerprint
    assert log_b.events[1].child_fingerprint == s2_b.fingerprint

    # DAG structure checks
    dag_a = log_a.get_dag()
    assert len(dag_a["edges"]) == 2
    dag_b = log_b.get_dag()
    assert len(dag_b["edges"]) == 2
    assert dag_b["fork_fingerprint"] == s1.fingerprint


def test_tracelog_from_simulation_trajectory() -> None:
    world = World(initial_state=_make_state(0, 100.0))
    scenario = Scenario(scenario_id="trace_test", horizon=3, samples=1, seed=42)
    engine = SimulationEngine()
    result = engine.run(world, scenario)

    traj = result.trajectories[0]
    log = TraceLog.from_trajectory(traj, stream_id="sim-run-1")

    assert len(log) == 3
    assert log.events[0].parent_fingerprint == traj.initial_state.fingerprint
    assert log.events[-1].child_fingerprint == traj.final_state.fingerprint
