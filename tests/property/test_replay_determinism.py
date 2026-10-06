"""Property-based determinism and replay equivalence tests.

Proves:
1. Deterministic replay: reconstruct any state/trajectory by folding events;
   prove replay == original logical trajectory.
2. Branching-as-DAG: forks preserve strict branch isolation.
"""

from __future__ import annotations

import pytest

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.durability.events import TraceLog
from ewm_engine.durability.replay import (
    fold_events,
    replay_trajectory,
    verify_trajectory_replay,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


@pytest.mark.property
@pytest.mark.parametrize("seed", [1, 42, 1337, 9999, 20261005])
@pytest.mark.parametrize("horizon", [1, 3, 5, 10])
def test_replay_equals_original_property(seed: int, horizon: int) -> None:
    """Property test: Replaying an event stream produces a bitwise and logically identical trajectory."""
    initial_state = WorldState(
        step=0,
        entities=[Entity(id="e1", type="node"), Entity(id="e2", type="node")],
        resources=[
            Resource(id="inventory", current=500.0, min_value=0.0, max_value=1000.0),
            Resource(id="cash", current=10000.0, min_value=0.0),
        ],
    )
    world = World(initial_state=initial_state)

    scenario = Scenario(
        scenario_id=f"replay_prop_s{seed}_h{horizon}",
        horizon=horizon,
        samples=1,
        seed=seed,
        scheduled_actions=(
            ScheduledAction(
                step=1, action=Action(id="act_order", type="order", parameters={"qty": 50.0})
            ),
        )
        if horizon > 1
        else (),
    )

    engine = SimulationEngine()
    result = engine.run(world, scenario)
    original_traj = result.trajectories[0]

    # Convert trajectory to TraceLog event stream
    log = TraceLog.from_trajectory(original_traj, stream_id=f"stream_{seed}_{horizon}")
    assert len(log.events) == horizon

    # Fold events to reconstruct final state
    folded_final = fold_events(initial_state=initial_state, events=log.events)
    assert folded_final.fingerprint == original_traj.final_state.fingerprint

    # Reconstruct trajectory
    replayed_traj = replay_trajectory(
        events=log.events,
        initial_state=initial_state,
        sample_id=original_traj.sample_id,
        seed=original_traj.seed,
    )

    # Prove replay == original
    assert verify_trajectory_replay(original_traj, replayed_traj) is True
    assert replayed_traj.initial_state.fingerprint == original_traj.initial_state.fingerprint
    assert replayed_traj.final_state.fingerprint == original_traj.final_state.fingerprint
    assert len(replayed_traj.steps) == len(original_traj.steps)

    for s_orig, s_repl in zip(original_traj.steps, replayed_traj.steps, strict=True):
        assert s_orig.step == s_repl.step
        assert s_orig.state_hash == s_repl.state_hash
        assert (
            s_orig.transition_result.next_state.fingerprint
            == s_repl.transition_result.next_state.fingerprint
        )


@pytest.mark.property
def test_branching_dag_isolation_property() -> None:
    """Property test: DAG branch mutations are strictly isolated."""
    s0 = WorldState(
        step=0,
        resources=[Resource(id="r1", current=100.0)],
    )
    world = World(initial_state=s0)
    scen = Scenario(scenario_id="dag_fork_test", horizon=2, samples=1, seed=42)

    res = SimulationEngine().run(world, scen)
    orig_log = TraceLog.from_trajectory(res.trajectories[0], stream_id="trunk")

    # Fork at step 0
    fork_1 = orig_log.branch("fork-1", at_step=0)
    fork_2 = orig_log.branch("fork-2", at_step=0)

    # Prove fork isolation: appending to fork_1 does not modify trunk or fork_2
    s1_child = fork_1.events[0].child_fingerprint
    next_s2_f1 = WorldState(step=2, resources=[Resource(id="r1", current=80.0)])
    next_s2_f2 = WorldState(step=2, resources=[Resource(id="r1", current=40.0)])

    from ewm_engine.durability.events import TransitionEvent

    fork_1.append(
        TransitionEvent(
            stream_id="fork-1",
            step=1,
            parent_fingerprint=s1_child,
            child_fingerprint=next_s2_f1.fingerprint,
            timestamp=1.0,
            state_snapshot=next_s2_f1,
        )
    )

    fork_2.append(
        TransitionEvent(
            stream_id="fork-2",
            step=1,
            parent_fingerprint=s1_child,
            child_fingerprint=next_s2_f2.fingerprint,
            timestamp=1.0,
            state_snapshot=next_s2_f2,
        )
    )

    assert len(orig_log) == 2
    assert len(fork_1) == 2
    assert len(fork_2) == 2

    assert fork_1.events[-1].child_fingerprint == next_s2_f1.fingerprint
    assert fork_2.events[-1].child_fingerprint == next_s2_f2.fingerprint
    assert fork_1.events[-1].child_fingerprint != fork_2.events[-1].child_fingerprint
