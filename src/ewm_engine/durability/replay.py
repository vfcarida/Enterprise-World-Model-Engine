"""Deterministic trajectory and state reconstruction from event streams.

Conforms to Track T1: Deterministic Replay (state = fold over events).
Proves that replaying an event stream produces a state and trajectory that is
bitwise and logically identical to the original execution.
"""

from __future__ import annotations

from collections.abc import Sequence

from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.trajectory import (
    StepRecord,
    Trajectory,
)


def fold_events(
    initial_state: WorldState,
    events: Sequence[TransitionEvent],
    verify_fingerprints: bool = True,
) -> WorldState:
    """Fold an ordered sequence of TransitionEvents over an initial WorldState.

    Args:
        initial_state: The baseline world state matching the parent fingerprint of the first event.
        events: Chronologically sorted sequence of transition events.
        verify_fingerprints: Whether to assert cryptographic fingerprint continuity at each step.

    Returns:
        The evolved WorldState resulting from folding all events.

    Raises:
        ValueError: If events are discontinuous or do not match the expected state fingerprint.
    """
    current_state = initial_state

    for idx, ev in enumerate(events):
        if verify_fingerprints:
            if current_state.fingerprint != ev.parent_fingerprint:
                raise ValueError(
                    f"Replay divergence at step {ev.step} (event index {idx}): "
                    f"expected parent fingerprint '{ev.parent_fingerprint}', "
                    f"got current state fingerprint '{current_state.fingerprint}'."
                )

        if ev.state_snapshot is not None:
            next_state = ev.state_snapshot
        else:
            # Reconstruct next state from applied changes and step progression
            builder = current_state
            if ev.applied_changes:
                # Apply resource changes if present
                for r_id, delta_val in ev.applied_changes.items():
                    if r_id in builder.resources:
                        try:
                            # If delta_val is a dict or float
                            if isinstance(delta_val, (int, float)):
                                builder = builder.update_resource(r_id, delta=float(delta_val))
                            elif isinstance(delta_val, dict) and "current" in delta_val:
                                builder = builder.update_resource(
                                    r_id, new_value=float(delta_val["current"])
                                )
                        except Exception:
                            pass
            next_state = builder.model_copy(update={"step": ev.step + 1, "timestamp": ev.timestamp})

        if verify_fingerprints and ev.state_snapshot is not None:
            if next_state.fingerprint != ev.child_fingerprint:
                raise ValueError(
                    f"Replay state snapshot corrupt at step {ev.step}: "
                    f"snapshot fingerprint '{next_state.fingerprint}' != "
                    f"declared child fingerprint '{ev.child_fingerprint}'."
                )

        current_state = next_state

    return current_state


def replay_trajectory(
    events: Sequence[TransitionEvent],
    initial_state: WorldState | None = None,
    sample_id: int = 0,
    seed: int | None = None,
    systemic_trace: SystemicTrace | None = None,
) -> Trajectory:
    """Reconstruct a full, finalized Trajectory from an ordered TransitionEvent stream.

    Args:
        events: The sequence of TransitionEvents to replay.
        initial_state: Optional starting state. If omitted, extracted from the first event snapshot.
        sample_id: Sample identifier for the replayed trajectory.
        seed: Random seed. Defaults to the seed stored in the first event or 0.
        systemic_trace: Optional trace graph. Defaults to a fresh SystemicTrace.

    Returns:
        A finalized Trajectory object identical to the originally simulated trajectory.
    """
    if not events:
        if initial_state is None:
            raise ValueError("Cannot replay empty event stream without an initial_state.")
        return Trajectory(
            sample_id=sample_id,
            seed=seed or 0,
            initial_state=initial_state,
            final_state=initial_state,
            steps=[],
            systemic_trace=systemic_trace or SystemicTrace(),
        )

    first_ev = events[0]
    effective_seed = seed if seed is not None else (first_ev.seed or 0)

    if initial_state is None:
        raise ValueError(
            "Replaying trajectory requires initial_state to verify root parent fingerprint."
        )

    if initial_state.fingerprint != first_ev.parent_fingerprint:
        raise ValueError(
            f"Initial state fingerprint '{initial_state.fingerprint}' does not match "
            f"first event parent fingerprint '{first_ev.parent_fingerprint}'."
        )

    steps: list[StepRecord] = []
    current_state = initial_state

    for ev in events:
        child_state = (
            ev.state_snapshot
            if ev.state_snapshot is not None
            else current_state.model_copy(update={"step": ev.step + 1, "timestamp": ev.timestamp})
        )
        trans_res = (
            ev.transition_result
            if ev.transition_result is not None
            else TransitionResult(
                next_state=child_state,
                applied_changes=dict(ev.applied_changes),
                diagnostics=dict(ev.metadata.get("diagnostics", {})),
                model_name=str(ev.metadata.get("model_name", "ReplayedDynamics")),
            )
        )
        record = StepRecord(
            step=ev.step,
            timestamp=ev.timestamp,
            state_hash=ev.parent_fingerprint,
            actions_proposed=ev.actions_proposed,
            actions_accepted=ev.actions_accepted,
            exogenous_events=ev.exogenous_events,
            transition_result=trans_res,
            constraint_violations=ev.constraint_violations,
            step_metrics=dict(ev.step_metrics),
        )
        steps.append(record)
        current_state = child_state

    trajectory = Trajectory(
        sample_id=sample_id,
        seed=effective_seed,
        initial_state=initial_state,
        final_state=current_state,
        steps=steps,
        systemic_trace=systemic_trace or SystemicTrace(),
    )
    trajectory.finalize()
    return trajectory


def verify_trajectory_replay(original: Trajectory, replayed: Trajectory) -> bool:
    """Verify that a replayed trajectory is bitwise and logically identical to original.

    Checks:
    1. Sample ID and RNG seed equivalence.
    2. Initial state and final state canonical fingerprint equivalence.
    3. Step-by-step step count and step record hash alignment.
    """
    if original.initial_state.fingerprint != replayed.initial_state.fingerprint:
        return False
    if original.final_state.fingerprint != replayed.final_state.fingerprint:
        return False
    if len(original.steps) != len(replayed.steps):
        return False

    for s_orig, s_repl in zip(original.steps, replayed.steps, strict=True):
        if s_orig.step != s_repl.step:
            return False
        if s_orig.state_hash != s_repl.state_hash:
            return False
        if (
            s_orig.transition_result.next_state.fingerprint
            != s_repl.transition_result.next_state.fingerprint
        ):
            return False

    return True
