"""Rollout execution unit for seeded, reproducible Monte Carlo simulations.

Extracts single-trajectory execution and single-step transitions into an isolated,
picklable module to support both serial and distributed backends.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ewm_engine.actors.base import ActorContext
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    StepRecord,
    Trajectory,
    TrajectoryStatus,
)

if TYPE_CHECKING:
    from ewm_engine.core.world import World


@dataclass(frozen=True)
class RolloutResult:
    """Encapsulates the output of a single Monte Carlo rollout execution.

    Attributes:
        sample_id: Index of this rollout within the scenario sample set.
        trajectory: Evolved trajectory containing all steps and systemic trace.
        constraint_evaluations: Total individual constraint evaluations performed.
        hard_violations: Total fatal hard violations encountered.
        soft_violations: Total non-fatal soft violations encountered.
        dynamics_transitions: Total state dynamics transition evaluations.
    """

    sample_id: int
    trajectory: Trajectory
    constraint_evaluations: int
    hard_violations: int
    soft_violations: int
    dynamics_transitions: int


def execute_step(
    world: World,
    state: WorldState,
    step_idx: int,
    rng: np.random.Generator,
    trace: SystemicTrace,
    scenario: Scenario,
    on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase], None] | None = None,
) -> tuple[StepRecord, WorldState, bool]:
    """Execute a single discrete simulation step within a rollout trajectory."""
    # 1. Sample exogenous shocks
    events: list[ExogenousEvent] = []
    for src in world.event_sources:
        sampled_events = src.sample(state=state, step=step_idx, rng=rng)
        for ev in sampled_events:
            events.append(ev)
            ev_node_id = f"event_{ev.id}_step_{step_idx}"
            trace.add_node(
                node_id=ev_node_id,
                step=step_idx,
                category="event",
                label=f"Exogenous Shock: {ev.type} (sev={ev.severity})",
                evidence_level=EvidenceLevel.STRUCTURAL,
                details=ev.parameters,
            )

    # 2. Query actors for proposed actions
    active_interventions = (scenario.intervention.id,) if scenario.intervention else ()
    context = ActorContext(
        step=step_idx,
        timestamp=state.timestamp,
        rng=rng,
        active_interventions=active_interventions,
    )

    proposed_actions: list[Action] = []
    for sa in scenario.scheduled_actions:
        if sa.step == step_idx:
            proposed_actions.append(sa.action)

    for actor in world.actors:
        for act in actor.act(state=state, context=context):
            proposed_actions.append(act)

    # 3. Pre-transition constraint validation on proposed actions
    accepted_actions, pre_results = world.constraints.validate_actions(
        state=state,
        actions=proposed_actions,
        on_evaluation=on_constraint_eval,
    )

    for act in proposed_actions:
        act_node_id = f"action_{act.id}_step_{step_idx}"
        if act in accepted_actions:
            trace.add_node(
                node_id=act_node_id,
                step=step_idx,
                category="action",
                label=f"Action: {act.type}",
                evidence_level=EvidenceLevel.INTERVENTIONAL,
                details=act.parameters,
            )
            if scenario.intervention is not None and "initial_intervention" in trace.nodes:
                trace.add_edge(
                    source="initial_intervention",
                    target=act_node_id,
                    step=step_idx,
                    relation="conditions",
                    evidence_level=EvidenceLevel.INTERVENTIONAL,
                )
        else:
            trace.add_node(
                node_id=act_node_id,
                step=step_idx,
                category="action",
                label=f"Rejected Action: {act.type}",
                evidence_level=EvidenceLevel.INTERVENTIONAL,
                details=act.parameters,
            )

    # 4. Dynamics transition
    if world.dynamics is not None:
        trans_result = world.dynamics.transition(
            state=state,
            actions=accepted_actions,
            exogenous_events=events,
            rng=rng,
        )
        raw_next_state = trans_result.next_state
    else:
        trans_result = TransitionResult(
            next_state=state,
            applied_changes={},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name="NoDynamics",
        )
        raw_next_state = state

    # Record state transition in systemic trace DAG if meaningful modifications occurred
    trans_node_id = f"trans_step_{step_idx}"
    if accepted_actions or events or trans_result.applied_changes:
        trace.add_node(
            node_id=trans_node_id,
            step=step_idx,
            category="state_change",
            label=f"Dynamics Transition: {trans_result.model_name}",
            evidence_level=trans_result.evidence_level,
            details=trans_result.applied_changes,
        )
        for act in accepted_actions:
            rel = str(
                act.parameters.get("trace_relation")
                or ("reroutes" if "reroute" in act.type else "drives")
            )
            trace.add_edge(
                source=f"action_{act.id}_step_{step_idx}",
                target=trans_node_id,
                step=step_idx,
                relation=rel,
                evidence_level=EvidenceLevel.INTERVENTIONAL,
            )
        for ev in events:
            trace.add_edge(
                source=f"event_{ev.id}_step_{step_idx}",
                target=trans_node_id,
                step=step_idx,
                relation="perturbs",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )

    # 5. Post-transition constraint validation on next state
    post_results = world.constraints.validate_state(
        state=raw_next_state,
        preceding_actions=accepted_actions,
        on_evaluation=on_constraint_eval,
    )
    all_violations = [*pre_results, *post_results]
    is_invalid = world.constraints.has_hard_violations(post_results)

    for viol in pre_results:
        viol_node_id = f"viol_{viol.constraint_id}_step_{step_idx}"
        trace.add_node(
            node_id=viol_node_id,
            step=step_idx,
            category="violation",
            label=f"Constraint Violation: {viol.constraint_id} ({viol.severity.value})",
            evidence_level=EvidenceLevel.STRUCTURAL,
            details={"message": viol.message, "values": viol.violating_values},
        )
        if viol.preceding_action_id:
            pred_act_id = f"action_{viol.preceding_action_id}_step_{step_idx}"
            if pred_act_id in trace.nodes:
                trace.add_edge(
                    source=viol_node_id,
                    target=pred_act_id,
                    step=step_idx,
                    relation="rejects",
                    evidence_level=EvidenceLevel.STRUCTURAL,
                )

    for viol in post_results:
        viol_node_id = f"viol_{viol.constraint_id}_step_{step_idx}"
        trace.add_node(
            node_id=viol_node_id,
            step=step_idx,
            category="violation",
            label=f"Constraint Violation: {viol.constraint_id} ({viol.severity.value})",
            evidence_level=EvidenceLevel.STRUCTURAL,
            details={"message": viol.message, "values": viol.violating_values},
        )
        if (
            viol.preceding_action_id
            and f"action_{viol.preceding_action_id}_step_{step_idx}" in trace.nodes
        ):
            trace.add_edge(
                source=f"action_{viol.preceding_action_id}_step_{step_idx}",
                target=viol_node_id,
                step=step_idx,
                relation="triggers_violation",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )
        elif trans_node_id in trace.nodes:
            trace.add_edge(
                source=trans_node_id,
                target=viol_node_id,
                step=step_idx,
                relation="leads_to_violation",
                evidence_level=EvidenceLevel.STRUCTURAL,
            )

    # 6. Advance step index and timestamp ONLY if not fatally invalid
    if is_invalid:
        final_next_state = raw_next_state
    else:
        final_next_state = raw_next_state.advance_time(1.0)

    # 7. Collect step metrics
    step_metrics: dict[str, float] = {}
    for r_id, r in final_next_state.resources.items():
        step_metrics[f"resource_{r_id}"] = r.current
        step_metrics[f"utilization_{r_id}"] = r.utilization

    for k, v in final_next_state.memory.items():
        if isinstance(v, (int, float)):
            step_metrics[f"mem_{k}"] = float(v)
            if k not in step_metrics:
                step_metrics[k] = float(v)

    step_metrics["violations_count"] = float(len(all_violations))
    step_metrics["hard_violations_count"] = float(
        sum(1 for v in all_violations if v.severity == ConstraintSeverity.HARD)
    )
    step_metrics["violations_penalty"] = world.constraints.total_penalty(all_violations)

    record = StepRecord(
        step=step_idx,
        timestamp=state.timestamp,
        state_hash=state.state_hash,
        actions_proposed=tuple(proposed_actions),
        actions_accepted=tuple(accepted_actions),
        exogenous_events=tuple(events),
        transition_result=trans_result,
        constraint_violations=tuple(all_violations),
        step_metrics=step_metrics,
    )

    return record, final_next_state, is_invalid


def execute_single_rollout(
    world: World,
    scenario: Scenario,
    sample_idx: int,
    child_seed: np.random.SeedSequence,
    *,
    on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None] | None = None,
    on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
) -> RolloutResult:
    """Execute a single Monte Carlo rollout trajectory given an immutable world and scenario snapshot."""
    rng = np.random.default_rng(child_seed)
    trace = SystemicTrace()

    # Initialize initial state, applying intervention if configured
    current_state = world.initial_state
    if scenario.intervention is not None:
        current_state = scenario.intervention.apply(current_state)
        trace.add_node(
            node_id="initial_intervention",
            step=0,
            category="intervention",
            label=f"Intervention: {scenario.intervention.description}",
            evidence_level=EvidenceLevel.INTERVENTIONAL,
            details=scenario.intervention.parameters,
        )

    seed_val = int(child_seed.generate_state(1, dtype=np.uint32)[0])
    trajectory = Trajectory(
        sample_id=sample_idx,
        seed=seed_val,
        initial_state=current_state,
        systemic_trace=trace,
    )

    constraint_evaluations = 0
    hard_violations = 0
    soft_violations = 0
    dynamics_transitions = 0

    # Step through scenario horizon
    for step_idx in range(scenario.horizon):

        def _step_on_eval(
            res: ConstraintResult,
            phase: ConstraintPhase,
            _s: int = sample_idx,
            _st: int = step_idx,
        ) -> None:
            nonlocal constraint_evaluations, hard_violations, soft_violations
            constraint_evaluations += 1
            if not res.satisfied:
                if res.severity == ConstraintSeverity.HARD:
                    hard_violations += 1
                else:
                    soft_violations += 1
            if on_constraint_eval is not None:
                on_constraint_eval(res, phase, _s, _st)

        step_record, current_state, is_invalid = execute_step(
            world=world,
            state=current_state,
            step_idx=step_idx,
            rng=rng,
            trace=trace,
            scenario=scenario,
            on_constraint_eval=_step_on_eval,
        )
        dynamics_transitions += 1

        if is_invalid:
            trajectory.append_step(step_record, current_state, status=TrajectoryStatus.INVALID)
            if on_step_completed is not None:
                on_step_completed(sample_idx, step_idx, step_record.state_hash, step_record, True)
            break
        else:
            trajectory.append_step(step_record, current_state)
            if on_step_completed is not None:
                on_step_completed(sample_idx, step_idx, step_record.state_hash, step_record, False)

    trajectory.finalize()
    return RolloutResult(
        sample_id=sample_idx,
        trajectory=trajectory,
        constraint_evaluations=constraint_evaluations,
        hard_violations=hard_violations,
        soft_violations=soft_violations,
        dynamics_transitions=dynamics_transitions,
    )
