"""Deterministic Monte Carlo simulation execution engine.

Reproducibility Contract (AC-004):
    Same logical initial state + scenario specification + component versions + master seed
    produces the exact same built-in logical trajectory across platforms and runs.

    Caveat: Neural/GPU/ML adapters carry an explicit caveat that floating-point non-determinism
    across heterogeneous hardware architectures is outside this bitwise guarantee.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
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
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.hooks.protocol import (
    ConstraintEvaluated,
    Hook,
    HookRegistry,
    RolloutCompleted,
    SimulationFinished,
    SimulationStarted,
    StepCompleted,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.metadata import ComponentVersion, Provenance
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import (
    SimulationResult,
    StepRecord,
    Trajectory,
    TrajectoryStatus,
)

if TYPE_CHECKING:
    from ewm_engine.core.world import World


class SimulationEngine:
    """Core simulation engine driving seeded, reproducible Monte Carlo rollouts.

    Coordinates exogenous shocks, agent actions, constraint checks,
    state dynamics, and systemic trace collection.
    """

    def __init__(self, hooks: HookRegistry | Sequence[Hook] | None = None) -> None:
        self.hooks = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks)

    def simulate(
        self,
        world: World,
        scenario: Scenario,
        hooks: HookRegistry | Sequence[Hook] | None = None,
    ) -> SimulationResult:
        """Alias for `run`, simulating rollouts for the specified world and scenario."""
        return self.run(world=world, scenario=scenario, hooks=hooks)

    def run(
        self,
        world: World,
        scenario: Scenario,
        hooks: HookRegistry | Sequence[Hook] | None = None,
    ) -> SimulationResult:
        """Execute simulation rollouts for the specified world and scenario."""
        if scenario.samples < 1:
            raise SimulationConfigurationError(
                f"Scenario samples must be >= 1, got {scenario.samples}"
            )
        if scenario.horizon < 1:
            raise SimulationConfigurationError(
                f"Scenario horizon must be >= 1, got {scenario.horizon}"
            )

        start_time = time.perf_counter()
        active_hooks = (
            hooks
            if isinstance(hooks, HookRegistry)
            else HookRegistry(hooks)
            if hooks is not None
            else self.hooks
        )

        active_hooks.emit(
            SimulationStarted(
                scenario_id=scenario.scenario_id,
                horizon=scenario.horizon,
                samples=scenario.samples,
                seed=scenario.seed,
                initial_state_fingerprint=world.initial_state.fingerprint,
                scenario=scenario,
            )
        )

        total_constraint_evaluations = 0
        total_hard_violations = 0
        total_soft_violations = 0
        total_dynamics_transitions = 0

        # 1. Prepare deterministic seed sequence
        seed_seq = np.random.SeedSequence(scenario.seed)
        child_seeds = seed_seq.spawn(scenario.samples)

        # 2. Assemble immutable simulation provenance
        components: list[ComponentVersion] = []
        if world.dynamics is not None:
            dyn_name = getattr(world.dynamics, "name", world.dynamics.__class__.__name__)
            dyn_ver = getattr(world.dynamics, "version", "1.0.0")
            components.append(ComponentVersion(component_id=dyn_name, component_version=dyn_ver))
            sub_models = getattr(world.dynamics, "models", None)
            if sub_models and isinstance(sub_models, Sequence):
                for sub in sub_models:
                    sub_id = getattr(sub, "name", sub.__class__.__name__)
                    sub_ver = getattr(sub, "version", "1.0.0")
                    components.append(
                        ComponentVersion(component_id=sub_id, component_version=sub_ver)
                    )

        for src in world.event_sources:
            src_id = getattr(src, "name", src.__class__.__name__)
            src_ver = getattr(src, "version", "1.0.0")
            components.append(ComponentVersion(component_id=src_id, component_version=src_ver))

        for actor in world.actors:
            actor_id = getattr(actor, "actor_id", actor.__class__.__name__)
            actor_ver = getattr(actor, "version", "1.0.0")
            components.append(ComponentVersion(component_id=actor_id, component_version=actor_ver))

        constraint_versions = tuple(
            ComponentVersion(component_id=c.constraint_id, component_version=c.version)
            for c in world.constraints
        )

        provenance = Provenance(
            engine_version="0.1.0",
            scenario_fingerprint=scenario.fingerprint,
            initial_state_fingerprint=world.initial_state.fingerprint,
            seed=scenario.seed,
            horizon=scenario.horizon,
            samples=scenario.samples,
            components=tuple(components),
            constraint_versions=constraint_versions,
            runtime_metadata={
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "intervention_id": scenario.intervention.id if scenario.intervention else None,
                **scenario.metadata,
            },
        )

        trajectories: list[Trajectory] = []

        # 3. Execute Monte Carlo sample trajectories
        for sample_idx, child_seed in enumerate(child_seeds):
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

            # Step through horizon
            for step_idx in range(scenario.horizon):
                curr_sample = sample_idx
                curr_step = step_idx

                def _on_eval(
                    res: ConstraintResult,
                    phase: ConstraintPhase,
                    _s: int = curr_sample,
                    _st: int = curr_step,
                ) -> None:
                    nonlocal \
                        total_constraint_evaluations, \
                        total_hard_violations, \
                        total_soft_violations
                    total_constraint_evaluations += 1
                    if not res.satisfied:
                        if res.severity == ConstraintSeverity.HARD:
                            total_hard_violations += 1
                        else:
                            total_soft_violations += 1
                    active_hooks.emit(
                        ConstraintEvaluated(
                            rollout_idx=_s,
                            step=_st,
                            phase=phase,
                            constraint_id=res.constraint_id,
                            severity=res.severity,
                            satisfied=res.satisfied,
                            result=res,
                        )
                    )

                step_record, current_state, is_invalid = self._execute_step(
                    world=world,
                    state=current_state,
                    step_idx=step_idx,
                    rng=rng,
                    trace=trace,
                    scenario=scenario,
                    on_constraint_eval=_on_eval,
                )
                total_dynamics_transitions += 1

                if is_invalid:
                    trajectory.append_step(
                        step_record, current_state, status=TrajectoryStatus.INVALID
                    )
                    active_hooks.emit(
                        StepCompleted(
                            rollout_idx=sample_idx,
                            step=step_idx,
                            state_hash=step_record.state_hash,
                            record=step_record,
                            is_invalid=True,
                        )
                    )
                    break
                else:
                    trajectory.append_step(step_record, current_state)
                    active_hooks.emit(
                        StepCompleted(
                            rollout_idx=sample_idx,
                            step=step_idx,
                            state_hash=step_record.state_hash,
                            record=step_record,
                            is_invalid=False,
                        )
                    )

            trajectory.finalize()
            trajectories.append(trajectory)
            active_hooks.emit(
                RolloutCompleted(
                    rollout_idx=sample_idx,
                    sample_id=sample_idx,
                    status=trajectory.status,
                    step_count=len(trajectory.steps),
                    trajectory=trajectory,
                )
            )

        duration = time.perf_counter() - start_time
        completed_count = sum(1 for t in trajectories if t.status == TrajectoryStatus.COMPLETED)
        invalid_count = sum(1 for t in trajectories if t.status == TrajectoryStatus.INVALID)
        step_count = sum(len(t.steps) for t in trajectories)
        trace_edge_count = sum(len(t.systemic_trace.edges) for t in trajectories)

        run_metrics = RunMetrics(
            simulation_duration_seconds=duration,
            rollout_count=len(trajectories),
            completed_rollout_count=completed_count,
            invalid_rollout_count=invalid_count,
            step_count=step_count,
            constraint_evaluation_count=total_constraint_evaluations,
            hard_violation_count=total_hard_violations,
            soft_violation_count=total_soft_violations,
            dynamics_transition_count=total_dynamics_transitions,
            trace_edge_count=trace_edge_count,
        )

        provenance_metadata = dict(provenance.runtime_metadata)
        provenance_metadata["metrics"] = run_metrics.model_dump()
        provenance_metadata["run_metrics"] = run_metrics.model_dump()
        provenance = provenance.model_copy(update={"runtime_metadata": provenance_metadata})

        result = SimulationResult(
            scenario=scenario,
            provenance=provenance,
            trajectories=trajectories,
            run_metrics=run_metrics,
        )

        active_hooks.emit(
            SimulationFinished(
                result=result,
                run_metrics=run_metrics,
                duration_seconds=duration,
            )
        )

        return result

    def verify_determinism(self, world: World, scenario: Scenario) -> bool:
        """Execute two independent runs of the scenario and verify logical bitwise determinism.

        Contract (AC-004):
            Same logical initial state + scenario + component versions + seed => same built-in trajectory.
        """
        res1 = self.run(world=world, scenario=scenario)
        res2 = self.run(world=world, scenario=scenario)
        if len(res1.trajectories) != len(res2.trajectories):
            return False
        for t1, t2 in zip(res1.trajectories, res2.trajectories, strict=True):
            if t1.seed != t2.seed or t1.status != t2.status:
                return False
            if t1.final_state.fingerprint != t2.final_state.fingerprint:
                return False
            if len(t1.steps) != len(t2.steps):
                return False
            for s1, s2 in zip(t1.steps, t2.steps, strict=True):
                if s1.state_hash != s2.state_hash or s1.step_metrics != s2.step_metrics:
                    return False
        return True

    def _execute_step(
        self,
        world: World,
        state: WorldState,
        step_idx: int,
        rng: np.random.Generator,
        trace: SystemicTrace,
        scenario: Scenario,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase], None] | None = None,
    ) -> tuple[StepRecord, WorldState, bool]:
        """Execute a single discrete simulation step."""
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
