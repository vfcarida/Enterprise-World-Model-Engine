"""Deterministic Monte Carlo simulation execution engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from ewm_engine.actors.base import ActorContext
from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.metadata import SimulationMetadata
from ewm_engine.provenance.trace import SystemicTrace
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

    def simulate(self, world: World, scenario: Scenario) -> SimulationResult:
        """Alias for `run`, simulating rollouts for the specified world and scenario."""
        return self.run(world=world, scenario=scenario)

    def run(self, world: World, scenario: Scenario) -> SimulationResult:
        """Execute simulation rollouts for the specified world and scenario."""
        # 1. Prepare deterministic seed sequence
        seed_seq = np.random.SeedSequence(scenario.seed)
        child_seeds = seed_seq.spawn(scenario.samples)

        # 2. Assemble immutable simulation metadata
        metadata = SimulationMetadata(
            scenario_id=scenario.scenario_id,
            world_hash=world.initial_state.state_hash,
            dynamics_name=getattr(world.dynamics, "name", "None"),
            constraint_versions={c.constraint_id: c.version for c in world.constraints},
            random_seed=scenario.seed,
            horizon=scenario.horizon,
            samples=scenario.samples,
            intervention_id=scenario.intervention.id if scenario.intervention else None,
            custom_metadata=scenario.metadata,
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

            if hasattr(child_seed, "entropy") and isinstance(child_seed.entropy, int):
                seed_val = child_seed.entropy
            elif (
                hasattr(child_seed, "entropy")
                and isinstance(child_seed.entropy, Sequence)
                and len(child_seed.entropy) > 0
            ):
                seed_val = int(child_seed.entropy[0])
            else:
                seed_val = scenario.seed + sample_idx
            trajectory = Trajectory(
                sample_id=sample_idx,
                seed=seed_val,
                initial_state=current_state,
                systemic_trace=trace,
            )

            # Step through horizon
            for step_idx in range(scenario.horizon):
                step_record, current_state, is_invalid = self._execute_step(
                    world=world,
                    state=current_state,
                    step_idx=step_idx,
                    rng=rng,
                    trace=trace,
                    scenario=scenario,
                )
                if is_invalid:
                    trajectory.append_step(
                        step_record, current_state, status=TrajectoryStatus.INVALID
                    )
                    break
                else:
                    trajectory.append_step(step_record, current_state)

            trajectory.finalize()
            trajectories.append(trajectory)

        return SimulationResult(scenario=scenario, metadata=metadata, trajectories=trajectories)

    def _execute_step(
        self,
        world: World,
        state: WorldState,
        step_idx: int,
        rng: np.random.Generator,
        trace: SystemicTrace,
        scenario: Scenario,
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
        for actor in world.actors:
            for act in actor.act(state=state, context=context):
                proposed_actions.append(act)

        # 3. Pre-transition constraint validation on proposed actions
        accepted_actions, pre_results = world.constraints.validate_actions(
            state=state,
            actions=proposed_actions,
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
                trace.add_edge(
                    source=f"action_{act.id}_step_{step_idx}",
                    target=trans_node_id,
                    relation="drives",
                    evidence_level=EvidenceLevel.INTERVENTIONAL,
                )
            for ev in events:
                trace.add_edge(
                    source=f"event_{ev.id}_step_{step_idx}",
                    target=trans_node_id,
                    relation="perturbs",
                    evidence_level=EvidenceLevel.STRUCTURAL,
                )

        # 5. Post-transition constraint validation on next state
        post_results = world.constraints.validate_state(
            state=raw_next_state,
            preceding_actions=accepted_actions,
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
                    relation="triggers_violation",
                    evidence_level=EvidenceLevel.STRUCTURAL,
                )
            elif trans_node_id in trace.nodes:
                trace.add_edge(
                    source=trans_node_id,
                    target=viol_node_id,
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
