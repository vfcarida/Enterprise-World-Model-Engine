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

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
)
from ewm_engine.core.state import WorldState
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
from ewm_engine.provenance.metadata import ComponentVersion, Provenance
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.executors import RolloutExecutor, resolve_executor
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.rollout import execute_step
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
    state dynamics, and systemic trace collection across pluggable execution backends.
    """

    def __init__(
        self,
        hooks: HookRegistry | Sequence[Hook] | None = None,
        executor: RolloutExecutor | str | None = None,
    ) -> None:
        """Initialize SimulationEngine.

        Args:
            hooks: Optional hook registry or collection of lifecycle hooks.
            executor: Optional default rollout executor backend ('serial',
                'multiprocessing', 'ray', or a RolloutExecutor instance).
                Defaults to SerialExecutor.
        """
        self.hooks = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks)
        self.executor = resolve_executor(executor)

    def simulate(
        self,
        world: World,
        scenario: Scenario,
        hooks: HookRegistry | Sequence[Hook] | None = None,
        executor: RolloutExecutor | str | None = None,
    ) -> SimulationResult:
        """Alias for `run`, simulating rollouts for the specified world and scenario."""
        return self.run(world=world, scenario=scenario, hooks=hooks, executor=executor)

    def run(
        self,
        world: World,
        scenario: Scenario,
        hooks: HookRegistry | Sequence[Hook] | None = None,
        executor: RolloutExecutor | str | None = None,
    ) -> SimulationResult:
        """Execute simulation rollouts for the specified world and scenario.

        Args:
            world: Initial world configuration containing entities, dynamics, and constraints.
            scenario: Scenario configuration defining horizon, sample count, and interventions.
            hooks: Optional runtime hooks overriding instance hooks.
            executor: Optional rollout executor backend overriding instance executor.

        Returns:
            A SimulationResult containing collected trajectories and execution metrics.
        """
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
        active_executor = self.executor if executor is None else resolve_executor(executor)

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

        # 3. Callbacks for in-process executors
        def _on_eval(
            res: ConstraintResult, phase: ConstraintPhase, rollout_idx: int, step_idx: int
        ) -> None:
            active_hooks.emit(
                ConstraintEvaluated(
                    rollout_idx=rollout_idx,
                    step=step_idx,
                    phase=phase,
                    constraint_id=res.constraint_id,
                    severity=res.severity,
                    satisfied=res.satisfied,
                    result=res,
                )
            )

        def _on_step(
            rollout_idx: int, step_idx: int, state_hash: str, record: StepRecord, is_invalid: bool
        ) -> None:
            active_hooks.emit(
                StepCompleted(
                    rollout_idx=rollout_idx,
                    step=step_idx,
                    state_hash=state_hash,
                    record=record,
                    is_invalid=is_invalid,
                )
            )

        # 4. Execute rollouts via chosen executor backend
        rollout_results = active_executor.execute(
            world=world,
            scenario=scenario,
            child_seeds=child_seeds,
            on_constraint_eval=_on_eval,
            on_step_completed=_on_step,
        )

        trajectories: list[Trajectory] = []
        total_constraint_evaluations = 0
        total_hard_violations = 0
        total_soft_violations = 0
        total_dynamics_transitions = 0

        # 5. Reassemble trajectories and emit RolloutCompleted hooks deterministically
        for r in rollout_results:
            total_constraint_evaluations += r.constraint_evaluations
            total_hard_violations += r.hard_violations
            total_soft_violations += r.soft_violations
            total_dynamics_transitions += r.dynamics_transitions
            trajectories.append(r.trajectory)
            active_hooks.emit(
                RolloutCompleted(
                    rollout_idx=r.sample_id,
                    sample_id=r.sample_id,
                    status=r.trajectory.status,
                    step_count=len(r.trajectory.steps),
                    trajectory=r.trajectory,
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

    def verify_determinism(
        self,
        world: World,
        scenario: Scenario,
        executor: RolloutExecutor | str | None = None,
    ) -> bool:
        """Execute two independent runs of the scenario and verify logical bitwise determinism.

        Contract (AC-004 / AC-P03-1):
            Same logical initial state + scenario + component versions + seed => same built-in trajectory.
        """
        res1 = self.run(world=world, scenario=scenario, executor=executor)
        res2 = self.run(world=world, scenario=scenario, executor=executor)
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
        """Execute a single discrete simulation step (delegates to rollout.execute_step)."""
        return execute_step(
            world=world,
            state=state,
            step_idx=step_idx,
            rng=rng,
            trace=trace,
            scenario=scenario,
            on_constraint_eval=on_constraint_eval,
        )
