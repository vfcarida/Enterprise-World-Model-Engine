"""Trajectory and simulation result data structures."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario

if TYPE_CHECKING:
    from ewm_engine.evaluation.comparison import ScenarioComparison
    from ewm_engine.evaluation.uncertainty import UncertaintyDistribution


class StepRecord(BaseModel):
    """Immutable audit record of a single simulation time step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(description="Step index (0, 1, 2, ...).")
    timestamp: float = Field(description="Continuous simulation time.")
    state_hash: str = Field(description="Cryptographic hash of the state prior to transition.")
    actions_proposed: tuple[Action, ...] = Field(default_factory=tuple)
    actions_accepted: tuple[Action, ...] = Field(default_factory=tuple)
    exogenous_events: tuple[ExogenousEvent, ...] = Field(default_factory=tuple)
    transition_result: TransitionResult
    constraint_violations: tuple[ConstraintResult, ...] = Field(default_factory=tuple)
    step_metrics: dict[str, float] = Field(default_factory=dict)


class TrajectoryStatus(StrEnum):
    """Terminal or execution status of an individual rollout trajectory.

    - COMPLETED: Rollout completed full horizon without fatal invariant failure.
    - INVALID: Rollout encountered fatal hard constraint violation; invalidated.
    - FAILED: Rollout failed due to runtime numerical instability or dynamic error.
    """

    COMPLETED = "completed"
    INVALID = "invalid"
    FAILED = "failed"


class Trajectory(BaseModel):
    """An individual rollout path generated during a Monte Carlo simulation run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Semantic schema version.")
    sample_id: int = Field(description="Sample identifier.")
    seed: int = Field(description="RNG seed used for this rollout.")
    initial_state: WorldState = Field(description="Starting world state.")
    status: TrajectoryStatus = Field(
        default=TrajectoryStatus.COMPLETED,
        description="Terminal or execution status of the trajectory.",
    )
    steps: list[StepRecord] = Field(
        default_factory=list,
        description="Sequence of executed simulation steps.",
    )
    systemic_trace: SystemicTrace = Field(
        default_factory=SystemicTrace,
        description="Systemic dependency and causal trace graph.",
    )
    final_state: WorldState = Field(
        description="The final state reached at the end of the trajectory.",
    )

    _finalized: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _set_default_final_state(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "final_state" not in data or data["final_state"] is None:
                if "initial_state" in data:
                    data["final_state"] = data["initial_state"]
        return data

    def __init__(
        self,
        sample_id: int,
        seed: int,
        initial_state: WorldState,
        systemic_trace: SystemicTrace | None = None,
        status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
        steps: list[StepRecord] | None = None,
        final_state: WorldState | None = None,
        schema_version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            schema_version=schema_version,
            sample_id=sample_id,
            seed=seed,
            initial_state=initial_state,
            systemic_trace=systemic_trace if systemic_trace is not None else SystemicTrace(),
            status=status,
            steps=steps or [],
            final_state=final_state if final_state is not None else initial_state,
            **kwargs,
        )

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_finalized", False) and name in {"status", "steps", "final_state"}:
            raise RuntimeError(f"Cannot modify {name} of a finalized Trajectory.")
        super().__setattr__(name, value)

    def finalize(self) -> None:
        """Lock the trajectory to make its status and state immutable after rollout completion."""
        object.__setattr__(self, "_finalized", True)

    @property
    def is_finalized(self) -> bool:
        """Whether the trajectory has completed and been finalized."""
        return self._finalized

    def append_step(
        self,
        step_record: StepRecord,
        resulting_state: WorldState,
        status: TrajectoryStatus | None = None,
    ) -> None:
        """Record an executed step and optionally set/propagate terminal status."""
        if self._finalized:
            raise RuntimeError("Cannot append step to a finalized Trajectory.")
        self.steps.append(step_record)
        object.__setattr__(self, "final_state", resulting_state)
        if status is not None:
            object.__setattr__(self, "status", status)

    @property
    def total_violations(self) -> int:
        """Total number of constraint violations across all steps."""
        return sum(len(step.constraint_violations) for step in self.steps)

    @property
    def hard_violations(self) -> int:
        """Number of fatal / hard constraint violations."""
        return sum(
            1
            for step in self.steps
            for v in step.constraint_violations
            if v.severity == ConstraintSeverity.HARD
        )

    def metric_series(self, metric_name: str) -> list[float]:
        """Extract a time series of values for a given step metric."""
        return [step.step_metrics.get(metric_name, 0.0) for step in self.steps]

    def final_metric(self, metric_name: str) -> float:
        """Get the final value of a metric at the end of the rollout."""
        if not self.steps:
            return 0.0
        return self.steps[-1].step_metrics.get(metric_name, 0.0)


class SimulationResult:
    """The complete result set of a multi-sample Monte Carlo simulation experiment."""

    def __init__(
        self,
        scenario: Scenario,
        provenance: Provenance | None = None,
        trajectories: list[Trajectory] | None = None,
        *,
        metadata: Provenance | None = None,
        run_metrics: RunMetrics | None = None,
    ) -> None:
        self.scenario = scenario
        actual_provenance = provenance if provenance is not None else metadata
        if actual_provenance is None:
            raise ValueError(
                "SimulationResult requires a valid Provenance or SimulationMetadata instance."
            )
        self.provenance: Provenance = actual_provenance
        self.metadata: Provenance = actual_provenance
        self.trajectories: list[Trajectory] = trajectories or []
        if run_metrics is not None:
            self.run_metrics: RunMetrics = run_metrics
        else:
            hard_violations = sum(
                sum(1 for v in s.constraint_violations if v.severity == ConstraintSeverity.HARD)
                for t in self.trajectories
                for s in t.steps
            )
            soft_violations = sum(
                sum(1 for v in s.constraint_violations if v.severity == ConstraintSeverity.SOFT)
                for t in self.trajectories
                for s in t.steps
            )
            dyn_transitions = sum(
                sum(1 for s in t.steps if s.transition_result is not None)
                for t in self.trajectories
            )
            total_edges = sum(len(t.systemic_trace.edges) for t in self.trajectories)
            total_steps = sum(len(t.steps) for t in self.trajectories)
            self.run_metrics = RunMetrics(
                simulation_duration_seconds=0.0,
                rollout_count=len(self.trajectories),
                completed_rollout_count=self.completed_count,
                invalid_rollout_count=self.invalid_count,
                step_count=total_steps,
                constraint_evaluation_count=hard_violations + soft_violations,
                hard_violation_count=hard_violations,
                soft_violation_count=soft_violations,
                dynamics_transition_count=dyn_transitions,
                trace_edge_count=total_edges,
            )

    @property
    def completed_count(self) -> int:
        """Count of rollouts that completed their full horizon without fatal invariant violations."""
        return sum(1 for t in self.trajectories if t.status == TrajectoryStatus.COMPLETED)

    @property
    def invalid_count(self) -> int:
        """Count of rollouts invalidated prematurely due to hard post-transition violations."""
        return sum(1 for t in self.trajectories if t.status == TrajectoryStatus.INVALID)

    @property
    def failed_count(self) -> int:
        """Count of rollouts that failed due to dynamic or numerical execution errors."""
        return sum(1 for t in self.trajectories if t.status == TrajectoryStatus.FAILED)

    def get_metric_distribution(self, metric_name: str) -> UncertaintyDistribution:
        """Compute typed UncertaintyDistribution (mean, std, quantiles, CVaR) for a metric across rollouts."""
        from ewm_engine.evaluation.uncertainty import summarize_distribution

        values = [t.final_metric(metric_name) for t in self.trajectories if t.steps]
        if not values:
            values = [0.0]
        return summarize_distribution(values)

    def metric_distribution(self, metric_name: str) -> dict[str, float]:
        """Compute statistical summary dictionary for a metric across rollouts."""
        dist = self.get_metric_distribution(metric_name)
        data = dist.model_dump()
        data["p50"] = dist.median
        return data

    def violation_rate(self) -> float:
        """Fraction of rollout trajectories that suffered constraint violations or became invalid."""
        if not self.trajectories:
            return 0.0
        violating = sum(
            1
            for t in self.trajectories
            if t.total_violations > 0 or t.status == TrajectoryStatus.INVALID
        )
        return violating / len(self.trajectories)

    def compare(
        self,
        baseline: SimulationResult,
        metrics: list[str] | None = None,
    ) -> ScenarioComparison:
        """Compare this simulation result against a baseline counterfactual."""
        from ewm_engine.evaluation.comparison import compare_scenarios

        return compare_scenarios(baseline=baseline, candidates=[self], metrics=metrics)
