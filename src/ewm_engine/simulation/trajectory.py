"""Trajectory and simulation result data structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.metadata import SimulationMetadata
from ewm_engine.provenance.trace import SystemicTrace
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


class Trajectory:
    """An individual rollout path generated during a Monte Carlo simulation run."""

    def __init__(
        self,
        sample_id: int,
        seed: int,
        initial_state: WorldState,
        systemic_trace: SystemicTrace | None = None,
    ) -> None:
        self.sample_id = sample_id
        self.seed = seed
        self.initial_state = initial_state
        self.steps: list[StepRecord] = []
        self.systemic_trace: SystemicTrace = systemic_trace or SystemicTrace()
        self._final_state: WorldState = initial_state

    def append_step(self, step_record: StepRecord, resulting_state: WorldState) -> None:
        """Record an executed step."""
        self.steps.append(step_record)
        self._final_state = resulting_state

    @property
    def final_state(self) -> WorldState:
        """The final state reached at the end of the trajectory."""
        return self._final_state

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
        metadata: SimulationMetadata,
        trajectories: list[Trajectory],
    ) -> None:
        self.scenario = scenario
        self.metadata = metadata
        self.trajectories = trajectories

    def get_metric_distribution(self, metric_name: str) -> UncertaintyDistribution:
        """Compute typed UncertaintyDistribution (mean, std, quantiles, CVaR) for a metric."""
        from ewm_engine.evaluation.uncertainty import summarize_distribution

        values = [t.final_metric(metric_name) for t in self.trajectories]
        return summarize_distribution(values)

    def metric_distribution(self, metric_name: str) -> dict[str, float]:
        """Compute statistical summary dictionary for a metric across rollouts."""
        dist = self.get_metric_distribution(metric_name)
        data = dist.model_dump()
        data["p50"] = dist.median
        return data

    def violation_rate(self) -> float:
        """Fraction of rollout trajectories that suffered at least one constraint violation."""
        if not self.trajectories:
            return 0.0
        violating = sum(1 for t in self.trajectories if t.total_violations > 0)
        return violating / len(self.trajectories)

    def compare(
        self,
        baseline: SimulationResult,
        metrics: list[str] | None = None,
    ) -> ScenarioComparison:
        """Compare this simulation result against a baseline counterfactual."""
        from ewm_engine.evaluation.comparison import compare_scenarios

        return compare_scenarios(baseline=baseline, candidates=[self], metrics=metrics)
