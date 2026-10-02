"""Unit tests for programmatic runtime metrics (RunMetrics) collection and attachment."""

from collections.abc import Sequence

import numpy as np

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import TrajectoryStatus


class SimpleStockDynamics:
    """Deterministic stock increment dynamics."""

    name: str = "SimpleStockDynamics"
    version: str = "1.0.0"

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: np.random.Generator,
    ) -> TransitionResult:
        delta = 10.0
        for act in actions:
            if act.type == "add":
                delta += float(act.parameters.get("amount", 0.0))
        next_s = state.update_resource("stock", delta=delta)
        return TransitionResult(
            next_state=next_s,
            applied_changes={"stock_delta": delta},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class SoftWarningConstraint:
    """Soft constraint checking threshold without invalidating rollout."""

    constraint_id: str = "soft_stock_warn"
    version: str = "1.0.0"
    severity: ConstraintSeverity = ConstraintSeverity.SOFT
    description: str = "Soft warning when stock exceeds 25."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        stock = state.resources.get("stock")
        val = stock.current if stock else 0.0
        if val > 25.0:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
                satisfied=False,
                message=f"Stock {val} exceeded 25",
            )
        return ConstraintResult(
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
            satisfied=True,
        )


class HardLimitConstraint:
    """Hard post-transition invariant triggering M2 rollout invalidation."""

    constraint_id: str = "hard_stock_cap"
    version: str = "1.0.0"
    severity: ConstraintSeverity = ConstraintSeverity.HARD
    description: str = "Hard cap at 35 units."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        stock = state.resources.get("stock")
        val = stock.current if stock else 0.0
        if val > 35.0:
            return ConstraintResult(
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
                satisfied=False,
                message=f"Stock {val} strictly exceeded hard limit 35.0",
                violating_resources=("stock",),
            )
        return ConstraintResult(
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
            satisfied=True,
        )


def test_run_metrics_on_successful_rollout() -> None:
    """Verify runtime metrics counts on a clean, completing simulation experiment."""
    world = World(
        state=WorldState(resources={"stock": Resource(id="stock", current=10.0, max_value=100.0)}),
        dynamics=SimpleStockDynamics(),
        constraints=[SoftWarningConstraint()],
    )
    # Horizon 3, Samples 2.
    # At step 0: stock 10 -> 20. Soft constraint satisfied.
    # At step 1: stock 20 -> 30. Soft constraint violated (soft violation count increments).
    # At step 2: stock 30 -> 40. Soft constraint violated.
    scenario = Scenario(
        scenario_id="clean_run",
        horizon=3,
        samples=2,
        seed=42,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    metrics: RunMetrics = result.run_metrics
    assert isinstance(metrics, RunMetrics)

    # Rollouts and steps
    assert metrics.rollout_count == 2
    assert metrics.completed_rollout_count == 2
    assert metrics.invalid_rollout_count == 0
    assert metrics.step_count == 6  # 2 rollouts * 3 steps

    # Timing
    assert metrics.simulation_duration_seconds > 0.0

    # Transitions
    assert metrics.dynamics_transition_count == 6  # 6 step transitions

    # Violations: 2 soft violations per rollout (steps 1 and 2), 0 hard violations
    assert metrics.hard_violation_count == 0
    assert metrics.soft_violation_count == 4  # 2 rollouts * 2 violations

    # Constraint evaluations:
    # 0 actions proposed => 0 pre evaluations.
    # 1 constraint * 3 steps * 2 rollouts = 6 evaluations.
    assert metrics.constraint_evaluation_count == 6

    # Systemic trace edges
    assert metrics.trace_edge_count >= 0

    # Provenance metadata integration
    prov_meta = result.provenance.runtime_metadata
    assert "metrics" in prov_meta
    assert prov_meta["metrics"]["rollout_count"] == 2
    assert prov_meta["metrics"]["step_count"] == 6
    assert prov_meta["metrics"]["completed_rollout_count"] == 2
    assert prov_meta["metrics"]["invalid_rollout_count"] == 0


def test_run_metrics_on_invalidated_rollout() -> None:
    """Verify runtime metrics increments invalid_rollout_count when M2 invalidation triggers."""
    world = World(
        state=WorldState(resources={"stock": Resource(id="stock", current=10.0, max_value=100.0)}),
        dynamics=SimpleStockDynamics(),
        constraints=[HardLimitConstraint()],
    )
    # Initial: 10
    # Step 0: stock becomes 20 (<= 35, valid)
    # Step 1: stock becomes 30 (<= 35, valid)
    # Step 2: stock becomes 40 (> 35, HARD POST-TRANSITION VIOLATION -> INVALIDATES)
    scenario = Scenario(
        scenario_id="invalidated_run",
        horizon=5,  # Horizon is 5, but should invalidate after step 2 (3 steps executed: 0, 1, 2)
        samples=1,
        seed=100,
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    metrics = result.run_metrics
    assert metrics.rollout_count == 1
    assert metrics.completed_rollout_count == 0
    assert metrics.invalid_rollout_count == 1  # M2 invalidation tracked
    assert metrics.step_count == 3  # halted early at step 2
    assert metrics.hard_violation_count == 1
    assert metrics.soft_violation_count == 0
    assert metrics.dynamics_transition_count == 3

    assert result.trajectories[0].status == TrajectoryStatus.INVALID


def test_run_metrics_data_sanitization_guardrail() -> None:
    """Guardrail: verify RunMetrics exposes only operational numbers without sensitive payloads."""
    world = World(
        state=WorldState(
            resources={"stock": Resource(id="stock", current=5.0, max_value=50.0)},
            context={"internal_secret_token": "SENSITIVE_12345"},
            memory={"user_ssn": "000-11-2222"},
        ),
        dynamics=SimpleStockDynamics(),
    )
    scenario = Scenario(
        scenario_id="sanitize_test",
        horizon=2,
        samples=1,
        seed=77,
        metadata={"confidential_note": "TOP_SECRET"},
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    dumped = result.run_metrics.model_dump()
    raw_str = str(dumped)

    # Ensure no secrets leak into RunMetrics
    assert "SENSITIVE" not in raw_str
    assert "000-11-2222" not in raw_str
    assert "TOP_SECRET" not in raw_str

    # Only numeric and expected keys present
    expected_keys = {
        "simulation_duration_seconds",
        "rollout_count",
        "completed_rollout_count",
        "invalid_rollout_count",
        "step_count",
        "constraint_evaluation_count",
        "hard_violation_count",
        "soft_violation_count",
        "dynamics_transition_count",
        "trace_edge_count",
    }
    assert set(dumped.keys()) == expected_keys
