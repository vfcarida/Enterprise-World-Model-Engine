"""Unit tests for planning layer rollout scorers and decision provenance (P08)."""

from __future__ import annotations

import pytest

from ewm_engine.constraints.results import ConstraintResult, ConstraintSeverity
from ewm_engine.core import Resource, WorldState
from ewm_engine.core.actions import Action
from ewm_engine.experimental.planning import (
    ConstraintPenalizedScorer,
    CVaRScorer,
    ExpectedObjectiveScorer,
    PlanningCandidate,
    UncertaintyPenalizedScorer,
)
from ewm_engine.simulation.trajectory import StepRecord, Trajectory, TrajectoryStatus


def _create_synthetic_trajectory(
    final_metric_value: float,
    metric_name: str = "profit",
    violations_count: int = 0,
    status: TrajectoryStatus = TrajectoryStatus.COMPLETED,
    sample_id: int = 0,
) -> Trajectory:
    """Helper creating a minimal Trajectory with prescribed final metric and violations."""
    s0 = WorldState(
        resources={
            "cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=1000.0),
        }
    )
    s1 = s0.update_resource("cash", delta=final_metric_value)

    violations = tuple(
        ConstraintResult(
            satisfied=False,
            constraint_id=f"capacity_limit_{i}",
            severity=ConstraintSeverity.SOFT,
            message=f"Synthetic violation {i}",
        )
        for i in range(violations_count)
    )

    from ewm_engine.dynamics.base import TransitionResult
    from ewm_engine.provenance.evidence import EvidenceLevel

    step = StepRecord(
        step=0,
        timestamp=1.0,
        state_hash=s0.state_hash,
        transition_result=TransitionResult(
            next_state=s1,
            applied_changes={},
            evidence_level=EvidenceLevel.STRUCTURAL,
        ),
        constraint_violations=violations,
        step_metrics={metric_name: final_metric_value},
    )

    traj = Trajectory(
        sample_id=sample_id,
        seed=100 + sample_id,
        initial_state=s0,
        status=status,
        steps=[step],
        final_state=s1,
    )
    return traj


def test_expected_objective_scorer_maximize_and_minimize() -> None:
    """ExpectedObjectiveScorer computes mathematical expectation with proper sign orientation."""
    trajs = [
        _create_synthetic_trajectory(10.0, sample_id=0),
        _create_synthetic_trajectory(20.0, sample_id=1),
        _create_synthetic_trajectory(30.0, sample_id=2),
    ]
    cand = PlanningCandidate(id="cand_a", name="Candidate A")

    # Maximize mode: score = +20.0
    scorer_max = ExpectedObjectiveScorer(metric="profit", minimize=False)
    res_max = scorer_max.score_rollouts(trajs, cand)
    assert res_max.score == pytest.approx(20.0)
    assert res_max.raw_objective == pytest.approx(20.0)
    assert res_max.details["mean"] == pytest.approx(20.0)
    assert res_max.details["min"] == 10.0
    assert res_max.details["max"] == 30.0

    # Minimize mode: score = -20.0
    scorer_min = ExpectedObjectiveScorer(metric="profit", minimize=True)
    res_min = scorer_min.score_rollouts(trajs, cand)
    assert res_min.score == pytest.approx(-20.0)
    assert res_min.raw_objective == pytest.approx(20.0)


def test_cvar_risk_averse_scorer_changes_decision_vs_expected_value() -> None:
    """CVaRScorer penalizes tail catastrophe, flipping preference against a high-mean gamble."""
    cand_safe = PlanningCandidate(id="cand_safe", name="Safe Strategy")
    cand_gamble = PlanningCandidate(id="cand_gamble", name="High-Risk Gamble")

    # Safe policy: steady return of 10.0 across all 10 rollouts
    safe_trajs = [_create_synthetic_trajectory(10.0, sample_id=i) for i in range(10)]

    # Gamble policy: 8 rollouts get 12.0, but 2 rollouts suffer severe crash of 2.0
    # Expected value: (8 * 12 + 2 * 2) / 10 = 100 / 10 = 10.0
    # Let's give gamble slightly higher expected value: 10.5
    # 8 rollouts with 12.5, 2 rollouts with 2.5 -> mean = (8*12.5 + 2*2.5)/10 = 105/10 = 10.5
    gamble_trajs = [_create_synthetic_trajectory(12.5, sample_id=i) for i in range(8)] + [
        _create_synthetic_trajectory(2.5, sample_id=8),
        _create_synthetic_trajectory(2.5, sample_id=9),
    ]

    expected_scorer = ExpectedObjectiveScorer(metric="profit", minimize=False)
    safe_exp = expected_scorer.score_rollouts(safe_trajs, cand_safe)
    gamble_exp = expected_scorer.score_rollouts(gamble_trajs, cand_gamble)

    # Expected value prefers gamble (10.5 > 10.0)
    assert gamble_exp.score > safe_exp.score

    # Risk-averse CVaR (alpha = 0.20, bottom 20% tail):
    # Safe tail is 10.0, Gamble tail is 2.5
    cvar_scorer = CVaRScorer(metric="profit", alpha=0.20, minimize=False)
    safe_cvar = cvar_scorer.score_rollouts(safe_trajs, cand_safe)
    gamble_cvar = cvar_scorer.score_rollouts(gamble_trajs, cand_gamble)

    assert safe_cvar.score == pytest.approx(10.0)
    assert gamble_cvar.score == pytest.approx(2.5)

    # Risk-averse CVaR flips preference: safe > gamble!
    assert safe_cvar.score > gamble_cvar.score


def test_cvar_scorer_minimization_mode() -> None:
    """CVaRScorer in minimize mode penalizes upper-tail extreme loss."""
    cand = PlanningCandidate(id="cost_cand", name="Cost Candidate")
    # Losses: 8 cases of 10.0 cost, 2 extreme cases of 100.0 cost
    trajs = [
        _create_synthetic_trajectory(10.0, metric_name="cost", sample_id=i) for i in range(8)
    ] + [
        _create_synthetic_trajectory(100.0, metric_name="cost", sample_id=8),
        _create_synthetic_trajectory(100.0, metric_name="cost", sample_id=9),
    ]

    scorer = CVaRScorer(metric="cost", alpha=0.20, minimize=True)
    res = scorer.score_rollouts(trajs, cand)

    # Tail losses (worst 20%) are 100.0. Negated score is -100.0
    assert res.score == pytest.approx(-100.0)
    assert res.raw_objective == pytest.approx(28.0)  # mean loss
    assert res.details["cvar"] == pytest.approx(100.0)


def test_uncertainty_penalized_scorer_changes_decision_vs_expected_value() -> None:
    """UncertaintyPenalizedScorer downweights high-variance alternatives."""
    cand_stable = PlanningCandidate(id="stable", name="Stable Strategy")
    cand_volatile = PlanningCandidate(id="volatile", name="Volatile Strategy")

    # Stable: zero variance around 20.0
    stable_trajs = [_create_synthetic_trajectory(20.0, sample_id=i) for i in range(10)]

    # Volatile: higher mean (22.0), but extreme variance (0.0 and 44.0)
    volatile_trajs = [_create_synthetic_trajectory(0.0, sample_id=i) for i in range(5)] + [
        _create_synthetic_trajectory(44.0, sample_id=i + 5) for i in range(5)
    ]

    expected_scorer = ExpectedObjectiveScorer(metric="profit", minimize=False)
    assert expected_scorer.score_rollouts(volatile_trajs, cand_volatile).score > (
        expected_scorer.score_rollouts(stable_trajs, cand_stable).score
    )

    # Uncertainty penalized scorer (lambda = 1.0 on std):
    # Volatile std is 22.0 -> Score = 22.0 - (1.0 * 22.0) = 0.0
    # Stable std is 0.0 -> Score = 20.0 - 0.0 = 20.0
    uncertainty_scorer = UncertaintyPenalizedScorer(
        metric="profit", uncertainty_weight=1.0, minimize=False
    )
    res_stable = uncertainty_scorer.score_rollouts(stable_trajs, cand_stable)
    res_volatile = uncertainty_scorer.score_rollouts(volatile_trajs, cand_volatile)

    assert res_stable.score == pytest.approx(20.0)
    assert res_volatile.score == pytest.approx(0.0)
    assert res_stable.score > res_volatile.score


def test_constraint_penalized_scorer_penalizes_violations_and_invalids() -> None:
    """ConstraintPenalizedScorer downweights candidates that stress world boundaries."""
    cand_clean = PlanningCandidate(id="clean", name="Clean Candidate")
    cand_violating = PlanningCandidate(id="violating", name="Violating Candidate")

    # Clean candidate: 40 profit, 0 violations
    clean_trajs = [
        _create_synthetic_trajectory(40.0, violations_count=0, sample_id=i) for i in range(4)
    ]

    # Violating candidate: 50 profit, but 2 violations per rollout
    violating_trajs = [
        _create_synthetic_trajectory(50.0, violations_count=2, sample_id=i) for i in range(4)
    ]

    scorer = ConstraintPenalizedScorer(
        base_metric="profit",
        violation_penalty_weight=15.0,
        minimize=False,
    )

    res_clean = scorer.score_rollouts(clean_trajs, cand_clean)
    res_violating = scorer.score_rollouts(violating_trajs, cand_violating)

    # Clean: 40 - 0 = 40
    assert res_clean.score == pytest.approx(40.0)
    # Violating: 50 - (15.0 * 2) = 50 - 30 = 20
    assert res_violating.score == pytest.approx(20.0)
    assert res_clean.score > res_violating.score


def test_planning_candidate_factory_methods() -> None:
    """PlanningCandidate factory helpers produce valid typed candidates."""
    act1 = Action(id="act1", type="restock", parameters={"amount": 10.0})
    act2 = Action(id="act2", type="audit", parameters={})

    cand_act = PlanningCandidate.from_action(act1)
    assert cand_act.id == "act1"
    assert cand_act.actions == (act1,)

    cand_acts = PlanningCandidate.from_actions([act1, act2], id="multi_act")
    assert cand_acts.id == "multi_act"
    assert len(cand_acts.actions) == 2
