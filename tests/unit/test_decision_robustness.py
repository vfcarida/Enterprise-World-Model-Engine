"""Unit tests for multi-robustness metrics and refusal-to-rank adjudication."""

from __future__ import annotations

import pytest

from ewm_engine.decision.robustness import (
    ActionRobustnessScore,
    DecisionComparisonResult,
    evaluate_decision_robustness,
)


def test_robustness_unanimous_winner() -> None:
    """When one action dominates across all criteria, engine declares a dominant winner."""
    # Action A has strictly lower cost across all scenarios
    outcomes = {
        "action_a": [10.0, 12.0, 11.0, 13.0, 10.0],
        "action_b": [25.0, 30.0, 28.0, 27.0, 29.0],
        "action_c": [50.0, 45.0, 60.0, 55.0, 48.0],
    }

    result = evaluate_decision_robustness(
        outcomes=outcomes,
        satisficing_threshold=15.0,
        higher_is_better=False,
    )

    assert isinstance(result, DecisionComparisonResult)
    assert result.is_conclusive is True
    assert result.refusal_to_rank is False
    assert result.dominant_action == "action_a"
    assert result.metric_winners["min_max_regret"] == "action_a"
    assert result.metric_winners["cvar_regret_90"] == "action_a"
    assert result.metric_winners["satisficing"] == "action_a"

    score_a = result.scores["action_a"]
    assert isinstance(score_a, ActionRobustnessScore)
    assert score_a.max_regret == 0.0
    assert score_a.satisficing_rate == 1.0


def test_robustness_epistemic_refusal_to_rank_when_metrics_disagree() -> None:
    """When criteria disagree on the winner, the engine refuses to declare a single winner (McPhail 2018)."""
    # Scenario 1 to 5:
    # Action A: low cost in 4 scenarios, but 1 extreme catastrophic outlier in scenario 5
    # Action B: moderate, consistent cost across all scenarios (best worst-case)
    # Action C: achieves target threshold in most scenarios but high tail regret
    outcomes = {
        # Action A: achieves satisficing (<=20) 4 out of 5 times, but huge worst-case (100)
        "action_a": [10.0, 12.0, 11.0, 14.0, 100.0],
        # Action B: never hits target <=20, but worst-case is only 30 (wins min-max regret)
        "action_b": [25.0, 26.0, 27.0, 28.0, 29.0],
    }

    # Satisficing threshold = 20.0
    result = evaluate_decision_robustness(
        outcomes=outcomes,
        satisficing_threshold=20.0,
        higher_is_better=False,
    )

    # In this setting:
    # - action_b has lower worst-case regret:
    #   scenarios best: [10, 12, 11, 14, 29]
    #   regret action_a: [0, 0, 0, 0, 71] -> max_regret = 71
    #   regret action_b: [15, 14, 16, 14, 0] -> max_regret = 16 (wins Min-Max Regret!)
    # - action_a meets threshold in 4/5 = 80% (wins Satisficing!)
    # Result: Criteria DISAGREE!
    assert result.is_conclusive is False
    assert result.refusal_to_rank is True
    assert result.dominant_action is None

    # Verify refusal rationale explicitly cites McPhail et al. and the trade-off
    assert "McPhail" in result.refusal_rationale
    assert "refuses to rank" in result.refusal_rationale
    assert result.metric_winners["min_max_regret"] == "action_b"
    assert result.metric_winners["satisficing"] == "action_a"


def test_robustness_higher_is_better_utility() -> None:
    """When higher_is_better=True, regret is max - actual, and satisficing is outcome >= threshold."""
    outcomes = {
        "policy_high": [100.0, 95.0, 110.0],
        "policy_low": [50.0, 60.0, 40.0],
    }

    result = evaluate_decision_robustness(
        outcomes=outcomes,
        satisficing_threshold=90.0,
        higher_is_better=True,
    )

    assert result.is_conclusive is True
    assert result.dominant_action == "policy_high"
    assert result.scores["policy_high"].satisficing_rate == 1.0
    assert result.scores["policy_low"].satisficing_rate == 0.0


def test_robustness_fingerprint_determinism() -> None:
    """Fingerprint must be deterministic across identical inputs."""
    outcomes = {
        "a": [1.0, 2.0, 3.0],
        "b": [2.0, 2.0, 2.0],
    }
    r1 = evaluate_decision_robustness(outcomes, satisficing_threshold=2.5)
    r2 = evaluate_decision_robustness(outcomes, satisficing_threshold=2.5)

    assert r1.fingerprint == r2.fingerprint
    assert len(r1.fingerprint) == 64


def test_robustness_validation_mismatched_lengths() -> None:
    """Mismatched scenario array lengths must raise ValueError."""
    with pytest.raises(ValueError, match="Mismatched scenario count"):
        evaluate_decision_robustness(
            outcomes={"a": [1.0, 2.0], "b": [1.0]},
            satisficing_threshold=1.5,
        )
