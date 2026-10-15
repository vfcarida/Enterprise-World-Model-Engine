"""Unit tests for Active Inference Expected Free Energy (EFE) action scoring."""

from __future__ import annotations

import pytest

from ewm_engine.decision.efe import ActiveInferenceScorer, ExpectedFreeEnergyScore


def test_efe_scorer_bounds_validation() -> None:
    """Weight epistemic must be in [0.0, 1.0]."""
    with pytest.raises(ValueError, match="weight_epistemic must be in"):
        ActiveInferenceScorer(weight_epistemic=1.5)

    with pytest.raises(ValueError, match="weight_epistemic must be in"):
        ActiveInferenceScorer(weight_epistemic=-0.1)


def test_efe_pragmatic_vs_epistemic_tradeoff() -> None:
    """Verify that EFE properly balances goal exploitation with information gain."""
    # Action 1 (Exploit): Low cost (mean = 10), but outcomes are completely flat (zero information about parameters)
    # Action 2 (Probe/Sense): Slightly higher cost (mean = 20), but outcomes vary strongly with the underlying parameter
    candidate_outcomes = {
        "action_exploit": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        "action_probe": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
    }
    parameter_points = [{"theta": float(i)} for i in range(8)]

    # 1. Pure exploitation (beta = 0.0)
    scorer_exploit = ActiveInferenceScorer(weight_epistemic=0.0)
    scores_exploit = scorer_exploit.score(
        candidate_outcomes, parameter_points, higher_is_better=False
    )

    assert isinstance(scores_exploit["action_exploit"], ExpectedFreeEnergyScore)
    # Action exploit has better (lower) mean cost, so pragmatic value = 1.0
    assert scores_exploit["action_exploit"].pragmatic_value == 1.0
    assert scores_exploit["action_probe"].pragmatic_value == 0.0
    assert scores_exploit["action_exploit"].total_efe > scores_exploit["action_probe"].total_efe

    # 2. Pure exploration / active sensing (beta = 1.0)
    scorer_explore = ActiveInferenceScorer(weight_epistemic=1.0)
    scores_explore = scorer_explore.score(
        candidate_outcomes, parameter_points, higher_is_better=False
    )

    # Action probe produces wide variance revealing theta, so epistemic value is higher
    assert (
        scores_explore["action_probe"].epistemic_value
        > scores_explore["action_exploit"].epistemic_value
    )
    assert scores_explore["action_probe"].total_efe > scores_explore["action_exploit"].total_efe

    # 3. Balanced active inference (beta = 0.3)
    scorer_balanced = ActiveInferenceScorer(weight_epistemic=0.3)
    scores_balanced = scorer_balanced.score(
        candidate_outcomes, parameter_points, higher_is_better=False
    )
    assert 0.0 <= scores_balanced["action_exploit"].total_efe <= 1.0
    assert 0.0 <= scores_balanced["action_probe"].total_efe <= 1.0
