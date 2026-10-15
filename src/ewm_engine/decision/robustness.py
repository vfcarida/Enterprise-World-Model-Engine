"""Decision robustness evaluation and multi-metric comparison.

Scientific Grounding:
    - Savage, L. J. (1951). "The theory of statistical decision." Journal of the
      American Statistical Association, 46(253), 55-67. (Min-Max Regret).
    - Starr, M. K. (1962). "Product design and decision theory." Prentice-Hall.
      (Domain criterion / satisficing under uncertainty).
    - Simon, H. A. (1955). "A behavioral model of rational choice." Quarterly Journal
      of Economics, 69(1), 99-118. (Satisficing).
    - Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk."
      Journal of Risk, 2, 21-42. (CVaR / expected shortfall).
    - McPhail, C., Maier, H. R., Kwakkel, J. H., Giuliani, M., Castelletti, A., &
      Zheng, F. (2018). "Robustness metrics: How choice of metric changes which
      policy wins." Environmental Modelling & Software, 107, 142-155.
    - Lempert, R. J., Popper, S. W., & Bankes, S. C. (2003). "Shaping the next one
      hundred years: new methods for quantitative, long-term policy analysis." RAND MR-1626.

Ethical & Epistemic Principle:
    Under deep uncertainty, different decision criteria rank policies differently.
    The engine refuses to collapse deep uncertainty into a single confident winner
    when robustness metrics disagree, explicitly alerting the decision-maker.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256, normalize_canonical_value


class ActionRobustnessScore(BaseModel):
    """Multi-metric robustness profile for a single candidate decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str = Field(description="Identifier of the evaluated action / policy.")
    max_regret: float = Field(description="Worst-case regret across all evaluated scenarios.")
    percentile_regret_90: float = Field(
        description="90th-percentile regret across evaluated scenarios."
    )
    cvar_regret_90: float = Field(
        description="Conditional Value at Risk (expected regret exceeding 90th percentile)."
    )
    satisficing_rate: float = Field(
        description="Proportion of scenarios meeting the satisficing threshold (0 to 1)."
    )
    mean_regret: float = Field(description="Expected / Laplace average regret across scenarios.")
    worst_case_outcome: float = Field(
        description="Worst-case raw performance outcome across scenarios."
    )
    best_case_outcome: float = Field(
        description="Best-case raw performance outcome across scenarios."
    )
    mean_outcome: float = Field(description="Average raw performance outcome across scenarios.")


class DecisionComparisonResult(BaseModel):
    """Side-by-side robustness evaluation and multi-metric comparison.

    Epistemic Guarantee:
        When multiple robustness criteria disagree on the optimal decision, the engine
        refuses to pick a single winner (`refusal_to_rank=True`, `dominant_action=None`),
        protecting decision-makers against artificial consensus and premature closure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: dict[str, ActionRobustnessScore] = Field(
        description="Robustness metric scores indexed by candidate action identifier."
    )
    metric_winners: dict[str, str] = Field(
        description="Winning candidate action under each evaluated robustness metric."
    )
    is_conclusive: bool = Field(
        description="True only if all evaluated robustness metrics unanimously agree on the winner."
    )
    refusal_to_rank: bool = Field(
        description="True if metrics disagree, triggering an explicit refusal to declare a single winner."
    )
    dominant_action: str | None = Field(
        default=None,
        description="Unanimously winning action, or None if robustness metrics disagree.",
    )
    refusal_rationale: str = Field(
        default="",
        description="Epistemic explanation for refusing to declare a single winner.",
    )
    satisficing_threshold: float = Field(
        description="Performance threshold used to evaluate Starr's satisficing criterion."
    )
    higher_is_better: bool = Field(
        default=False,
        description="Whether higher raw performance metric indicates better outcome.",
    )
    fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint of the robustness evaluation."
    )


def evaluate_decision_robustness(
    outcomes: Mapping[str, Sequence[float] | np.ndarray],
    satisficing_threshold: float,
    higher_is_better: bool = False,
    percentile: float = 0.90,
) -> DecisionComparisonResult:
    """Evaluate candidate decisions across multiple robustness metrics.

    Args:
        outcomes: Mapping of action_id to sequence of evaluated outcomes across scenarios.
            All actions must have been evaluated on the identical sequence of scenarios.
        satisficing_threshold: Threshold defining acceptable/satisfactory performance.
        higher_is_better: True if outcome represents reward/utility; False if cost/loss.
        percentile: Quantile for percentile regret and CVaR (default: 0.90).

    Returns:
        DecisionComparisonResult containing multi-metric scores and agreement audit.
    """
    if not outcomes:
        raise ValueError("Cannot evaluate decision robustness with empty outcomes.")

    action_ids = list(outcomes.keys())
    first_len = len(outcomes[action_ids[0]])
    for act_id in action_ids:
        if len(outcomes[act_id]) != first_len:
            raise ValueError(
                f"Mismatched scenario count: '{act_id}' has {len(outcomes[act_id])}, expected {first_len}."
            )

    n_scenarios = first_len
    if n_scenarios == 0:
        raise ValueError("Cannot evaluate decision robustness over zero scenarios.")

    # 1. Construct performance matrix: rows = actions, cols = scenarios
    perf_matrix = np.zeros((len(action_ids), n_scenarios), dtype=np.float64)
    for i, act_id in enumerate(action_ids):
        perf_matrix[i, :] = np.asarray(outcomes[act_id], dtype=np.float64)

    # 2. Compute Regret Matrix
    # If lower is better (cost): baseline per scenario is min_a cost(a, s)
    # Regret = cost(a, s) - min_a cost(a, s) >= 0
    # If higher is better (utility): baseline per scenario is max_a utility(a, s)
    # Regret = max_a utility(a, s) - utility(a, s) >= 0
    if not higher_is_better:
        best_per_scenario = np.min(perf_matrix, axis=0)
        regret_matrix = perf_matrix - best_per_scenario[np.newaxis, :]
    else:
        best_per_scenario = np.max(perf_matrix, axis=0)
        regret_matrix = best_per_scenario[np.newaxis, :] - perf_matrix

    # 3. Calculate metrics per candidate action
    scores: dict[str, ActionRobustnessScore] = {}

    for i, act_id in enumerate(action_ids):
        act_regrets = regret_matrix[i, :]
        act_outcomes = perf_matrix[i, :]

        # Min-max regret
        max_r = float(np.max(act_regrets))

        # Percentile regret (e.g. 90th percentile)
        p90_r = float(np.quantile(act_regrets, percentile))

        # CVaR (expected regret given regret >= p90)
        tail_regrets = act_regrets[act_regrets >= p90_r]
        cvar_r = float(np.mean(tail_regrets)) if len(tail_regrets) > 0 else p90_r

        # Satisficing rate (Starr's domain criterion)
        if not higher_is_better:
            sat_count = int(np.sum(act_outcomes <= satisficing_threshold))
        else:
            sat_count = int(np.sum(act_outcomes >= satisficing_threshold))
        sat_rate = float(sat_count / n_scenarios)

        # Averages & extrema
        mean_r = float(np.mean(act_regrets))
        worst_out = float(np.max(act_outcomes) if not higher_is_better else np.min(act_outcomes))
        best_out = float(np.min(act_outcomes) if not higher_is_better else np.max(act_outcomes))
        mean_out = float(np.mean(act_outcomes))

        scores[act_id] = ActionRobustnessScore(
            action_id=act_id,
            max_regret=max_r,
            percentile_regret_90=p90_r,
            cvar_regret_90=cvar_r,
            satisficing_rate=sat_rate,
            mean_regret=mean_r,
            worst_case_outcome=worst_out,
            best_case_outcome=best_out,
            mean_outcome=mean_out,
        )

    # 4. Identify winning candidate under each metric
    # Lower is better for all regret metrics; higher is better for satisficing rate
    winner_mmr = min(action_ids, key=lambda a: scores[a].max_regret)
    winner_cvar = min(action_ids, key=lambda a: scores[a].cvar_regret_90)
    winner_sat = max(action_ids, key=lambda a: scores[a].satisficing_rate)
    winner_mean_regret = min(action_ids, key=lambda a: scores[a].mean_regret)

    metric_winners: dict[str, str] = {
        "min_max_regret": winner_mmr,
        "cvar_regret_90": winner_cvar,
        "satisficing": winner_sat,
        "mean_regret": winner_mean_regret,
    }

    # 5. Check for metric unanimity vs disagreement (Refusal to Rank)
    primary_winners = {winner_mmr, winner_cvar, winner_sat}
    if len(primary_winners) == 1:
        # All three primary metrics agree!
        is_conclusive = True
        refusal_to_rank = False
        dominant_action: str | None = winner_mmr
        refusal_rationale = (
            f"All robustness criteria (Min-Max Regret, CVaR Regret, Satisficing) unanimously "
            f"favor candidate '{dominant_action}'."
        )
    else:
        # Disagreement detected across criteria: Epistemic Refusal to Rank
        is_conclusive = False
        refusal_to_rank = True
        dominant_action = None
        refusal_rationale = (
            f"Robustness metrics disagree across candidate decisions (McPhail et al., 2018). "
            f"Min-Max Regret favors '{winner_mmr}' (minimizing worst-case surprise), "
            f"CVaR(90) Regret favors '{winner_cvar}' (minimizing tail regret), and "
            f"Starr's Satisficing favors '{winner_sat}' (maximizing probability of meeting target). "
            f"Under deep uncertainty, declaring a single winner would impose an ungrounded risk "
            f"posture; the engine explicitly refuses to rank and surfaces the trade-off to the decision-maker."
        )

    # 6. Canonical SHA-256 fingerprint
    fp_dict: dict[str, Any] = {
        "actions": sorted(action_ids),
        "scores": {k: normalize_canonical_value(v.model_dump()) for k, v in scores.items()},
        "metric_winners": metric_winners,
        "is_conclusive": is_conclusive,
        "refusal_to_rank": refusal_to_rank,
        "satisficing_threshold": satisficing_threshold,
        "n_scenarios": n_scenarios,
    }
    eval_fingerprint = canonical_sha256(fp_dict)

    return DecisionComparisonResult(
        scores=scores,
        metric_winners=metric_winners,
        is_conclusive=is_conclusive,
        refusal_to_rank=refusal_to_rank,
        dominant_action=dominant_action,
        refusal_rationale=refusal_rationale,
        satisficing_threshold=satisficing_threshold,
        higher_is_better=higher_is_better,
        fingerprint=eval_fingerprint,
    )
