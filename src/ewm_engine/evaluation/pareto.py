"""Multi-objective evaluation and Pareto frontier analysis for counterfactual scenarios."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ObjectiveDirection(StrEnum):
    """Optimization direction for an evaluation metric."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveSpec(BaseModel):
    """Specification of an objective metric, direction, and optional preference weight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Name of the metric to optimize.")
    direction: ObjectiveDirection = Field(
        default=ObjectiveDirection.MINIMIZE,
        description="Optimization direction: 'minimize' (e.g. costs, violations) or 'maximize' (e.g. throughput, equity).",
    )
    weight: float | None = Field(
        default=None,
        description="Optional scalarization weight (must be > 0.0 if specified).",
    )


class ParetoFrontier(BaseModel):
    """Result of multi-objective Pareto analysis across scenarios.

    Epistemic Note:
        The Pareto frontier identifies non-dominated candidates under simulation dynamics.
        In socio-technical world models, trade-offs are intrinsic. No single scenario
        is declared an automated 'winner' unless explicit decision-maker preferences
        or utility weights are provided.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objectives: list[ObjectiveSpec] = Field(description="Evaluated objective specifications.")
    frontier: list[str] = Field(
        description="Scenario names comprising the non-dominated Pareto frontier."
    )
    dominated_by: dict[str, list[str]] = Field(
        description="Mapping from each scenario to scenarios that strictly dominate it."
    )
    dominates: dict[str, list[str]] = Field(
        description="Mapping from each scenario to scenarios that it strictly dominates."
    )
    scores: dict[str, dict[str, float]] = Field(
        description="Mean score per scenario per objective."
    )
    weighted_scores: dict[str, float] | None = Field(
        default=None,
        description="Optional scalarized weighted utility scores when weights are specified.",
    )
    epistemic_warning: str = Field(
        default=(
            "Pareto frontier identifies non-dominated candidates under simulation dynamics. "
            "No single winner is declared without explicit decision-maker preferences/weights."
        ),
        description="Epistemic humility guardrail against arbitrary automated scenario winners.",
    )


def compute_pareto_frontier(
    scores: dict[str, dict[str, float]],
    objectives: Sequence[ObjectiveSpec],
) -> ParetoFrontier:
    """Compute the non-dominated Pareto frontier across scenarios.

    Args:
        scores: Mapping of {scenario_name: {metric_name: value}}.
        objectives: Sequence of ObjectiveSpec objects defining directions and weights.

    Returns:
        A ParetoFrontier containing non-dominated scenarios and dominance relationships.
    """
    scenario_names = list(scores.keys())
    obj_list = list(objectives)

    dominated_by: dict[str, list[str]] = {name: [] for name in scenario_names}
    dominates: dict[str, list[str]] = {name: [] for name in scenario_names}

    # Evaluate pairwise dominance
    for i, name_a in enumerate(scenario_names):
        scores_a = scores[name_a]
        for j, name_b in enumerate(scenario_names):
            if i == j:
                continue
            scores_b = scores[name_b]

            # A dominates B if A is at least as good in all objectives and strictly better in at least one
            at_least_as_good = True
            strictly_better = False

            for obj in obj_list:
                m = obj.metric_name
                val_a = scores_a.get(m, 0.0)
                val_b = scores_b.get(m, 0.0)

                if obj.direction == ObjectiveDirection.MINIMIZE:
                    if val_a > val_b:
                        at_least_as_good = False
                        break
                    elif val_a < val_b:
                        strictly_better = True
                else:  # MAXIMIZE
                    if val_a < val_b:
                        at_least_as_good = False
                        break
                    elif val_a > val_b:
                        strictly_better = True

            if at_least_as_good and strictly_better:
                dominates[name_a].append(name_b)
                dominated_by[name_b].append(name_a)

    # Frontier consists of all scenarios with empty dominated_by
    frontier = [name for name in scenario_names if not dominated_by[name]]

    # Optional scalarized weighted scoring if all objectives have weights
    all_weighted = all(obj.weight is not None and obj.weight > 0 for obj in obj_list)
    weighted_scores: dict[str, float] | None = None

    if all_weighted and scenario_names:
        total_weight = sum(obj.weight for obj in obj_list if obj.weight is not None)
        weighted_scores = {}

        # Compute min/max per metric for normalization
        metric_bounds: dict[str, tuple[float, float]] = {}
        for obj in obj_list:
            m = obj.metric_name
            vals = [scores[name].get(m, 0.0) for name in scenario_names]
            min_v = min(vals)
            max_v = max(vals)
            metric_bounds[m] = (min_v, max_v)

        for name in scenario_names:
            score_sum = 0.0
            for obj in obj_list:
                m = obj.metric_name
                val = scores[name].get(m, 0.0)
                min_v, max_v = metric_bounds[m]
                w = obj.weight or 1.0

                if max_v == min_v:
                    normalized = 1.0
                elif obj.direction == ObjectiveDirection.MINIMIZE:
                    normalized = (max_v - val) / (max_v - min_v)
                else:
                    normalized = (val - min_v) / (max_v - min_v)

                score_sum += w * normalized

            weighted_scores[name] = float(score_sum / total_weight) if total_weight > 0 else 0.0

    return ParetoFrontier(
        objectives=obj_list,
        frontier=frontier,
        dominated_by=dominated_by,
        dominates=dominates,
        scores=scores,
        weighted_scores=weighted_scores,
    )
