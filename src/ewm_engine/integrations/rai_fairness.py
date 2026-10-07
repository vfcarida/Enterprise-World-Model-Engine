"""Fairlearn adapter for responsible AI evaluation of simulation actors and policies.

This module provides an optional integration with Fairlearn to evaluate algorithmic
fairness, demographic parity, and equalized odds across actor actions and trajectory outcomes.

Requires `ewm-engine[fairness]`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field


class GroupMetricSummary(BaseModel):
    """Fairness metric summary for a specific group."""

    group_value: str = Field(description="Group or cohort identifier")
    sample_count: int = Field(description="Number of observations in group")
    selection_rate: float = Field(description="Decision or positive outcome rate")
    accuracy_or_metric: float = Field(default=0.0, description="Group-specific performance metric")


class FairnessAuditResult(BaseModel):
    """Structured audit results from Fairlearn analysis."""

    sensitive_feature: str = Field(
        description="Name of the protected or sensitive attribute evaluated"
    )
    demographic_parity_difference: float = Field(
        description="Max difference in selection rate between groups (0.0 is perfect parity)"
    )
    equalized_odds_difference: float = Field(
        description="Max difference in TPR/FPR between groups (0.0 is perfect equality)"
    )
    by_group: list[GroupMetricSummary] = Field(
        default_factory=list, description="Per-group breakdown of rates and counts"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context or solver details"
    )

    def summary(self) -> str:
        """Render a concise text summary of fairness diagnostics."""
        lines = [
            f"=== Responsible AI Fairness Audit: {self.sensitive_feature} ===",
            f"Demographic Parity Difference: {self.demographic_parity_difference:.4f}",
            f"Equalized Odds Difference:     {self.equalized_odds_difference:.4f}",
            "Group Breakdown:",
        ]
        for g in self.by_group:
            lines.append(
                f"  - [{g.group_value}] n={g.sample_count}, selection_rate={g.selection_rate:.3f}, metric={g.accuracy_or_metric:.3f}"
            )
        return "\n".join(lines)


class FairnessAuditAdapter:
    """Adapter bridging simulation traces and actor decisions into Fairlearn."""

    def __init__(self) -> None:
        self._check_dependency()

    @staticmethod
    def _check_dependency() -> None:
        """Verify that fairlearn is installed."""
        try:
            import fairlearn.metrics  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Fairlearn is not installed. To use the fairness audit adapter, "
                "install the fairness extra: pip install ewm-engine[fairness]"
            ) from exc

    def audit_policy_decisions(
        self,
        y_true: Sequence[int | float],
        y_pred: Sequence[int | float],
        sensitive_features: Sequence[Any],
        sensitive_feature_name: str = "cohort",
    ) -> FairnessAuditResult:
        """Run fairness audit on simulation actor decisions or predictions.

        Args:
            y_true: Ground truth or benchmark target outcomes (0 or 1).
            y_pred: Simulated actor actions or predicted decisions (0 or 1).
            sensitive_features: Cohort or protected attributes for each observation.
            sensitive_feature_name: Name of the attribute being evaluated.

        Returns:
            Structured FairnessAuditResult.
        """
        self._check_dependency()
        from fairlearn.metrics import (
            MetricFrame,
            demographic_parity_difference,
            equalized_odds_difference,
            selection_rate,
        )

        dp_diff = float(
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        )
        eo_diff = float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        )

        # Frame breakdown
        frame = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        by_group: list[GroupMetricSummary] = []
        by_group_dict: Mapping[Any, Any] = frame.by_group.to_dict()["selection_rate"]
        for grp, sel_rate in by_group_dict.items():
            cnt = sum(1 for sf in sensitive_features if str(sf) == str(grp))
            by_group.append(
                GroupMetricSummary(
                    group_value=str(grp),
                    sample_count=cnt,
                    selection_rate=float(sel_rate),
                )
            )

        return FairnessAuditResult(
            sensitive_feature=sensitive_feature_name,
            demographic_parity_difference=dp_diff,
            equalized_odds_difference=eo_diff,
            by_group=by_group,
        )
