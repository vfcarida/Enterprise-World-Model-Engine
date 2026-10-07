"""Counterfactual explanation adapter for enterprise world model decisions and invariant boundaries.

This module provides an optional integration with DiCE (Diverse Counterfactual Explanations)
and InterpretML to identify minimal feature shifts required to alter world model outcomes.

Requires `ewm-engine[explain]`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field


class CounterfactualExplanation(BaseModel):
    """A single counterfactual world-state scenario."""

    original_features: dict[str, float] = Field(description="Original world state variable values")
    counterfactual_features: dict[str, float] = Field(
        description="Perturbed variable values achieving the alternative outcome"
    )
    predicted_outcome: float = Field(
        description="Model prediction or decision under perturbed state"
    )
    features_changed: list[str] = Field(
        default_factory=list, description="Variables altered between original and counterfactual"
    )
    distance: float = Field(default=0.0, description="Normalized perturbation distance")


class CounterfactualExplanationResult(BaseModel):
    """Collection of counterfactual explanations for a world state decision."""

    target_outcome: float = Field(description="Desired target outcome or invariant-safe state")
    total_counterfactuals: int = Field(description="Number of valid counterfactual paths found")
    explanations: list[CounterfactualExplanation] = Field(
        default_factory=list, description="List of generated counterfactual explanations"
    )
    summary_message: str = Field(default="", description="High-level interpretation guidance")

    def summary(self) -> str:
        """Render a human-readable summary of counterfactual paths."""
        lines = [
            f"=== Counterfactual Decision Paths (Target: {self.target_outcome}) ===",
            f"Found {self.total_counterfactuals} counterfactual scenario(s):",
        ]
        for idx, cf in enumerate(self.explanations, 1):
            deltas = [
                f"{k}: {cf.original_features.get(k, 0.0):.2f} -> {v:.2f}"
                for k, v in cf.counterfactual_features.items()
                if k in cf.features_changed
            ]
            lines.append(f"  Path {idx} (distance={cf.distance:.3f}): {', '.join(deltas)}")
        return "\n".join(lines)


class CounterfactualExplainerAdapter:
    """Adapter for generating actionable counterfactual interventions using DiCE."""

    def __init__(self) -> None:
        self._check_dependency()

    @staticmethod
    def _check_dependency() -> None:
        """Verify that dice-ml is installed."""
        try:
            import dice_ml  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "DiCE is not installed. To use the counterfactual explainer adapter, "
                "install the explain extra: pip install ewm-engine[explain]"
            ) from exc

    def explain_counterfactual(
        self,
        model_wrapper: Any,
        query_instance: dict[str, float],
        continuous_features: Sequence[str],
        outcome_name: str = "outcome",
        desired_class: int = 1,
        total_cfs: int = 2,
    ) -> CounterfactualExplanationResult:
        """Generate counterfactual explanations for a query instance.

        Args:
            model_wrapper: Model or surrogate predicting scenario outcomes.
            query_instance: Current world state feature dictionary.
            continuous_features: Names of continuous world state features.
            outcome_name: Name of target variable.
            desired_class: Desired binary target (e.g., 0 for no violation, 1 for safe).
            total_cfs: Number of diverse counterfactuals to generate.

        Returns:
            CounterfactualExplanationResult containing diverse alternative scenarios.
        """
        self._check_dependency()
        import dice_ml
        import pandas as pd

        # Prepare single row dataframe
        df_query = pd.DataFrame([query_instance])
        d = dice_ml.Data(
            dataframe=df_query,
            continuous_features=list(continuous_features),
            outcome_name=outcome_name,
        )
        m = dice_ml.Model(model=model_wrapper, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        dice_exp = exp.generate_counterfactuals(
            df_query, total_CFs=total_cfs, desired_class=desired_class
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df

        explanations: list[CounterfactualExplanation] = []
        if cf_df is not None:
            for _, row in cf_df.iterrows():
                cf_dict = {str(k): float(v) for k, v in row.items() if k != outcome_name}
                changed = [
                    k for k, v in cf_dict.items() if abs(v - query_instance.get(k, v)) > 1e-5
                ]
                dist = sum(abs(cf_dict[k] - query_instance.get(k, 0.0)) for k in changed)
                explanations.append(
                    CounterfactualExplanation(
                        original_features=query_instance,
                        counterfactual_features=cf_dict,
                        predicted_outcome=float(desired_class),
                        features_changed=changed,
                        distance=dist,
                    )
                )

        return CounterfactualExplanationResult(
            target_outcome=float(desired_class),
            total_counterfactuals=len(explanations),
            explanations=explanations,
            summary_message="Generated via DiCE diverse counterfactual optimization.",
        )
