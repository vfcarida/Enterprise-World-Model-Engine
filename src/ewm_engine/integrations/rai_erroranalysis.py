"""ErrorAnalysis adapter for identifying error cohorts in world model simulations.

This module provides an optional integration with the Microsoft Responsible AI
Toolbox (erroranalysis) to identify sub-populations and feature spaces where
the world model shows elevated prediction error or policy divergence.

Requires `ewm-engine[erroranalysis]`.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field


class ErrorCohort(BaseModel):
    """High-error subset identified in simulation runs."""

    cohort_name: str = Field(description="Descriptor or condition defining the cohort")
    sample_size: int = Field(description="Number of instances falling into this cohort")
    error_rate: float = Field(description="Error rate or mean squared residual in cohort")
    feature_conditions: list[str] = Field(
        default_factory=list, description="Split conditions characterizing this subspace"
    )


class ErrorAnalysisResult(BaseModel):
    """Structured breakdown of model error patterns."""

    total_samples: int = Field(description="Total number of evaluated rollout states")
    overall_error_rate: float = Field(description="Global error rate across all states")
    top_error_cohorts: list[ErrorCohort] = Field(
        default_factory=list, description="Cohorts exhibiting significantly elevated error"
    )

    def summary(self) -> str:
        """Render a readable summary of error cohorts."""
        lines = [
            "=== Responsible AI Error Analysis Cohorts ===",
            f"Evaluated {self.total_samples} samples (Overall Error Rate: {self.overall_error_rate:.4f})",
            "Identified High-Error Cohorts:",
        ]
        for c in self.top_error_cohorts:
            lines.append(
                f"  - [{c.cohort_name}] n={c.sample_size}, error_rate={c.error_rate:.4f} "
                f"Conditions: {', '.join(c.feature_conditions)}"
            )
        return "\n".join(lines)


class ErrorAnalysisAdapter:
    """Adapter for cohort-based error discovery using erroranalysis."""

    def __init__(self) -> None:
        self._check_dependency()

    @staticmethod
    def _check_dependency() -> None:
        """Verify that erroranalysis is installed."""
        try:
            import erroranalysis._internal  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "erroranalysis is not installed. To use the error analysis adapter, "
                "install the erroranalysis extra: pip install ewm-engine[erroranalysis]"
            ) from exc

    def analyze_errors(
        self,
        features: Sequence[dict[str, float]],
        y_true: Sequence[float | int],
        y_pred: Sequence[float | int],
        feature_names: Sequence[str],
        max_depth: int = 3,
        num_leaves: int = 4,
    ) -> ErrorAnalysisResult:
        """Construct an error tree to uncover high-error regimes.

        Args:
            features: List of state feature dictionaries.
            y_true: True values or verified benchmark targets.
            y_pred: Model predictions.
            feature_names: Feature names to include in the tree.
            max_depth: Maximum tree depth for error splits.
            num_leaves: Maximum number of leaves in tree.

        Returns:
            ErrorAnalysisResult with detected cohorts.
        """
        self._check_dependency()
        import numpy as np
        import pandas as pd
        from erroranalysis._internal.error_analyzer import ModelAnalyzer

        df = pd.DataFrame(features)
        y_t = np.array(y_true)
        y_p = np.array(y_pred)

        analyzer = ModelAnalyzer(
            model=None,
            dataset=df[list(feature_names)],
            true_y=y_t,
            pred_y=y_p,
            feature_names=list(feature_names),
            categorical_features=[],
            model_task="classification" if len(set(y_t)) <= 5 else "regression",
        )

        tree = analyzer.compute_error_tree(max_depth=max_depth, num_leaves=num_leaves)
        overall_error = (
            float(np.mean(y_t != y_p)) if len(set(y_t)) <= 5 else float(np.mean((y_t - y_p) ** 2))
        )

        cohorts: list[ErrorCohort] = []
        # Extract tree leaves if available
        if isinstance(tree, list):
            for idx, node in enumerate(tree):
                if node.get("leaf", False):
                    cohorts.append(
                        ErrorCohort(
                            cohort_name=f"Cohort_Leaf_{idx}",
                            sample_size=int(node.get("size", 0)),
                            error_rate=float(node.get("metric_value", 0.0)),
                            feature_conditions=[str(node.get("condition", ""))],
                        )
                    )

        return ErrorAnalysisResult(
            total_samples=len(y_true),
            overall_error_rate=overall_error,
            top_error_cohorts=cohorts,
        )
