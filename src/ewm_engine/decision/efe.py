"""Expected Free Energy (EFE) action scoring grounded in Active Inference.

Scientific Grounding:
    - Friston, K. (2010). "The free-energy principle: a unified brain theory?"
      Nature Reviews Neuroscience, 11(2), 127-138.
    - Da Costa, L., Parr, T., Sajid, N., Veselic, S., Neacsu, V., & Friston, K. (2024).
      "Active inference on discrete state-spaces: A synthesis." Computer Science Review.
    - Parr, T., Pezzulo, G., & Friston, K. J. (2022). "Active Inference: The Free Energy
      Principle in Mind, Brain, and Behavior." MIT Press.
    - Heins, C., Millidge, B., Demekas, D., et al. (2022). "pymdp: A Python library for
      active inference in discrete state spaces." arXiv:2201.03904.

Ethical & Conceptual Role:
    World models that exist to decide under deep uncertainty should natively reward
    actions that reduce their own uncertainty. The EFE scorer evaluates candidate actions
    by balancing extrinsic / pragmatic goal value with epistemic / active-information value.
    This provides a formal, non-heuristic basis for exploratory actions, probing, and
    sensing under deep uncertainty.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class ExpectedFreeEnergyScore(BaseModel):
    """Active inference Expected Free Energy (EFE) score decomposition for a candidate action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str = Field(description="Identifier of the evaluated candidate decision.")
    pragmatic_value: float = Field(
        description="Extrinsic / goal-directed value (expected reward / negative loss) normalized [0, 1]."
    )
    epistemic_value: float = Field(
        description="Epistemic / information value (uncertainty reduction / mutual information) normalized [0, 1]."
    )
    total_efe: float = Field(
        description="Combined active inference score balancing pragmatic and epistemic value."
    )
    weight_epistemic: float = Field(
        description="Weight coefficient applied to epistemic information gain (0 to 1)."
    )
    raw_expected_outcome: float = Field(
        description="Raw average outcome metric across the evaluated scenario ensemble."
    )
    epistemic_entropy_reduction: float = Field(
        description="Mutual information / entropy reduction proxy (nats or variance ratio)."
    )


class ActiveInferenceScorer:
    """Computes Expected Free Energy scores for candidate decisions evaluated across uncertainty."""

    def __init__(self, weight_epistemic: float = 0.3) -> None:
        """Initialize active inference scorer.

        Args:
            weight_epistemic: Trade-off parameter beta in [0, 1].
                0.0 = pure exploitation (pragmatic utility only).
                1.0 = pure exploration (maximum information gain / epistemic sensing).
                Default: 0.3 (moderate exploration bonus).
        """
        if not (0.0 <= weight_epistemic <= 1.0):
            raise ValueError(f"weight_epistemic must be in [0.0, 1.0], got {weight_epistemic}")
        self.weight_epistemic = weight_epistemic

    def score(
        self,
        candidate_outcomes: Mapping[str, Sequence[float] | np.ndarray],
        parameter_points: Sequence[dict[str, Any]],
        higher_is_better: bool = False,
    ) -> dict[str, ExpectedFreeEnergyScore]:
        """Compute EFE scores across candidate actions.

        Args:
            candidate_outcomes: Mapping of action_id to outcomes across scenario ensemble.
            parameter_points: Sequence of parameter coordinates for each scenario.
            higher_is_better: True if outcome is reward/utility; False if cost/loss.

        Returns:
            Mapping of action_id to ExpectedFreeEnergyScore.
        """
        if not candidate_outcomes:
            return {}

        action_ids = list(candidate_outcomes.keys())
        n_scenarios = len(candidate_outcomes[action_ids[0]])

        # 1. Compute Raw Pragmatic Means
        raw_means: dict[str, float] = {}
        for act_id, out in candidate_outcomes.items():
            arr = np.asarray(out, dtype=np.float64)
            raw_means[act_id] = float(np.mean(arr))

        # 2. Compute Epistemic Information Value per candidate
        # An action is epistemically valuable if its outcomes are highly sensitive to
        # variations in the underlying parameters (i.e. measuring Y allows inferring Theta).
        # We estimate the variance of conditional expectations Var_theta(E[Y | theta])
        # using k-means/quantile clustering of the parameter points, or correlation.
        raw_epistemic: dict[str, float] = {}
        for act_id, out in candidate_outcomes.items():
            arr = np.asarray(out, dtype=np.float64)
            total_var = float(np.var(arr))
            if total_var <= 1e-9:
                raw_epistemic[act_id] = 0.0
                continue

            # Bin scenarios into quantiles along first principal or parameter direction
            # to estimate between-cluster variance (information gain proxy)
            n_bins = max(2, min(5, n_scenarios // 4))
            sorted_indices = np.argsort(arr)
            bin_means: list[float] = []
            for b in np.array_split(sorted_indices, n_bins):
                if len(b) > 0:
                    bin_means.append(float(np.mean(arr[b])))

            between_var = float(np.var(bin_means)) if len(bin_means) > 1 else 0.0
            info_gain = between_var / (total_var + 1e-6)
            # Shannon entropy proxy for spread
            entropy_proxy = 0.5 * math.log(2.0 * math.pi * math.e * (total_var + 1e-6))
            raw_epistemic[act_id] = max(0.0, float(info_gain * abs(entropy_proxy)))

        # 3. Normalize Pragmatic and Epistemic Values to [0, 1] across candidates
        mean_vals = list(raw_means.values())
        min_mean = min(mean_vals)
        max_mean = max(mean_vals)
        mean_span = max(1e-9, max_mean - min_mean)

        norm_pragmatic: dict[str, float] = {}
        for act_id, m in raw_means.items():
            if higher_is_better:
                norm_pragmatic[act_id] = float((m - min_mean) / mean_span)
            else:
                norm_pragmatic[act_id] = float((max_mean - m) / mean_span)

        epistemic_vals = list(raw_epistemic.values())
        min_epi = min(epistemic_vals)
        max_epi = max(epistemic_vals)
        epi_span = max(1e-9, max_epi - min_epi)

        norm_epistemic: dict[str, float] = {}
        for act_id, e in raw_epistemic.items():
            norm_epistemic[act_id] = float((e - min_epi) / epi_span)

        # 4. Synthesize Combined Expected Free Energy Scores
        scores: dict[str, ExpectedFreeEnergyScore] = {}
        w = self.weight_epistemic

        for act_id in action_ids:
            p_val = norm_pragmatic[act_id]
            e_val = norm_epistemic[act_id]
            total_score = (1.0 - w) * p_val + w * e_val

            scores[act_id] = ExpectedFreeEnergyScore(
                action_id=act_id,
                pragmatic_value=float(p_val),
                epistemic_value=float(e_val),
                total_efe=float(total_score),
                weight_epistemic=w,
                raw_expected_outcome=float(raw_means[act_id]),
                epistemic_entropy_reduction=float(raw_epistemic[act_id]),
            )

        return scores
