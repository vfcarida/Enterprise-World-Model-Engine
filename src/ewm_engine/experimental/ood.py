"""Out-of-Distribution (OOD) and regime-shift detection for simulation rollouts.

[EXPERIMENTAL] This module provides epistemic-uncertainty diagnostics that flag when a
world state or rollout trajectory enters an ungrounded extrapolation regime:
    - Distance-to-support boundaries
    - Mahalanobis multivariate correlation anomalies
    - Trajectory groundedness fractions

Epistemic Note:
    OOD detection surfaces epistemic risk in simulation provenance without classifying
    the state as an invariant violation. Real-world dynamics may explore novel regimes,
    but models calibrated on historical support have ungrounded predictive validity.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.learned import TransitionDataset
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.trajectory import Trajectory


class OODStatus(StrEnum):
    """Categorical classification of state support groundedness."""

    GROUNDED = "grounded"
    MARGINAL = "marginal"
    UNGROUNDED = "ungrounded"


class OODStepReport(BaseModel):
    """Diagnostic audit of a single state transition's groundedness."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(description="Simulation step index.")
    state_hash: str = Field(description="SHA-256 cryptographic state hash.")
    score: float = Field(
        description="Normalized anomaly/extrapolation score (0.0 = centered in support)."
    )
    threshold: float = Field(description="Decision boundary for ungrounded regime classification.")
    is_ood: bool = Field(description="True if state exceeds support threshold.")
    status: OODStatus = Field(description="Categorical support classification.")
    offending_features: dict[str, float] = Field(
        default_factory=dict,
        description="Features that exceeded support limits with their relative deviations.",
    )
    details: dict[str, Any] = Field(default_factory=dict)


class OODTrajectoryReport(BaseModel):
    """Summary of groundedness across an entire rollout trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: int = Field(description="Rollout sample identifier.")
    total_steps: int = Field(description="Total steps evaluated.")
    grounded_steps: int = Field(description="Number of steps within verified training support.")
    ood_steps: int = Field(description="Number of steps flagged as out-of-distribution.")
    grounded_fraction: float = Field(
        description="Proportion of trajectory steps that remained within grounded support [0.0, 1.0].",
    )
    first_ood_step: int | None = Field(
        default=None,
        description="First step index where trajectory entered ungrounded regime, if any.",
    )
    max_anomaly_score: float = Field(description="Maximum anomaly score observed along trajectory.")
    step_reports: tuple[OODStepReport, ...] = Field(default_factory=tuple)


@runtime_checkable
class OODDetector(Protocol):
    """Protocol for detecting when world states leave grounded empirical support."""

    def fit(self, states: Sequence[WorldState] | TransitionDataset) -> None:
        """Calibrate support boundaries using reference in-distribution states."""
        ...

    def score_state(self, state: WorldState) -> float:
        """Compute continuous anomaly/novelty score for a state."""
        ...

    def is_ood(self, state: WorldState) -> bool:
        """Check whether state exceeds empirical support threshold."""
        ...

    def evaluate_trajectory(self, trajectory: Trajectory) -> OODTrajectoryReport:
        """Evaluate an entire rollout trajectory and compute grounded fraction."""
        ...

    def annotate_trace(self, trace: SystemicTrace, report: OODTrajectoryReport) -> None:
        """Attach OOD diagnostic metadata to a systemic dependency trace without modifying EvidenceLevels."""
        ...


class SupportBoundaryOODDetector:
    """Detects OOD regimes via per-resource bounding boxes and rule consistency."""

    def __init__(
        self,
        tolerance_fraction: float = 0.05,
        target_resources: Sequence[str] | None = None,
    ) -> None:
        self.tolerance_fraction = tolerance_fraction
        self.target_resources = tuple(target_resources) if target_resources is not None else None
        self._bounds: dict[str, tuple[float, float, float]] = {}  # min, max, range
        self._known_rules: set[str] = set()
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, states: Sequence[WorldState] | TransitionDataset) -> None:
        if isinstance(states, TransitionDataset):
            collected_states = [sample.state for sample in states]
            collected_states.extend(sample.next_state for sample in states)
        else:
            collected_states = list(states)

        if not collected_states:
            raise ValueError("SupportBoundaryOODDetector.fit requires at least one WorldState.")

        res_values: dict[str, list[float]] = {}
        all_rules: set[str] = set()

        for s in collected_states:
            for r_id, r in s.resources.items():
                if self.target_resources is None or r_id in self.target_resources:
                    res_values.setdefault(r_id, []).append(r.current)
            all_rules.update(k for k, v in s.active_rules.items() if v)

        self._bounds.clear()
        for r_id, vals in res_values.items():
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            span = max(1e-6, max_v - min_v)
            self._bounds[r_id] = (min_v, max_v, span)

        self._known_rules = all_rules
        self._is_fitted = True

    def score_state(self, state: WorldState) -> float:
        if not self._is_fitted:
            raise RuntimeError("SupportBoundaryOODDetector must be fit before scoring.")

        max_deviation = 0.0
        for r_id, (min_v, max_v, span) in self._bounds.items():
            r = state.resources.get(r_id)
            if r is None:
                continue
            val = r.current
            if val < min_v:
                dev = (min_v - val) / span
                max_deviation = max(max_deviation, dev)
            elif val > max_v:
                dev = (val - max_v) / span
                max_deviation = max(max_deviation, dev)

        # Novel rule check: active rule not in calibration set adds penalty
        current_active = {k for k, v in state.active_rules.items() if v}
        novel_rules = current_active - self._known_rules
        if novel_rules:
            max_deviation += float(len(novel_rules))

        return float(max_deviation)

    def is_ood(self, state: WorldState) -> bool:
        score = self.score_state(state)
        return score > self.tolerance_fraction

    def evaluate_step(self, state: WorldState, step_idx: int) -> OODStepReport:
        score = self.score_state(state)
        is_flagged = score > self.tolerance_fraction

        offending: dict[str, float] = {}
        for r_id, (min_v, max_v, span) in self._bounds.items():
            r = state.resources.get(r_id)
            if r is None:
                continue
            val = r.current
            if val < min_v:
                offending[r_id] = (min_v - val) / span
            elif val > max_v:
                offending[r_id] = (val - max_v) / span

        if score <= self.tolerance_fraction:
            status = OODStatus.GROUNDED
        elif score <= self.tolerance_fraction * 2.0:
            status = OODStatus.MARGINAL
        else:
            status = OODStatus.UNGROUNDED

        return OODStepReport(
            step=step_idx,
            state_hash=state.state_hash,
            score=score,
            threshold=self.tolerance_fraction,
            is_ood=is_flagged,
            status=status,
            offending_features=offending,
        )

    def evaluate_trajectory(self, trajectory: Trajectory) -> OODTrajectoryReport:
        step_reports: list[OODStepReport] = []
        first_ood: int | None = None
        ood_count = 0
        max_score = 0.0

        for step in trajectory.steps:
            rep = self.evaluate_step(step.transition_result.next_state, step.step)
            step_reports.append(rep)
            max_score = max(max_score, rep.score)
            if rep.is_ood:
                ood_count += 1
                if first_ood is None:
                    first_ood = step.step

        total = len(step_reports)
        grounded_count = total - ood_count
        fraction = (grounded_count / total) if total > 0 else 1.0

        return OODTrajectoryReport(
            sample_id=trajectory.sample_id,
            total_steps=total,
            grounded_steps=grounded_count,
            ood_steps=ood_count,
            grounded_fraction=float(fraction),
            first_ood_step=first_ood,
            max_anomaly_score=float(max_score),
            step_reports=tuple(step_reports),
        )

    def annotate_trace(self, trace: SystemicTrace, report: OODTrajectoryReport) -> None:
        """Annotate systemic trace with ungrounded regime metadata without changing EvidenceLevel."""
        for step_rep in report.step_reports:
            if step_rep.is_ood:
                node_id = f"ood_step_{step_rep.step}"
                trace.add_node(
                    node_id=node_id,
                    step=step_rep.step,
                    category="ood_regime_shift",
                    label=f"OOD regime shift at step {step_rep.step} (score={step_rep.score:.2f})",
                    evidence_level=EvidenceLevel.PREDICTIVE,
                    details={
                        "score": step_rep.score,
                        "offending_features": step_rep.offending_features,
                        "grounded_fraction": report.grounded_fraction,
                    },
                )


class MahalanobisOODDetector:
    """Multivariate covariance distance detector capturing correlation structure violations."""

    def __init__(
        self,
        threshold_quantile: float = 0.99,
        target_resources: Sequence[str] | None = None,
        ridge_regularization: float = 1e-4,
    ) -> None:
        self.threshold_quantile = threshold_quantile
        self.target_resources = tuple(target_resources) if target_resources is not None else None
        self.ridge_regularization = ridge_regularization
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._resource_keys: tuple[str, ...] = ()
        self._threshold: float = 0.0
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, states: Sequence[WorldState] | TransitionDataset) -> None:
        if isinstance(states, TransitionDataset):
            collected_states = [sample.state for sample in states]
            collected_states.extend(sample.next_state for sample in states)
        else:
            collected_states = list(states)

        if not collected_states:
            raise ValueError("MahalanobisOODDetector requires at least one state.")

        if self.target_resources is not None:
            keys = tuple(sorted(self.target_resources))
        else:
            keys = tuple(sorted(collected_states[0].resources.keys()))

        self._resource_keys = keys
        dim = len(keys)
        if dim == 0:
            raise ValueError("No resources available for Mahalanobis detector.")

        data = np.zeros((len(collected_states), dim), dtype=float)
        for i, s in enumerate(collected_states):
            for j, k in enumerate(keys):
                res = s.resources.get(k)
                data[i, j] = res.current if res is not None else 0.0

        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        if dim == 1:
            cov = np.array([[float(cov)]])

        # Apply ridge regularization to guarantee positive-definiteness
        reg_cov = cov + (np.eye(dim) * self.ridge_regularization)
        inv_cov = np.linalg.pinv(reg_cov)

        # Compute empirical Mahalanobis distances to establish threshold
        diff = data - mean
        distances = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))
        thresh = float(np.percentile(distances, 100.0 * self.threshold_quantile))

        self._mean = mean
        self._inv_cov = inv_cov
        self._threshold = max(1e-3, thresh)
        self._is_fitted = True

    def score_state(self, state: WorldState) -> float:
        if not self._is_fitted or self._mean is None or self._inv_cov is None:
            raise RuntimeError("MahalanobisOODDetector must be fit before scoring.")

        x = np.zeros(len(self._resource_keys), dtype=float)
        for j, k in enumerate(self._resource_keys):
            res = state.resources.get(k)
            x[j] = res.current if res is not None else 0.0

        diff = x - self._mean
        dist = float(np.sqrt(np.dot(diff @ self._inv_cov, diff)))
        # Return distance normalized by threshold
        return dist / self._threshold

    def is_ood(self, state: WorldState) -> bool:
        return self.score_state(state) > 1.0

    def evaluate_trajectory(self, trajectory: Trajectory) -> OODTrajectoryReport:
        step_reports: list[OODStepReport] = []
        first_ood: int | None = None
        ood_count = 0
        max_score = 0.0

        for step in trajectory.steps:
            state = step.transition_result.next_state
            score = self.score_state(state)
            is_flagged = score > 1.0
            max_score = max(max_score, score)

            if is_flagged:
                ood_count += 1
                if first_ood is None:
                    first_ood = step.step
                status = OODStatus.UNGROUNDED
            else:
                status = OODStatus.GROUNDED

            rep = OODStepReport(
                step=step.step,
                state_hash=state.state_hash,
                score=score,
                threshold=1.0,
                is_ood=is_flagged,
                status=status,
            )
            step_reports.append(rep)

        total = len(step_reports)
        grounded_count = total - ood_count
        fraction = (grounded_count / total) if total > 0 else 1.0

        return OODTrajectoryReport(
            sample_id=trajectory.sample_id,
            total_steps=total,
            grounded_steps=grounded_count,
            ood_steps=ood_count,
            grounded_fraction=float(fraction),
            first_ood_step=first_ood,
            max_anomaly_score=float(max_score),
            step_reports=tuple(step_reports),
        )

    def annotate_trace(self, trace: SystemicTrace, report: OODTrajectoryReport) -> None:
        for step_rep in report.step_reports:
            if step_rep.is_ood:
                node_id = f"mahalanobis_ood_{step_rep.step}"
                trace.add_node(
                    node_id=node_id,
                    step=step_rep.step,
                    category="ood_covariance_novelty",
                    label=f"Covariance anomaly at step {step_rep.step} (normalized D_M={step_rep.score:.2f})",
                    evidence_level=EvidenceLevel.PREDICTIVE,
                    details={
                        "score": step_rep.score,
                        "grounded_fraction": report.grounded_fraction,
                    },
                )
