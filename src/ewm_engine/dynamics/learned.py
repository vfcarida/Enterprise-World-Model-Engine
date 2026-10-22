"""Machine learning extension interfaces and empirical dynamics protocols.

[EXPERIMENTAL] Learned dynamics models and residual estimators are currently experimental.
They require explicit verification against structural conservation laws before deployment.
See ewm_engine.experimental for experimental re-exports.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.cards.models import DatasetCard, ModelCard
from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import (
    DynamicsModel,
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    ResidualFeasibilityRecord,
    TransitionResult,
)
from ewm_engine.dynamics.deterministic import _build_context_diagnostics
from ewm_engine.provenance.evidence import EvidenceLevel


class TransitionSample(BaseModel):
    """A single observed transition record: (S_t, A_t, E_t, S_{t+1})."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: WorldState
    actions: tuple[Action, ...] = Field(default_factory=tuple)
    events: tuple[ExogenousEvent, ...] = Field(default_factory=tuple)
    next_state: WorldState


class TransitionDataset:
    """In-memory dataset of historical or recorded world-state transitions."""

    def __init__(
        self,
        samples: Sequence[TransitionSample] | None = None,
        *,
        split_metadata: dict[str, Any] | None = None,
    ) -> None:
        self._samples: list[TransitionSample] = list(samples or [])
        self.split_metadata: dict[str, Any] = dict(split_metadata or {})

    def add(self, sample: TransitionSample) -> None:
        """Add an observed transition sample."""
        self._samples.append(sample)

    def extend(self, samples: Sequence[TransitionSample]) -> None:
        """Add multiple observed transition samples."""
        self._samples.extend(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> TransitionSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[TransitionSample]:
        return iter(self._samples)

    @property
    def fingerprint(self) -> str:
        """Canonical SHA-256 cryptographic fingerprint of the transition dataset."""
        payload = [
            {
                "state_fp": s.state.fingerprint,
                "state_step": s.state.step,
                "state_timestamp": s.state.timestamp,
                "actions": [a.model_dump(mode="json") for a in s.actions],
                "events": [e.model_dump(mode="json") for e in s.events],
                "next_state_fp": s.next_state.fingerprint,
                "next_state_step": s.next_state.step,
                "next_state_timestamp": s.next_state.timestamp,
            }
            for s in self._samples
        ]
        return canonical_sha256(payload)

    @staticmethod
    def verify_no_temporal_leakage(
        train_dataset: TransitionDataset,
        eval_dataset: TransitionDataset,
    ) -> None:
        """Verify that no training transition occurs chronologically after an evaluation transition.

        Raises:
            ValueError: If `max(train.timestamp) > min(eval.timestamp)` or
                `max(train.step) > min(eval.step)`.
        """
        if len(train_dataset) == 0 or len(eval_dataset) == 0:
            return

        max_train_ts = max(max(s.state.timestamp, s.next_state.timestamp) for s in train_dataset)
        min_eval_ts = min(s.state.timestamp for s in eval_dataset)

        max_train_step = max(s.state.step for s in train_dataset)
        min_eval_step = min(s.state.step for s in eval_dataset)

        if max_train_ts > min_eval_ts or max_train_step > min_eval_step:
            raise ValueError(
                "Temporal leakage detected between train and eval splits: "
                f"max_train_timestamp={max_train_ts} > min_eval_timestamp={min_eval_ts} "
                f"(max_train_step={max_train_step}, min_eval_step={min_eval_step}). "
                "Use strategy='temporal' or dataset.temporal_split() for sequential trajectories."
            )

    def temporal_split(
        self, train_ratio: float = 0.8
    ) -> tuple[TransitionDataset, TransitionDataset]:
        """Split dataset chronologically into past (train) and future (eval) subsets.

        Sorts transitions by `(state.timestamp, state.step)` and enforces strict
        temporal-leakage protection so future states never leak into training.
        """
        return self.split(
            train_ratio=train_ratio,
            seed=0,
            strategy="temporal",
            prevent_temporal_leakage=True,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        *,
        strategy: str = "random",
        prevent_temporal_leakage: bool = False,
    ) -> tuple[TransitionDataset, TransitionDataset]:
        """Split dataset into train and eval subsets deterministically.

        Args:
            train_ratio: Fraction of samples assigned to training (0.0 to 1.0).
            seed: Deterministic random seed for shuffling when `strategy='random'`.
            strategy: Split strategy: `'temporal'` (chronological order with leakage
                check) or `'random'` (seeded permutation for i.i.d. transitions).
            prevent_temporal_leakage: If True, verifies that no training sample has a
                timestamp/step exceeding the earliest evaluation sample.

        Returns:
            Tuple of (train_dataset, eval_dataset).
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        if strategy not in ("random", "temporal"):
            raise ValueError(
                f"Unknown split strategy {strategy!r}; expected 'temporal' or 'random'."
            )
        import numpy as np

        n = len(self._samples)
        if n == 0:
            return TransitionDataset(), TransitionDataset()

        parent_fp = self.fingerprint
        split_idx = int(n * train_ratio)

        if strategy == "temporal":
            ordered = sorted(
                enumerate(self._samples),
                key=lambda item: (
                    item[1].state.timestamp,
                    item[1].state.step,
                    item[0],
                ),
            )
            train_samples = [s for _, s in ordered[:split_idx]]
            test_samples = [s for _, s in ordered[split_idx:]]
            # If boundary timestamps overlap, ensure strict non-leakage check runs
            enforce_leakage = True
        else:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n)
            train_indices = indices[:split_idx]
            test_indices = indices[split_idx:]
            train_samples = [self._samples[i] for i in train_indices]
            test_samples = [self._samples[i] for i in test_indices]
            enforce_leakage = prevent_temporal_leakage

        train_ds = TransitionDataset(
            train_samples,
            split_metadata={
                "role": "train",
                "strategy": strategy,
                "train_ratio": train_ratio,
                "seed": seed,
                "parent_dataset_fingerprint": parent_fp,
            },
        )
        eval_ds = TransitionDataset(
            test_samples,
            split_metadata={
                "role": "eval",
                "strategy": strategy,
                "train_ratio": train_ratio,
                "seed": seed,
                "parent_dataset_fingerprint": parent_fp,
            },
        )

        if enforce_leakage:
            self.verify_no_temporal_leakage(train_ds, eval_ds)

        return train_ds, eval_ds

    def to_dataset_card(
        self,
        dataset_id: str = "transition_dataset",
        name: str = "Observed World-State Transition Dataset",
        *,
        description: str = "Recorded (S_t, A_t, E_t, S_{t+1}) state transition tuples.",
    ) -> DatasetCard:
        """Build a Datasheets-for-Datasets compliant DatasetCard for this dataset."""
        resource_keys: set[str] = set()
        upstream_fps: list[str] = []
        for s in self._samples:
            resource_keys.update(s.state.resources.keys())
            resource_keys.update(s.next_state.resources.keys())
        if self._samples:
            upstream_fps.append(self._samples[0].state.fingerprint)
            upstream_fps.append(self._samples[-1].next_state.fingerprint)

        return DatasetCard(
            artifact_fingerprint=self.fingerprint,
            dataset_id=dataset_id,
            name=name,
            description=description,
            collection_process=(
                f"EWM TransitionDataset ({self.split_metadata.get('strategy', 'full')} split)"
            ),
            num_trajectories=1 if self._samples else 0,
            num_steps=len(self._samples),
            features=sorted(resource_keys),
            intended_use=[
                "Fitting and evaluating residual learned dynamics models under conservation constraints.",
                "Offline transition error and out-of-distribution (OOD) residual diagnostics.",
            ],
            biases_and_limitations=[
                "Observational transition coverage is limited to the action/state support visited during data collection.",
                "Sequential transitions require chronological splitting (strategy='temporal') to prevent temporal lookahead leakage.",
            ],
            provenance_fingerprints=upstream_fps,
        )

    def to_numpy(
        self,
        resource_ids: Sequence[str],
        action_type: str | Sequence[str] | None = None,
        action_param: str = "value",
    ) -> tuple[Any, Any, Any]:
        """Export dataset as NumPy arrays: (states, actions, next_states).

        Args:
            resource_ids: List of resource identifiers to include in feature vectors.
            action_type: Action type(s) to extract (if None, sums across all actions).
            action_param: Parameter key to extract from action (default: 'value').

        Returns:
            Tuple of (states_arr, actions_arr, next_states_arr) as np.ndarray (float64).
        """
        import numpy as np

        n = len(self._samples)
        d = len(resource_ids)
        states = np.zeros((n, d), dtype=np.float64)
        next_states = np.zeros((n, d), dtype=np.float64)

        if isinstance(action_type, (list, tuple)):
            act_cols = len(action_type)
            actions = np.zeros((n, act_cols), dtype=np.float64)
            for i, sample in enumerate(self._samples):
                for j, rid in enumerate(resource_ids):
                    r_curr = sample.state.get_resource(rid)
                    states[i, j] = r_curr.current if r_curr is not None else 0.0
                    r_next = sample.next_state.get_resource(rid)
                    next_states[i, j] = r_next.current if r_next is not None else 0.0

                for k, atype in enumerate(action_type):
                    act_val = 0.0
                    for act in sample.actions:
                        if act.type == atype:
                            val = act.parameters.get(action_param, 1.0)
                            try:
                                act_val += float(val)
                            except (ValueError, TypeError):
                                act_val += 1.0
                    actions[i, k] = act_val
        else:
            actions = np.zeros((n, 1), dtype=np.float64)
            for i, sample in enumerate(self._samples):
                for j, rid in enumerate(resource_ids):
                    r_curr = sample.state.get_resource(rid)
                    states[i, j] = r_curr.current if r_curr is not None else 0.0
                    r_next = sample.next_state.get_resource(rid)
                    next_states[i, j] = r_next.current if r_next is not None else 0.0

                act_val = 0.0
                for act in sample.actions:
                    if action_type is None or act.type == action_type:
                        val = act.parameters.get(action_param, 1.0)
                        try:
                            act_val += float(val)
                        except (ValueError, TypeError):
                            act_val += 1.0
                actions[i, 0] = act_val

        return states, actions, next_states

    def batch(
        self, batch_size: int, shuffle: bool = False, seed: int = 42
    ) -> Iterator[TransitionDataset]:
        """Yield mini-datasets of size batch_size."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        import numpy as np

        n = len(self._samples)
        if n == 0:
            return

        if shuffle:
            rng = np.random.default_rng(seed)
            indices = list(rng.permutation(n))
        else:
            indices = list(range(n))

        for start in range(0, n, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield TransitionDataset([self._samples[i] for i in batch_indices])


@runtime_checkable
class LearnedDynamics(DynamicsModel, Protocol):
    """Protocol for learned or data-driven dynamics models."""

    def fit(self, dataset: TransitionDataset) -> None:
        """Fit or fine-tune model parameters on observed transition data."""
        ...


class LinearResidualDynamics:
    """Minimal learned baseline modeling linear resource deltas from actions and shocks.

    Validates the LearnedDynamics interface without requiring heavyweight ML frameworks.
    Enforces explicit InfeasibleResidualPolicy (never silent clamping) and emits
    cryptographic config/model/data fingerprints and Mitchell-complete ModelCards.
    """

    def __init__(
        self,
        target_resource: ResourceId,
        action_type: str,
        name: str = "LinearResidualDynamics",
        *,
        version: str = "1.0.0",
        infeasible_policy: InfeasibleResidualPolicy = InfeasibleResidualPolicy.PROJECT,
        seed: int = 42,
    ) -> None:
        self.target_resource = target_resource
        self.action_type = action_type
        self.name = name
        self.version = version
        self.seed = seed
        self.action_weight: float = 1.0
        self.bias: float = 0.0
        self.residual_std: float = 0.0
        self.is_fitted: bool = False
        self.infeasible_policy = infeasible_policy
        self.training_dataset_fingerprint: str | None = None
        self.training_split_metadata: dict[str, Any] = {}
        self.training_metrics: dict[str, float] = {}
        self.eval_metrics: dict[str, float] = {}
        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.LEARNED,
            required_context_keys=(),
            evidence_level=EvidenceLevel.PREDICTIVE,
            source="linear_ols_estimator",
            uncertainty_contract="empirical_gaussian_residual_std",
            invariants_affected=(f"resource_bounds:{target_resource}",),
            failure_semantics=infeasible_policy,
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        """Return formal mechanism provenance metadata."""
        return self._provenance

    @property
    def config_fingerprint(self) -> str:
        """Cryptographic SHA-256 fingerprint of model hyperparameters and configuration."""
        return canonical_sha256(
            {
                "model_class": "LinearResidualDynamics",
                "name": self.name,
                "version": self.version,
                "target_resource": self.target_resource,
                "action_type": self.action_type,
                "infeasible_policy": self.infeasible_policy.value,
                "seed": self.seed,
            }
        )

    @property
    def model_fingerprint(self) -> str:
        """Cryptographic SHA-256 fingerprint of config, fitted parameters, and training data."""
        return canonical_sha256(
            {
                "config_fingerprint": self.config_fingerprint,
                "is_fitted": self.is_fitted,
                "action_weight": round(self.action_weight, 12),
                "bias": round(self.bias, 12),
                "residual_std": round(self.residual_std, 12),
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
            }
        )

    def fit(self, dataset: TransitionDataset) -> None:
        """Fit linear parameters (weight, bias) via closed-form ordinary least squares."""
        import numpy as np

        if len(dataset) == 0:
            return

        deltas: list[float] = []
        action_vals: list[float] = []

        for sample in dataset:
            if (
                self.target_resource not in sample.state.resources
                or self.target_resource not in sample.next_state.resources
            ):
                continue
            curr_val = sample.state.get_resource(self.target_resource).current
            next_val = sample.next_state.get_resource(self.target_resource).current
            delta = next_val - curr_val

            act_val = 0.0
            for act in sample.actions:
                if act.type == self.action_type:
                    act_val += float(act.get("value", 1.0))

            deltas.append(delta)
            action_vals.append(act_val)

        if not deltas:
            return

        x = np.array(action_vals, dtype=float)
        y = np.array(deltas, dtype=float)

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.var(x))

        if x_var > 1e-12:
            cov_xy = float(np.mean((x - x_mean) * (y - y_mean)))
            self.action_weight = cov_xy / x_var
            self.bias = y_mean - (self.action_weight * x_mean)
        else:
            self.action_weight = 0.0
            self.bias = y_mean

        preds = (self.action_weight * x) + self.bias
        residuals = y - preds
        train_rmse = float(np.sqrt(np.mean(residuals**2)))
        train_mae = float(np.mean(np.abs(residuals)))
        self.residual_std = float(np.std(residuals))

        self.training_dataset_fingerprint = dataset.fingerprint
        self.training_split_metadata = dict(dataset.split_metadata)
        self.training_metrics = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "residual_std": self.residual_std,
            "n_train_samples": float(len(deltas)),
        }
        self.is_fitted = True

    def evaluate(
        self,
        eval_dataset: TransitionDataset,
        *,
        train_dataset: TransitionDataset | None = None,
        enforce_no_leakage: bool = True,
    ) -> dict[str, float]:
        """Evaluate fitted model on a held-out evaluation dataset with leakage checks."""
        import numpy as np

        if train_dataset is not None and enforce_no_leakage:
            TransitionDataset.verify_no_temporal_leakage(train_dataset, eval_dataset)

        sq_errors: list[float] = []
        abs_errors: list[float] = []
        infeasible_count = 0
        rng = np.random.default_rng(self.seed)

        for sample in eval_dataset:
            if (
                self.target_resource not in sample.state.resources
                or self.target_resource not in sample.next_state.resources
            ):
                continue
            res = self.transition(sample.state, sample.actions, sample.events, rng)
            pred_next = res.next_state.get_resource(self.target_resource).current
            true_next = sample.next_state.get_resource(self.target_resource).current
            err = pred_next - true_next
            sq_errors.append(err * err)
            abs_errors.append(abs(err))
            feas = res.diagnostics.get("residual_feasibility", {})
            if isinstance(feas, dict) and not feas.get("is_feasible", True):
                infeasible_count += 1

        n_eval = len(sq_errors)
        if n_eval == 0:
            metrics = {
                "eval_rmse": 0.0,
                "eval_mae": 0.0,
                "eval_infeasible_rate": 0.0,
                "n_eval_samples": 0.0,
            }
        else:
            metrics = {
                "eval_rmse": float(np.sqrt(np.mean(sq_errors))),
                "eval_mae": float(np.mean(abs_errors)),
                "eval_infeasible_rate": float(infeasible_count) / float(n_eval),
                "n_eval_samples": float(n_eval),
            }
        self.eval_metrics = metrics
        return dict(metrics)

    def to_model_card(self) -> ModelCard:
        """Build a Mitchell-complete ModelCard documenting provenance, split, metrics, and OOD policy."""
        combined_metrics: dict[str, float | str] = {
            "action_weight": round(self.action_weight, 6),
            "bias": round(self.bias, 6),
            "residual_std": round(self.residual_std, 6),
        }
        for k, v in self.training_metrics.items():
            combined_metrics[k] = round(v, 6)
        for k, v in self.eval_metrics.items():
            combined_metrics[k] = round(v, 6)

        return ModelCard(
            artifact_fingerprint=self.model_fingerprint,
            maturity="experimental",
            model_id=self.name,
            model_name=self.name,
            version=self.version,
            description=(
                f"Closed-form ordinary least squares residual dynamics model for "
                f"resource '{self.target_resource}' conditioned on action '{self.action_type}'."
            ),
            model_details={
                "architecture": "Linear OLS Residual Estimator",
                "config_fingerprint": self.config_fingerprint,
                "model_fingerprint": self.model_fingerprint,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
                "seed": self.seed,
                "evidence_level": EvidenceLevel.PREDICTIVE.value,
                "uncertainty_contract": self._provenance.uncertainty_contract,
                "ood_and_infeasible_behavior": (
                    f"Enforces InfeasibleResidualPolicy.{self.infeasible_policy.name} "
                    f"({self.infeasible_policy.value}) when predicted state exits "
                    f"[{self.target_resource}.min_value, {self.target_resource}.max_value]."
                ),
            },
            intended_use=[
                f"Predicting single-step linear resource deltas on '{self.target_resource}' within observed action support.",
                "Serving as an interpretable, zero-extra-dependency residual baseline alongside structural constraints.",
            ],
            assumptions=[
                f"Resource delta for '{self.target_resource}' is approximately affine in '{self.action_type}' action magnitude.",
                "Residual noise is homoscedastic with standard deviation captured by `residual_std`.",
            ],
            out_of_scope=[
                "Unconstrained extrapolation outside physical resource bounds without InfeasibleResidualPolicy enforcement.",
                "Nonlinear regime shifts or unidentifiable confounding without structural causal graph validation.",
            ],
            limitations=[
                "Linear specification cannot capture saturation or higher-order interaction terms.",
                "Predictive evidence tier (EvidenceLevel.PREDICTIVE); does not by itself establish causal identifiability.",
            ],
            training_data=[
                f"training_dataset_fingerprint:{self.training_dataset_fingerprint or 'unfitted'}",
                f"split_strategy:{self.training_split_metadata.get('strategy', 'unspecified')}",
            ],
            eval_data=[
                "Chronological held-out transition split verified via TransitionDataset.verify_no_temporal_leakage."
            ],
            constraints_exercised=[f"resource_bounds:{self.target_resource}"],
            metrics=combined_metrics,
            quantitative_analyses={
                "uncertainty": {
                    "residual_std": self.residual_std,
                    "uncertainty_contract": self._provenance.uncertainty_contract,
                },
                "ood_policy": self.infeasible_policy.value,
            },
            provenance_sources=[
                self.config_fingerprint,
                self.model_fingerprint,
            ]
            + ([self.training_dataset_fingerprint] if self.training_dataset_fingerprint else []),
        )

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        act_val = 0.0
        for act in actions:
            if act.type == self.action_type:
                act_val += float(act.get("value", 1.0))

        predicted_delta = (self.action_weight * act_val) + self.bias
        res = state.get_resource(self.target_resource)
        proposed_val = res.current + predicted_delta
        is_feasible = res.min_value <= proposed_val <= res.max_value
        violations: tuple[str, ...] = (
            ()
            if is_feasible
            else (
                f"resource_bounds_violation:{self.target_resource}"
                f"({proposed_val:.6f} not in [{res.min_value}, {res.max_value}])",
            )
        )

        if is_feasible:
            applied_delta = predicted_delta
            next_state = state.update_resource(
                self.target_resource,
                delta=applied_delta,
                clamp=False,
                enforce_bounds=True,
            )
            action_taken = "accepted"
            l2_dist = 0.0
        elif self.infeasible_policy == InfeasibleResidualPolicy.PROJECT:
            projected_val = max(res.min_value, min(proposed_val, res.max_value))
            applied_delta = projected_val - res.current
            l2_dist = abs(applied_delta - predicted_delta)
            next_state = state.update_resource(
                self.target_resource,
                delta=applied_delta,
                clamp=False,
                enforce_bounds=True,
            )
            action_taken = "projected"
        elif self.infeasible_policy == InfeasibleResidualPolicy.REJECT:
            applied_delta = 0.0
            l2_dist = abs(predicted_delta)
            next_state = state.clone()
            action_taken = "rejected"
        elif self.infeasible_policy == InfeasibleResidualPolicy.ABSTAIN:
            applied_delta = 0.0
            l2_dist = 0.0
            next_state = state.clone()
            action_taken = "abstained"
        else:
            applied_delta = 0.0
            l2_dist = 0.0
            next_state = state.clone()
            action_taken = "invalidated"

        feasibility = ResidualFeasibilityRecord(
            is_feasible=is_feasible,
            policy=self.infeasible_policy,
            action_taken=action_taken,
            raw_deltas={self.target_resource: predicted_delta},
            applied_deltas={self.target_resource: applied_delta}
            if action_taken in ("accepted", "projected")
            else {},
            l2_projection_distance=l2_dist,
            violations=violations,
        )

        diag = _build_context_diagnostics(
            self._provenance,
            runtime_context,
            extra={
                "is_fitted": self.is_fitted,
                "weight": self.action_weight,
                "residual_feasibility": feasibility.model_dump(mode="json"),
            },
        )
        if action_taken == "abstained":
            diag["abstained"] = True
        elif action_taken == "invalidated":
            diag["invalidated"] = True

        return TransitionResult(
            next_state=next_state,
            applied_changes={
                "predicted_delta": predicted_delta,
                "applied_delta": applied_delta,
                "action_val": act_val,
            },
            evidence_level=EvidenceLevel.PREDICTIVE,
            diagnostics=diag,
            model_name=self.name,
        )
