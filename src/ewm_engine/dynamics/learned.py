"""Machine learning extension interfaces and empirical dynamics protocols.

[EXPERIMENTAL] Learned dynamics models and residual estimators are currently experimental.
They require explicit verification against structural conservation laws before deployment.
See ewm_engine.experimental for experimental re-exports.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
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

    def __init__(self, samples: Sequence[TransitionSample] | None = None) -> None:
        self._samples: list[TransitionSample] = list(samples or [])

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

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> tuple[TransitionDataset, TransitionDataset]:
        """Split dataset into train and test subsets deterministically.

        Args:
            train_ratio: Fraction of samples assigned to training (0.0 to 1.0).
            seed: Deterministic random seed for shuffling.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        import numpy as np

        n = len(self._samples)
        if n == 0:
            return TransitionDataset(), TransitionDataset()

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)
        split_idx = int(n * train_ratio)

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        train_samples = [self._samples[i] for i in train_indices]
        test_samples = [self._samples[i] for i in test_indices]
        return TransitionDataset(train_samples), TransitionDataset(test_samples)

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
    """

    def __init__(
        self,
        target_resource: ResourceId,
        action_type: str,
        name: str = "LinearResidualDynamics",
    ) -> None:
        self.target_resource = target_resource
        self.action_type = action_type
        self.name = name
        self.action_weight: float = 1.0
        self.bias: float = 0.0
        self.is_fitted: bool = False

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

        self.is_fitted = True

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        act_val = 0.0
        for act in actions:
            if act.type == self.action_type:
                act_val += float(act.get("value", 1.0))

        predicted_delta = (self.action_weight * act_val) + self.bias
        next_state = state.update_resource(self.target_resource, delta=predicted_delta, clamp=True)

        return TransitionResult(
            next_state=next_state,
            applied_changes={"predicted_delta": predicted_delta, "action_val": act_val},
            evidence_level=EvidenceLevel.PREDICTIVE,
            diagnostics={"is_fitted": self.is_fitted, "weight": self.action_weight},
            model_name=self.name,
        )
