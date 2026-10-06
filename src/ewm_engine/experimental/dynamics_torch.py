"""PyTorch neural residual dynamics baseline for learned transition modeling.

[EXPERIMENTAL] This module provides a neural network dynamics baseline conforming to
the LearnedDynamics protocol. It is intended for experimental evaluation and benchmarking
against structural ground truth models.

Requires the 'ml' optional extra: pip install 'ewm-engine[ml]'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.dynamics.learned import LearnedDynamics, TransitionDataset
from ewm_engine.exceptions import DynamicsError, SimulationConfigurationError
from ewm_engine.experimental.dynamics_eval import symlog
from ewm_engine.provenance.evidence import EvidenceLevel


def _load_torch() -> tuple[Any, Any, Any]:
    """Lazily import PyTorch modules, raising SimulationConfigurationError if absent."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        return torch, nn, optim
    except ImportError as exc:
        raise SimulationConfigurationError(
            "PyTorch is required for TorchNeuralResidualDynamics. "
            "Install with: pip install 'ewm-engine[ml]'"
        ) from exc


class TorchNeuralResidualDynamics(LearnedDynamics):
    """Neural residual transition model parameterized by a multi-layer perceptron (MLP).

    Conforms to the LearnedDynamics protocol. Predicts state deltas \\(\\Delta s_t\\) conditioned
    on current state \\(s_t\\) and actions \\(a_t\\):
        \\(\\hat{s}_{t+1} = \\text{clamp}(s_t + \\text{MLP}(\\text{symlog}([s_t, a_t])), [min, max])\\)

    Uses DreamerV3-style symlog transformations for scale-invariant representation
    across disparate organizational resource metrics. Outputs are clamped to declared
    resource capacity and non-negativity bounds.

    Evidence level is strictly PREDICTIVE.

    Determinism note:
        On CPU, training is deterministic given fixed seed. Multi-threaded BLAS or GPU
        execution may exhibit minor numerical drift (scoped determinism, ADR-012).
    """

    def __init__(
        self,
        target_resources: Sequence[ResourceId],
        action_types: Sequence[str] | str | None = None,
        action_param: str = "value",
        hidden_dims: tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        use_symlog: bool = True,
        weight_decay: float = 1e-4,
        seed: int = 42,
        name: str = "TorchNeuralResidualDynamics",
    ) -> None:
        self.target_resources = list(target_resources)
        if isinstance(action_types, str):
            self.action_types: list[str] | None = [action_types]
        elif action_types is not None:
            self.action_types = list(action_types)
        else:
            self.action_types = None
        self.action_param = action_param
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_symlog = use_symlog
        self.weight_decay = weight_decay
        self.seed = seed
        self.name = name

        self.is_fitted: bool = False
        self._model: Any = None
        self._final_loss: float | None = None

    def fit(self, dataset: TransitionDataset) -> None:
        """Fit neural network weights on observed transition dataset."""
        if len(dataset) == 0:
            return

        torch, nn, optim = _load_torch()

        # Enforce deterministic seed
        torch.manual_seed(self.seed)

        # Extract features
        states_np, actions_np, next_states_np = dataset.to_numpy(
            resource_ids=self.target_resources,
            action_type=self.action_types,
            action_param=self.action_param,
        )

        deltas_np = next_states_np - states_np

        if self.use_symlog:
            states_feat = symlog(states_np)
            actions_feat = symlog(actions_np)
        else:
            states_feat = states_np
            actions_feat = actions_np

        import numpy as np

        x_np = np.concatenate([states_feat, actions_feat], axis=1).astype(np.float32)
        y_np = deltas_np.astype(np.float32)

        x_tensor = torch.from_numpy(x_np)
        y_tensor = torch.from_numpy(y_np)

        in_dim = x_np.shape[1]
        out_dim = y_np.shape[1]

        # Build PyTorch sequential MLP
        layers: list[Any] = []
        prev_dim = in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, out_dim))

        model = nn.Sequential(*layers)
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()

        model.train()
        n_samples = len(x_tensor)
        n_batches = max(1, (n_samples + self.batch_size - 1) // self.batch_size)

        for _ in range(self.epochs):
            # Mini-batch shuffle
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for b in range(n_batches):
                indices = perm[b * self.batch_size : (b + 1) * self.batch_size]
                batch_x = x_tensor[indices]
                batch_y = y_tensor[indices]

                optimizer.zero_grad()
                pred_deltas = model(batch_x)
                loss = criterion(pred_deltas, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

        model.eval()
        self._model = model
        self._final_loss = epoch_loss / n_batches
        self.is_fitted = True

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        """Compute the next state transition using the fitted neural model."""
        if not self.is_fitted or self._model is None:
            raise DynamicsError(
                f"TorchNeuralResidualDynamics '{self.name}' must be fitted before calling transition."
            )

        torch, _, _ = _load_torch()
        import numpy as np

        d_res = len(self.target_resources)
        s_vec = np.zeros((1, d_res), dtype=np.float32)
        for j, rid in enumerate(self.target_resources):
            r = state.get_resource(rid)
            s_vec[0, j] = r.current if r is not None else 0.0

        if self.action_types is not None:
            a_vec = np.zeros((1, len(self.action_types)), dtype=np.float32)
            for k, atype in enumerate(self.action_types):
                act_val = 0.0
                for act in actions:
                    if act.type == atype:
                        val = act.parameters.get(self.action_param, 1.0)
                        try:
                            act_val += float(val)
                        except (ValueError, TypeError):
                            act_val += 1.0
                a_vec[0, k] = act_val
        else:
            a_vec = np.zeros((1, 1), dtype=np.float32)
            act_val = 0.0
            for act in actions:
                val = act.parameters.get(self.action_param, 1.0)
                try:
                    act_val += float(val)
                except (ValueError, TypeError):
                    act_val += 1.0
            a_vec[0, 0] = act_val

        if self.use_symlog:
            s_feat = symlog(s_vec)
            a_feat = symlog(a_vec)
        else:
            s_feat = s_vec
            a_feat = a_vec

        x_feat = np.concatenate([s_feat, a_feat], axis=1).astype(np.float32)
        x_tensor = torch.from_numpy(x_feat)

        with torch.no_grad():
            pred_deltas_tensor = self._model(x_tensor)
            pred_deltas = pred_deltas_tensor.numpy()[0]

        next_state = state
        applied_deltas: dict[str, float] = {}

        for j, rid in enumerate(self.target_resources):
            delta = float(pred_deltas[j])
            applied_deltas[rid] = delta
            # Respect declared bounds by enforcing clamp=True
            next_state = next_state.update_resource(rid, delta=delta, clamp=True)

        return TransitionResult(
            next_state=next_state,
            applied_changes={"predicted_deltas": applied_deltas},
            evidence_level=EvidenceLevel.PREDICTIVE,
            diagnostics={
                "is_fitted": self.is_fitted,
                "final_loss": self._final_loss,
                "epochs": self.epochs,
            },
            model_name=self.name,
        )
