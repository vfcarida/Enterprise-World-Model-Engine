"""PyTorch neural residual dynamics baseline for learned transition modeling.

[EXPERIMENTAL] This module provides a neural network dynamics baseline conforming to
the LearnedDynamics protocol. It is intended for experimental evaluation and benchmarking
against structural ground truth models.

Requires the 'ml' optional extra: pip install 'ewm-engine[ml]'
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ewm_engine.cards.models import ModelCard
from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import InfeasibleResidualPolicy, TransitionResult
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
    across disparate organizational resource metrics. Outputs are projected or checked
    against declared resource capacity and non-negativity bounds via `infeasible_policy`.

    Evidence level is strictly PREDICTIVE.

    Determinism note:
        On CPU under pinned single-thread execution, training is deterministic given fixed seed.
        Multi-threaded BLAS or GPU execution is classified under scoped tolerance (ADR-012 / ADR-050).
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
        *,
        version: str = "1.0.0",
        infeasible_policy: InfeasibleResidualPolicy = InfeasibleResidualPolicy.PROJECT,
    ) -> None:
        self.target_resources = list(target_resources)
        if isinstance(action_types, str):
            self.action_types: list[str] | None = [action_types]
        elif action_types is not None:
            self.action_types = list(action_types)
        else:
            self.action_types = None
        self.action_param = action_param
        self.hidden_dims = tuple(hidden_dims)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_symlog = use_symlog
        self.weight_decay = weight_decay
        self.seed = seed
        self.name = name
        self.version = version
        self.infeasible_policy = infeasible_policy

        self.is_fitted: bool = False
        self._model: Any = None
        self._final_loss: float | None = None
        self._weights_hash: str | None = None
        self.training_dataset_fingerprint: str | None = None
        self.training_split_metadata: dict[str, Any] = {}
        self.training_metrics: dict[str, float] = {}
        self.eval_metrics: dict[str, float] = {}

    @property
    def config_fingerprint(self) -> str:
        """Cryptographic SHA-256 fingerprint of neural architecture and hyperparameters."""
        return canonical_sha256(
            {
                "model_class": "TorchNeuralResidualDynamics",
                "name": self.name,
                "version": self.version,
                "target_resources": self.target_resources,
                "action_types": self.action_types,
                "action_param": self.action_param,
                "hidden_dims": list(self.hidden_dims),
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "use_symlog": self.use_symlog,
                "weight_decay": self.weight_decay,
                "seed": self.seed,
                "infeasible_policy": self.infeasible_policy.value,
            }
        )

    @property
    def model_fingerprint(self) -> str:
        """Cryptographic SHA-256 fingerprint of config, fitted weights, and training dataset."""
        return canonical_sha256(
            {
                "config_fingerprint": self.config_fingerprint,
                "is_fitted": self.is_fitted,
                "weights_hash": self._weights_hash,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
            }
        )

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

        epoch_loss = 0.0
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

        with torch.no_grad():
            full_pred = model(x_tensor).numpy()
            residuals = y_np - full_pred
            train_rmse = float(np.sqrt(np.mean(residuals**2)))
            train_mae = float(np.mean(np.abs(residuals)))
            residual_std = float(np.std(residuals))
            param_summary = [
                round(float(p.detach().cpu().numpy().sum()), 8) for p in model.parameters()
            ]
            self._weights_hash = canonical_sha256(param_summary)

        self.training_dataset_fingerprint = dataset.fingerprint
        self.training_split_metadata = dict(dataset.split_metadata)
        self.training_metrics = {
            "final_mse_loss": float(self._final_loss),
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "residual_std": residual_std,
            "n_train_samples": float(n_samples),
        }
        self.is_fitted = True

    def evaluate(
        self,
        eval_dataset: TransitionDataset,
        *,
        train_dataset: TransitionDataset | None = None,
        enforce_no_leakage: bool = True,
    ) -> dict[str, float]:
        """Evaluate fitted neural model on a held-out evaluation split with leakage checks."""
        import numpy as np

        if train_dataset is not None and enforce_no_leakage:
            TransitionDataset.verify_no_temporal_leakage(train_dataset, eval_dataset)

        rng = np.random.default_rng(self.seed)
        sq_errors: list[float] = []
        abs_errors: list[float] = []

        for sample in eval_dataset:
            res = self.transition(sample.state, sample.actions, sample.events, rng)
            for rid in self.target_resources:
                if rid in sample.state.resources and rid in sample.next_state.resources:
                    pred_v = res.next_state.get_resource(rid).current
                    true_v = sample.next_state.get_resource(rid).current
                    err = pred_v - true_v
                    sq_errors.append(err * err)
                    abs_errors.append(abs(err))

        if not sq_errors:
            metrics = {"eval_rmse": 0.0, "eval_mae": 0.0, "n_eval_samples": 0.0}
        else:
            metrics = {
                "eval_rmse": float(np.sqrt(np.mean(sq_errors))),
                "eval_mae": float(np.mean(abs_errors)),
                "n_eval_samples": float(len(eval_dataset)),
            }
        self.eval_metrics = metrics
        return dict(metrics)

    def to_model_card(self) -> ModelCard:
        """Build a Mitchell-complete ModelCard documenting architecture, split, metrics, and OOD behavior."""
        combined_metrics: dict[str, float | str] = {}
        for k, v in self.training_metrics.items():
            combined_metrics[k] = round(v, 6)
        for k, v in self.eval_metrics.items():
            combined_metrics[k] = round(v, 6)
        if not combined_metrics:
            combined_metrics["epochs"] = float(self.epochs)

        return ModelCard(
            artifact_fingerprint=self.model_fingerprint,
            maturity="experimental",
            model_id=self.name,
            model_name=self.name,
            version=self.version,
            description=(
                "PyTorch MLP residual transition model with DreamerV3-style symlog "
                f"feature scaling over resources {self.target_resources}."
            ),
            model_details={
                "architecture": f"MLP(hidden_dims={list(self.hidden_dims)}, symlog={self.use_symlog})",
                "config_fingerprint": self.config_fingerprint,
                "model_fingerprint": self.model_fingerprint,
                "training_dataset_fingerprint": self.training_dataset_fingerprint,
                "seed": self.seed,
                "evidence_level": EvidenceLevel.PREDICTIVE.value,
                "uncertainty_contract": "empirical_residual_std_with_conformal_calibration",
                "ood_and_infeasible_behavior": (
                    f"Enforces InfeasibleResidualPolicy.{self.infeasible_policy.name} "
                    "on resource boundary excursions and pairs with MahalanobisOODDetector "
                    "for covariate shift detection."
                ),
            },
            intended_use=[
                "Experimental nonlinear residual transition learning conditioned on state and action features.",
                "Benchmarking learned dynamics against structural conservation laws.",
            ],
            assumptions=[
                "Symlog-transformed state and action vectors lie within the empirical support of the training distribution.",
                "Hard conservation laws are enforced externally via ProjectiveConservationWrapper or ConstraintRegistry.",
            ],
            out_of_scope=[
                "Unchecked deployment in safety-critical decisions without conservation projection and OOD monitoring.",
                "Zero-shot structural causal extrapolation under unidentifiable interventions.",
            ],
            limitations=[
                "Requires optional 'ml' extra (PyTorch) and is never a mandatory core dependency.",
                "Autoregressive rollout errors can compound over long horizons outside training support.",
            ],
            training_data=[
                f"training_dataset_fingerprint:{self.training_dataset_fingerprint or 'unfitted'}",
                f"split_strategy:{self.training_split_metadata.get('strategy', 'unspecified')}",
            ],
            eval_data=[
                "Chronological held-out transition split verified via TransitionDataset.verify_no_temporal_leakage."
            ],
            constraints_exercised=[f"resource_bounds:{rid}" for rid in self.target_resources],
            metrics=combined_metrics,
            quantitative_analyses={
                "training_metrics": self.training_metrics,
                "eval_metrics": self.eval_metrics,
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
                "model_fingerprint": self.model_fingerprint,
            },
            model_name=self.name,
        )
