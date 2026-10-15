"""Grey-box dynamics combining known symbolic rules with learned residual estimators.

[EXPERIMENTAL] Implements the Universal Differential Equation (UDE) framing for enterprise systems:
    s_{t+1} = f_{known}(s_t, a_t, e_t) + delta_{learned}(s_t, a_t, e_t)

Key invariants:
- Zero-residual reduction: If the learned residual is zero or absent, the grey-box reduces
  bit-for-bit to the symbolic kernel, producing identical cryptographic state fingerprints (S00 Invariant 5).
- Residual-only learning: Machine learning models predict ONLY the discrepancy over explicit
  known rules, never hallucinating full ungrounded next-states.
- Architectural conservation: Residual deltas pass through an optional ConservationProjectionLayer
  (KKT-hPINN / Port-Hamiltonian projection) guaranteeing zero violation of declared physical/accounting laws.
- Epistemic integrity: Pure known executions retain STRUCTURAL evidence; applying a learned
  residual sets evidence level to PREDICTIVE.

Anchor papers:
- Rackauckas et al. "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385
- KKT-hPINN: Hard Linear Equality Constraints for Neural Networks, arXiv:2402.07251
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator, ResourceId
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.dynamics.learned import TransitionDataset
from ewm_engine.experimental.conservation import ConservationProjectionLayer
from ewm_engine.provenance.evidence import EvidenceLevel


@runtime_checkable
class ResidualEstimator(Protocol):
    """Protocol for models predicting residuals over known symbolic dynamics."""

    def predict_residual(
        self,
        state: WorldState,
        actions: Sequence[Action],
        events: Sequence[ExogenousEvent],
        known_next_state: WorldState,
    ) -> dict[ResourceId, float]:
        """Predict the resource deltas correcting known dynamics toward reality."""
        ...

    def fit(self, dataset: TransitionDataset, known_model: DynamicsModel) -> None:
        """Fit parameters on the residual discrepancy: delta = s_{observed} - s_{known}."""
        ...


class ZeroResidualEstimator:
    """Null estimator predicting zero residual everywhere.

    Guarantees strict bit-identical reduction of the grey-box to the symbolic kernel.
    """

    def predict_residual(
        self,
        state: WorldState,
        actions: Sequence[Action],
        events: Sequence[ExogenousEvent],
        known_next_state: WorldState,
    ) -> dict[ResourceId, float]:
        return {}

    def fit(self, dataset: TransitionDataset, known_model: DynamicsModel) -> None:
        pass


class CallableResidualEstimator:
    """Wraps a pure Python callable for lightweight testing and custom residual policies."""

    def __init__(
        self,
        fn: Callable[[WorldState, Sequence[Action], Sequence[ExogenousEvent], WorldState], dict[ResourceId, float]],
    ) -> None:
        self._fn = fn

    def predict_residual(
        self,
        state: WorldState,
        actions: Sequence[Action],
        events: Sequence[ExogenousEvent],
        known_next_state: WorldState,
    ) -> dict[ResourceId, float]:
        return self._fn(state, actions, events, known_next_state)

    def fit(self, dataset: TransitionDataset, known_model: DynamicsModel) -> None:
        pass


class LinearResidualEstimator:
    """Linear regression baseline fitting residual discrepancies on state/action features.

    Learns weights W and bias b such that:
        residual = W @ [state_features, action_features] + b
    where training targets are:
        target_residual = s_{sample.next_state} - s_{known.next_state}
    """

    def __init__(
        self,
        target_resources: Sequence[ResourceId],
        action_types: Sequence[str] | str | None = None,
        action_param: str = "value",
        name: str = "LinearResidualEstimator",
    ) -> None:
        self.target_resources = tuple(target_resources)
        if isinstance(action_types, str):
            self.action_types: tuple[str, ...] | None = (action_types,)
        elif action_types is not None:
            self.action_types = tuple(action_types)
        else:
            self.action_types = None
        self.action_param = action_param
        self.name = name

        self.is_fitted: bool = False
        self._weights: np.ndarray | None = None
        self._bias: np.ndarray | None = None

    def fit(self, dataset: TransitionDataset, known_model: DynamicsModel) -> None:
        """Fit weights on the residual between dataset next_states and known_model next_states."""
        if len(dataset) == 0:
            return

        dummy_rng = np.random.default_rng(0)
        x_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []

        for sample in dataset:
            # Compute known model's prediction for this sample
            known_res = known_model.transition(
                state=sample.state,
                actions=sample.actions,
                exogenous_events=sample.events,
                rng=dummy_rng,
            )

            # Features: state resource levels + action values
            feat_s = [
                sample.state.get_resource(rid).current if rid in sample.state.resources else 0.0
                for rid in self.target_resources
            ]
            feat_a: list[float] = []
            if self.action_types is not None:
                for atype in self.action_types:
                    val = 0.0
                    for act in sample.actions:
                        if act.type == atype:
                            val += float(act.get(self.action_param, 1.0))
                    feat_a.append(val)
            else:
                val = 0.0
                for act in sample.actions:
                    val += float(act.get(self.action_param, 1.0))
                feat_a.append(val)

            feat = np.array(feat_s + feat_a, dtype=np.float64)

            # Target: actual next_state - known next_state
            residuals: list[float] = []
            for rid in self.target_resources:
                actual = (
                    sample.next_state.get_resource(rid).current
                    if rid in sample.next_state.resources
                    else 0.0
                )
                pred_known = (
                    known_res.next_state.get_resource(rid).current
                    if rid in known_res.next_state.resources
                    else 0.0
                )
                residuals.append(actual - pred_known)

            x_rows.append(feat)
            y_rows.append(np.array(residuals, dtype=np.float64))

        X = np.array(x_rows, dtype=np.float64)
        Y = np.array(y_rows, dtype=np.float64)

        # Add intercept column
        n_samples, n_feats = X.shape
        X_design = np.hstack([X, np.ones((n_samples, 1), dtype=np.float64)])

        # Ridge / OLS fit: (X^T X + alpha I)^{-1} X^T Y
        alpha = 1e-6
        reg = alpha * np.eye(n_feats + 1)
        reg[-1, -1] = 0.0  # Do not regularize bias
        coeffs = np.linalg.pinv(X_design.T @ X_design + reg) @ (X_design.T @ Y)

        self._weights = coeffs[:-1, :].T  # shape (n_targets, n_feats)
        self._bias = coeffs[-1, :]  # shape (n_targets,)
        self.is_fitted = True

    def predict_residual(
        self,
        state: WorldState,
        actions: Sequence[Action],
        events: Sequence[ExogenousEvent],
        known_next_state: WorldState,
    ) -> dict[ResourceId, float]:
        if not self.is_fitted or self._weights is None or self._bias is None:
            return {}

        feat_s = [
            state.get_resource(rid).current if rid in state.resources else 0.0
            for rid in self.target_resources
        ]
        feat_a: list[float] = []
        if self.action_types is not None:
            for atype in self.action_types:
                val = 0.0
                for act in actions:
                    if act.type == atype:
                        val += float(act.get(self.action_param, 1.0))
                feat_a.append(val)
        else:
            val = 0.0
            for act in actions:
                val += float(act.get(self.action_param, 1.0))
            feat_a.append(val)

        feat = np.array(feat_s + feat_a, dtype=np.float64)
        pred = (self._weights @ feat) + self._bias

        return {rid: float(pred[i]) for i, rid in enumerate(self.target_resources)}


class GreyBoxDynamics:
    """Universal Differential Equation (UDE) grey-box dynamics model.

    Orchestrates the hybrid combination:
        s_{t+1} = f_{known}(s_t, a_t, e_t) + Proj(delta_{learned}(s_t, a_t, e_t))

    When residual_estimator is None or predicts zeros, this model strictly reduces
    to the symbolic kernel f_{known}, producing bit-for-bit identical state hashes and fingerprints.
    """

    def __init__(
        self,
        known_model: DynamicsModel,
        residual_estimator: ResidualEstimator | None = None,
        conservation_layer: ConservationProjectionLayer | None = None,
        name: str = "GreyBoxDynamics",
    ) -> None:
        self.known_model = known_model
        self.residual_estimator = residual_estimator
        self.conservation_layer = conservation_layer
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        # Step 1: Compute known symbolic transition
        known_result = self.known_model.transition(
            state=state,
            actions=actions,
            exogenous_events=exogenous_events,
            rng=rng,
        )

        # Fast path & invariant 5 preservation: if no residual model, return exact known result
        if self.residual_estimator is None:
            return known_result

        # Step 2: Query learned residual model
        raw_residuals = self.residual_estimator.predict_residual(
            state=state,
            actions=actions,
            events=exogenous_events,
            known_next_state=known_result.next_state,
        )

        # Check for strict zero residual
        non_zero_residuals = {
            rid: delta for rid, delta in raw_residuals.items() if abs(delta) > 1e-12
        }
        if not non_zero_residuals:
            # Bit-for-bit fingerprint-identical to known result
            return known_result

        # Step 3: Pass residual through architectural conservation layer (if present)
        if self.conservation_layer is not None:
            projected_residuals = self.conservation_layer.project_deltas(
                current_state=known_result.next_state,
                raw_deltas=non_zero_residuals,
            )
        else:
            projected_residuals = non_zero_residuals

        # Step 4: Apply projected deltas to known next state
        next_state = known_result.next_state
        applied_deltas: dict[str, float] = {}

        for rid, delta in projected_residuals.items():
            if abs(delta) > 1e-12 and rid in next_state.resources:
                applied_deltas[rid] = delta
                next_state = next_state.update_resource(rid, delta=delta, clamp=True)

        # If clamping or projection resulted in no effective change, maintain bit identity
        if not applied_deltas:
            return known_result

        combined_changes = dict(known_result.applied_changes)
        combined_changes["known_model"] = known_result.model_name
        combined_changes["learned_residual"] = applied_deltas
        combined_changes["raw_residual"] = non_zero_residuals

        combined_diagnostics = dict(known_result.diagnostics)
        combined_diagnostics["has_conservation_projection"] = self.conservation_layer is not None
        combined_diagnostics["residual_estimator"] = type(self.residual_estimator).__name__

        return TransitionResult(
            next_state=next_state,
            applied_changes=combined_changes,
            evidence_level=EvidenceLevel.PREDICTIVE,
            diagnostics=combined_diagnostics,
            model_name=self.name,
        )
