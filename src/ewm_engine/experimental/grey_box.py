"""Grey-box dynamics combining known symbolic rules with learned residual estimators.

[EXPERIMENTAL] Implements the Universal Differential Equation (UDE) framing for enterprise systems:
    s_{t+1} = f_{known}(s_t, a_t, e_t | K_t) + Proj_C(delta_{learned}(s_t, a_t, e_t | K_t))

Key invariants:
- Zero-residual reduction: If the learned residual is zero or absent, the grey-box reduces
  bit-for-bit to the symbolic kernel, producing identical cryptographic state fingerprints (S00 Invariant 5).
- Residual-only learning: Machine learning models predict ONLY the discrepancy over explicit
  known rules, never hallucinating full ungrounded next-states.
- Architectural conservation & no silent repair: Residual deltas pass through an optional
  ConservationProjectionLayer and explicit feasibility checks against resource bounds and hard
  constraints. Infeasible proposals are explicitly projected, rejected, abstained, or invalidated
  per InfeasibleResidualPolicy — never silently repaired.
- Epistemic integrity: Pure known executions retain STRUCTURAL evidence; applying a learned
  residual sets evidence level to PREDICTIVE.

Anchor papers:
- Rackauckas et al. "Universal Differential Equations for Scientific Machine Learning", arXiv:2001.04385
- KKT-hPINN: Hard Linear Equality Constraints for Neural Networks, arXiv:2402.07251
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ewm_engine.constraints.base import Constraint
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
    ValidityDomain,
)
from ewm_engine.dynamics.deterministic import _build_context_diagnostics
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
        fn: Callable[
            [WorldState, Sequence[Action], Sequence[ExogenousEvent], WorldState],
            dict[ResourceId, float],
        ],
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


def _invoke_dynamics(
    model: DynamicsModel,
    state: WorldState,
    actions: Sequence[Action],
    exogenous_events: Sequence[ExogenousEvent],
    rng: RandomGenerator,
    runtime_context: ContextSnapshot | None,
) -> TransitionResult:
    """Invoke a DynamicsModel, passing runtime_context if supported."""
    if runtime_context is not None:
        try:
            sig = inspect.signature(model.transition)
            if "runtime_context" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                fn: Any = model.transition
                res: TransitionResult = fn(
                    state=state,
                    actions=actions,
                    exogenous_events=exogenous_events,
                    rng=rng,
                    runtime_context=runtime_context,
                )
                return res
        except (ValueError, TypeError):
            pass
    return model.transition(
        state=state,
        actions=actions,
        exogenous_events=exogenous_events,
        rng=rng,
    )


class GreyBoxDynamics:
    """Universal Differential Equation (UDE) constraint-preserving hybrid dynamics model.

    Orchestrates the hybrid combination:
        s_{t+1} = f_{known}(s_t, a_t, e_t | K_t) + Proj_C(delta_{learned}(s_t, a_t, e_t | K_t))

    When residual_estimator is None or predicts zeros, this model strictly reduces
    to the symbolic kernel f_{known}, producing bit-for-bit identical state hashes and fingerprints.
    When the learned residual proposes an infeasible transition (violating conservation laws,
    resource capacity bounds, or explicit hard constraints), silent repair is forbidden:
    the proposal is explicitly projected, rejected, abstained, or invalidated according to
    ``infeasible_policy`` and logged in ``TransitionResult.residual_feasibility``.
    """

    def __init__(
        self,
        known_model: DynamicsModel,
        residual_estimator: ResidualEstimator | None = None,
        conservation_layer: ConservationProjectionLayer | None = None,
        name: str = "GreyBoxDynamics",
        *,
        infeasible_policy: InfeasibleResidualPolicy = InfeasibleResidualPolicy.PROJECT,
        constraints: Sequence[Constraint] = (),
        version: str = "1.0.0",
        required_context_keys: Sequence[str] | None = None,
        validity_domain: ValidityDomain | None = None,
    ) -> None:
        self.known_model = known_model
        self.residual_estimator = residual_estimator
        self.conservation_layer = conservation_layer
        self.name = name
        self.infeasible_policy = infeasible_policy
        self.constraints = tuple(constraints)

        known_prov = getattr(known_model, "mechanism_provenance", None)
        if required_context_keys is not None:
            req_keys = tuple(required_context_keys)
        elif isinstance(known_prov, MechanismProvenance):
            req_keys = known_prov.required_context_keys
        else:
            req_keys = ()

        invariants: list[str] = ["resource_bounds"]
        if conservation_layer is not None:
            invariants.append("linear_conservation_laws")
        for c in self.constraints:
            invariants.append(f"constraint:{c.constraint_id}")

        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.HYBRID,
            validity_domain=validity_domain or ValidityDomain(),
            required_context_keys=req_keys,
            evidence_level=(
                EvidenceLevel.STRUCTURAL if residual_estimator is None else EvidenceLevel.PREDICTIVE
            ),
            source=f"hybrid({getattr(known_model, 'name', type(known_model).__name__)} + {type(residual_estimator).__name__ if residual_estimator else 'None'})",
            uncertainty_contract="hybrid_structural_plus_residual",
            invariants_affected=tuple(invariants),
            failure_semantics=infeasible_policy,
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        """Return formal mechanism provenance metadata."""
        return self._provenance

    def _evaluate_constraints(
        self,
        candidate_state: WorldState,
        actions: Sequence[Action],
    ) -> list[str]:
        from ewm_engine.constraints.results import ConstraintPhase

        violations: list[str] = []
        for constraint in self.constraints:
            res = constraint.evaluate(
                state=candidate_state,
                actions=actions,
                phase=ConstraintPhase.POST_TRANSITION,
            )
            if not res.satisfied:
                violations.append(f"constraint_violation:{res.constraint_id}:{res.message}")
        return violations

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        # Step 1: Compute known symbolic transition
        known_result = _invoke_dynamics(
            self.known_model,
            state=state,
            actions=actions,
            exogenous_events=exogenous_events,
            rng=rng,
            runtime_context=runtime_context,
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

        # Filter to target resources present in state with non-zero delta
        non_zero_residuals = {
            rid: float(delta)
            for rid, delta in raw_residuals.items()
            if abs(float(delta)) > 1e-12 and rid in known_result.next_state.resources
        }
        if not non_zero_residuals:
            # Bit-for-bit fingerprint-identical to known result
            return known_result

        base_state = known_result.next_state
        violations: list[str] = []

        # Step 3: Check conservation law feasibility
        if self.conservation_layer is not None:
            conserved_residuals = self.conservation_layer.project_deltas(
                current_state=base_state,
                raw_deltas=non_zero_residuals,
            )
            for rid, raw_d in non_zero_residuals.items():
                proj_d = conserved_residuals.get(rid, 0.0)
                if abs(proj_d - raw_d) > 1e-9:
                    violations.append(
                        f"conservation_law_violation:{rid}(raw={raw_d:.6f},projected={proj_d:.6f})"
                    )
        else:
            conserved_residuals = dict(non_zero_residuals)

        # Step 4: Check resource capacity bounds on the raw proposal
        for rid, raw_d in non_zero_residuals.items():
            res = base_state.get_resource(rid)
            proposed_val = res.current + raw_d
            if proposed_val < res.min_value - 1e-9 or proposed_val > res.max_value + 1e-9:
                violations.append(
                    f"resource_bounds_violation:{rid}"
                    f"({proposed_val:.6f} not in [{res.min_value}, {res.max_value}])"
                )

        # Step 5: Check explicit post-transition constraints if raw bounds were satisfied
        if not violations and self.constraints:
            raw_candidate = base_state
            for rid, raw_d in non_zero_residuals.items():
                raw_candidate = raw_candidate.update_resource(
                    rid, delta=raw_d, clamp=False, enforce_bounds=True
                )
            violations.extend(
                self._evaluate_constraints(
                    candidate_state=raw_candidate,
                    actions=actions,
                )
            )

        is_feasible = len(violations) == 0

        # Step 6: Execute explicit feasibility policy (never silent clamping)
        if is_feasible:
            next_state = base_state
            applied_deltas: dict[str, float] = {}
            for rid, raw_d in non_zero_residuals.items():
                applied_deltas[rid] = raw_d
                next_state = next_state.update_resource(
                    rid, delta=raw_d, clamp=False, enforce_bounds=True
                )
            action_taken = "accepted"
            l2_dist = 0.0
        elif self.infeasible_policy == InfeasibleResidualPolicy.PROJECT:
            next_state = base_state
            applied_deltas = {}
            for rid, cons_d in conserved_residuals.items():
                if rid not in next_state.resources:
                    continue
                res = next_state.get_resource(rid)
                proj_val = max(res.min_value, min(res.current + cons_d, res.max_value))
                eff_d = proj_val - res.current
                applied_deltas[rid] = eff_d
                next_state = next_state.update_resource(
                    rid, delta=eff_d, clamp=False, enforce_bounds=True
                )

            l2_dist = math.sqrt(
                sum(
                    (applied_deltas.get(rid, 0.0) - raw_d) ** 2
                    for rid, raw_d in non_zero_residuals.items()
                )
            )

            # Verify explicit hard constraints after projection
            post_proj_violations = self._evaluate_constraints(
                candidate_state=next_state,
                actions=actions,
            )
            if post_proj_violations:
                violations.extend(post_proj_violations)
                next_state = base_state
                applied_deltas = {}
                l2_dist = math.sqrt(sum(d * d for d in non_zero_residuals.values()))
                action_taken = "rejected"
            else:
                action_taken = "projected"
        elif self.infeasible_policy == InfeasibleResidualPolicy.REJECT:
            next_state = base_state
            applied_deltas = {}
            l2_dist = math.sqrt(sum(d * d for d in non_zero_residuals.values()))
            action_taken = "rejected"
        elif self.infeasible_policy == InfeasibleResidualPolicy.ABSTAIN:
            next_state = state.clone()
            applied_deltas = {}
            l2_dist = 0.0
            action_taken = "abstained"
        else:
            next_state = base_state
            applied_deltas = {}
            l2_dist = 0.0
            action_taken = "invalidated"

        feasibility_record = ResidualFeasibilityRecord(
            is_feasible=is_feasible,
            policy=self.infeasible_policy,
            action_taken=action_taken,
            raw_deltas=non_zero_residuals,
            applied_deltas=applied_deltas,
            l2_projection_distance=l2_dist,
            violations=tuple(violations),
        )

        combined_changes = dict(known_result.applied_changes)
        combined_changes["known_model"] = known_result.model_name
        combined_changes["learned_residual"] = applied_deltas
        combined_changes["raw_residual"] = non_zero_residuals
        combined_changes["residual_action"] = action_taken

        combined_diagnostics = _build_context_diagnostics(
            self._provenance,
            runtime_context,
            extra=dict(known_result.diagnostics),
        )
        combined_diagnostics["has_conservation_projection"] = self.conservation_layer is not None
        combined_diagnostics["residual_estimator"] = type(self.residual_estimator).__name__
        combined_diagnostics["residual_feasibility"] = feasibility_record.model_dump(mode="json")
        if action_taken == "abstained":
            combined_diagnostics["abstained"] = True
        elif action_taken == "invalidated":
            combined_diagnostics["invalidated"] = True

        return TransitionResult(
            next_state=next_state,
            applied_changes=combined_changes,
            evidence_level=EvidenceLevel.PREDICTIVE,
            diagnostics=combined_diagnostics,
            model_name=self.name,
        )


HybridDynamics = GreyBoxDynamics
