"""Hybrid Transition Router (`Pre-rules -> ML -> Post-rules`) and Causal Transition Engine.

Separates deterministic structural compliance rules from uncertain learned predictions,
enforces pre-post numerical `NaN`/`Inf` safeguards, rejects 100% of out-of-bounds or
invalid ML hallucinations via structural rollback, and implements Pearl's `do(X_t = x)`
graph surgery via `engine.branch().do(action)` and `Intervention` context managers.

Scientific Grounding:
    - Peters, J., Janzing, D., & Schölkopf, B. (2017): "Elements of Causal Inference:
      Foundations and Learning Algorithms", MIT Press.
    - Pearl, J. (2009): "Causality: Models, Reasoning, and Inference", Cambridge University Press.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.causal.intervention import (
    CausalExecutionLogEntry,
    InterventionContext,
    get_active_interventions,
    normalize_node_name,
    parse_intervention_targets,
    sever_inbound_edges,
)
from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.scm import CausalGraph
from ewm_engine.core.state import WorldState
from ewm_engine.core.time import EventLog
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import (
    DynamicsModel,
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    ResidualFeasibilityRecord,
    TransitionResult,
)
from ewm_engine.provenance.evidence import EvidenceLevel


class NumericalInstabilityError(ValueError):
    """Raised when a learned ML module emits numerical `NaN` or `Inf` before structural evaluation."""


class CalibratedPrediction(BaseModel):
    """Output produced by a learned ML predictor before post-rule structural verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Predicted resource deltas indexed by resource ID.",
    )
    std_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated predictive standard deviations indexed by resource ID.",
    )
    proposed_actions: tuple[Action, ...] = Field(
        default_factory=tuple,
        description="Optional actions proposed/hallucinated by the learned policy or dynamics.",
    )
    calibration_temperature: float = Field(
        default=1.0,
        description="Temperature scaling factor (> 0) applied to predictive uncertainty.",
    )


class CalibratedStateDistribution(BaseModel):
    """Calibrated predictive distribution over next-step world state resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated posterior/predictive mean per resource ID.",
    )
    std: dict[str, float] = Field(
        default_factory=dict,
        description="Calibrated predictive standard deviation per resource ID.",
    )
    lower_95: dict[str, float] = Field(
        default_factory=dict,
        description="Lower 95% calibrated predictive interval bound per resource ID.",
    )
    upper_95: dict[str, float] = Field(
        default_factory=dict,
        description="Upper 95% calibrated predictive interval bound per resource ID.",
    )
    quantiles: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Calibrated predictive quantiles (p05, p50, p95) per resource ID.",
    )
    calibration_temperature: float = Field(
        default=1.0,
        gt=0.0,
        description="Temperature scaling factor used to calibrate predictive intervals.",
    )
    is_calibrated: bool = Field(
        default=True,
        description="True when predictive intervals are calibrated.",
    )

    def sample(self, rng: RandomGenerator | None = None) -> dict[str, float]:
        """Draw a single sample from the calibrated Gaussian predictive distribution."""
        gen = rng if rng is not None else np.random.default_rng(0)
        drawn: dict[str, float] = {}
        for rid, mu in self.mean.items():
            sigma = self.std.get(rid, 0.0)
            if sigma > 0.0:
                drawn[rid] = float(gen.normal(loc=mu, scale=sigma))
            else:
                drawn[rid] = float(mu)
        return drawn


@runtime_checkable
class LearnedPredictorProtocol(Protocol):
    """Protocol for learned ML modules wrapped by `HybridTransitionRouter`."""

    def predict(
        self,
        state: WorldState,
        actions: Sequence[Action],
        structural_state: WorldState,
    ) -> CalibratedPrediction | Mapping[str, float]:
        """Predict resource deltas (or a CalibratedPrediction) given state and actions."""
        ...


StructuralPreRuleFn = Callable[[WorldState, Sequence[Action]], tuple[bool, str]]
StructuralPostRuleFn = Callable[[WorldState, WorldState, Sequence[Action]], tuple[bool, str]]
StructuralEquationFn = Callable[
    [Mapping[str, float], Sequence[Action], Mapping[str, float]],
    float,
]


def _check_finite_number(value: Any, label: str) -> float:
    """Validate that a learned numeric output is neither NaN nor Inf before structural evaluation."""
    try:
        fval = float(value)
    except (TypeError, ValueError) as exc:
        raise NumericalInstabilityError(
            f"Learned module produced non-numeric value for {label!r}: {value!r}"
        ) from exc
    if math.isnan(fval) or math.isinf(fval):
        raise NumericalInstabilityError(
            f"Numerical instability detected in learned module before structural evaluation: "
            f"{label}={fval}"
        )
    return fval


def validate_learned_prediction_no_nan(
    prediction: CalibratedPrediction | Mapping[str, Any],
) -> CalibratedPrediction:
    """Inspect a learned prediction for `NaN`/`Inf` before structural evaluation.

    Raises:
        NumericalInstabilityError: If any delta, std, temperature, or action parameter is NaN/Inf.
    """
    if isinstance(prediction, CalibratedPrediction):
        clean_means: dict[str, float] = {}
        for k, v in prediction.mean_deltas.items():
            clean_means[str(k)] = _check_finite_number(v, f"mean_deltas[{k!r}]")
        clean_stds: dict[str, float] = {}
        for k, v in prediction.std_deltas.items():
            sval = _check_finite_number(v, f"std_deltas[{k!r}]")
            if sval < 0.0:
                raise NumericalInstabilityError(
                    f"Learned module returned negative std for {k!r}: {sval}"
                )
            clean_stds[str(k)] = sval
        temp = _check_finite_number(
            prediction.calibration_temperature, "calibration_temperature"
        )
        if temp <= 0.0:
            raise NumericalInstabilityError(
                f"calibration_temperature must be > 0, got {temp}"
            )
        for act in prediction.proposed_actions:
            for pk, pv in act.parameters.items():
                if isinstance(pv, (int, float)):
                    _check_finite_number(pv, f"proposed_action[{act.type}].{pk}")
        return CalibratedPrediction(
            mean_deltas=clean_means,
            std_deltas=clean_stds,
            proposed_actions=prediction.proposed_actions,
            calibration_temperature=temp,
        )

    if isinstance(prediction, Mapping):
        raw_means: dict[str, float] = {}
        raw_stds: dict[str, float] = {}
        proposed_acts: list[Action] = []
        temp_val = 1.0

        if "mean_deltas" in prediction and isinstance(prediction["mean_deltas"], Mapping):
            for k, v in prediction["mean_deltas"].items():
                raw_means[str(k)] = _check_finite_number(v, f"mean_deltas[{k!r}]")
            if "std_deltas" in prediction and isinstance(prediction["std_deltas"], Mapping):
                for k, v in prediction["std_deltas"].items():
                    sval = _check_finite_number(v, f"std_deltas[{k!r}]")
                    if sval < 0.0:
                        raise NumericalInstabilityError(
                            f"Learned module returned negative std for {k!r}: {sval}"
                        )
                    raw_stds[str(k)] = sval
            if "proposed_actions" in prediction and isinstance(
                prediction["proposed_actions"], Sequence
            ):
                for act in prediction["proposed_actions"]:
                    if isinstance(act, Action):
                        for pk, pv in act.parameters.items():
                            if isinstance(pv, (int, float)):
                                _check_finite_number(pv, f"proposed_action[{act.type}].{pk}")
                        proposed_acts.append(act)
            if "calibration_temperature" in prediction:
                temp_val = _check_finite_number(
                    prediction["calibration_temperature"], "calibration_temperature"
                )
        else:
            for k, v in prediction.items():
                raw_means[str(k)] = _check_finite_number(v, f"delta[{k!r}]")

        return CalibratedPrediction(
            mean_deltas=raw_means,
            std_deltas=raw_stds,
            proposed_actions=tuple(proposed_acts),
            calibration_temperature=temp_val,
        )

    raise TypeError(
        f"Unsupported learned prediction type: {type(prediction).__name__}"
    )


class TransitionStepOutcome(BaseModel):
    """Immutable outcome of a `HybridTransitionRouter.step` or `TransitionEngine.step` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    next_state: WorldState = Field(
        description="The resulting immutable WorldState at step t+1."
    )
    structural_state: WorldState = Field(
        description="The deterministic structural state prior to ML residual application."
    )
    is_deterministic: bool = Field(
        description="True if only structural rules applied; False when ML distribution is used."
    )
    distribution: CalibratedStateDistribution | None = Field(
        default=None,
        description="Calibrated predictive state distribution when ML is used; None if purely structural.",
    )
    pre_rule_rejected: bool = Field(
        default=False,
        description="True if Pre-rules rejected one or more invalid enterprise actions.",
    )
    post_rule_rejected: bool = Field(
        default=False,
        description="True if Post-rules rejected an out-of-bounds or invalid ML prediction.",
    )
    ml_rejected: bool = Field(
        default=False,
        description="True if the learned ML prediction was rejected (by NaN safeguard or Post-rules).",
    )
    rolled_back: bool = Field(
        default=False,
        description="True if the transition rolled back an invalid ML proposal or rejected action.",
    )
    nan_caught_before_post_rules: bool = Field(
        default=False,
        description="True if the operational safeguard caught NaN/Inf from ML before Post-rules.",
    )
    rejected_actions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of enterprise actions rejected during Pre-rules or Post-rules.",
    )
    pre_rule_violations: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Diagnostic messages from Pre-rules violations.",
    )
    post_rule_violations: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Diagnostic messages from Post-rules violations.",
    )
    forced_interventions: dict[str, Any] = Field(
        default_factory=dict,
        description="Active do(X_t = x) or context-managed forced state node values.",
    )
    severed_edges: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Inbound causal graph edges (u -> X_t) severed during this step.",
    )
    confounding_disabled: bool = Field(
        default=False,
        description="True when observational confounding into intervened nodes was disabled.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Epistemic evidence tier of the resulting transition.",
    )
    causal_log_entry: CausalExecutionLogEntry = Field(
        description="Audit record of causal graph topology and severed edges during this step."
    )
    transition_result: TransitionResult = Field(
        description="Underlying TransitionResult for compatibility with DynamicsModel consumers."
    )
    abstained: bool = Field(
        default=False,
        description="True if the learned predictor abstained due to OOD or ConfidenceInterval > threshold.",
    )
    abstention_reason: str | None = Field(
        default=None,
        description="Diagnostic reason when abstained is True.",
    )
    confidence_interval: Any | None = Field(
        default=None,
        description="ConfidenceInterval emitted by the learned predictor, if available.",
    )
    fallback_executed: bool = Field(
        default=False,
        description="True if fallback structural rules executed upon abstention.",
    )
    fallback_retry_depth: int = Field(
        default=0,
        ge=0,
        le=1,
        description="Number of fallback retries executed (strictly bounded by max_retry_depth = 1).",
    )
    fallback_failed: bool = Field(
        default=False,
        description="True if the custom fallback structural handler failed after max_retry_depth = 1.",
    )


class HybridTransitionRouter:
    """Three-stage hybrid transition router (`Pre-rules -> ML -> Post-rules`).

    Guarantees:
    1. **Pre-rules**: Evaluates structural action constraints (`ConstraintPhase.PRE_ACTION` and
       custom `pre_rules`), rejecting invalid enterprise actions before state evolution, and
       computes the deterministic structural state $S_{t+1}^{\\text{struct}}$ over the causal DAG
       while respecting `do(X_t = x)` edge severing and `Intervention` context managers.
    2. **ML Inference, Calibration & OOD Abstention**: When a learned module is present, executes
       prediction, inspects all numeric outputs for `NaN`/`Inf` *before* structural `Post-rules`
       evaluation (`NumericalInstabilityError`), and evaluates `ConfidenceInterval > threshold`
       and OOD / variance bounds (`AbstainException`). Upon abstention, executes safe structural
       fallback with maximum retry depth = 1.
    3. **Post-rules & 100% Rollback**: Evaluates resource bounds `[min_value, max_value]`,
       `ConstraintPhase.POST_TRANSITION` constraints, ML-hallucinated actions, and custom
       `post_rules`. Any out-of-bounds or invalid ML prediction is 100% rejected and rolled
       back to the valid structural state.
    """

    def __init__(
        self,
        *,
        structural_model: DynamicsModel | None = None,
        structural_equations: Mapping[str, StructuralEquationFn] | None = None,
        ml_predictor: (
            LearnedPredictorProtocol
            | DynamicsModel
            | Callable[[WorldState, Sequence[Action], WorldState], Any]
            | Any
            | None
        ) = None,
        constraints: ConstraintRegistry | Sequence[Constraint] | None = None,
        pre_rules: Sequence[StructuralPreRuleFn] = (),
        post_rules: Sequence[StructuralPostRuleFn] = (),
        causal_graph: CausalGraph | None = None,
        default_std: float = 0.05,
        calibration_temperature: float = 1.0,
        threshold: float = float("inf"),
        ood_threshold: float = float("inf"),
        variance_threshold: float = float("inf"),
        fallback_structural_fn: (
            Callable[[WorldState, Sequence[Action]], WorldState] | DynamicsModel | None
        ) = None,
        max_fallback_retries: int = 1,
        raise_on_abstain: bool = False,
        raise_on_nan: bool = False,
        raise_on_pre_rule_violation: bool = False,
        delta_t: float = 1.0,
        name: str = "HybridTransitionRouter",
        version: str = "1.0.0",
    ) -> None:
        self.structural_model = structural_model
        self.structural_equations: dict[str, StructuralEquationFn] = dict(
            structural_equations or {}
        )
        self.ml_predictor = ml_predictor
        if constraints is None:
            self.constraints = ConstraintRegistry()
        elif isinstance(constraints, ConstraintRegistry):
            self.constraints = constraints
        else:
            self.constraints = ConstraintRegistry(list(constraints))
        self.pre_rules: tuple[StructuralPreRuleFn, ...] = tuple(pre_rules)
        self.post_rules: tuple[StructuralPostRuleFn, ...] = tuple(post_rules)
        self.causal_graph = (
            CausalGraph(
                nodes=set(causal_graph.nodes),
                edges=set(causal_graph.edges),
                unobserved_nodes=set(causal_graph.unobserved_nodes),
            )
            if causal_graph is not None
            else CausalGraph()
        )
        self.default_std = float(default_std)
        self.calibration_temperature = float(calibration_temperature)
        self.threshold = float(threshold)
        self.ood_threshold = float(ood_threshold)
        self.variance_threshold = float(variance_threshold)
        self.fallback_structural_fn = fallback_structural_fn
        # Operational Safeguard (EWM-003 Section 11): Prevent infinite loops; max retry depth = 1
        self.max_fallback_retries = min(max(0, int(max_fallback_retries)), 1)
        self.raise_on_abstain = bool(raise_on_abstain)
        self.raise_on_nan = bool(raise_on_nan)
        self.raise_on_pre_rule_violation = bool(raise_on_pre_rule_violation)
        self.delta_t = float(delta_t)
        self.name = name
        self.version = version

        self._provenance = MechanismProvenance(
            mechanism_id=name,
            version=version,
            kind=MechanismKind.KNOWN_DETERMINISTIC if ml_predictor is None else MechanismKind.HYBRID,
            evidence_level=(
                EvidenceLevel.STRUCTURAL if ml_predictor is None else EvidenceLevel.PREDICTIVE
            ),
            source="hybrid_transition_router_pre_ml_post",
            uncertainty_contract=(
                "point_deterministic"
                if ml_predictor is None
                else "calibrated_gaussian_with_structural_post_rules"
            ),
            invariants_affected=("pre_action_rules", "resource_bounds", "post_transition_rules"),
            failure_semantics=InfeasibleResidualPolicy.REJECT,
        )

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        """Return formal mechanism provenance metadata."""
        return self._provenance

    def clone(self) -> HybridTransitionRouter:
        """Create an isolated copy of this router with a deep-copied CausalGraph and ConstraintRegistry."""
        return HybridTransitionRouter(
            structural_model=self.structural_model,
            structural_equations=dict(self.structural_equations),
            ml_predictor=self.ml_predictor,
            constraints=self.constraints.clone(),
            pre_rules=self.pre_rules,
            post_rules=self.post_rules,
            causal_graph=CausalGraph(
                nodes=set(self.causal_graph.nodes),
                edges=set(self.causal_graph.edges),
                unobserved_nodes=set(self.causal_graph.unobserved_nodes),
            ),
            default_std=self.default_std,
            calibration_temperature=self.calibration_temperature,
            threshold=self.threshold,
            ood_threshold=self.ood_threshold,
            variance_threshold=self.variance_threshold,
            fallback_structural_fn=self.fallback_structural_fn,
            max_fallback_retries=self.max_fallback_retries,
            raise_on_abstain=self.raise_on_abstain,
            raise_on_nan=self.raise_on_nan,
            raise_on_pre_rule_violation=self.raise_on_pre_rule_violation,
            delta_t=self.delta_t,
            name=self.name,
            version=self.version,
        )

    @staticmethod
    def _normalize_actions(
        action: Action | Sequence[Action] | None,
    ) -> list[Action]:
        if action is None:
            return []
        if isinstance(action, Action):
            return [action]
        return list(action)

    def _evaluate_pre_rules(
        self,
        state: WorldState,
        proposed_actions: Sequence[Action],
    ) -> tuple[list[Action], list[str], list[str]]:
        """Phase 1A: Evaluate structural Pre-rules on proposed enterprise actions."""
        accepted_actions, reg_results = self.constraints.validate_actions(
            state=state,
            actions=proposed_actions,
        )
        violations: list[str] = [
            f"pre_constraint:{r.constraint_id}:{r.message}"
            for r in reg_results
            if not r.satisfied and r.severity == ConstraintSeverity.HARD
        ]
        rejected_ids: list[str] = [
            a.id for a in proposed_actions if a not in accepted_actions
        ]

        # Built-in structural enterprise action hygiene checks & custom pre_rules
        filtered_accepted: list[Action] = []
        for act in accepted_actions:
            act_ok = True
            if act.parameters.get("invalid", False) is True:
                act_ok = False
                violations.append(f"pre_rule:invalid_action_flag:{act.id}({act.type})")
            target_res = act.parameters.get("resource_id") or act.parameters.get("target_resource")
            if isinstance(target_res, str) and target_res not in state.resources:
                act_ok = False
                violations.append(
                    f"pre_rule:unknown_target_resource:{act.id}({target_res!r})"
                )
            delta_param = act.parameters.get("delta")
            if isinstance(delta_param, (int, float)):
                if math.isnan(float(delta_param)) or math.isinf(float(delta_param)):
                    act_ok = False
                    violations.append(f"pre_rule:non_finite_action_delta:{act.id}")
                elif isinstance(target_res, str) and target_res in state.resources:
                    res = state.get_resource(target_res)
                    proposed_val = res.current + float(delta_param)
                    if proposed_val < res.min_value - 1e-9 or proposed_val > res.max_value + 1e-9:
                        act_ok = False
                        violations.append(
                            f"pre_rule:action_exceeds_resource_bounds:{act.id}:"
                            f"{target_res}({proposed_val:.4f} not in [{res.min_value}, {res.max_value}])"
                        )

            if act_ok:
                filtered_accepted.append(act)
            elif act.id not in rejected_ids:
                rejected_ids.append(act.id)

        for rule_fn in self.pre_rules:
            ok, msg = rule_fn(state, filtered_accepted)
            if not ok:
                violations.append(f"pre_rule_fn:{msg}")
                for a in filtered_accepted:
                    if a.id not in rejected_ids:
                        rejected_ids.append(a.id)
                filtered_accepted = []
                break

        if violations and self.raise_on_pre_rule_violation:
            raise ValueError("; ".join(violations))

        return filtered_accepted, rejected_ids, violations

    def _compute_structural_baseline(
        self,
        state: WorldState,
        accepted_actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        u_noise: Mapping[str, float],
        active_graph: CausalGraph,
        forced_nodes: Mapping[str, Any],
        rng: RandomGenerator,
    ) -> WorldState:
        """Phase 1B: Compute deterministic structural transition respecting `do(X_t = x)` surgery."""
        current = state

        # 1. Force intervened nodes at step t first so downstream children read forced X_t = x
        for node_key, forced_val in forced_nodes.items():
            clean_key = node_key.split(":", 1)[1] if node_key.startswith("resource:") else node_key
            if clean_key in current.resources and isinstance(forced_val, (int, float)):
                current = current.update_resource(
                    clean_key,
                    new_value=float(forced_val),
                    clamp=False,
                    enforce_bounds=True,
                )
            elif node_key.startswith("memory:") or clean_key not in current.resources:
                mem_key = node_key.split(":", 1)[1] if node_key.startswith("memory:") else node_key
                current = current.with_memory(mem_key, forced_val)

        # 2. If a structural DynamicsModel is configured, run it
        if self.structural_model is not None:
            res = self.structural_model.transition(
                state=current,
                actions=accepted_actions,
                exogenous_events=exogenous_events,
                rng=rng,
            )
            current = res.next_state

        # 3. Apply structural equations over the causal graph (Y_{t+1} = f(PA_Y(t), A_t, U_t))
        if self.structural_equations:
            # Snapshot t values (with forced X_t = x already applied)
            snapshot_vals: dict[str, float] = {
                rid: r.current for rid, r in current.resources.items()
            }
            for mk, mv in current.memory.items():
                if isinstance(mv, (int, float)):
                    snapshot_vals[mk] = float(mv)

            # Evaluate equations in declaration / topological order
            # First update any non-intervened root/parent nodes, then downstream nodes
            ordered_targets = self._topological_sort_targets(
                list(self.structural_equations.keys()), active_graph
            )
            for target in ordered_targets:
                clean_target = (
                    target.split(":", 1)[1] if target.startswith("resource:") else target
                )
                norm_target = normalize_node_name(target, active_graph)
                if (
                    target in forced_nodes
                    or clean_target in forced_nodes
                    or norm_target in forced_nodes
                ):
                    # Intervened node: inbound edges were severed; skip structural equation!
                    continue

                eq_fn = self.structural_equations[target]
                # Only supply values from parents that STILL have an unsevered directed edge -> target!
                if norm_target in active_graph.nodes:
                    valid_parents = active_graph.parents(norm_target) | {
                        p.split(":", 1)[1] if p.startswith("resource:") else p
                        for p in active_graph.parents(norm_target)
                    }
                    parent_Context = {
                        k: v
                        for k, v in snapshot_vals.items()
                        if k in valid_parents or k == clean_target
                    }
                else:
                    parent_Context = dict(snapshot_vals)

                new_val = float(eq_fn(parent_Context, accepted_actions, u_noise))
                snapshot_vals[clean_target] = new_val
                snapshot_vals[norm_target] = new_val
                if clean_target in current.resources:
                    current = current.update_resource(
                        clean_target,
                        new_value=new_val,
                        clamp=False,
                        enforce_bounds=True,
                    )
                else:
                    current = current.with_memory(clean_target, new_val)
        elif self.structural_model is None:
            # Default action delta application for resources not locked by do()
            for act in accepted_actions:
                target_res = act.parameters.get("resource_id") or act.parameters.get(
                    "target_resource"
                )
                delta_val = act.parameters.get("delta")
                if (
                    isinstance(target_res, str)
                    and target_res in current.resources
                    and target_res not in forced_nodes
                    and isinstance(delta_val, (int, float))
                ):
                    current = current.update_resource(
                        target_res,
                        delta=float(delta_val),
                        clamp=False,
                        enforce_bounds=True,
                    )

        # 4. Re-assert forced nodes so neither structural_model nor equations can overwrite do(X_t = x)
        for node_key, forced_val in forced_nodes.items():
            clean_key = node_key.split(":", 1)[1] if node_key.startswith("resource:") else node_key
            if clean_key in current.resources and isinstance(forced_val, (int, float)):
                current = current.update_resource(
                    clean_key,
                    new_value=float(forced_val),
                    clamp=False,
                    enforce_bounds=True,
                )
            elif node_key.startswith("memory:") or clean_key not in current.resources:
                mem_key = node_key.split(":", 1)[1] if node_key.startswith("memory:") else node_key
                current = current.with_memory(mem_key, forced_val)

        return current

    @staticmethod
    def _topological_sort_targets(
        targets: Sequence[str],
        graph: CausalGraph,
    ) -> list[str]:
        """Order structural equation targets topologically according to `graph` edges."""
        target_list = list(targets)
        target_set = set(target_list)
        in_degree: dict[str, int] = dict.fromkeys(target_list, 0)
        adj: dict[str, list[str]] = {t: [] for t in target_list}

        for u, v in graph.edges:
            if u in target_set and v in target_set and u != v:
                adj[u].append(v)
                in_degree[v] += 1

        queue = [t for t in target_list if in_degree[t] == 0]
        ordered: list[str] = []
        while queue:
            curr = queue.pop(0)
            ordered.append(curr)
            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        for t in target_list:
            if t not in ordered:
                ordered.append(t)
        return ordered

    def _invoke_ml_predictor(
        self,
        state: WorldState,
        accepted_actions: Sequence[Action],
        structural_state: WorldState,
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> tuple[CalibratedPrediction, Any]:
        """Query the configured ML predictor, validate against NaN/Inf, and enforce OOD/CI abstention."""
        from ewm_engine.dynamics.calibration import (
            AbstainException,
            ConfidenceInterval,
        )

        assert self.ml_predictor is not None
        raw_out: Any
        if hasattr(self.ml_predictor, "predict") and callable(self.ml_predictor.predict):
            pred_fn: Any = self.ml_predictor.predict
            try:
                raw_out = pred_fn(state, accepted_actions, structural_state)
            except TypeError:
                try:
                    raw_out = pred_fn(state, accepted_actions)
                except TypeError:
                    raw_out = pred_fn(state)
        elif hasattr(self.ml_predictor, "predict_residual") and callable(
            self.ml_predictor.predict_residual
        ):
            fn_res: Any = self.ml_predictor.predict_residual
            raw_out = fn_res(state, accepted_actions, exogenous_events, structural_state)
        elif isinstance(self.ml_predictor, DynamicsModel):
            trans_res = self.ml_predictor.transition(
                state=structural_state,
                actions=accepted_actions,
                exogenous_events=exogenous_events,
                rng=rng,
            )
            deltas: dict[str, float] = {}
            for rid, r_next in trans_res.next_state.resources.items():
                r_base = structural_state.resources.get(rid)
                base_val = r_base.current if r_base is not None else 0.0
                deltas[rid] = r_next.current - base_val
            std_val = float(getattr(self.ml_predictor, "residual_std", self.default_std))
            stds = {rid: max(1e-6, std_val) for rid in deltas}
            raw_out = CalibratedPrediction(
                mean_deltas=deltas,
                std_deltas=stds,
                calibration_temperature=self.calibration_temperature,
            )
        elif callable(self.ml_predictor):
            fn_pred: Any = self.ml_predictor
            try:
                raw_out = fn_pred(state, accepted_actions, structural_state)
            except TypeError:
                raw_out = fn_pred(state)
        else:
            raise TypeError(
                f"Unsupported ml_predictor type: {type(self.ml_predictor).__name__}"
            )

        ci_obj: ConfidenceInterval | None = None
        if isinstance(raw_out, tuple) and len(raw_out) == 2:
            raw_pred, second_elem = raw_out
            if isinstance(second_elem, ConfidenceInterval):
                ci_obj = second_elem
        else:
            raw_pred = raw_out

        calibrated_pred = validate_learned_prediction_no_nan(raw_pred)

        if ci_obj is None:
            z_90 = 1.6448536269514722
            hw_map = {
                rid: z_90 * calibrated_pred.std_deltas.get(rid, self.default_std)
                for rid in calibrated_pred.mean_deltas
            }
            w_map = {rid: 2.0 * hw for rid, hw in hw_map.items()}
            lo_map = {
                rid: mu - hw_map[rid]
                for rid, mu in calibrated_pred.mean_deltas.items()
            }
            hi_map = {
                rid: mu + hw_map[rid]
                for rid, mu in calibrated_pred.mean_deltas.items()
            }
            ci_obj = ConfidenceInterval(
                lower=lo_map,
                upper=hi_map,
                half_width=hw_map,
                width=w_map,
                max_width=max(w_map.values(), default=0.0),
                max_half_width=max(hw_map.values(), default=0.0),
                ensemble_variance=dict.fromkeys(calibrated_pred.mean_deltas, 0.0),
                max_ensemble_variance=0.0,
                ood_score=0.0,
                nominal_coverage=0.90,
            )

        if not math.isinf(self.ood_threshold) and ci_obj.ood_score > self.ood_threshold:
            raise AbstainException(
                f"Router OOD threshold exceeded: ood_score={ci_obj.ood_score:.4f} > "
                f"ood_threshold={self.ood_threshold:.4f}",
                confidence_interval=ci_obj,
                ood_score=ci_obj.ood_score,
                ensemble_variance=ci_obj.max_ensemble_variance,
                threshold=self.ood_threshold,
                trigger="router_ood_threshold",
            )
        if (
            not math.isinf(self.variance_threshold)
            and ci_obj.max_ensemble_variance > self.variance_threshold
        ):
            raise AbstainException(
                f"Router ensemble variance threshold exceeded: "
                f"Var(mu)={ci_obj.max_ensemble_variance:.4f} > "
                f"variance_threshold={self.variance_threshold:.4f}",
                confidence_interval=ci_obj,
                ood_score=ci_obj.ood_score,
                ensemble_variance=ci_obj.max_ensemble_variance,
                threshold=self.variance_threshold,
                trigger="router_variance_threshold",
            )
        if ci_obj > self.threshold:
            raise AbstainException(
                f"Router ConfidenceInterval threshold exceeded: "
                f"effective_uncertainty={ci_obj.effective_uncertainty:.4f} > "
                f"threshold={self.threshold:.4f}",
                confidence_interval=ci_obj,
                ood_score=ci_obj.ood_score,
                ensemble_variance=ci_obj.max_ensemble_variance,
                threshold=self.threshold,
                trigger="router_confidence_interval_threshold",
            )

        return calibrated_pred, ci_obj

    def _execute_fallback_with_retry_guard(
        self,
        state: WorldState,
        accepted_actions: Sequence[Action],
        structural_state: WorldState,
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> tuple[WorldState, int, bool]:
        """Execute safe structural fallback upon `AbstainException` with max retry depth = 1.

        Operational Safeguard (EWM-003 Section 11):
            Prevents infinite loops if the fallback structural handler also fails.
            Maximum retry depth is strictly capped at 1 (`0` initial attempt + at most `1` retry).
        """
        if self.fallback_structural_fn is None:
            return structural_state, 0, False

        max_retries = min(max(0, self.max_fallback_retries), 1)
        retry_depth = 0
        for attempt in range(max_retries + 1):
            retry_depth = attempt
            try:
                if isinstance(self.fallback_structural_fn, DynamicsModel):
                    res = self.fallback_structural_fn.transition(
                        state=state,
                        actions=accepted_actions,
                        exogenous_events=exogenous_events,
                        rng=rng,
                    )
                    return res.next_state, retry_depth, False
                if callable(self.fallback_structural_fn):
                    fb_state = self.fallback_structural_fn(state, accepted_actions)
                    return fb_state, retry_depth, False
            except Exception:
                if attempt >= max_retries:
                    return structural_state, retry_depth, True

        return structural_state, retry_depth, True

    def _build_calibrated_distribution(
        self,
        center_state: WorldState,
        std_map: Mapping[str, float],
        forced_nodes: Mapping[str, Any],
        temperature: float,
    ) -> CalibratedStateDistribution:
        """Build a CalibratedStateDistribution around `center_state`."""
        mean_dict: dict[str, float] = {}
        std_dict: dict[str, float] = {}
        lower_95: dict[str, float] = {}
        upper_95: dict[str, float] = {}
        quantiles: dict[str, dict[str, float]] = {}

        z_95 = 1.959963984540054
        z_90 = 1.6448536269514722

        for rid, res in center_state.resources.items():
            mu = float(res.current)
            if rid in forced_nodes or f"resource:{rid}" in forced_nodes:
                sigma = 0.0
            else:
                base_sigma = float(std_map.get(rid, self.default_std))
                sigma = max(0.0, base_sigma * temperature)
            mean_dict[rid] = mu
            std_dict[rid] = sigma
            lo = max(res.min_value, mu - z_95 * sigma)
            hi = min(res.max_value, mu + z_95 * sigma)
            lower_95[rid] = lo
            upper_95[rid] = hi
            quantiles[rid] = {
                "p05": max(res.min_value, mu - z_90 * sigma),
                "p50": mu,
                "p95": min(res.max_value, mu + z_90 * sigma),
            }

        return CalibratedStateDistribution(
            mean=mean_dict,
            std=std_dict,
            lower_95=lower_95,
            upper_95=upper_95,
            quantiles=quantiles,
            calibration_temperature=temperature,
            is_calibrated=True,
        )

    def step(
        self,
        state: WorldState,
        action: Action | Sequence[Action] | None = None,
        *,
        exogenous_events: Sequence[ExogenousEvent] = (),
        u_noise: Mapping[str, float] | None = None,
        active_do_interventions: Mapping[str, Any] | None = None,
        causal_graph: CausalGraph | None = None,
        rng: RandomGenerator | None = None,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionStepOutcome:
        """Execute the 3-stage `Pre-rules -> ML -> Post-rules` transition pipeline."""
        active_rng = rng if rng is not None else np.random.default_rng(state.step)
        noise_map = dict(u_noise or {})
        proposed_actions = self._normalize_actions(action)

        # Collect active interventions from engine `do()` and context-managed `with Intervention(...)`
        forced_nodes: dict[str, Any] = dict(active_do_interventions or {})
        for ctx_iv in get_active_interventions():
            for k, v in ctx_iv.forced_nodes().items():
                forced_nodes[k] = copy.deepcopy(v)

        base_graph = causal_graph if causal_graph is not None else self.causal_graph
        if forced_nodes:
            step_graph, causal_log = sever_inbound_edges(
                base_graph,
                list(forced_nodes.keys()),
                step=state.step,
                forced_values=forced_nodes,
                operation="step_execution",
            )
        else:
            step_graph = CausalGraph(
                nodes=set(base_graph.nodes),
                edges=set(base_graph.edges),
                unobserved_nodes=set(base_graph.unobserved_nodes),
            )
            causal_log = CausalExecutionLogEntry(
                step=state.step,
                operation="step_execution",
                intervened_variables=(),
                forced_values={},
                edges_before=tuple(sorted(base_graph.edges)),
                severed_inbound_edges=(),
                edges_after=tuple(sorted(base_graph.edges)),
                confounding_disabled=False,
                notes="Standard observational/structural step execution (no active do() intervention).",
            )

        # =====================================================================
        # PHASE 1: PRE-RULES (Structural Action Validation & Baseline Evolution)
        # =====================================================================
        accepted_actions, rejected_ids, pre_violations = self._evaluate_pre_rules(
            state=state,
            proposed_actions=proposed_actions,
        )
        pre_rule_rejected = len(pre_violations) > 0

        if pre_rule_rejected and len(accepted_actions) == 0 and len(proposed_actions) > 0:
            # All proposed enterprise actions were invalid and rejected by Pre-rules
            structural_state = self._compute_structural_baseline(
                state=state,
                accepted_actions=(),
                exogenous_events=exogenous_events,
                u_noise=noise_map,
                active_graph=step_graph,
                forced_nodes=forced_nodes,
                rng=active_rng,
            )
            final_struct = structural_state.advance_time(delta_t=self.delta_t, steps=1)
            dist = (
                self._build_calibrated_distribution(
                    final_struct, {}, forced_nodes, self.calibration_temperature
                )
                if self.ml_predictor is not None
                else None
            )
            ev_level = (
                EvidenceLevel.INTERVENTIONAL
                if forced_nodes
                else (
                    EvidenceLevel.STRUCTURAL
                    if self.ml_predictor is None
                    else EvidenceLevel.PREDICTIVE
                )
            )
            tr = TransitionResult(
                next_state=final_struct,
                applied_changes={"pre_rule_rejected": True, "rejected_actions": rejected_ids},
                evidence_level=ev_level,
                diagnostics={
                    "pre_rule_rejected": True,
                    "pre_rule_violations": pre_violations,
                    "severed_edges": list(causal_log.severed_inbound_edges),
                },
                model_name=self.name,
            )
            return TransitionStepOutcome(
                next_state=final_struct,
                structural_state=final_struct,
                is_deterministic=(self.ml_predictor is None),
                distribution=dist,
                pre_rule_rejected=True,
                post_rule_rejected=False,
                ml_rejected=False,
                rolled_back=True,
                nan_caught_before_post_rules=False,
                rejected_actions=tuple(rejected_ids),
                pre_rule_violations=tuple(pre_violations),
                post_rule_violations=(),
                forced_interventions=forced_nodes,
                severed_edges=causal_log.severed_inbound_edges,
                confounding_disabled=causal_log.confounding_disabled,
                evidence_level=ev_level,
                causal_log_entry=causal_log,
                transition_result=tr,
            )

        structural_state = self._compute_structural_baseline(
            state=state,
            accepted_actions=accepted_actions,
            exogenous_events=exogenous_events,
            u_noise=noise_map,
            active_graph=step_graph,
            forced_nodes=forced_nodes,
            rng=active_rng,
        )

        # If no ML predictor is configured, return deterministic structural outcome
        if self.ml_predictor is None:
            final_state = structural_state.advance_time(delta_t=self.delta_t, steps=1)
            ev_level = (
                EvidenceLevel.INTERVENTIONAL if forced_nodes else EvidenceLevel.STRUCTURAL
            )
            tr = TransitionResult(
                next_state=final_state,
                applied_changes={
                    "forced_interventions": forced_nodes,
                    "accepted_actions": [a.id for a in accepted_actions],
                },
                evidence_level=ev_level,
                diagnostics={
                    "pipeline": "pre_rules_only_deterministic",
                    "severed_edges": list(causal_log.severed_inbound_edges),
                    "confounding_disabled": causal_log.confounding_disabled,
                },
                model_name=self.name,
            )
            return TransitionStepOutcome(
                next_state=final_state,
                structural_state=final_state,
                is_deterministic=True,
                distribution=None,
                pre_rule_rejected=pre_rule_rejected,
                post_rule_rejected=False,
                ml_rejected=False,
                rolled_back=False,
                nan_caught_before_post_rules=False,
                rejected_actions=tuple(rejected_ids),
                pre_rule_violations=tuple(pre_violations),
                post_rule_violations=(),
                forced_interventions=forced_nodes,
                severed_edges=causal_log.severed_inbound_edges,
                confounding_disabled=causal_log.confounding_disabled,
                evidence_level=ev_level,
                causal_log_entry=causal_log,
                transition_result=tr,
            )

        # =====================================================================
        # PHASE 2: ML INFERENCE, NaN/Inf SAFEGUARD & OOD/CI ABSTENTION GATE
        # =====================================================================
        from ewm_engine.dynamics.calibration import AbstainException

        nan_caught = False
        abstained = False
        abstention_reason: str | None = None
        ci_obj: Any = None
        fallback_executed = False
        fallback_retry_depth = 0
        fallback_failed = False
        post_violations: list[str] = []
        ml_rejected = False
        post_rule_rejected = False
        rolled_back = False

        try:
            calibrated_pred, ci_obj = self._invoke_ml_predictor(
                state=state,
                accepted_actions=accepted_actions,
                structural_state=structural_state,
                exogenous_events=exogenous_events,
                rng=active_rng,
            )
        except AbstainException as exc:
            abstained = True
            abstention_reason = exc.reason
            ci_obj = exc.confidence_interval
            ml_rejected = True
            rolled_back = True
            fallback_executed = True
            post_violations.append(f"abstain:{exc.reason}")
            if self.raise_on_abstain:
                raise
            structural_state, fallback_retry_depth, fallback_failed = (
                self._execute_fallback_with_retry_guard(
                    state=state,
                    accepted_actions=accepted_actions,
                    structural_state=structural_state,
                    exogenous_events=exogenous_events,
                    rng=active_rng,
                )
            )
            calibrated_pred = CalibratedPrediction(
                mean_deltas={},
                std_deltas={},
                proposed_actions=(),
                calibration_temperature=self.calibration_temperature,
            )
        except NumericalInstabilityError as exc:
            nan_caught = True
            ml_rejected = True
            rolled_back = True
            post_violations.append(f"nan_safeguard:{exc}")
            if self.raise_on_nan:
                raise
            calibrated_pred = CalibratedPrediction(
                mean_deltas={},
                std_deltas={},
                proposed_actions=(),
                calibration_temperature=self.calibration_temperature,
            )

        # =====================================================================
        # PHASE 3: POST-RULES (Structural Post-Constraints & Hallucination Rollback)
        # =====================================================================
        raw_ml_deltas: dict[str, float] = {}
        if not nan_caught and not abstained:
            # 3A. Validate any actions proposed/hallucinated by the ML module
            if calibrated_pred.proposed_actions:
                ml_acc, ml_pre_res = self.constraints.validate_actions(
                    state=structural_state,
                    actions=calibrated_pred.proposed_actions,
                )
                for r in ml_pre_res:
                    if not r.satisfied and r.severity == ConstraintSeverity.HARD:
                        post_violations.append(
                            f"post_rule:ml_hallucinated_action_violation:{r.constraint_id}:{r.message}"
                        )
                for ml_act in calibrated_pred.proposed_actions:
                    if ml_act not in ml_acc or ml_act.parameters.get("invalid", False) is True:
                        if ml_act.id not in rejected_ids:
                            rejected_ids.append(ml_act.id)
                        post_violations.append(
                            f"post_rule:ml_invalid_action_rejected:{ml_act.id}({ml_act.type})"
                        )
                    target_res = ml_act.parameters.get("resource_id") or ml_act.parameters.get(
                        "target_resource"
                    )
                    if isinstance(target_res, str) and target_res not in structural_state.resources:
                        if ml_act.id not in rejected_ids:
                            rejected_ids.append(ml_act.id)
                        post_violations.append(
                            f"post_rule:ml_hallucinated_unknown_resource:{ml_act.id}({target_res!r})"
                        )
                    delta_p = ml_act.parameters.get("delta")
                    if (
                        isinstance(target_res, str)
                        and target_res in structural_state.resources
                        and isinstance(delta_p, (int, float))
                    ):
                        raw_ml_deltas[target_res] = raw_ml_deltas.get(target_res, 0.0) + float(
                            delta_p
                        )

            # 3B. Collect ML resource deltas (ignoring nodes locked by active Intervention / do())
            for rid, delta in calibrated_pred.mean_deltas.items():
                clean_rid = rid.split(":", 1)[1] if rid.startswith("resource:") else rid
                if (
                    rid in forced_nodes
                    or clean_rid in forced_nodes
                    or f"resource:{clean_rid}" in forced_nodes
                ):
                    # Active Intervention / do() overrides ML dynamics for forced nodes
                    continue
                if clean_rid not in structural_state.resources:
                    post_violations.append(
                        f"post_rule:unknown_resource_in_ml_prediction:{clean_rid!r}"
                    )
                    continue
                raw_ml_deltas[clean_rid] = raw_ml_deltas.get(clean_rid, 0.0) + float(delta)

            # 3C. Check resource bounds [min_value, max_value] on all proposed ML deltas
            for rid, delta in raw_ml_deltas.items():
                res_obj = structural_state.get_resource(rid)
                proposed_val = res_obj.current + delta
                if (
                    proposed_val < res_obj.min_value - 1e-9
                    or proposed_val > res_obj.max_value + 1e-9
                ):
                    post_violations.append(
                        f"post_rule:resource_bounds_violation:{rid}"
                        f"({proposed_val:.6f} not in [{res_obj.min_value}, {res_obj.max_value}])"
                    )

            # 3D. If within resource bounds, build candidate state and evaluate POST_TRANSITION constraints
            if not post_violations:
                candidate_state = structural_state
                for rid, delta in raw_ml_deltas.items():
                    candidate_state = candidate_state.update_resource(
                        rid,
                        delta=delta,
                        clamp=False,
                        enforce_bounds=True,
                    )
                all_acts = [*accepted_actions, *calibrated_pred.proposed_actions]
                post_constraint_results = self.constraints.validate_state(
                    state=candidate_state,
                    preceding_actions=all_acts,
                )
                for r in post_constraint_results:
                    if not r.satisfied and r.severity == ConstraintSeverity.HARD:
                        post_violations.append(
                            f"post_constraint:{r.constraint_id}:{r.message}"
                        )
                for post_fn in self.post_rules:
                    ok, msg = post_fn(structural_state, candidate_state, all_acts)
                    if not ok:
                        post_violations.append(f"post_rule_fn:{msg}")
            else:
                candidate_state = structural_state

            if post_violations:
                post_rule_rejected = True
                ml_rejected = True
                rolled_back = True
                accepted_post_ml_state = structural_state
            else:
                accepted_post_ml_state = candidate_state
        else:
            accepted_post_ml_state = structural_state

        final_next_state = accepted_post_ml_state.advance_time(delta_t=self.delta_t, steps=1)
        final_structural_state = structural_state.advance_time(delta_t=self.delta_t, steps=1)

        applied_deltas = (
            {}
            if ml_rejected
            else {rid: float(d) for rid, d in raw_ml_deltas.items() if abs(float(d)) > 0.0}
        )
        l2_dist = (
            math.sqrt(sum(d * d for d in raw_ml_deltas.values()))
            if ml_rejected
            else 0.0
        )
        feasibility = ResidualFeasibilityRecord(
            is_feasible=not ml_rejected,
            policy=(
                InfeasibleResidualPolicy.ABSTAIN
                if abstained
                else InfeasibleResidualPolicy.REJECT
            ),
            action_taken="abstained" if abstained else ("rejected" if ml_rejected else "accepted"),
            raw_deltas=raw_ml_deltas,
            applied_deltas=applied_deltas,
            l2_projection_distance=l2_dist,
            violations=tuple(post_violations),
        )

        distribution = self._build_calibrated_distribution(
            center_state=final_next_state,
            std_map=calibrated_pred.std_deltas,
            forced_nodes=forced_nodes,
            temperature=calibrated_pred.calibration_temperature,
        )
        ev_level = (
            EvidenceLevel.INTERVENTIONAL
            if forced_nodes
            else (EvidenceLevel.STRUCTURAL if abstained else EvidenceLevel.PREDICTIVE)
        )
        tr = TransitionResult(
            next_state=final_next_state,
            applied_changes={
                "raw_ml_deltas": raw_ml_deltas,
                "applied_ml_deltas": applied_deltas,
                "ml_rejected": ml_rejected,
                "abstained": abstained,
                "rolled_back": rolled_back,
                "forced_interventions": forced_nodes,
            },
            evidence_level=ev_level,
            diagnostics={
                "pipeline": "pre_rules_ml_post_rules",
                "pre_rule_rejected": pre_rule_rejected,
                "post_rule_rejected": post_rule_rejected,
                "ml_rejected": ml_rejected,
                "abstained": abstained,
                "abstention_reason": abstention_reason,
                "fallback_executed": fallback_executed,
                "fallback_retry_depth": fallback_retry_depth,
                "fallback_failed": fallback_failed,
                "rolled_back": rolled_back,
                "nan_caught_before_post_rules": nan_caught,
                "residual_feasibility": feasibility.model_dump(mode="json"),
                "severed_edges": list(causal_log.severed_inbound_edges),
                "confounding_disabled": causal_log.confounding_disabled,
            },
            model_name=self.name,
        )

        return TransitionStepOutcome(
            next_state=final_next_state,
            structural_state=final_structural_state,
            is_deterministic=False,
            distribution=distribution,
            pre_rule_rejected=pre_rule_rejected,
            post_rule_rejected=post_rule_rejected,
            ml_rejected=ml_rejected,
            rolled_back=rolled_back,
            nan_caught_before_post_rules=nan_caught,
            rejected_actions=tuple(rejected_ids),
            pre_rule_violations=tuple(pre_violations),
            post_rule_violations=tuple(post_violations),
            forced_interventions=forced_nodes,
            severed_edges=causal_log.severed_inbound_edges,
            confounding_disabled=causal_log.confounding_disabled,
            evidence_level=ev_level,
            causal_log_entry=causal_log,
            transition_result=tr,
            abstained=abstained,
            abstention_reason=abstention_reason,
            confidence_interval=ci_obj,
            fallback_executed=fallback_executed,
            fallback_retry_depth=fallback_retry_depth,
            fallback_failed=fallback_failed,
        )

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        """`DynamicsModel` protocol implementation wrapping `step()` for backwards compatibility."""
        outcome = self.step(
            state=state,
            action=actions,
            exogenous_events=exogenous_events,
            rng=rng,
            runtime_context=runtime_context,
        )
        return outcome.transition_result


class TransitionEngine:
    """Causal and hybrid state-transition engine supporting `engine.branch().do(action)`.

    Contracts:
    1. `TransitionEngine.step(state, action)` yields a deterministic outcome
       (`outcome.is_deterministic == True`, `outcome.distribution is None`) if only
       structural rules apply, and a calibrated distribution
       (`outcome.is_deterministic == False`, `outcome.distribution is not None`) if ML is used.
    2. `engine.branch().do(action)` spawns an isolated branch, severs all inbound edges
       $(u \\to X_t)$ to the intervened variables in the causal graph ($Y_{t+1} = f(X_t, U_t)$),
       and disables observational confounding mechanisms into $X_t$.
    3. `legacy_step(state, action)` wraps the new `HybridTransitionRouter` for backwards compatibility.
    """

    def __init__(
        self,
        router: HybridTransitionRouter | None = None,
        *,
        initial_state: WorldState | None = None,
        causal_graph: CausalGraph | None = None,
        structural_equations: Mapping[str, StructuralEquationFn] | None = None,
        ml_predictor: (
            LearnedPredictorProtocol
            | DynamicsModel
            | Callable[[WorldState, Sequence[Action], WorldState], CalibratedPrediction | Mapping[str, float]]
            | Any
            | None
        ) = None,
        constraints: ConstraintRegistry | Sequence[Constraint] | None = None,
        pre_rules: Sequence[StructuralPreRuleFn] = (),
        post_rules: Sequence[StructuralPostRuleFn] = (),
        threshold: float = float("inf"),
        ood_threshold: float = float("inf"),
        variance_threshold: float = float("inf"),
        fallback_structural_fn: (
            Callable[[WorldState, Sequence[Action]], WorldState] | DynamicsModel | None
        ) = None,
        max_fallback_retries: int = 1,
        raise_on_abstain: bool = False,
        raise_on_nan: bool = False,
        branch_id: str = "main",
    ) -> None:
        if router is not None:
            self.router = router.clone()
            if causal_graph is not None:
                self.router.causal_graph = CausalGraph(
                    nodes=set(causal_graph.nodes),
                    edges=set(causal_graph.edges),
                    unobserved_nodes=set(causal_graph.unobserved_nodes),
                )
        else:
            self.router = HybridTransitionRouter(
                structural_equations=structural_equations,
                ml_predictor=ml_predictor,
                constraints=constraints,
                pre_rules=pre_rules,
                post_rules=post_rules,
                causal_graph=causal_graph,
                threshold=threshold,
                ood_threshold=ood_threshold,
                variance_threshold=variance_threshold,
                fallback_structural_fn=fallback_structural_fn,
                max_fallback_retries=max_fallback_retries,
                raise_on_abstain=raise_on_abstain,
                raise_on_nan=raise_on_nan,
            )

        self.branch_id = branch_id
        self._current_state: WorldState | None = (
            initial_state.clone() if initial_state is not None else None
        )
        self.event_log: EventLog = EventLog(
            initial_state=self._current_state if self._current_state is not None else WorldState(),
            branch_id=branch_id,
        )
        self._active_do_interventions: dict[str, Any] = {}
        self._causal_execution_log: list[CausalExecutionLogEntry] = []

    @property
    def causal_graph(self) -> CausalGraph:
        """Return the engine's active CausalGraph."""
        return self.router.causal_graph

    @causal_graph.setter
    def causal_graph(self, graph: CausalGraph) -> None:
        self.router.causal_graph = graph

    @property
    def current_state(self) -> WorldState | None:
        """Return an immutable snapshot of the engine's current state."""
        return self._current_state.clone() if self._current_state is not None else None

    @property
    def current_step(self) -> int:
        """Return the current discrete time step index."""
        return self._current_state.step if self._current_state is not None else 0

    @property
    def active_do_interventions(self) -> dict[str, Any]:
        """Return a deep copy of active `do(X_t = x)` interventions on this engine branch."""
        return copy.deepcopy(self._active_do_interventions)

    @property
    def causal_execution_log(self) -> tuple[CausalExecutionLogEntry, ...]:
        """Return the immutable log of causal graph surgeries and step executions."""
        return tuple(self._causal_execution_log)

    def branch(self, branch_id: str | None = None) -> TransitionEngine:
        """Spawn a leak-proof counterfactual branch of this `TransitionEngine`."""
        new_branch_id = branch_id or f"{self.branch_id}_branch_{len(self._causal_execution_log)}"
        branched = TransitionEngine(
            router=self.router.clone(),
            initial_state=self._current_state.clone() if self._current_state is not None else None,
            branch_id=new_branch_id,
        )
        branched._active_do_interventions = copy.deepcopy(self._active_do_interventions)
        branched._causal_execution_log = list(self._causal_execution_log)
        return branched

    def do(
        self,
        action: Action | Intervention | Mapping[str, Any],
    ) -> TransitionEngine:
        """Apply Pearl's `do(X_t = x)` operator on this engine branch.

        1. Parses target variables $X_t$ and forced values $x$ from `action`.
        2. Severs all inbound directed edges $(u \\to X_t)$ in `self.causal_graph`,
           explicitly disabling observational confounding mechanisms into $X_t$.
        3. Locks $X_t = x$ on the current state and active intervention registry.
        4. Appends a `CausalExecutionLogEntry` documenting the severed edges.
        """
        raw_targets = parse_intervention_targets(action)
        mutilated_graph, log_entry = sever_inbound_edges(
            self.router.causal_graph,
            list(raw_targets.keys()),
            step=self.current_step,
            forced_values=raw_targets,
            operation="do_surgery",
        )
        self.router.causal_graph = mutilated_graph
        self._causal_execution_log.append(log_entry)

        for k, v in raw_targets.items():
            norm_k = normalize_node_name(k, self.router.causal_graph)
            clean_k = k.split(":", 1)[1] if k.startswith("resource:") else k
            self._active_do_interventions[norm_k] = copy.deepcopy(v)
            self._active_do_interventions[clean_k] = copy.deepcopy(v)

        if self._current_state is not None:
            iv = Intervention(
                id=f"do_{len(self._causal_execution_log)}",
                description="Pearl do() graph surgery",
                target_type="do",
                parameters=raw_targets,
            )
            self._current_state = iv.apply_forced_nodes(self._current_state)

        return self

    def intervene(
        self,
        intervention: Intervention | Action | Mapping[str, Any],
        *,
        intervention_id: str = "ctx_intervention",
        description: str = "Context-managed causal node override",
    ) -> InterventionContext:
        """Return an `InterventionContext` context manager bound to this engine."""
        return InterventionContext(
            intervention=intervention,
            engine=self,
            intervention_id=intervention_id,
            description=description,
        )

    def step(
        self,
        state: WorldState | None = None,
        action: Action | Sequence[Action] | None = None,
        *,
        exogenous_events: Sequence[ExogenousEvent] = (),
        u_noise: Mapping[str, float] | None = None,
        rng: RandomGenerator | None = None,
    ) -> TransitionStepOutcome:
        """Execute a single transition step through the `HybridTransitionRouter`.

        Yields a deterministic outcome (`is_deterministic=True`, `distribution=None`)
        if only structural rules apply, and a calibrated distribution
        (`is_deterministic=False`, `distribution=CalibratedStateDistribution`) if ML is used.
        """
        base_state = state if state is not None else self._current_state
        if base_state is None:
            raise ValueError(
                "TransitionEngine.step() requires a `state` argument or `initial_state` on the engine."
            )

        outcome = self.router.step(
            state=base_state,
            action=action,
            exogenous_events=exogenous_events,
            u_noise=u_noise,
            active_do_interventions=self._active_do_interventions,
            causal_graph=self.router.causal_graph,
            rng=rng,
        )
        self._causal_execution_log.append(outcome.causal_log_entry)
        self._current_state = outcome.next_state

        primary_action = (
            action
            if isinstance(action, Action)
            else (action[0] if isinstance(action, Sequence) and len(action) > 0 else None)
        )
        if outcome.next_state.step > self.event_log.current_step:
            self.event_log.append(outcome.next_state, action=primary_action)

        return outcome

    def legacy_step(
        self,
        state: WorldState,
        action: Action | Sequence[Action] | None = None,
        *,
        exogenous_events: Sequence[ExogenousEvent] = (),
        u_noise: Mapping[str, float] | None = None,
        rng: RandomGenerator | None = None,
    ) -> WorldState:
        """Backwards-compatible `step()` wrapper returning the next `WorldState` directly."""
        outcome = self.step(
            state=state,
            action=action,
            exogenous_events=exogenous_events,
            u_noise=u_noise,
            rng=rng,
        )
        return outcome.next_state


__all__ = [
    "CalibratedPrediction",
    "CalibratedStateDistribution",
    "HybridTransitionRouter",
    "LearnedPredictorProtocol",
    "NumericalInstabilityError",
    "StructuralEquationFn",
    "StructuralPostRuleFn",
    "StructuralPreRuleFn",
    "TransitionEngine",
    "TransitionStepOutcome",
    "validate_learned_prediction_no_nan",
]
