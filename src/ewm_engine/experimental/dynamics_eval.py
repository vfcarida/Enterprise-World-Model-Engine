"""Evaluation harness and benchmark protocol for learned and empirical dynamics models.

Evaluates:
1. One-step prediction error (per-resource and aggregate MAE, RMSE, MAPE, DreamerV3 Symlog).
2. Multi-step autoregressive rollout divergence against reference ground truth.
3. Systemic invariant preservation (non-negativity, capacity bounds, mass conservation).
4. Stochastic prediction calibration (CRPS and quantile coverage).
5. Interventional error under held-out / shifted action distributions.

Reference Anchors:
- Physics-IQ (arXiv:2501.09038): Surface error does not imply systemic understanding.
- Sora Physical-Law Critique (arXiv:2411.02385): Testing invariants, not just pixels/metrics.
- DreamerV3 (arXiv:2301.04104): Symlog normalization for heterogeneous org metrics.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.world import World
from ewm_engine.dynamics.base import DynamicsModel
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.evaluation.uncertainty import compute_crps
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Symlog & Symexp Normalization (DreamerV3, arXiv:2301.04104)
# ---------------------------------------------------------------------------


def symlog(x: float | np.ndarray) -> Any:
    """Symmetric logarithmic transformation: sign(x) * ln(|x| + 1).

    Compresses wide dynamic ranges (e.g. monetary millions vs staffing units)
    while preserving sign and gradient behavior near zero.
    """
    arr = np.asarray(x, dtype=np.float64)
    res = np.sign(arr) * np.log1p(np.abs(arr))
    return float(res) if np.isscalar(x) else res


def symexp(y: float | np.ndarray) -> Any:
    """Inverse of symlog: sign(y) * (exp(|y|) - 1)."""
    arr = np.asarray(y, dtype=np.float64)
    res = np.sign(arr) * np.expm1(np.abs(arr))
    return float(res) if np.isscalar(y) else res


# ---------------------------------------------------------------------------
# Dataset Helpers & Simulation Collection
# ---------------------------------------------------------------------------


def from_trajectory(trajectory: Trajectory) -> TransitionDataset:
    """Extract observed transitions (S_t, A_t, E_t, S_{t+1}) from a simulation trajectory."""
    dataset = TransitionDataset()
    curr_state = trajectory.initial_state

    for step in trajectory.steps:
        next_s = step.transition_result.next_state
        sample = TransitionSample(
            state=curr_state,
            actions=tuple(step.actions_accepted),
            events=tuple(step.exogenous_events),
            next_state=next_s,
        )
        dataset.add(sample)
        curr_state = next_s

    return dataset


def collect_transition_dataset(
    world: World,
    scenario: Scenario,
    engine: SimulationEngine | None = None,
) -> TransitionDataset:
    """Simulate rollouts on a world and scenario, packaging all transitions into a TransitionDataset."""
    sim_engine = engine or SimulationEngine()
    result = sim_engine.run(world, scenario)
    dataset = TransitionDataset()

    for traj in result.trajectories:
        traj_dataset = from_trajectory(traj)
        for sample in traj_dataset:
            dataset.add(sample)

    return dataset


# ---------------------------------------------------------------------------
# Structured Evaluation Metric Result Models
# ---------------------------------------------------------------------------


class InvariantCheckResult(BaseModel):
    """Evaluation of structural and organizational invariant preservation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_transitions_checked: int = Field(ge=0)
    violations_count: int = Field(ge=0)
    violation_rate: float = Field(ge=0.0, le=1.0)
    bounds_violations: int = Field(ge=0)
    non_negativity_violations: int = Field(default=0, ge=0)
    conservation_violations: int = Field(ge=0)
    violating_resources: list[str] = Field(default_factory=list)
    passed: bool = Field(description="True if zero invariant violations occurred.")
    violation_details: list[dict[str, Any]] = Field(default_factory=list)


class OneStepErrorResult(BaseModel):
    """One-step state prediction accuracy metrics across resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mae: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    mape: float = Field(ge=0.0)
    symlog_error: float = Field(
        ge=0.0, description="Mean absolute error under DreamerV3 symlog scaling."
    )
    per_resource_mae: dict[str, float] = Field(default_factory=dict)
    per_resource_rmse: dict[str, float] = Field(default_factory=dict)
    per_resource_symlog: dict[str, float] = Field(default_factory=dict)


class RolloutDivergenceResult(BaseModel):
    """Multi-step compounding divergence between evaluated model and reference trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(ge=1)
    step_divergences: list[float] = Field(description="L2 state divergence at each step.")
    mean_divergence: float = Field(ge=0.0)
    final_divergence: float = Field(ge=0.0)
    max_divergence: float = Field(ge=0.0)
    drift_rate: float = Field(description="Average divergence growth per rollout step.")


class CalibrationResult(BaseModel):
    """Probabilistic prediction calibration metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crps: float = Field(ge=0.0, description="Continuous Ranked Probability Score.")
    coverage: float = Field(
        ge=0.0, le=1.0, description="Empirical fraction of truths in [p05, p95] band."
    )
    nominal_coverage: float = Field(default=0.90)
    is_calibrated: bool = Field(
        description="Whether empirical coverage is within acceptable tolerance."
    )


class InterventionalShiftResult(BaseModel):
    """Evaluates generalization and degradation under out-of-distribution interventional actions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    in_distribution_mae: float = Field(ge=0.0)
    interventional_mae: float = Field(ge=0.0)
    interventional_gap: float = Field(description="interventional_mae minus in_distribution_mae.")
    gap_ratio: float = Field(description="interventional_mae / in_distribution_mae.")
    shift_detected: bool = Field(
        description="True if model error statistically degrades under interventional shift."
    )


class DynamicsEvaluationReport(BaseModel):
    """Comprehensive evaluation report measuring a dynamics model across all axes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    evaluated_at: float = Field(default_factory=time.time)
    one_step: OneStepErrorResult
    invariants: InvariantCheckResult
    rollout: RolloutDivergenceResult | None = None
    calibration: CalibrationResult | None = None
    interventional: InterventionalShiftResult | None = None
    overall_valid: bool = Field(
        description="Consistency gate: True only if invariants pass AND numeric error is within bounds."
    )

    def summary_table(self) -> str:
        """Render a formatted, human-readable evaluation summary."""
        lines = [
            f"=== Dynamics Model Evaluation Report: {self.model_name} ===",
            f"Overall Status: {'PASSED' if self.overall_valid else 'FAILED / REJECTED'}",
            "",
            "1. Systemic Invariants & Physical Constraints Gate:",
            f"   - Total Checked: {self.invariants.total_transitions_checked}",
            f"   - Violations: {self.invariants.violations_count} (Rate: {self.invariants.violation_rate * 100:.2f}%)",
            f"   - Bounds Violations: {self.invariants.bounds_violations}",
            f"   - Conservation Violations: {self.invariants.conservation_violations}",
            f"   - Gate Outcome: {'PASS' if self.invariants.passed else 'FAIL (CRITICAL REJECTION)'}",
            "",
            "2. One-Step Predictive Accuracy:",
            f"   - Aggregate MAE: {self.one_step.mae:.4f}",
            f"   - Aggregate RMSE: {self.one_step.rmse:.4f}",
            f"   - Symlog Error (Scale-Invariant): {self.one_step.symlog_error:.4f}",
            f"   - Aggregate MAPE: {self.one_step.mape * 100:.2f}%",
        ]

        if self.one_step.per_resource_mae:
            lines.append("   - Per-Resource Breakdown:")
            for r_id, mae in self.one_step.per_resource_mae.items():
                sym = self.one_step.per_resource_symlog.get(r_id, 0.0)
                lines.append(f"       * {r_id:<20}: MAE = {mae:.4f} | Symlog = {sym:.4f}")

        if self.rollout is not None:
            lines.extend(
                [
                    "",
                    "3. Multi-Step Autoregressive Rollout Divergence:",
                    f"   - Horizon: {self.rollout.horizon} steps",
                    f"   - Mean Divergence: {self.rollout.mean_divergence:.4f}",
                    f"   - Final Divergence: {self.rollout.final_divergence:.4f}",
                    f"   - Drift Rate (Compounding Error): {self.rollout.drift_rate:+.4f}/step",
                ]
            )

        if self.calibration is not None:
            lines.extend(
                [
                    "",
                    "4. Stochastic Prediction Calibration:",
                    f"   - CRPS: {self.calibration.crps:.4f}",
                    f"   - Empirical Coverage [p05, p95]: {self.calibration.coverage * 100:.1f}% (Nominal: {self.calibration.nominal_coverage * 100:.1f}%)",
                    f"   - Calibrated: {self.calibration.is_calibrated}",
                ]
            )

        if self.interventional is not None:
            lines.extend(
                [
                    "",
                    "5. Interventional Distribution Shift Detection:",
                    f"   - In-Distribution MAE: {self.interventional.in_distribution_mae:.4f}",
                    f"   - Interventional MAE: {self.interventional.interventional_mae:.4f}",
                    f"   - Generalization Gap: {self.interventional.interventional_gap:+.4f} (Ratio: {self.interventional.gap_ratio:.2f}x)",
                    f"   - Shift Degradation Detected: {self.interventional.shift_detected}",
                ]
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Evaluation Harness Functions
# ---------------------------------------------------------------------------


def evaluate_one_step(
    model: DynamicsModel,
    dataset: TransitionDataset,
    conservation_groups: Sequence[Sequence[str]] | None = None,
    seed: int = 42,
) -> tuple[OneStepErrorResult, InvariantCheckResult]:
    """Evaluate one-step predictive accuracy and systemic invariant preservation."""
    if len(dataset) == 0:
        raise ValueError("Cannot evaluate model on empty TransitionDataset.")

    rng = np.random.default_rng(seed)
    resource_ids = sorted(dataset[0].state.resources.keys())

    abs_errors: dict[str, list[float]] = {r: [] for r in resource_ids}
    sq_errors: dict[str, list[float]] = {r: [] for r in resource_ids}
    pct_errors: dict[str, list[float]] = {r: [] for r in resource_ids}
    sym_errors: dict[str, list[float]] = {r: [] for r in resource_ids}

    bounds_violations = 0
    non_negativity_violations = 0
    conservation_violations = 0
    violations_count = 0
    violating_res_set: set[str] = set()
    violation_details: list[dict[str, Any]] = []

    for idx, sample in enumerate(dataset):
        # Predict next state using model
        transition_res = model.transition(
            state=sample.state,
            actions=sample.actions,
            exogenous_events=sample.events,
            rng=rng,
        )
        pred_state = transition_res.next_state
        true_state = sample.next_state

        step_had_violation = False

        # 1. Invariant: Check bounds on all resources
        for r_id in resource_ids:
            if r_id not in pred_state.resources:
                continue
            r_pred = pred_state.get_resource(r_id)

            # Non-negativity and max_value bounds check
            val = r_pred.current
            min_v = r_pred.min_value if r_pred.min_value is not None else -np.inf
            max_v = r_pred.max_value if r_pred.max_value is not None else np.inf

            if val < (min_v - 1e-6) or val > (max_v + 1e-6):
                bounds_violations += 1
                step_had_violation = True
                violating_res_set.add(r_id)
                if val < -1e-6 and min_v >= 0.0:
                    non_negativity_violations += 1
                violation_details.append(
                    {
                        "sample_idx": idx,
                        "type": "bounds_violation",
                        "resource": r_id,
                        "value": val,
                        "bounds": [min_v, max_v],
                    }
                )

        # 2. Invariant: Conservation of mass across declared groups
        if conservation_groups:
            for group in conservation_groups:
                initial_sum = sum(
                    sample.state.get_resource(r).current
                    for r in group
                    if sample.state.get_resource(r) is not None
                )
                pred_sum = sum(
                    pred_state.get_resource(r).current
                    for r in group
                    if pred_state.get_resource(r) is not None
                )
                if abs(initial_sum - pred_sum) > 1e-4:
                    conservation_violations += 1
                    step_had_violation = True
                    for r in group:
                        violating_res_set.add(r)
                    violation_details.append(
                        {
                            "sample_idx": idx,
                            "type": "conservation_violation",
                            "group": list(group),
                            "initial_sum": initial_sum,
                            "pred_sum": pred_sum,
                            "delta": pred_sum - initial_sum,
                        }
                    )

        if step_had_violation:
            violations_count += 1

        # 3. Numeric accuracy calculations
        for r_id in resource_ids:
            true_r = true_state.get_resource(r_id)
            pred_r = pred_state.get_resource(r_id)
            y = true_r.current if true_r is not None else 0.0
            y_hat = pred_r.current if pred_r is not None else 0.0

            diff = abs(y - y_hat)
            abs_errors[r_id].append(diff)
            sq_errors[r_id].append(diff**2)
            sym_errors[r_id].append(abs(symlog(y) - symlog(y_hat)))

            denom = abs(y) if abs(y) > 1e-6 else 1.0
            pct_errors[r_id].append(diff / denom)

    # Aggregate accuracy
    per_res_mae = {r: float(np.mean(abs_errors[r])) for r in resource_ids}
    per_res_rmse = {r: float(np.sqrt(np.mean(sq_errors[r]))) for r in resource_ids}
    per_res_symlog = {r: float(np.mean(sym_errors[r])) for r in resource_ids}

    mean_mae = float(np.mean(list(per_res_mae.values())))
    mean_rmse = float(np.mean(list(per_res_rmse.values())))
    mean_mape = float(np.mean([np.mean(pct_errors[r]) for r in resource_ids]))
    mean_symlog = float(np.mean(list(per_res_symlog.values())))

    one_step_result = OneStepErrorResult(
        mae=round(mean_mae, 6),
        rmse=round(mean_rmse, 6),
        mape=round(mean_mape, 6),
        symlog_error=round(mean_symlog, 6),
        per_resource_mae={r: round(v, 6) for r, v in per_res_mae.items()},
        per_resource_rmse={r: round(v, 6) for r, v in per_res_rmse.items()},
        per_resource_symlog={r: round(v, 6) for r, v in per_res_symlog.items()},
    )

    n_samples = len(dataset)
    violation_rate = violations_count / n_samples
    invariant_result = InvariantCheckResult(
        total_transitions_checked=n_samples,
        violations_count=violations_count,
        violation_rate=round(violation_rate, 4),
        bounds_violations=bounds_violations,
        non_negativity_violations=non_negativity_violations,
        conservation_violations=conservation_violations,
        violating_resources=sorted(violating_res_set),
        passed=violations_count == 0,
        violation_details=violation_details,
    )

    return one_step_result, invariant_result


def evaluate_rollout_divergence(
    model: DynamicsModel,
    reference_world: World,
    scenario: Scenario,
    engine: SimulationEngine | None = None,
) -> RolloutDivergenceResult:
    """Evaluate multi-step autoregressive rollout divergence against a ground-truth world."""
    sim_engine = engine or SimulationEngine()

    # 1. Run reference ground-truth simulation
    ref_result = sim_engine.run(reference_world, scenario)
    ref_traj = ref_result.trajectories[0]

    # 2. Construct test world and swap in the evaluated dynamics model
    test_world = World(
        state=reference_world.initial_state,
        dynamics=model,
        constraints=reference_world.constraints.clone(),
        actors=list(reference_world.actors),
        event_sources=list(reference_world.event_sources),
    )
    test_result = sim_engine.run(test_world, scenario)
    test_traj = test_result.trajectories[0]

    horizon = min(len(ref_traj.steps), len(test_traj.steps))
    if horizon == 0:
        return RolloutDivergenceResult(
            horizon=0,
            step_divergences=[],
            mean_divergence=0.0,
            final_divergence=0.0,
            max_divergence=0.0,
            drift_rate=0.0,
        )

    resource_ids = sorted(reference_world.initial_state.resources.keys())
    step_divergences: list[float] = []

    for h in range(horizon):
        ref_s = ref_traj.steps[h].transition_result.next_state
        test_s = test_traj.steps[h].transition_result.next_state

        v_ref = np.array([ref_s.get_resource(r).current for r in resource_ids])
        v_test = np.array([test_s.get_resource(r).current for r in resource_ids])

        dist = float(np.linalg.norm(v_ref - v_test))
        step_divergences.append(round(dist, 6))

    mean_div = float(np.mean(step_divergences))
    final_div = step_divergences[-1]
    max_div = float(np.max(step_divergences))

    # Linear drift rate (slope of error divergence over time)
    if horizon > 1:
        x = np.arange(horizon)
        slope, _ = np.polyfit(x, step_divergences, 1)
        drift_rate = float(slope)
    else:
        drift_rate = 0.0

    return RolloutDivergenceResult(
        horizon=horizon,
        step_divergences=step_divergences,
        mean_divergence=round(mean_div, 6),
        final_divergence=round(final_div, 6),
        max_divergence=round(max_div, 6),
        drift_rate=round(drift_rate, 6),
    )


def evaluate_stochastic_calibration(
    model: DynamicsModel,
    test_dataset: TransitionDataset,
    samples: int = 50,
    seed: int = 42,
) -> CalibrationResult:
    """Measure quantile coverage and CRPS calibration for stochastic dynamics models."""
    if len(test_dataset) == 0:
        raise ValueError("Cannot calibrate on empty dataset.")

    rng = np.random.default_rng(seed)
    resource_ids = sorted(test_dataset[0].state.resources.keys())
    crps_values: list[float] = []
    covered_count = 0
    total_checks = 0

    for sample in test_dataset:
        # Sample multiple stochastic transitions
        predicted_vals: dict[str, list[float]] = {r: [] for r in resource_ids}
        for _ in range(samples):
            res = model.transition(
                state=sample.state,
                actions=sample.actions,
                exogenous_events=sample.events,
                rng=rng,
            )
            for r_id in resource_ids:
                predicted_vals[r_id].append(res.next_state.get_resource(r_id).current)

        for r_id in resource_ids:
            truth = sample.next_state.get_resource(r_id).current
            preds = np.array(predicted_vals[r_id])

            # CRPS via empirical CDF
            crps_values.append(compute_crps(preds.tolist(), truth))

            # Empirical [p05, p95] coverage
            p05, p95 = np.percentile(preds, [5.0, 95.0])
            if p05 <= truth <= p95:
                covered_count += 1
            total_checks += 1

    mean_crps = float(np.mean(crps_values)) if crps_values else 0.0
    empirical_coverage = covered_count / total_checks if total_checks > 0 else 0.0
    is_calibrated = 0.75 <= empirical_coverage <= 0.98

    return CalibrationResult(
        crps=round(mean_crps, 6),
        coverage=round(empirical_coverage, 4),
        nominal_coverage=0.90,
        is_calibrated=is_calibrated,
    )


def evaluate_interventional_shift(
    model: DynamicsModel,
    in_dist_dataset: TransitionDataset,
    out_dist_dataset: TransitionDataset,
    seed: int = 42,
) -> InterventionalShiftResult:
    """Detect whether dynamics model error degrades under out-of-distribution interventional actions."""
    in_err, _ = evaluate_one_step(model, in_dist_dataset, seed=seed)
    out_err, _ = evaluate_one_step(model, out_dist_dataset, seed=seed)

    gap = out_err.mae - in_err.mae
    ratio = out_err.mae / max(1e-6, in_err.mae)
    # Shift is detected if interventional error is at least 15% higher
    shift_detected = out_err.mae > (in_err.mae * 1.15)

    return InterventionalShiftResult(
        in_distribution_mae=round(in_err.mae, 6),
        interventional_mae=round(out_err.mae, 6),
        interventional_gap=round(gap, 6),
        gap_ratio=round(ratio, 4),
        shift_detected=shift_detected,
    )


def evaluate_dynamics_model(
    model: DynamicsModel,
    dataset: TransitionDataset,
    reference_world: World | None = None,
    rollout_scenario: Scenario | None = None,
    out_dist_dataset: TransitionDataset | None = None,
    conservation_groups: Sequence[Sequence[str]] | None = None,
    stochastic_samples: int = 0,
    seed: int = 42,
) -> DynamicsEvaluationReport:
    """Execute complete multi-faceted evaluation of any DynamicsModel.

    Mandatory Consistency Gate:
    Models that violate physical bounds or conservation laws are flagged as failed,
    independent of numeric prediction error.
    """
    model_name = getattr(model, "name", type(model).__name__)

    # 1. One-step predictive error and systemic invariant verification
    one_step_res, invariant_res = evaluate_one_step(
        model=model,
        dataset=dataset,
        conservation_groups=conservation_groups,
        seed=seed,
    )

    # 2. Multi-step autoregressive rollout divergence (optional if world/scenario provided)
    rollout_res = None
    if reference_world is not None and rollout_scenario is not None:
        rollout_res = evaluate_rollout_divergence(
            model=model,
            reference_world=reference_world,
            scenario=rollout_scenario,
        )

    # 3. Calibration (optional if stochastic_samples > 0)
    calibration_res = None
    if stochastic_samples > 0:
        calibration_res = evaluate_stochastic_calibration(
            model=model,
            test_dataset=dataset,
            samples=stochastic_samples,
            seed=seed,
        )

    # 4. Interventional shift detection (optional if out-of-dist dataset provided)
    interventional_res = None
    if out_dist_dataset is not None and len(out_dist_dataset) > 0:
        interventional_res = evaluate_interventional_shift(
            model=model,
            in_dist_dataset=dataset,
            out_dist_dataset=out_dist_dataset,
            seed=seed,
        )

    # Overall validity requires passing the invariant gate
    overall_valid = invariant_res.passed and (one_step_res.symlog_error < 10.0)

    return DynamicsEvaluationReport(
        model_name=model_name,
        one_step=one_step_res,
        invariants=invariant_res,
        rollout=rollout_res,
        calibration=calibration_res,
        interventional=interventional_res,
        overall_valid=overall_valid,
    )
