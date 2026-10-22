"""Canonical world-model baselines and statistical evaluation helpers for EWM benchmarks.

Implements the four mandatory baseline paradigms required to test the enterprise
world-model hypothesis without rigging outcomes:
1. `rules-only` (explicit deterministic rules, fixed K_0, zero learned residual)
2. `learned-only` (empirical ridge/polynomial regression on training transitions, unconstrained)
3. `static hybrid` (structural rules + feasible-projected learned residual, fixed K_0)
4. `hybrid + runtime context discovery` (structural rules + feasible-projected learned residual
   + runtime ContextSnapshot discovery + belief filtering + competence abstention)
"""

from __future__ import annotations

import hashlib
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.protocol import (
    BaselineEvaluationRecord,
    BaselineParadigm,
    BenchmarkTaskType,
    MetricDimension,
    MetricStatisticalSummary,
    ShiftProtocolRecord,
    apply_holm_bonferroni,
)

TRAIN_DOMAIN_TAG = 0xE03_1001
EVAL_DOMAIN_TAG = 0xE03_2002
CONFIG_DOMAIN_TAG = 0xE03_3003


def derive_partition_seeds(
    master_seed: int,
    n_eval_seeds: int = 5,
) -> tuple[int, tuple[int, ...], int]:
    """Derive cryptographically isolated train, eval, and held-out config seeds.

    Guarantees hard seed-partition separation so training generator draws can never
    collide with evaluation scenario draws.
    """
    train_seq = np.random.SeedSequence([int(master_seed), TRAIN_DOMAIN_TAG])
    eval_seq = np.random.SeedSequence([int(master_seed), EVAL_DOMAIN_TAG])
    cfg_seq = np.random.SeedSequence([int(master_seed), CONFIG_DOMAIN_TAG])

    train_seed = int(train_seq.generate_state(1)[0])
    eval_children = eval_seq.spawn(max(1, n_eval_seeds))
    eval_seeds = tuple(int(c.generate_state(1)[0]) for c in eval_children)
    cfg_seed = int(cfg_seq.generate_state(1)[0])

    # Ensure disjointness even in adversarial seed collisions
    if train_seed in eval_seeds:
        train_seed = (train_seed + 7919) & 0x7FFFFFFF
    return train_seed, eval_seeds, cfg_seed


def compute_dataset_scenario_fingerprint(
    family_name: str,
    train_fingerprints: Sequence[str],
    eval_fingerprints: Sequence[str],
    parameters: Mapping[str, Any],
) -> str:
    """Compute a canonical SHA-256 fingerprint over train/eval state hashes and parameters."""
    h = hashlib.sha256()
    h.update(family_name.encode("utf-8"))
    h.update(repr(sorted(parameters.items())).encode("utf-8"))
    for fp in train_fingerprints:
        h.update(fp.encode("utf-8"))
    h.update(b"|EVAL|")
    for fp in eval_fingerprints:
        h.update(fp.encode("utf-8"))
    return h.hexdigest()


class PolynomialRidgeRegressor:
    """Zero-dependency polynomial ridge regressor for empirical transition baselines.

    Fits degree-2 polynomial features with L2 regularization using closed-form SVD/Cholesky
    via NumPy, providing fast, deterministic empirical fitting without external ML libraries.
    """

    def __init__(self, l2_penalty: float = 1e-2, include_quadratic: bool = True) -> None:
        self.l2_penalty = float(l2_penalty)
        self.include_quadratic = bool(include_quadratic)
        self._weights: np.ndarray | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._residual_std: np.ndarray | None = None
        self._support_min: np.ndarray | None = None
        self._support_max: np.ndarray | None = None

    def _expand(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        parts = [np.ones((x.shape[0], 1), dtype=float), x]
        if self.include_quadratic:
            parts.append(x * x)
            # Pairwise interactions for first 4 features if available
            d = min(4, x.shape[1])
            for i in range(d):
                for j in range(i + 1, d):
                    parts.append((x[:, i] * x[:, j]).reshape(-1, 1))
        return np.hstack(parts)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit polynomial ridge regression mapping X (N, D) -> Y (N, K)."""
        X_arr = np.asarray(X, dtype=float)
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)

        self._support_min = np.min(X_arr, axis=0)
        self._support_max = np.max(X_arr, axis=0)
        self._feature_mean = np.mean(X_arr, axis=0)
        self._feature_std = np.where(np.std(X_arr, axis=0) > 1e-8, np.std(X_arr, axis=0), 1.0)

        X_norm = (X_arr - self._feature_mean) / self._feature_std
        Phi = self._expand(X_norm)
        if self.l2_penalty <= 0.0:
            self._weights = np.linalg.lstsq(Phi, Y_arr, rcond=None)[0]
        else:
            reg = self.l2_penalty * np.eye(Phi.shape[1], dtype=float)
            reg[0, 0] = 0.0  # Do not regularize bias term
            self._weights = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ Y_arr)

        preds = Phi @ self._weights
        residuals = Y_arr - preds
        self._residual_std = np.maximum(1e-3, np.std(residuals, axis=0))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict target vector for feature vector x."""
        if self._weights is None or self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("PolynomialRidgeRegressor must be fitted before calling predict().")
        x_arr = np.asarray(x, dtype=float)
        single = x_arr.ndim == 1
        if single:
            x_arr = x_arr.reshape(1, -1)
        x_norm = (x_arr - self._feature_mean) / self._feature_std
        phi = self._expand(x_norm)
        out = phi @ self._weights
        return np.asarray(out[0] if single else out, dtype=float)

    @property
    def residual_std(self) -> np.ndarray:
        if self._residual_std is None:
            return np.array([1.0], dtype=float)
        return self._residual_std

    def is_out_of_support(self, x: np.ndarray, margin_factor: float = 0.25) -> bool:
        """Return True if input feature vector falls outside training support box + margin."""
        if self._support_min is None or self._support_max is None:
            return False
        x_arr = np.asarray(x, dtype=float).ravel()
        span = np.maximum(1.0, self._support_max - self._support_min)
        low = self._support_min - margin_factor * span
        high = self._support_max + margin_factor * span
        return bool(np.any(x_arr < low) or np.any(x_arr > high))


@dataclass
class PerSeedBaselineMetrics:
    """Accumulator for per-seed metrics across all 13 orthogonal dimensions."""

    paradigm: BaselineParadigm
    model_name: str
    seeds: list[int] = field(default_factory=list)
    dimension_values: dict[MetricDimension, list[float]] = field(
        default_factory=lambda: {d: [] for d in MetricDimension}
    )
    model_spreads: dict[MetricDimension, list[float]] = field(
        default_factory=lambda: {d: [] for d in MetricDimension}
    )
    raw_accum: dict[str, list[float]] = field(default_factory=dict)
    task_accum: dict[BenchmarkTaskType, list[float]] = field(default_factory=dict)
    invariant_violations: int = 0
    failed_runs: int = 0
    total_runs: int = 0
    failure_modes: list[str] = field(default_factory=list)
    shift_protocol: ShiftProtocolRecord | None = None

    def record_seed(
        self,
        *,
        seed: int,
        dimensions: Mapping[MetricDimension, float],
        spreads: Mapping[MetricDimension, float] | None = None,
        raw: Mapping[str, float] | None = None,
        tasks: Mapping[BenchmarkTaskType, float] | None = None,
        violations: int = 0,
        failed: bool = False,
    ) -> None:
        self.seeds.append(int(seed))
        self.total_runs += 1
        if failed:
            self.failed_runs += 1
        self.invariant_violations += int(violations)
        for dim in MetricDimension:
            val = float(dimensions.get(dim, 0.0))
            self.dimension_values[dim].append(val)
            if spreads and dim in spreads:
                self.model_spreads[dim].append(float(spreads[dim]))
        if raw:
            for k, v in raw.items():
                self.raw_accum.setdefault(k, []).append(float(v))
        if tasks:
            for t_k, t_v in tasks.items():
                self.task_accum.setdefault(t_k, []).append(float(t_v))


def measure_TimedStep(
    fn: Callable[[], Any],
    n_steps: int,
) -> tuple[Any, float, float, float]:
    """Execute `fn`, returning (result, latency_ms_per_step, memory_kb, throughput_steps_per_sec)."""
    steps = max(1, int(n_steps))
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = fn()
    finally:
        elapsed = max(1e-7, time.perf_counter() - t0)
        _current, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    # Deterministic floor/normalization on timing & memory so metrics are realistic and positive
    latency_ms = float((elapsed * 1000.0) / steps)
    memory_kb = max(0.5, float(peak_bytes) / 1024.0)
    throughput = float(steps / elapsed)
    return result, latency_ms, memory_kb, throughput


def build_baseline_evaluation_records(
    *,
    accumulators: Sequence[PerSeedBaselineMetrics],
    task_types: Sequence[BenchmarkTaskType],
    is_deterministic_oracle: bool,
    oracle_targets: Mapping[MetricDimension, float] | None = None,
    oracle_tolerance: float = 1e-12,
    bootstrap_seed: int = 42,
) -> tuple[BaselineEvaluationRecord, ...]:
    """Compile per-seed accumulators into BaselineEvaluationRecords with Holm-Bonferroni correction."""
    rules_acc = next(
        (a for a in accumulators if a.paradigm == BaselineParadigm.RULES_ONLY),
        accumulators[0],
    )

    # First pass: build raw MetricStatisticalSummary objects
    temp_summaries: dict[BaselineParadigm, dict[MetricDimension, MetricStatisticalSummary]] = {}
    for acc in accumulators:
        dim_map: dict[MetricDimension, MetricStatisticalSummary] = {}
        for dim_idx, dim in enumerate(MetricDimension):
            vals = acc.dimension_values[dim]
            ref_vals = rules_acc.dimension_values[dim]
            spreads = acc.model_spreads.get(dim)

            if is_deterministic_oracle:
                target = oracle_targets.get(dim) if oracle_targets else None
                ref_val = ref_vals[0] if ref_vals else None
                dim_map[dim] = MetricStatisticalSummary.from_deterministic(
                    dimension=dim,
                    value=vals[0] if vals else 0.0,
                    seed=acc.seeds[0] if acc.seeds else bootstrap_seed,
                    oracle_target=target,
                    oracle_tolerance=oracle_tolerance,
                    model_uncertainty_spread=spreads[0] if spreads else 0.0,
                    paired_reference_value=ref_val
                    if acc.paradigm != BaselineParadigm.RULES_ONLY
                    else None,
                    failed_runs=acc.failed_runs,
                    total_runs=acc.total_runs,
                )
            else:
                dim_map[dim] = MetricStatisticalSummary.from_stochastic(
                    dimension=dim,
                    seeds=acc.seeds,
                    values=vals,
                    model_spreads=spreads,
                    paired_reference_values=(
                        ref_vals if acc.paradigm != BaselineParadigm.RULES_ONLY else None
                    ),
                    failed_runs=acc.failed_runs,
                    total_runs=acc.total_runs,
                    bootstrap_seed=bootstrap_seed + dim_idx * 31,
                )
        temp_summaries[acc.paradigm] = dim_map

    # Second pass: apply Holm-Bonferroni across non-reference baselines per dimension
    if not is_deterministic_oracle:
        for dim in MetricDimension:
            raw_ps: dict[str, float] = {}
            for acc in accumulators:
                if acc.paradigm == BaselineParadigm.RULES_ONLY:
                    continue
                s = temp_summaries[acc.paradigm][dim]
                if s.raw_p_value is not None:
                    raw_ps[acc.paradigm.value] = s.raw_p_value
            adj_ps = apply_holm_bonferroni(raw_ps)
            for acc in accumulators:
                if acc.paradigm.value in adj_ps:
                    old_s = temp_summaries[acc.paradigm][dim]
                    temp_summaries[acc.paradigm][dim] = old_s.model_copy(
                        update={"holm_adjusted_p_value": adj_ps[acc.paradigm.value]}
                    )

    records: list[BaselineEvaluationRecord] = []
    for acc in accumulators:
        raw_means = {k: float(np.mean(v)) for k, v in acc.raw_accum.items() if v}
        task_means = {k: float(np.mean(v)) for k, v in acc.task_accum.items() if v}
        records.append(
            BaselineEvaluationRecord(
                paradigm=acc.paradigm,
                model_name=acc.model_name,
                task_types_evaluated=tuple(task_types),
                dimension_summaries=temp_summaries[acc.paradigm],
                task_scores=task_means,
                raw_metrics=raw_means,
                invariants_passed=(acc.invariant_violations == 0),
                invariant_violations=acc.invariant_violations,
                failed_runs=acc.failed_runs,
                total_runs=acc.total_runs,
                failure_modes_exposed=tuple(dict.fromkeys(acc.failure_modes)),
                shift_protocol=acc.shift_protocol,
            )
        )

    return tuple(records)
