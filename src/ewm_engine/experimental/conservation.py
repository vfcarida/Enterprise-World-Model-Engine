"""Architectural conservation layer for hybrid and grey-box enterprise world models.

[EXPERIMENTAL] This module implements exact linear equality and manifold projection
layers (KKT-hPINN / Port-Hamiltonian style). Violations of declared enterprise conservation
laws (resource mass conservation, double-entry balance, zero-sum transfers) are made
structurally impossible by construction, rather than rejected post-hoc.

Anchor papers:
- KKT-hPINN (hard linear equality constraints): arXiv:2402.07251
- Stable Port-Hamiltonian Neural Networks: arXiv:2502.02480
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ResourceId
from ewm_engine.dynamics.base import DynamicsModel
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample


@runtime_checkable
class ConservationLaw(Protocol):
    """Protocol for enterprise conservation laws and structural invariants."""

    @property
    def name(self) -> str:
        """Identifier of the conservation law."""
        ...

    @property
    def resource_ids(self) -> tuple[ResourceId, ...]:
        """Resources governed by this conservation law."""
        ...

    def evaluate_violation(
        self,
        state_before: WorldState,
        state_after: WorldState,
    ) -> float:
        """Compute the scalar magnitude of conservation violation (0.0 if strictly conserved)."""
        ...

    def is_satisfied(
        self,
        state_before: WorldState,
        state_after: WorldState,
        tolerance: float = 1e-7,
    ) -> bool:
        """Return True if violation is within acceptable numerical tolerance."""
        ...


class LinearConservationLaw:
    """Linear conservation equality constraint over resource deltas: A @ delta = b.

    Examples:
        - Mass conservation across transfers: sum(delta_i) = 0  (A = [[1, 1, ...]], b = [0])
        - Double-entry balance: delta_assets - delta_liabilities - delta_equity = 0
    """

    def __init__(
        self,
        matrix: np.ndarray | Sequence[Sequence[float]],
        resource_ids: Sequence[ResourceId],
        target: np.ndarray | Sequence[float] | None = None,
        name: str = "LinearConservationLaw",
    ) -> None:
        self._matrix = np.asarray(matrix, dtype=np.float64)
        if self._matrix.ndim == 1:
            self._matrix = self._matrix.reshape(1, -1)
        self._resource_ids = tuple(resource_ids)
        if self._matrix.shape[1] != len(self._resource_ids):
            raise ValueError(
                f"Matrix columns ({self._matrix.shape[1]}) must match resource count ({len(self._resource_ids)})"
            )

        if target is None:
            self._target = np.zeros(self._matrix.shape[0], dtype=np.float64)
        else:
            self._target = np.asarray(target, dtype=np.float64)
            if self._target.shape[0] != self._matrix.shape[0]:
                raise ValueError(
                    f"Target length ({self._target.shape[0]}) must match matrix rows ({self._matrix.shape[0]})"
                )

        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def resource_ids(self) -> tuple[ResourceId, ...]:
        return self._resource_ids

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def target(self) -> np.ndarray:
        return self._target

    @classmethod
    def zero_sum(
        cls,
        resource_ids: Sequence[ResourceId],
        name: str = "ZeroSumConservation",
    ) -> LinearConservationLaw:
        """Create a zero-sum conservation law: sum(delta_i) = 0."""
        n = len(resource_ids)
        matrix = np.ones((1, n), dtype=np.float64)
        return cls(matrix=matrix, resource_ids=resource_ids, name=name)

    @classmethod
    def double_entry(
        cls,
        debit_resources: Sequence[ResourceId],
        credit_resources: Sequence[ResourceId],
        name: str = "DoubleEntryBalance",
    ) -> LinearConservationLaw:
        """Create a double-entry equality: sum(debit) - sum(credit) = 0."""
        all_res = list(debit_resources) + list(credit_resources)
        row = [1.0] * len(debit_resources) + [-1.0] * len(credit_resources)
        matrix = np.array([row], dtype=np.float64)
        return cls(matrix=matrix, resource_ids=all_res, name=name)

    def evaluate_violation(
        self,
        state_before: WorldState,
        state_after: WorldState,
    ) -> float:
        deltas = np.zeros(len(self._resource_ids), dtype=np.float64)
        for i, rid in enumerate(self._resource_ids):
            before = state_before.get_resource(rid).current
            after = state_after.get_resource(rid).current
            deltas[i] = after - before

        residual = (self._matrix @ deltas) - self._target
        return float(np.linalg.norm(residual, ord=2))

    def is_satisfied(
        self,
        state_before: WorldState,
        state_after: WorldState,
        tolerance: float = 1e-7,
    ) -> bool:
        return self.evaluate_violation(state_before, state_after) <= tolerance


class ConservationProjectionLayer:
    """Architectural projection layer enforcing linear conservation laws by construction.

    Implements KKT-hPINN nullspace projection:
        P = I - A^T (A A^T)^{-1} A
        delta_proj = P @ delta_raw + A^T (A A^T)^{-1} b

    By construction, A @ delta_proj == b identically up to numerical floating-point precision,
    making conservation violations mathematically impossible for ANY neural or empirical output.

    Also supports conservative capacity and non-negativity boundary scaling:
    When bounds [r_min, r_max] would be violated, delta_proj is scaled by alpha in [0, 1]
    such that bounds are strictly satisfied. Because A @ (alpha * delta_proj) = alpha * (A @ delta_proj) = 0,
    linear conservation is simultaneously preserved.
    """

    def __init__(
        self,
        laws: Sequence[LinearConservationLaw],
        enforce_capacity_bounds: bool = True,
        tolerance: float = 1e-9,
        name: str = "ConservationProjectionLayer",
    ) -> None:
        self.laws = list(laws)
        self.enforce_capacity_bounds = enforce_capacity_bounds
        self.tolerance = tolerance
        self.name = name

        # Collect unique ordered resources
        res_set: list[ResourceId] = []
        for law in self.laws:
            for rid in law.resource_ids:
                if rid not in res_set:
                    res_set.append(rid)
        self.all_resource_ids = tuple(res_set)
        self.d = len(self.all_resource_ids)
        self._rid_to_idx = {rid: i for i, rid in enumerate(self.all_resource_ids)}

        # Compile global linear system A_global @ delta = b_global
        if self.laws and self.d > 0:
            rows: list[np.ndarray] = []
            targets: list[float] = []
            for law in self.laws:
                m = law.matrix.shape[0]
                for row_idx in range(m):
                    global_row = np.zeros(self.d, dtype=np.float64)
                    for local_col, rid in enumerate(law.resource_ids):
                        global_col = self._rid_to_idx[rid]
                        global_row[global_col] = law.matrix[row_idx, local_col]
                    rows.append(global_row)
                    targets.append(float(law.target[row_idx]))

            self._A = np.array(rows, dtype=np.float64)
            self._b = np.array(targets, dtype=np.float64)

            # Precompute Moore-Penrose pseudo-inverse projection operator
            # P = I - A_pinv @ A
            A_pinv = np.linalg.pinv(self._A)
            self._P = np.eye(self.d, dtype=np.float64) - (A_pinv @ self._A)
            self._offset = A_pinv @ self._b
        else:
            self._A = np.zeros((0, self.d), dtype=np.float64)
            self._b = np.zeros(0, dtype=np.float64)
            self._P = np.eye(self.d, dtype=np.float64)
            self._offset = np.zeros(self.d, dtype=np.float64)

    def project_deltas(
        self,
        current_state: WorldState,
        raw_deltas: dict[ResourceId, float],
    ) -> dict[ResourceId, float]:
        """Project raw/unconstrained deltas onto the linear conservation manifold.

        Args:
            current_state: WorldState prior to applying deltas (used for capacity checks).
            raw_deltas: Dictionary of unconstrained resource deltas.

        Returns:
            Dictionary of projected deltas strictly satisfying all declared conservation laws.
        """
        if self.d == 0 or len(self.laws) == 0:
            return dict(raw_deltas)

        # Assemble delta vector
        delta_vec = np.zeros(self.d, dtype=np.float64)
        for rid, val in raw_deltas.items():
            if rid in self._rid_to_idx:
                delta_vec[self._rid_to_idx[rid]] = float(val)

        # Apply orthogonal KKT projection
        proj_vec = (self._P @ delta_vec) + self._offset

        # Enforce boundary constraints via conservative alpha scaling if enabled
        if self.enforce_capacity_bounds:
            proj_vec = self._apply_boundary_scaling(current_state, proj_vec)

        # Build output dictionary
        output: dict[ResourceId, float] = dict(raw_deltas)
        for rid, idx in self._rid_to_idx.items():
            output[rid] = float(proj_vec[idx])

        return output

    def _apply_boundary_scaling(
        self,
        current_state: WorldState,
        proj_vec: np.ndarray,
    ) -> np.ndarray:
        """Find maximal scalar alpha in [0, 1] preserving linear conservation while obeying bounds."""
        alpha = 1.0

        for rid, idx in self._rid_to_idx.items():
            delta_i = proj_vec[idx]
            if abs(delta_i) < 1e-12:
                continue
            if rid not in current_state.resources:
                continue

            res = current_state.get_resource(rid)
            current = res.current
            min_val = res.min_value
            max_val = res.max_value

            if delta_i > 0 and not np.isinf(max_val):
                headroom = max(0.0, max_val - current)
                if delta_i > headroom:
                    alpha_i = headroom / delta_i
                    alpha = min(alpha, alpha_i)
            elif delta_i < 0:
                available = max(0.0, current - min_val)
                if abs(delta_i) > available:
                    alpha_i = available / abs(delta_i)
                    alpha = min(alpha, alpha_i)

        alpha = max(0.0, min(1.0, alpha))
        # Guard against 1-ULP IEEE-754 rounding overshoot in `proj_vec * alpha`
        for _ in range(4):
            overshoot = False
            scaled = proj_vec * alpha
            for rid, idx in self._rid_to_idx.items():
                if rid not in current_state.resources:
                    continue
                res = current_state.get_resource(rid)
                new_val = res.current + float(scaled[idx])
                if new_val < res.min_value or (
                    not np.isinf(res.max_value) and new_val > res.max_value
                ):
                    overshoot = True
                    break
            if not overshoot or alpha <= 0.0:
                break
            alpha = float(np.nextafter(alpha, 0.0))

        return proj_vec * alpha


class PortHamiltonianConservationLayer:
    """Port-Hamiltonian parameterized layer for conservative/dissipative dynamics.

    Implements:
        delta = (J - R) @ grad_H
    where J = -J^T (skew-symmetric lossless routing matrix) and R >= 0 (positive semi-definite
    dissipation/frictional matrix).

    Guarantees that total Hamiltonian energy H(x) is non-increasing (Lyapunov stability):
        d/dt H <= 0 by construction.
    """

    def __init__(
        self,
        resource_ids: Sequence[ResourceId],
        dissipation_rate: float = 0.0,
        name: str = "PortHamiltonianLayer",
    ) -> None:
        self.resource_ids = tuple(resource_ids)
        self.d = len(self.resource_ids)
        self.dissipation_rate = max(0.0, dissipation_rate)
        self.name = name

        # Construct default skew-symmetric cyclic circulation J
        self._J = np.zeros((self.d, self.d), dtype=np.float64)
        for i in range(self.d - 1):
            self._J[i, i + 1] = 1.0
            self._J[i + 1, i] = -1.0
        if self.d > 2:
            self._J[self.d - 1, 0] = 1.0
            self._J[0, self.d - 1] = -1.0

        # Construct symmetric positive semi-definite dissipation matrix R
        self._R = np.eye(self.d, dtype=np.float64) * self.dissipation_rate

    @property
    def skew_symmetric_matrix(self) -> np.ndarray:
        return self._J

    @property
    def dissipation_matrix(self) -> np.ndarray:
        return self._R

    def compute_flow(
        self,
        current_levels: np.ndarray | Sequence[float],
    ) -> np.ndarray:
        """Compute Port-Hamiltonian flow vector: delta = (J - R) @ x."""
        x = np.asarray(current_levels, dtype=np.float64)
        return (self._J - self._R) @ x


class ConservationHonestyReport(BaseModel):
    """Honesty metric evaluating structural constraint satisfaction rate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(description="Name of the dynamics model evaluated.")
    samples_evaluated: int = Field(description="Total number of transition steps tested.")
    violations_detected: int = Field(
        description="Number of transition steps with conservation violations."
    )
    violation_rate: float = Field(
        description="Empirical violation frequency in [0.0, 1.0]. 0.0 indicates structural invariance."
    )
    max_absolute_violation: float = Field(
        description="Worst-case scalar magnitude of conservation violation."
    )
    mean_absolute_violation: float = Field(
        description="Average scalar magnitude of conservation violation."
    )
    is_structurally_conserved: bool = Field(
        description="True if violation rate is strictly 0.0 within numerical tolerance."
    )
    laws_checked: tuple[str, ...] = Field(
        default_factory=tuple, description="Identifiers of conservation laws audited."
    )
    unconstrained_comparison: dict[str, Any] | None = Field(
        default=None,
        description="Optional comparative metrics against an unconstrained learned baseline.",
    )


def evaluate_conservation_honesty(
    model: DynamicsModel,
    dataset: TransitionDataset | Sequence[TransitionSample],
    laws: Sequence[ConservationLaw],
    tolerance: float = 1e-6,
    unconstrained_baseline: DynamicsModel | None = None,
) -> ConservationHonestyReport:
    """Evaluate and report the conservation satisfaction honesty metric.

    Audits whether any transition output violates declared conservation laws.
    For models equipped with an architectural ConservationProjectionLayer,
    violation_rate is structurally zero.
    """
    rng = np.random.default_rng(42)
    violations = 0
    max_viol = 0.0
    total_viol = 0.0
    count = 0

    for sample in dataset:
        count += 1
        res = model.transition(
            state=sample.state,
            actions=sample.actions,
            exogenous_events=sample.events,
            rng=rng,
        )

        step_max_viol = 0.0
        for law in laws:
            viol = law.evaluate_violation(sample.state, res.next_state)
            if viol > step_max_viol:
                step_max_viol = viol

        if step_max_viol > tolerance:
            violations += 1

        total_viol += step_max_viol
        if step_max_viol > max_viol:
            max_viol = step_max_viol

    total_samples = max(1, count)
    violation_rate = violations / total_samples
    mean_viol = total_viol / total_samples

    unconstrained_comp: dict[str, Any] | None = None
    if unconstrained_baseline is not None:
        base_rep = evaluate_conservation_honesty(
            model=unconstrained_baseline,
            dataset=dataset,
            laws=laws,
            tolerance=tolerance,
        )
        unconstrained_comp = {
            "model_name": base_rep.model_name,
            "violation_rate": base_rep.violation_rate,
            "max_absolute_violation": base_rep.max_absolute_violation,
            "mean_absolute_violation": base_rep.mean_absolute_violation,
        }

    return ConservationHonestyReport(
        model_name=getattr(model, "name", "DynamicsModel"),
        samples_evaluated=count,
        violations_detected=violations,
        violation_rate=violation_rate,
        max_absolute_violation=max_viol,
        mean_absolute_violation=mean_viol,
        is_structurally_conserved=(violation_rate == 0.0),
        laws_checked=tuple(law.name for law in laws),
        unconstrained_comparison=unconstrained_comp,
    )
