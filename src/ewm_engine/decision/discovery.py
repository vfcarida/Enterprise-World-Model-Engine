"""Scenario discovery via Patient Rule Induction Method (PRIM) and CART.

Scientific Grounding:
    - Friedman, J. H., & Fisher, N. I. (1999). "Bump hunting in high-dimensional data."
      Statistics and Computing, 9(2), 123-143. (Foundational PRIM paper).
    - Bryant, B. P., & Lempert, R. J. (2010). "Thinking through the unthinkable:
      scenario discovery for robust decision making." Technological Forecasting and
      Social Change, 77(1), 34-49. (RDM scenario discovery with PRIM/CART).
    - Kwakkel, J. H. (2017). "The Exploratory Modelling Workbench: An open source toolkit
      for exploratory modelling, scenario discovery, and robust decision making."
      Environmental Modelling & Software, 96, 239-250.

Conforms to Track T4: Core Substrate (Zero-dep, pure NumPy/stdlib).
Optional adapter for EMA Workbench (`[decision]` extra).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256, normalize_canonical_value
from ewm_engine.experimentation.params import ParameterSpace


def is_ema_workbench_available() -> bool:
    """Return True if ema_workbench is installed and importable."""
    try:
        import ema_workbench  # noqa: F401

        return True
    except ImportError:
        return False


class VulnerabilityBox(BaseModel):
    """An interpretable hyper-rectangular subspace identifying policy vulnerabilities.

    Attributes:
        box_id: Unique identifier for the box.
        bounds: Mapping of parameter name to [lower, upper] interval.
        coverage: Proportion of all system failures captured within this box (recall).
        density: Proportion of samples inside the box that are failures (precision).
        support: Number of samples falling within this box.
        total_samples: Total number of evaluated samples in the dataset.
        restricted_dimensions: Dimensions whose bounds are strictly narrower than the space.
        quasi_p_value: Significance of density enhancement over random binomial subsampling.
        description: Human-readable predicate describing the failure conditions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    box_id: str = Field(description="Unique identifier for the vulnerability box.")
    bounds: dict[str, tuple[float, float]] = Field(
        description="Restricted interval [lower, upper] for each parameter dimension."
    )
    coverage: float = Field(description="Proportion of total failures contained in box (0 to 1).")
    density: float = Field(description="Proportion of samples in box that are failures (0 to 1).")
    support: int = Field(description="Number of samples contained within the box.")
    total_samples: int = Field(description="Total sample count across the ensemble.")
    restricted_dimensions: tuple[str, ...] = Field(
        description="Parameters whose boundaries are restricted relative to full space."
    )
    quasi_p_value: float = Field(
        default=1.0,
        description="Quasi-p-value evaluating density enhancement significance.",
    )
    description: str = Field(default="", description="Human-readable rule representation.")

    def is_inside(self, point: dict[str, Any]) -> bool:
        """Check whether a concrete parameter point falls within this box."""
        for dim, (low, high) in self.bounds.items():
            if dim in point:
                val = float(point[dim])
                if val < low or val > high:
                    return False
        return True

    def format_rule(self) -> str:
        """Generate an interpretable logical expression representing the box."""
        if not self.restricted_dimensions:
            return "ALL_REGIONS (Unrestricted)"
        clauses: list[str] = []
        for dim in self.restricted_dimensions:
            low, high = self.bounds[dim]
            clauses.append(f"({low:.2f} <= {dim} <= {high:.2f})")
        return " AND ".join(clauses)


class PeelingStep(BaseModel):
    """Audit record of a single peeling iteration in PRIM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(description="Iteration step number.")
    coverage: float = Field(description="Coverage at this step.")
    density: float = Field(description="Density at this step.")
    support: int = Field(description="Number of points remaining inside the box.")
    peeled_dimension: str = Field(description="Parameter peeled at this step.")
    peeled_direction: Literal["lower", "upper"] = Field(
        description="Whether lower or upper quantile was removed."
    )
    peeled_bound_value: float = Field(description="New boundary value adopted.")


class ScenarioDiscoveryResult(BaseModel):
    """Audit result of scenario discovery over an ensemble rollout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy_id: str = Field(description="Identifier of the evaluated decision / strategy.")
    boxes: tuple[VulnerabilityBox, ...] = Field(
        description="Discovered vulnerability boxes ordered by significance."
    )
    primary_box: VulnerabilityBox | None = Field(
        default=None,
        description="Top-ranked vulnerability box balancing coverage and density.",
    )
    peeling_trajectory: tuple[PeelingStep, ...] = Field(
        default_factory=tuple,
        description="Full trajectory of peeling iterations.",
    )
    algorithm: Literal["prim", "cart", "hybrid", "ema_workbench"] = Field(
        default="prim",
        description="Discovery algorithm utilized.",
    )
    discovery_fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint ensuring reproducible discovery."
    )
    total_failures: int = Field(description="Total count of failure events in the ensemble.")
    total_samples: int = Field(description="Total ensemble sample count.")


def _calculate_quasi_p_value(k: int, n: int, total_k: int, total_n: int) -> float:
    """Compute quasi-p-value via normal approximation to hypergeometric test."""
    if total_n <= 0 or n <= 0 or total_k <= 0 or k <= 0:
        return 1.0
    p_null = total_k / total_n
    expected_k = n * p_null
    variance = n * p_null * (1.0 - p_null) * max(0.0, (total_n - n) / max(1, total_n - 1))
    if variance <= 1e-9:
        return 0.0 if k > expected_k else 1.0
    z = (k - expected_k) / math.sqrt(variance)
    # One-sided survival probability P(Z >= z)
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def run_prim(
    points: Sequence[dict[str, Any]],
    targets: Sequence[int | bool | float],
    parameter_space: ParameterSpace,
    alpha: float = 0.05,
    min_support_ratio: float = 0.05,
    threshold: float = 0.5,
    strategy_id: str = "default_strategy",
) -> ScenarioDiscoveryResult:
    """Execute Patient Rule Induction Method (PRIM) in pure NumPy.

    Args:
        points: Sequence of parameter dictionaries from the ensemble rollout.
        targets: Binary indicator of failure (1/True) or continuous metric >= threshold.
        parameter_space: Specification of parameter bounds and types.
        alpha: Peeling fraction per iteration (default: 0.05 = 5%).
        min_support_ratio: Minimum fraction of points required in candidate box.
        threshold: Binarization threshold if targets are continuous.
        strategy_id: Identifier of the strategy being analyzed.

    Returns:
        Deterministic ScenarioDiscoveryResult containing discovered vulnerability boxes.
    """
    n_samples = len(points)
    if n_samples == 0:
        raise ValueError("Cannot run scenario discovery on empty points sequence.")

    # 1. Binarize targets: 1 = failure/vulnerability, 0 = success
    y = np.array([1 if float(t) >= threshold else 0 for t in targets], dtype=np.int32)
    total_failures = int(np.sum(y))

    if total_failures == 0:
        # No failures observed: return benign trivial box
        bounds: dict[str, tuple[float, float]] = {}
        for p in parameter_space.parameters:
            if p.bounds is not None:
                bounds[p.name] = (float(p.bounds[0]), float(p.bounds[1]))
            else:
                bounds[p.name] = (0.0, 1.0)
        box = VulnerabilityBox(
            box_id=f"box_{strategy_id}_zero_failures",
            bounds=bounds,
            coverage=0.0,
            density=0.0,
            support=n_samples,
            total_samples=n_samples,
            restricted_dimensions=(),
            quasi_p_value=1.0,
            description="No failures observed in ensemble.",
        )
        fp = canonical_sha256(
            {
                "strategy": strategy_id,
                "algorithm": "prim",
                "boxes": [box.model_dump()],
                "space": parameter_space.space_hash,
            }
        )
        return ScenarioDiscoveryResult(
            strategy_id=strategy_id,
            boxes=(box,),
            primary_box=box,
            algorithm="prim",
            discovery_fingerprint=fp,
            total_failures=0,
            total_samples=n_samples,
        )

    # 2. Extract numeric feature matrix for dimensions with numeric bounds
    dims: list[str] = []
    initial_bounds: dict[str, tuple[float, float]] = {}
    for p in parameter_space.parameters:
        if p.name in points[0] and p.bounds is not None:
            dims.append(p.name)
            initial_bounds[p.name] = (float(p.bounds[0]), float(p.bounds[1]))

    X = np.zeros((n_samples, len(dims)), dtype=np.float64)
    for i, pt in enumerate(points):
        for j, dim in enumerate(dims):
            X[i, j] = float(pt[dim])

    # 3. Patient Rule Induction Loop (Peeling)
    current_bounds = {dim: [low, high] for dim, (low, high) in initial_bounds.items()}
    min_support = max(5, int(n_samples * min_support_ratio))

    trajectory: list[PeelingStep] = []
    boxes_on_path: list[tuple[dict[str, tuple[float, float]], float, float, int]] = []

    # Initial state
    mask = np.ones(n_samples, dtype=bool)
    init_support = n_samples
    init_density = float(total_failures / n_samples)
    boxes_on_path.append(
        (
            {dim: (initial_bounds[dim][0], initial_bounds[dim][1]) for dim in dims},
            1.0,
            init_density,
            init_support,
        )
    )

    step_num = 0
    while True:
        current_support = int(np.sum(mask))
        if current_support <= min_support:
            break

        best_dim_idx = -1
        best_direction: Literal["lower", "upper"] = "lower"
        best_new_bound = 0.0
        best_density = -1.0
        best_mask = mask

        # Try peeling bottom and top alpha quantile for each dimension
        for j, _dim in enumerate(dims):
            vals_in_box = X[mask, j]
            if len(vals_in_box) == 0:
                continue

            # Peel lower quantile
            q_low = float(np.quantile(vals_in_box, alpha))
            cand_mask_low = mask & (X[:, j] >= q_low)
            cand_support_low = int(np.sum(cand_mask_low))

            if cand_support_low >= min_support and cand_support_low < current_support:
                cand_failures_low = int(np.sum(y[cand_mask_low]))
                cand_dens_low = cand_failures_low / cand_support_low
                if cand_dens_low > best_density:
                    best_density = cand_dens_low
                    best_dim_idx = j
                    best_direction = "lower"
                    best_new_bound = q_low
                    best_mask = cand_mask_low

            # Peel upper quantile
            q_high = float(np.quantile(vals_in_box, 1.0 - alpha))
            cand_mask_high = mask & (X[:, j] <= q_high)
            cand_support_high = int(np.sum(cand_mask_high))

            if cand_support_high >= min_support and cand_support_high < current_support:
                cand_failures_high = int(np.sum(y[cand_mask_high]))
                cand_dens_high = cand_failures_high / cand_support_high
                if cand_dens_high > best_density:
                    best_density = cand_dens_high
                    best_dim_idx = j
                    best_direction = "upper"
                    best_new_bound = q_high
                    best_mask = cand_mask_high

        if best_dim_idx == -1 or np.array_equal(best_mask, mask):
            # No valid peel improved or reduced points
            break

        # Apply best peel
        step_num += 1
        dim_name = dims[best_dim_idx]
        if best_direction == "lower":
            current_bounds[dim_name][0] = best_new_bound
        else:
            current_bounds[dim_name][1] = best_new_bound

        mask = best_mask
        new_support = int(np.sum(mask))
        new_failures = int(np.sum(y[mask]))
        new_coverage = new_failures / total_failures if total_failures > 0 else 0.0
        new_density = new_failures / new_support if new_support > 0 else 0.0

        peel_step = PeelingStep(
            step=step_num,
            coverage=float(new_coverage),
            density=float(new_density),
            support=new_support,
            peeled_dimension=dim_name,
            peeled_direction=best_direction,
            peeled_bound_value=best_new_bound,
        )
        trajectory.append(peel_step)

        frozen_bounds = {
            dim: (float(current_bounds[dim][0]), float(current_bounds[dim][1])) for dim in dims
        }
        boxes_on_path.append((frozen_bounds, new_coverage, new_density, new_support))

    # 4. Select Primary Vulnerability Box (maximize F1 score balancing coverage & density)
    best_f1 = -1.0
    best_candidate_idx = 0

    for idx, (_, cov, dens, supp) in enumerate(boxes_on_path):
        if supp < min_support:
            continue
        # F_1 score = 2 * (coverage * density) / (coverage + density)
        denom = cov + dens
        f1 = (2.0 * cov * dens / denom) if denom > 1e-9 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_candidate_idx = idx

    selected_bounds, sel_cov, sel_dens, sel_supp = boxes_on_path[best_candidate_idx]

    # Find restricted dimensions (bounds strictly tighter than initial)
    restricted: list[str] = []
    for dim in dims:
        orig_l, orig_h = initial_bounds[dim]
        box_l, box_h = selected_bounds[dim]
        # Restrict if width changed by more than 1% of initial span
        span = max(1e-6, orig_h - orig_l)
        if (box_l - orig_l) / span > 0.01 or (orig_h - box_h) / span > 0.01:
            restricted.append(dim)

    # Compute quasi-p-value
    k_box = round(sel_dens * sel_supp)
    quasi_p = _calculate_quasi_p_value(k_box, sel_supp, total_failures, n_samples)

    primary_box = VulnerabilityBox(
        box_id=f"box_{strategy_id}_prim_1",
        bounds=selected_bounds,
        coverage=float(sel_cov),
        density=float(sel_dens),
        support=sel_supp,
        total_samples=n_samples,
        restricted_dimensions=tuple(restricted),
        quasi_p_value=quasi_p,
        description=f"PRIM discovery box ({len(restricted)} restricted dims, density={sel_dens:.2f})",
    )

    fingerprint_dict = {
        "strategy": strategy_id,
        "algorithm": "prim",
        "primary_box": normalize_canonical_value(primary_box.model_dump()),
        "peeling_steps": len(trajectory),
        "total_failures": total_failures,
        "total_samples": n_samples,
    }
    discovery_fp = canonical_sha256(fingerprint_dict)

    return ScenarioDiscoveryResult(
        strategy_id=strategy_id,
        boxes=(primary_box,),
        primary_box=primary_box,
        peeling_trajectory=tuple(trajectory),
        algorithm="prim",
        discovery_fingerprint=discovery_fp,
        total_failures=total_failures,
        total_samples=n_samples,
    )


def run_cart(
    points: Sequence[dict[str, Any]],
    targets: Sequence[int | bool | float],
    parameter_space: ParameterSpace,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    threshold: float = 0.5,
    strategy_id: str = "default_strategy",
) -> ScenarioDiscoveryResult:
    """Execute CART (Classification and Regression Tree) rule extraction in pure NumPy.

    Recursively splits on single features to maximize Gini impurity reduction,
    extracting high-failure leaves as rectangular vulnerability boxes.
    """
    n_samples = len(points)
    if n_samples == 0:
        raise ValueError("Cannot run scenario discovery on empty points sequence.")

    y = np.array([1 if float(t) >= threshold else 0 for t in targets], dtype=np.int32)
    total_failures = int(np.sum(y))

    if total_failures == 0:
        return run_prim(
            points,
            targets,
            parameter_space,
            threshold=threshold,
            strategy_id=strategy_id,
        )

    dims: list[str] = []
    initial_bounds: dict[str, tuple[float, float]] = {}
    for p in parameter_space.parameters:
        if p.name in points[0] and p.bounds is not None:
            dims.append(p.name)
            initial_bounds[p.name] = (float(p.bounds[0]), float(p.bounds[1]))

    X = np.zeros((n_samples, len(dims)), dtype=np.float64)
    for i, pt in enumerate(points):
        for j, dim in enumerate(dims):
            X[i, j] = float(pt[dim])

    # Recursive Tree node
    discovered_boxes: list[VulnerabilityBox] = []

    def _gini(labels: np.ndarray) -> float:
        if len(labels) == 0:
            return 0.0
        p = float(np.mean(labels))
        return 1.0 - (p * p + (1.0 - p) * (1.0 - p))

    def _split_node(
        mask: np.ndarray,
        current_bounds: dict[str, list[float]],
        depth: int,
    ) -> None:
        node_samples = int(np.sum(mask))
        if node_samples <= min_samples_leaf or depth >= max_depth:
            # Leaf node: evaluate if high failure density
            node_failures = int(np.sum(y[mask]))
            if node_samples > 0:
                density = node_failures / node_samples
                baseline_density = total_failures / n_samples
                if density > baseline_density and node_failures > 0:
                    coverage = node_failures / total_failures
                    frozen_bounds = {
                        dim: (bounds[0], bounds[1]) for dim, bounds in current_bounds.items()
                    }
                    restricted = [
                        dim
                        for dim in dims
                        if frozen_bounds[dim][0] > initial_bounds[dim][0]
                        or frozen_bounds[dim][1] < initial_bounds[dim][1]
                    ]
                    box = VulnerabilityBox(
                        box_id=f"box_{strategy_id}_cart_{len(discovered_boxes) + 1}",
                        bounds=frozen_bounds,
                        coverage=float(coverage),
                        density=float(density),
                        support=node_samples,
                        total_samples=n_samples,
                        restricted_dimensions=tuple(restricted),
                        quasi_p_value=_calculate_quasi_p_value(
                            node_failures, node_samples, total_failures, n_samples
                        ),
                        description=f"CART leaf at depth {depth} (density={density:.2f})",
                    )
                    discovered_boxes.append(box)
            return

        current_gini = _gini(y[mask])
        best_gain = -1.0
        best_dim_idx = -1
        best_threshold = 0.0
        best_left_mask = mask
        best_right_mask = mask

        for j in range(len(dims)):
            feature_vals = np.unique(X[mask, j])
            if len(feature_vals) <= 1:
                continue
            # Candidate split thresholds (midpoints)
            split_candidates = (feature_vals[:-1] + feature_vals[1:]) / 2.0
            for split_val in split_candidates:
                left = mask & (X[:, j] <= split_val)
                right = mask & (X[:, j] > split_val)
                n_left = int(np.sum(left))
                n_right = int(np.sum(right))
                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue
                gain = current_gini - (
                    (n_left / node_samples) * _gini(y[left])
                    + (n_right / node_samples) * _gini(y[right])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_dim_idx = j
                    best_threshold = float(split_val)
                    best_left_mask = left
                    best_right_mask = right

        if best_gain <= 1e-6 or best_dim_idx == -1:
            # Cannot meaningfully split further
            _split_node(mask, current_bounds, max_depth)
            return

        dim_name = dims[best_dim_idx]

        # Recurse left
        left_bounds = {k: list(v) for k, v in current_bounds.items()}
        left_bounds[dim_name][1] = min(left_bounds[dim_name][1], best_threshold)
        _split_node(best_left_mask, left_bounds, depth + 1)

        # Recurse right
        right_bounds = {k: list(v) for k, v in current_bounds.items()}
        right_bounds[dim_name][0] = max(right_bounds[dim_name][0], best_threshold)
        _split_node(best_right_mask, right_bounds, depth + 1)

    initial_tree_bounds = {dim: [low, high] for dim, (low, high) in initial_bounds.items()}
    _split_node(np.ones(n_samples, dtype=bool), initial_tree_bounds, depth=0)

    # Sort boxes by density descending then coverage descending
    discovered_boxes.sort(key=lambda b: (b.density, b.coverage), reverse=True)
    primary = discovered_boxes[0] if discovered_boxes else None

    fp = canonical_sha256(
        {
            "strategy": strategy_id,
            "algorithm": "cart",
            "boxes_count": len(discovered_boxes),
            "primary": primary.model_dump() if primary else None,
            "total_failures": total_failures,
        }
    )

    return ScenarioDiscoveryResult(
        strategy_id=strategy_id,
        boxes=tuple(discovered_boxes),
        primary_box=primary,
        algorithm="cart",
        discovery_fingerprint=fp,
        total_failures=total_failures,
        total_samples=n_samples,
    )
