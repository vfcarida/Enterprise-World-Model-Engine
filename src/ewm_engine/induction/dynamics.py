"""Dynamics induction via SINDy and symbolic regression.

Scientific Grounding:
    - Brunton, Proctor, & Kutz (2016, PNAS, DOI:10.1073/pnas.1517384113):
      "Discovering governing equations from data by sparse identification of nonlinear dynamical systems" (SINDy).
    - Cranmer (2023, arXiv:2305.01582): "PySR: High-Performance Symbolic Regression in Python and Julia."
    - Champion et al. (2019, PNAS, DOI:10.1073/pnas.1906995116):
      Data-driven discovery of coordinates and governing equations.

Epistemic Invariant:
    Dynamics models learned from finite trajectory samples carry empirical status (CALIBRATED,
    PREDICTIVE, or HYPOTHETICAL depending on R^2 and out-of-sample fit). They are never
    promoted to STRUCTURAL invariants without explicit conservation stress testing.
    Default execution is 100% LLM-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.dynamics.learned import LinearResidualDynamics, TransitionDataset, TransitionSample
from ewm_engine.provenance.evidence import EvidenceLevel

if TYPE_CHECKING:
    from ewm_engine.induction.report import InductionReport


class InducedRateLaw(BaseModel):
    """Interpretable, closed-form governing differential or difference equation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_variable: str = Field(description="Name of the evolving state variable.")
    feature_variables: tuple[str, ...] = Field(
        description="Variables appearing in the governing rate law."
    )
    expression: str = Field(
        description="Human-readable mathematical expression (e.g. 'd(X)/dt = -0.05*X + 0.8*A')."
    )
    coefficients: dict[str, float] = Field(
        description="Sparse regression coefficients for each library term."
    )
    r_squared: float = Field(
        ge=0.0, le=1.0, description="Coefficient of determination quantifying empirical fit."
    )
    mse: float = Field(ge=0.0, description="Mean squared prediction error on transition deltas.")
    complexity: int = Field(description="Number of non-zero terms in the rate law.")
    evidence_level: EvidenceLevel = Field(
        description="Epistemic standing derived strictly from goodness of fit."
    )
    sample_count: int = Field(description="Number of observed transitions used for fitting.")
    engine: str = Field(
        default="sindy_stlsq", description="Induction engine (sindy_stlsq, pysindy, or pysr)."
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the rate law."
    )

    def to_linear_residual_dynamics(self) -> LinearResidualDynamics:
        """Compile rate law into a LinearResidualDynamics model when linear."""
        action_weight = 1.0
        for k, v in self.coefficients.items():
            if "action" in k:
                action_weight = v
                break

        bias = self.coefficients.get("1", 0.0)
        dyn = LinearResidualDynamics(
            target_resource=self.target_variable,
            action_type="default",
            name=f"InducedDynamics_{self.target_variable}",
        )
        dyn.action_weight = action_weight
        dyn.bias = bias
        return dyn


# ---------------------------------------------------------------------------
# SINDy Feature Library and STLSQ Sparse Regression Engine
# ---------------------------------------------------------------------------


def _build_feature_library(
    X: np.ndarray,
    feature_names: list[str],
    degree: int = 2,
    include_bias: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Construct polynomial and interaction feature library Theta(X)."""
    n_samples, n_features = X.shape
    cols = []
    names = []

    if include_bias:
        cols.append(np.ones((n_samples, 1), dtype=np.float64))
        names.append("1")

    # Degree 1: linear terms
    for j in range(n_features):
        cols.append(X[:, j : j + 1])
        names.append(feature_names[j])

    # Degree 2: quadratic terms and pairwise interactions
    if degree >= 2:
        for j in range(n_features):
            for k in range(j, n_features):
                cols.append((X[:, j] * X[:, k])[:, np.newaxis])
                if j == k:
                    names.append(f"{feature_names[j]}^2")
                else:
                    names.append(f"{feature_names[j]}*{feature_names[k]}")

    Theta = np.hstack(cols)
    return Theta, names


def run_sindy_stlsq(
    X: np.ndarray,
    Y_dot: np.ndarray,
    feature_names: list[str],
    target_name: str,
    degree: int = 2,
    threshold: float = 0.05,
    alpha_ridge: float = 1e-4,
    max_iter: int = 10,
) -> InducedRateLaw:
    """Sequential Thresholded Least Squares (STLSQ) SINDy solver in pure NumPy.

    Minimizes: ||Y_dot - Theta Xi||_2^2 + alpha ||Xi||_2^2, iteratively zeroing out
    coefficients smaller than threshold.
    """
    n_samples = X.shape[0]
    Theta, lib_names = _build_feature_library(X, feature_names, degree=degree)
    n_lib = Theta.shape[1]

    # Initial Ridge regression estimate
    reg_eye = alpha_ridge * np.eye(n_lib)
    try:
        Xi = np.linalg.solve(Theta.T @ Theta + reg_eye, Theta.T @ Y_dot)
    except np.linalg.LinAlgError:
        Xi, _, _, _ = np.linalg.lstsq(Theta, Y_dot, rcond=None)

    # Iterative thresholding (STLSQ)
    for _ in range(max_iter):
        small_indices = np.abs(Xi) < threshold
        Xi[small_indices] = 0.0
        active_indices = ~small_indices

        if not np.any(active_indices):
            # Fall back to largest coefficient to prevent degenerate zero model
            best_idx = int(np.argmax(np.abs(Xi)))
            active_indices[best_idx] = True

        Theta_active = Theta[:, active_indices]
        try:
            Xi_active = np.linalg.solve(
                Theta_active.T @ Theta_active + alpha_ridge * np.eye(int(np.sum(active_indices))),
                Theta_active.T @ Y_dot,
            )
            Xi[active_indices] = Xi_active
        except Exception:
            break

    # Goodness-of-fit evaluation
    y_pred = Theta @ Xi
    ss_res = float(np.sum((Y_dot - y_pred) ** 2))
    ss_tot = float(np.sum((Y_dot - np.mean(Y_dot)) ** 2))
    mse = float(ss_res / max(n_samples, 1))

    if ss_tot > 1e-9:
        r2 = max(0.0, min(1.0, 1.0 - (ss_res / ss_tot)))
    else:
        r2 = 1.0 if ss_res < 1e-6 else 0.0

    # Build formula string and non-zero coefficient map
    active_terms: dict[str, float] = {}
    term_strings: list[str] = []
    for idx, name in enumerate(lib_names):
        coef = float(Xi[idx])
        if abs(coef) >= 1e-4:
            active_terms[name] = round(coef, 4)
            sign = "+" if coef >= 0 and term_strings else ("-" if coef < 0 else "")
            val_abs = abs(coef)
            if name == "1":
                term_strings.append(f"{sign} {val_abs:.3f}".strip())
            else:
                term_strings.append(f"{sign} {val_abs:.3f}*{name}".strip())

    formula = " + ".join(term_strings) if term_strings else "0.0"
    formula = formula.replace(" + -", " - ")
    expr = f"d({target_name})/dt = {formula}"

    # Determine EvidenceLevel
    if r2 >= 0.95:
        level = EvidenceLevel.CALIBRATED
    elif r2 >= 0.70:
        level = EvidenceLevel.PREDICTIVE
    else:
        level = EvidenceLevel.HYPOTHETICAL

    fp = canonical_sha256(
        {
            "target": target_name,
            "expression": expr,
            "coefficients": sorted(active_terms.items()),
            "r2": round(r2, 4),
        }
    )

    used_features = [
        f for f in feature_names if any(f in term for term in active_terms if term != "1")
    ]

    return InducedRateLaw(
        target_variable=target_name,
        feature_variables=tuple(used_features),
        expression=expr,
        coefficients=active_terms,
        r_squared=round(r2, 4),
        mse=round(mse, 6),
        complexity=len(active_terms),
        evidence_level=level,
        sample_count=n_samples,
        engine="sindy_stlsq",
        fingerprint=fp,
    )


def induce_dynamics(
    trajectories_or_dataset: Any,
    target_resources: Sequence[str] | None = None,
    action_types: Sequence[str] | None = None,
    degree: int = 2,
    threshold: float = 0.05,
    seed: int = 42,
) -> tuple[list[InducedRateLaw], InductionReport]:
    """Public high-level API for dynamics and rate-law induction.

    Ingests state transition trajectories or a TransitionDataset, discovers
    parsimonious governing differential equations via SINDy, and builds an auditable InductionReport.
    """
    from ewm_engine.induction.report import (
        InducedRateLawSummary,
        InductionReport,
    )

    # 1. Standardize input into arrays
    if isinstance(trajectories_or_dataset, TransitionDataset):
        dataset = trajectories_or_dataset
        if not dataset:
            raise ValueError("Empty TransitionDataset provided for dynamics induction.")
        # Infer resource IDs if not provided
        first_sample: TransitionSample = dataset[0]
        res_ids = (
            list(target_resources)
            if target_resources is not None
            else sorted(first_sample.state.resources.keys())
        )
        states, actions, next_states = dataset.to_numpy(
            resource_ids=res_ids,
            action_type=action_types,
        )
        feature_names = [*res_ids]
        if action_types:
            feature_names.extend([f"action_{a}" for a in action_types])
        elif actions.shape[1] > 0:
            feature_names.append("action_input")
        X = np.hstack([states, actions])
        Y_next = next_states
    elif isinstance(trajectories_or_dataset, (list, tuple)) and len(trajectories_or_dataset) > 0:
        # Check if list of TransitionSample or list of dicts
        first_item = trajectories_or_dataset[0]
        if isinstance(first_item, TransitionSample):
            dataset = TransitionDataset(trajectories_or_dataset)
            return induce_dynamics(
                dataset,
                target_resources=target_resources,
                action_types=action_types,
                degree=degree,
                threshold=threshold,
                seed=seed,
            )
        elif isinstance(first_item, dict):
            # Dict rows: keys are variables
            keys = sorted(first_item.keys())
            res_ids = list(target_resources) if target_resources is not None else keys
            arr = np.array([[float(d.get(k, 0.0)) for k in keys] for d in trajectories_or_dataset])
            # Finite differences between consecutive rows
            X = arr[:-1]
            Y_next = arr[1:]
            feature_names = keys
        else:
            raise ValueError(f"Unsupported trajectory sequence item type: {type(first_item)}")
    elif isinstance(trajectories_or_dataset, np.ndarray):
        arr = trajectories_or_dataset
        res_ids = (
            list(target_resources)
            if target_resources is not None
            else [f"res_{i}" for i in range(arr.shape[1])]
        )
        X = arr[:-1]
        Y_next = arr[1:]
        feature_names = res_ids
    else:
        raise ValueError(f"Unsupported trajectories input format: {type(trajectories_or_dataset)}")

    # Compute derivative / discrete change deltas: Delta Y = Y_{t+1} - Y_t
    deltas = Y_next - X[:, : Y_next.shape[1]]

    # 2. Fit SINDy models for each target resource
    rate_laws: list[InducedRateLaw] = []
    for idx, target_name in enumerate(res_ids):
        y_dot = deltas[:, idx]
        rate_law = run_sindy_stlsq(
            X=X,
            Y_dot=y_dot,
            feature_names=feature_names,
            target_name=target_name,
            degree=degree,
            threshold=threshold,
        )
        rate_laws.append(rate_law)

    # 3. Build InductionReport
    summaries = [
        InducedRateLawSummary(
            target_variable=r.target_variable,
            expression=r.expression,
            r_squared=r.r_squared,
            complexity=r.complexity,
            evidence_level=r.evidence_level.value,
        )
        for r in rate_laws
    ]

    evidence_dist: dict[str, int] = {}
    for r in rate_laws:
        lvl = r.evidence_level.value
        evidence_dist[lvl] = evidence_dist.get(lvl, 0) + 1

    report = InductionReport(
        source="dynamics_trajectory_dataset",
        source_type="dynamics",
        seed=seed,
        entities_count=len(res_ids),
        resources_count=len(res_ids),
        evidence_distribution=evidence_dist,
        induced_rate_laws=summaries,
        metadata={
            "degree": degree,
            "threshold": threshold,
            "sample_count": X.shape[0],
        },
    )

    return rate_laws, report
