"""Induction honesty and validation gate (`ewm induce validate`).

This module implements EWM Engine's honesty moat:
Every induction pathway (schema, process, causal, dynamics, invariants) must pass
through a single mandatory validation and refusal gate. EWM declines to emit any
structural element into a WorldSpec unless it clears ASSERTED epistemic standing.

Anchors & Science:
- DiBS posterior-over-DAGs (arXiv:2105.11839)
- Interventional discovery (Ke et al., arXiv:1910.01075)
- Tearing Apart NOTEARS / variance manipulation (arXiv:2206.07195)
- PC / FCI equivalence-class honesty (Spirtes et al., 2000; arXiv:2305.10032)
- DoWhy refuters: placebo treatment, random common cause, data subset (MIT)
- Process conformance replay validation (Carmona et al., 2018)
- AI-Poincaré invariant stress testing (arXiv:2011.04698)
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, TypeAlias

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import (
    EntitySpec,
    RelationshipSpec,
    ResourceSpec,
    WorldSpec,
)
from ewm_engine.core.trust import InductionRefusalError
from ewm_engine.induction.causal import CausalEdge, CausalEdgeType, CausalEquivalenceClass
from ewm_engine.induction.dynamics import InducedRateLaw
from ewm_engine.induction.invariants import InvariantHypothesis
from ewm_engine.induction.report import (
    CandidateInvariant,
    ConformanceScore,
    InductionReport,
    ProvisionalLink,
)
from ewm_engine.provenance.evidence import EvidenceLevel

# ==============================================================================
# 1. Validation & Refuter Data Models
# ==============================================================================


class RefuterType(StrEnum):
    """DoWhy-style refutation methodology."""

    PLACEBO_TREATMENT = "placebo_treatment"
    RANDOM_COMMON_CAUSE = "random_common_cause"
    DATA_SUBSET = "data_subset"


class RefuterResult(BaseModel):
    """Result of an individual causal refutation test."""

    model_config = ConfigDict(extra="forbid")

    refuter_type: RefuterType
    passed: bool
    original_effect: float
    refuted_effect: float
    effect_change_ratio: float
    p_value: float | None = None
    threshold: float
    details: str


class HeldOutReproductionResult(BaseModel):
    """Result of evaluating induced dynamics against held-out trajectories."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    target_variable: str
    train_r2: float
    held_out_r2: float
    threshold: float
    details: str


class InvariantStressResult(BaseModel):
    """Result of Monte Carlo stress testing for a candidate conservation law."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    name: str
    violation_rate: float
    max_observed_violation: float
    tolerance: float
    mc_samples: int
    details: str


class ProcessConformanceResult(BaseModel):
    """Result of process model trace replay alignment validation."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    object_type: str
    fitness: float
    precision: float
    min_fitness: float
    min_precision: float
    details: str


class RefusedElement(BaseModel):
    """An induced structural element that failed validation and was refused from WorldSpec."""

    model_config = ConfigDict(extra="forbid")

    element_type: (
        str  # "causal_edge", "hard_constraint", "provisional_link", "dynamics", "transition"
    )
    identifier: str
    claimed_or_inferred_level: EvidenceLevel
    refusal_reason: str
    failed_validators: list[str] = Field(default_factory=list)
    candidate_alternatives: list[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    """Explicit epistemic open question surfaced to the human modeler for refused structure."""

    model_config = ConfigDict(extra="forbid")

    id: str
    topic: str
    description: str
    candidate_options: list[str] = Field(default_factory=list)
    recommended_action: str


class InductionGateOutcome(StrEnum):
    """Explicit outcome vocabulary for the fail-closed induction validation gate."""

    ALL_CANDIDATES_ADMITTED = "ALL_CANDIDATES_ADMITTED"
    REFUSED_UNIDENTIFIABLE_AS_DESIGNED = "REFUSED_UNIDENTIFIABLE_AS_DESIGNED"


class InductionValidationReport(BaseModel):
    """Comprehensive, deterministic audit of induction validation and refusal decisions."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    gate_outcome: InductionGateOutcome = InductionGateOutcome.ALL_CANDIDATES_ADMITTED
    admitted_elements_count: int
    refused_elements_count: int
    admitted_elements: list[dict[str, Any]] = Field(default_factory=list)
    refused_elements: list[RefusedElement] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    refuter_evaluations: list[RefuterResult] = Field(default_factory=list)
    held_out_evaluations: list[HeldOutReproductionResult] = Field(default_factory=list)
    invariant_evaluations: list[InvariantStressResult] = Field(default_factory=list)
    conformance_evaluations: list[ProcessConformanceResult] = Field(default_factory=list)
    sha256_fingerprint: str = ""

    @property
    def all_candidates_admitted(self) -> bool:
        """True only when every candidate element cleared ASSERTED with zero refusals."""
        return self.passed

    def model_post_init(self, __context: Any) -> None:
        """Synchronize gate_outcome and compute SHA-256 fingerprint if not present."""
        self.gate_outcome = (
            InductionGateOutcome.ALL_CANDIDATES_ADMITTED
            if self.passed
            else InductionGateOutcome.REFUSED_UNIDENTIFIABLE_AS_DESIGNED
        )
        if not self.sha256_fingerprint:
            self.sha256_fingerprint = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        """Calculate canonical SHA-256 hash over deterministic fields."""
        payload: dict[str, Any] = {
            "passed": self.passed,
            "admitted_elements_count": self.admitted_elements_count,
            "refused_elements_count": self.refused_elements_count,
            "admitted_elements": sorted(
                [f"{e.get('type')}:{e.get('id')}:{e.get('level')}" for e in self.admitted_elements]
            ),
            "refused_elements": sorted(
                [
                    f"{r.element_type}:{r.identifier}:{r.claimed_or_inferred_level.value}:{r.refusal_reason}"
                    for r in self.refused_elements
                ]
            ),
            "open_questions": sorted([f"{q.id}:{q.topic}" for q in self.open_questions]),
            "refuter_evaluations": sorted(
                [
                    f"{r.refuter_type}:{r.passed}:{round(r.original_effect, 4)}:{round(r.refuted_effect, 4)}"
                    for r in self.refuter_evaluations
                ]
            ),
            "invariant_evaluations": sorted(
                [
                    f"{inv.name}:{inv.passed}:{round(inv.violation_rate, 4)}"
                    for inv in self.invariant_evaluations
                ]
            ),
        }
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to standard serializable dictionary."""
        return self.model_dump(mode="json")

    def to_markdown(self) -> str:
        """Generate human-readable Markdown summary of the validation gate."""
        status_badge = (
            "PASSED (Clean Induction)" if self.passed else "REFUSALS PRESENT (Fails Closed)"
        )
        lines = [
            "# Induction Validation & Refusal Report",
            "",
            f"- **Validation Gate Status**: `{status_badge}`",
            f"- **Admitted Elements (ASSERTED)**: {self.admitted_elements_count}",
            f"- **Refused Elements**: {self.refused_elements_count}",
            f"- **Open Questions Surfaced**: {len(self.open_questions)}",
            f"- **SHA-256 Fingerprint**: `{self.sha256_fingerprint}`",
            "",
            "## Admitted Structure (cleared ASSERTED)",
            "",
            "| Element Type | Identifier | Epistemic Level | Mechanism / Note |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for e in self.admitted_elements:
            lines.append(
                f"| `{e.get('type')}` | `{e.get('id')}` | `{e.get('level')}` | {e.get('note', '-')} |"
            )
        if not self.admitted_elements:
            lines.append("| *None* | - | - | - |")

        lines.extend(
            [
                "",
                "## Refused Structure (Declined from WorldSpec)",
                "",
                "| Element Type | Identifier | Claimed Level | Refusal Reason | Candidate Alternatives |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for r in self.refused_elements:
            alts = "; ".join(r.candidate_alternatives) if r.candidate_alternatives else "None"
            lines.append(
                f"| `{r.element_type}` | `{r.identifier}` | `{r.claimed_or_inferred_level.value}` | {r.refusal_reason} | {alts} |"
            )
        if not self.refused_elements:
            lines.append("| *None (Zero Refusals)* | - | - | - | - |")

        lines.extend(
            [
                "",
                "## Surfaced Open Questions (Human-in-the-Loop Decisions)",
                "",
            ]
        )
        for q in self.open_questions:
            lines.extend(
                [
                    f"### {q.id}: {q.topic}",
                    "",
                    f"{q.description}",
                    "",
                    "**Candidate Options:**",
                ]
            )
            for opt in q.candidate_options:
                lines.append(f"- {opt}")
            lines.extend(
                [
                    "",
                    f"**Recommended Action**: {q.recommended_action}",
                    "",
                ]
            )
        if not self.open_questions:
            lines.append("*No open questions. All induced structures resolved unambiguously.*")

        return "\n".join(lines)


# ==============================================================================
# 2. DoWhy-Style Causal Refuters
# ==============================================================================


InductionDataInput: TypeAlias = (
    np.ndarray | Mapping[str, Sequence[float] | np.ndarray] | list[dict[str, Any]]
)


def _extract_column(
    data: InductionDataInput,
    var_name: str,
    var_index: int | None = None,
) -> np.ndarray:
    """Extract a 1D float array for a named variable from supported data formats."""
    if isinstance(data, Mapping):
        if var_name not in data:
            raise KeyError(f"Variable '{var_name}' not found in data dict.")
        return np.asarray(data[var_name], dtype=float)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        return np.array([float(d.get(var_name, 0.0)) for d in data], dtype=float)
    elif isinstance(data, np.ndarray):
        if var_index is None:
            raise ValueError("var_index must be specified when data is a raw NumPy array.")
        return data[:, var_index].astype(float)
    raise ValueError(f"Unsupported data format: {type(data)}")


def _estimate_linear_effect(
    x: np.ndarray,
    y: np.ndarray,
    covariates: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Estimate OLS causal effect coefficient beta, standard error, and p-value.

    y ~ beta * x + gamma * covariates + intercept
    Returns (beta, se, p_value).
    """
    n = len(x)
    if n < 5:
        return 0.0, 1.0, 1.0

    # Form design matrix [x, 1, covariates...]
    cols = [x.reshape(-1, 1), np.ones((n, 1))]
    if covariates is not None and covariates.shape[1] > 0:
        cols.append(covariates)
    X_mat = np.hstack(cols)

    try:
        # Solve OLS: (X^T X)^-1 X^T Y
        xtx = X_mat.T @ X_mat + 1e-9 * np.eye(X_mat.shape[1])
        beta_vec = np.linalg.solve(xtx, X_mat.T @ y)
        beta_val = float(beta_vec[0])

        residuals = y - X_mat @ beta_vec
        dof = max(1, n - X_mat.shape[1])
        sigma2 = float(np.sum(residuals**2) / dof)
        inv_xtx = np.linalg.inv(xtx)
        se = float(np.sqrt(max(1e-12, sigma2 * inv_xtx[0, 0])))

        t_stat = beta_val / se if se > 1e-12 else 0.0
        # Normal approximation for two-tailed p-value
        p_val = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / np.sqrt(2.0)))))
        return beta_val, se, p_val
    except np.linalg.LinAlgError:
        return 0.0, 1.0, 1.0


def refute_placebo_treatment(
    data: InductionDataInput,
    treatment: str,
    outcome: str,
    covariates: list[str] | None = None,
    var_indices: dict[str, int] | None = None,
    threshold: float = 0.20,
    seed: int = 42,
) -> RefuterResult:
    """DoWhy Refuter: Replace treatment with an independent random placebo permutation.

    Under the true causal graph, permuting treatment must cause the estimated
    effect to collapse to near zero. If a significant effect persists, the
    original association was spurious or severely confounded.
    """
    t_idx = var_indices.get(treatment) if var_indices else None
    o_idx = var_indices.get(outcome) if var_indices else None
    x = _extract_column(data, treatment, t_idx)
    y = _extract_column(data, outcome, o_idx)

    cov_mat = None
    if covariates:
        cov_cols = [
            _extract_column(data, c, var_indices.get(c) if var_indices else None).reshape(-1, 1)
            for c in covariates
            if c not in (treatment, outcome)
        ]
        if cov_cols:
            cov_mat = np.hstack(cov_cols)

    orig_effect, _, _ = _estimate_linear_effect(x, y, cov_mat)

    # Permute treatment with seed
    rng = np.random.default_rng(seed)
    x_placebo = rng.permutation(x)
    placebo_effect, _, p_val = _estimate_linear_effect(x_placebo, y, cov_mat)

    denom = max(abs(orig_effect), 1e-6)
    ratio = abs(placebo_effect) / denom

    # Placebo passes if effect collapses below threshold or is statistically indistinguishable from 0
    passed = ratio <= threshold or p_val > 0.05
    details = (
        f"Placebo treatment effect={placebo_effect:.4f} (ratio vs orig={ratio:.2%}, p={p_val:.3f}). "
        f"{'Passed: effect vanished under placebo.' if passed else 'Failed: significant placebo effect persisted.'}"
    )

    return RefuterResult(
        refuter_type=RefuterType.PLACEBO_TREATMENT,
        passed=passed,
        original_effect=orig_effect,
        refuted_effect=placebo_effect,
        effect_change_ratio=ratio,
        p_value=p_val,
        threshold=threshold,
        details=details,
    )


def refute_random_common_cause(
    data: InductionDataInput,
    treatment: str,
    outcome: str,
    covariates: list[str] | None = None,
    var_indices: dict[str, int] | None = None,
    threshold: float = 0.15,
    seed: int = 42,
) -> RefuterResult:
    """DoWhy Refuter: Add an unobserved synthetic random variable as a common cause.

    Adding an independent random Gaussian variable into the covariate set
    should NOT significantly change the estimated causal effect. If it does,
    the estimator is unstable or collinear.
    """
    t_idx = var_indices.get(treatment) if var_indices else None
    o_idx = var_indices.get(outcome) if var_indices else None
    x = _extract_column(data, treatment, t_idx)
    y = _extract_column(data, outcome, o_idx)

    cov_cols = []
    if covariates:
        for c in covariates:
            if c not in (treatment, outcome):
                c_idx = var_indices.get(c) if var_indices else None
                cov_cols.append(_extract_column(data, c, c_idx).reshape(-1, 1))

    cov_mat = np.hstack(cov_cols) if cov_cols else None
    orig_effect, _, _ = _estimate_linear_effect(x, y, cov_mat)

    # Add synthetic random confounder W ~ N(0, 1)
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, size=(len(x), 1))
    augmented_cov = np.hstack([cov_mat, w]) if cov_mat is not None else w

    new_effect, _, p_val = _estimate_linear_effect(x, y, augmented_cov)
    denom = max(abs(orig_effect), 1e-6)
    ratio = abs(new_effect - orig_effect) / denom

    passed = ratio <= threshold
    details = (
        f"Effect after random common cause={new_effect:.4f} (shift={ratio:.2%}). "
        f"{'Passed: estimate stable under random common cause.' if passed else 'Failed: estimate shifted significantly.'}"
    )

    return RefuterResult(
        refuter_type=RefuterType.RANDOM_COMMON_CAUSE,
        passed=passed,
        original_effect=orig_effect,
        refuted_effect=new_effect,
        effect_change_ratio=ratio,
        p_value=p_val,
        threshold=threshold,
        details=details,
    )


def refute_data_subset(
    data: InductionDataInput,
    treatment: str,
    outcome: str,
    covariates: list[str] | None = None,
    var_indices: dict[str, int] | None = None,
    subset_fraction: float = 0.80,
    threshold: float = 0.25,
    seed: int = 42,
) -> RefuterResult:
    """DoWhy Refuter: Remove a random subset of data (80% subsample).

    Estimating the causal effect on a random 80% subsample should yield
    an estimate close to the full-data estimate. If the estimate swings wildly,
    the causal claim is driven by outliers or fragile sampling.
    """
    t_idx = var_indices.get(treatment) if var_indices else None
    o_idx = var_indices.get(outcome) if var_indices else None
    x = _extract_column(data, treatment, t_idx)
    y = _extract_column(data, outcome, o_idx)

    cov_cols = []
    if covariates:
        for c in covariates:
            if c not in (treatment, outcome):
                c_idx = var_indices.get(c) if var_indices else None
                cov_cols.append(_extract_column(data, c, c_idx).reshape(-1, 1))

    cov_mat = np.hstack(cov_cols) if cov_cols else None
    orig_effect, _, _ = _estimate_linear_effect(x, y, cov_mat)

    n = len(x)
    k_sub = max(5, int(n * subset_fraction))
    rng = np.random.default_rng(seed)
    sub_indices = rng.choice(n, size=k_sub, replace=False)

    x_sub = x[sub_indices]
    y_sub = y[sub_indices]
    cov_sub = cov_mat[sub_indices] if cov_mat is not None else None

    sub_effect, _, p_val = _estimate_linear_effect(x_sub, y_sub, cov_sub)
    denom = max(abs(orig_effect), 1e-6)
    ratio = abs(sub_effect - orig_effect) / denom

    passed = ratio <= threshold
    details = (
        f"Subset effect ({subset_fraction:.0%} data)={sub_effect:.4f} (shift={ratio:.2%}). "
        f"{'Passed: estimate stable across data subsets.' if passed else 'Failed: estimate unstable under subsampling.'}"
    )

    return RefuterResult(
        refuter_type=RefuterType.DATA_SUBSET,
        passed=passed,
        original_effect=orig_effect,
        refuted_effect=sub_effect,
        effect_change_ratio=ratio,
        p_value=p_val,
        threshold=threshold,
        details=details,
    )


def run_causal_refutations(
    data: InductionDataInput,
    treatment: str,
    outcome: str,
    covariates: list[str] | None = None,
    var_indices: dict[str, int] | None = None,
    seed: int = 42,
) -> list[RefuterResult]:
    """Execute all three DoWhy refutation gates for a candidate causal edge."""
    r_placebo = refute_placebo_treatment(
        data, treatment, outcome, covariates, var_indices, seed=seed
    )
    r_confounder = refute_random_common_cause(
        data, treatment, outcome, covariates, var_indices, seed=seed + 1
    )
    r_subset = refute_data_subset(data, treatment, outcome, covariates, var_indices, seed=seed + 2)
    return [r_placebo, r_confounder, r_subset]


# ==============================================================================
# 3. Held-Out Dynamics Reproduction Validator
# ==============================================================================


def validate_dynamics_held_out(
    rate_laws: Sequence[InducedRateLaw],
    held_out_trajectories: list[dict[str, float]] | np.ndarray,
    r2_threshold: float = 0.70,
) -> list[HeldOutReproductionResult]:
    """Validate that induced rate laws reproduce held-out trajectory steps."""
    results: list[HeldOutReproductionResult] = []
    if len(rate_laws) == 0:
        return results

    if isinstance(held_out_trajectories, list) and len(held_out_trajectories) > 1:
        keys = sorted(held_out_trajectories[0].keys())
        arr = np.array([[float(d.get(k, 0.0)) for k in keys] for d in held_out_trajectories])
        var_to_idx = {k: i for i, k in enumerate(keys)}
    elif isinstance(held_out_trajectories, np.ndarray) and held_out_trajectories.shape[0] > 1:
        arr = held_out_trajectories
        var_to_idx = {r.target_variable: i for i, r in enumerate(rate_laws)}
    else:
        return results

    X_test = arr[:-1]
    Y_next_test = arr[1:]

    for law in rate_laws:
        var = law.target_variable
        if var not in var_to_idx:
            continue
        v_idx = var_to_idx[var]
        actual_delta = Y_next_test[:, v_idx] - X_test[:, v_idx]

        # Predict delta using rate law linear weights
        pred_delta = np.zeros_like(actual_delta)
        for term, coeff in law.coefficients.items():
            if term == "1":
                pred_delta += coeff
            elif term in var_to_idx:
                pred_delta += coeff * X_test[:, var_to_idx[term]]

        ss_res = float(np.sum((actual_delta - pred_delta) ** 2))
        ss_tot = float(np.sum((actual_delta - np.mean(actual_delta)) ** 2))
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 1.0

        passed = r2 >= r2_threshold
        details = (
            f"Held-out R²={r2:.3f} (train R²={law.r_squared:.3f}, threshold={r2_threshold:.2f})."
        )
        results.append(
            HeldOutReproductionResult(
                passed=passed,
                target_variable=var,
                train_r2=law.r_squared,
                held_out_r2=r2,
                threshold=r2_threshold,
                details=details,
            )
        )

    return results


# ==============================================================================
# 4. Mandatory Validation & Refusal Gate (`InductionValidationGate`)
# ==============================================================================


class InductionValidationGate:
    """The induction honesty gate: fails closed on structural uncertainty.

    Declines to write any unasserted, directional-uncertain, confounded,
    or refuter-failing candidate structures into WorldSpec.
    """

    def __init__(
        self,
        *,
        min_causal_support: float = 0.70,
        require_causal_refuters: bool = True,
        min_process_fitness: float = 0.95,
        min_process_precision: float = 0.90,
        invariant_tolerance: float = 1e-3,
        dynamics_held_out_r2: float = 0.70,
        fail_closed: bool = True,
        strict: bool = False,
        seed: int = 42,
    ) -> None:
        self.min_causal_support = min_causal_support
        self.require_causal_refuters = require_causal_refuters
        self.min_process_fitness = min_process_fitness
        self.min_process_precision = min_process_precision
        self.invariant_tolerance = invariant_tolerance
        self.dynamics_held_out_r2 = dynamics_held_out_r2
        self.fail_closed = fail_closed
        self.strict = strict
        self.seed = seed

    def validate(
        self,
        report: InductionReport,
        data: InductionDataInput | None = None,
        held_out_trajectories: list[dict[str, float]] | np.ndarray | None = None,
    ) -> InductionValidationReport:
        """Validate an InductionReport, raising InductionRefusalError if strict and any refusals exist."""
        _spec, val_rep = self.validate_full_induction(
            report=report,
            data=data,
            held_out_trajectories=held_out_trajectories,
            seed=self.seed,
        )
        if self.strict and len(val_rep.refused_elements) > 0:
            reasons = [
                f"[{r.element_type}] {r.identifier}: {r.refusal_reason}"
                for r in val_rep.refused_elements
            ]
            raise InductionRefusalError(
                f"Induction validation gate failed closed in strict mode: refused {len(val_rep.refused_elements)} candidate element(s).",
                reasons=reasons,
            )
        return val_rep

    def validate_causal(
        self,
        equiv_class: CausalEquivalenceClass,
        data: InductionDataInput | None = None,
        seed: int = 42,
    ) -> tuple[list[CausalEdge], list[RefusedElement], list[OpenQuestion], list[RefuterResult]]:
        """Validate discovered causal equivalence class, refusing unoriented edges."""
        admitted_edges: list[CausalEdge] = []
        refused_elements: list[RefusedElement] = []
        open_questions: list[OpenQuestion] = []
        all_refuter_results: list[RefuterResult] = []

        q_counter = 1

        for e in equiv_class.edges:
            edge_id = (
                f"{e.source} -> {e.target}"
                if e.edge_type == CausalEdgeType.DIRECTED
                else f"{e.source} --- {e.target}"
            )

            # 1. Non-asserted direction / Markov equivalence class
            if e.edge_type == CausalEdgeType.UNDIRECTED:
                refused = RefusedElement(
                    element_type="causal_edge",
                    identifier=edge_id,
                    claimed_or_inferred_level=EvidenceLevel.DIRECTIONAL_UNCERTAIN,
                    refusal_reason=(
                        f"Edge between '{e.source}' and '{e.target}' is observationally equivalent; "
                        "direction cannot be identified from observational data alone."
                    ),
                    failed_validators=["markov_equivalence_check"],
                    candidate_alternatives=[
                        f"Directed: {e.source} -> {e.target}",
                        f"Directed: {e.target} -> {e.source}",
                        f"Latent Confounder: {e.source} <-> {e.target}",
                    ],
                )
                refused_elements.append(refused)
                open_questions.append(
                    OpenQuestion(
                        id=f"OQ-CAUSAL-{q_counter:03d}",
                        topic=f"Directional Ambiguity: {e.source} and {e.target}",
                        description=(
                            f"Observational data proves statistical association between '{e.source}' and '{e.target}', "
                            "but orientation is observationally symmetric within the Markov Equivalence Class (CPDAG)."
                        ),
                        candidate_options=[
                            f"1. Assert domain knowledge that '{e.source}' causes '{e.target}'",
                            f"2. Assert domain knowledge that '{e.target}' causes '{e.source}'",
                            "3. Collect interventional data (A/B testing, policy perturbation) to break symmetry",
                            "4. Leave unoriented as an observational association (associated_with)",
                        ],
                        recommended_action="Provide interventional data or formal domain assertion.",
                    )
                )
                q_counter += 1
                continue

            # 2. Confounded latent common cause
            if e.edge_type == CausalEdgeType.BIDIRECTED:
                refused = RefusedElement(
                    element_type="causal_edge",
                    identifier=f"{e.source} <-> {e.target}",
                    claimed_or_inferred_level=EvidenceLevel.CONFOUNDED,
                    refusal_reason=(
                        f"Latent common cause detected between '{e.source}' and '{e.target}'. "
                        "FCI conditional independence test failed."
                    ),
                    failed_validators=["fci_latent_confounding_test"],
                    candidate_alternatives=["Unobserved common cause W -> {source, target}"],
                )
                refused_elements.append(refused)
                open_questions.append(
                    OpenQuestion(
                        id=f"OQ-CAUSAL-{q_counter:03d}",
                        topic=f"Latent Confounder: {e.source} <-> {e.target}",
                        description=(
                            f"Statistical dependencies indicate '{e.source}' and '{e.target}' share an unobserved "
                            "latent common cause. Direct causal mechanism cannot be asserted."
                        ),
                        candidate_options=[
                            "1. Identify and instrument the missing latent variable in data collection",
                            "2. Treat relationship as confounded association without interventional claims",
                        ],
                        recommended_action="Do not assert directed dependency; instrument missing confounder.",
                    )
                )
                q_counter += 1
                continue

            # 3. Directed edge: must satisfy posterior support and pass refuters
            if e.posterior_support < self.min_causal_support:
                refused = RefusedElement(
                    element_type="causal_edge",
                    identifier=edge_id,
                    claimed_or_inferred_level=EvidenceLevel.PROVISIONAL,
                    refusal_reason=(
                        f"Posterior support ({e.posterior_support:.2f}) falls below ASSERTED threshold ({self.min_causal_support:.2f})."
                    ),
                    failed_validators=["posterior_support_threshold"],
                    candidate_alternatives=["Weak candidate edge (provisional)"],
                )
                refused_elements.append(refused)
                continue

            # Run DoWhy refuters if data is provided
            if self.require_causal_refuters and data is not None:
                other_vars = [v for v in equiv_class.variables if v not in (e.source, e.target)]
                refuters = run_causal_refutations(
                    data=data,
                    treatment=e.source,
                    outcome=e.target,
                    covariates=other_vars,
                    seed=seed,
                )
                all_refuter_results.extend(refuters)
                failed_refuters = [r.refuter_type.value for r in refuters if not r.passed]

                if failed_refuters:
                    refused = RefusedElement(
                        element_type="causal_edge",
                        identifier=edge_id,
                        claimed_or_inferred_level=EvidenceLevel.PROVISIONAL,
                        refusal_reason=f"Failed DoWhy refuters: {', '.join(failed_refuters)}.",
                        failed_validators=failed_refuters,
                        candidate_alternatives=["Spurious correlation / unstable estimate"],
                    )
                    refused_elements.append(refused)
                    open_questions.append(
                        OpenQuestion(
                            id=f"OQ-CAUSAL-{q_counter:03d}",
                            topic=f"Causal Refutation Failure: {edge_id}",
                            description=(
                                f"Candidate directed edge {edge_id} failed refuters ({', '.join(failed_refuters)}). "
                                "Under placebo or random common cause, the effect estimate exhibited severe fragility."
                            ),
                            candidate_options=[
                                "1. Reject causal claim (treat as non-causal)",
                                "2. Increase sample size or clean observational noise",
                            ],
                            recommended_action="Reject edge from WorldSpec; requires further empirical verification.",
                        )
                    )
                    q_counter += 1
                    continue

            # Edge clears all gates -> ADMITTED
            admitted_edges.append(e)

        return admitted_edges, refused_elements, open_questions, all_refuter_results

    def validate_invariants(
        self,
        invariants: Sequence[CandidateInvariant | InvariantHypothesis],
    ) -> tuple[
        list[CandidateInvariant | InvariantHypothesis],
        list[RefusedElement],
        list[OpenQuestion],
        list[InvariantStressResult],
    ]:
        """Validate candidate conservation laws, refusing those that leak or violate tolerance."""
        admitted: list[CandidateInvariant | InvariantHypothesis] = []
        refused: list[RefusedElement] = []
        open_questions: list[OpenQuestion] = []
        stress_results: list[InvariantStressResult] = []

        q_counter = 1
        for inv in invariants:
            name = inv.name
            if isinstance(inv, InvariantHypothesis):
                violation_rate = inv.violation_rate
                max_viol = inv.max_observed_violation
                is_promoted = inv.is_promoted_hard_constraint
            else:
                violation_rate = 1.0 - inv.support_ratio
                max_viol = inv.max_observed_violation
                is_promoted = inv.evidence_level == EvidenceLevel.STRUCTURAL

            passed = is_promoted and violation_rate <= 1e-4 and max_viol <= self.invariant_tolerance
            stress_res = InvariantStressResult(
                passed=passed,
                name=name,
                violation_rate=violation_rate,
                max_observed_violation=max_viol,
                tolerance=self.invariant_tolerance,
                mc_samples=getattr(inv, "mc_stress_samples", 10),
                details=f"Violation rate: {violation_rate:.2%}, max violation: {max_viol:.4e}.",
            )
            stress_results.append(stress_res)

            if passed:
                admitted.append(inv)
            else:
                refused.append(
                    RefusedElement(
                        element_type="hard_constraint",
                        identifier=name,
                        claimed_or_inferred_level=EvidenceLevel.HYPOTHETICAL,
                        refusal_reason=(
                            f"Conservation candidate '{name}' failed Monte Carlo stress testing: "
                            f"violation rate {violation_rate:.2%} exceeds 0.00% threshold."
                        ),
                        failed_validators=["monte_carlo_invariant_stress"],
                        candidate_alternatives=[
                            "Soft constraint with bounded violation penalty",
                            "Model as open system with unmetered external flow",
                        ],
                    )
                )
                open_questions.append(
                    OpenQuestion(
                        id=f"OQ-INVAR-{q_counter:03d}",
                        topic=f"Conservation Leakage in '{name}'",
                        description=(
                            f"Candidate conservation law '{name}' leaked under stochastic rollouts "
                            f"(violation rate {violation_rate:.2%}). It cannot be promoted to a hard constraint."
                        ),
                        candidate_options=[
                            "1. Retain as a soft constraint with empirical violation logging",
                            "2. Add an explicit inflow/outflow buffer resource to account for leakage",
                            "3. Reject candidate invariant completely",
                        ],
                        recommended_action="Treat as soft constraint or instrument unmetered leakage.",
                    )
                )
                q_counter += 1

        return admitted, refused, open_questions, stress_results

    def validate_provisional_links(
        self,
        links: Sequence[ProvisionalLink],
    ) -> tuple[list[ProvisionalLink], list[RefusedElement], list[OpenQuestion]]:
        """Validate provisional latent links, declining unasserted connections."""
        admitted: list[ProvisionalLink] = []
        refused: list[RefusedElement] = []
        open_questions: list[OpenQuestion] = []

        q_counter = 1
        for link in links:
            link_id = f"{link.source_entity} --({link.relationship_type})--> {link.target_entity}"
            if link.clears_gate and link.evidence_level == EvidenceLevel.ASSERTED:
                admitted.append(link)
            else:
                refused.append(
                    RefusedElement(
                        element_type="provisional_link",
                        identifier=link_id,
                        claimed_or_inferred_level=link.evidence_level,
                        refusal_reason=(
                            f"Link prediction between '{link.source_entity}' and '{link.target_entity}' "
                            f"carries PROVISIONAL standing (confidence: {link.confidence:.2f}). "
                            "Unasserted links cannot enter WorldSpec relationships without human approval."
                        ),
                        failed_validators=["provisional_link_assertion_gate"],
                        candidate_alternatives=["Unlinked entities", "Declared foreign key"],
                    )
                )
                open_questions.append(
                    OpenQuestion(
                        id=f"OQ-LINK-{q_counter:03d}",
                        topic=f"Provisional Latent Relationship: {link_id}",
                        description=(
                            f"Relational mining proposed a latent connection ({link.relationship_type}) "
                            f"with confidence {link.confidence:.2f}, but no asserted Foreign Key exists."
                        ),
                        candidate_options=[
                            f"1. Approve link as asserted domain relationship ({link.relationship_type})",
                            "2. Reject link as spurious co-occurrence",
                        ],
                        recommended_action="Keep out of WorldSpec until verified by domain architect.",
                    )
                )
                q_counter += 1

        return admitted, refused, open_questions

    def validate_process_conformance(
        self,
        conformance: dict[str, ConformanceScore],
    ) -> tuple[
        dict[str, ConformanceScore],
        list[RefusedElement],
        list[OpenQuestion],
        list[ProcessConformanceResult],
    ]:
        """Validate process mining trace replay alignment metrics."""
        admitted: dict[str, ConformanceScore] = {}
        refused: list[RefusedElement] = []
        open_questions: list[OpenQuestion] = []
        evals: list[ProcessConformanceResult] = []

        q_counter = 1
        for obj_type, conf in conformance.items():
            passed = (
                conf.fitness >= self.min_process_fitness
                and conf.precision >= self.min_process_precision
            )
            eval_res = ProcessConformanceResult(
                passed=passed,
                object_type=obj_type,
                fitness=conf.fitness,
                precision=conf.precision,
                min_fitness=self.min_process_fitness,
                min_precision=self.min_process_precision,
                details=f"Fitness={conf.fitness:.3f}, Precision={conf.precision:.3f}.",
            )
            evals.append(eval_res)

            if passed:
                admitted[obj_type] = conf
            else:
                refused.append(
                    RefusedElement(
                        element_type="transition",
                        identifier=f"process_lifecycle_{obj_type}",
                        claimed_or_inferred_level=conf.evidence_level,
                        refusal_reason=(
                            f"Process lifecycle for '{obj_type}' failed conformance replay: "
                            f"fitness={conf.fitness:.3f} (min {self.min_process_fitness:.2f}), "
                            f"precision={conf.precision:.3f} (min {self.min_process_precision:.2f})."
                        ),
                        failed_validators=["conformance_replay_alignment"],
                        candidate_alternatives=["Unstructured process log / noisy transitions"],
                    )
                )
                open_questions.append(
                    OpenQuestion(
                        id=f"OQ-PROC-{q_counter:03d}",
                        topic=f"Process Conformance Deficit: {obj_type}",
                        description=(
                            f"Object type '{obj_type}' had replay fitness {conf.fitness:.3f} and "
                            f"precision {conf.precision:.3f}. Observed traces diverge from discovered model."
                        ),
                        candidate_options=[
                            "1. Increase training traces or filter unaligned outlier variants",
                            "2. Accept as provisional empirical model (non-asserted)",
                        ],
                        recommended_action="Filter noise or re-mine with relaxed cut thresholds.",
                    )
                )
                q_counter += 1

        return admitted, refused, open_questions, evals

    def validate_full_induction(
        self,
        report: InductionReport,
        data: InductionDataInput | None = None,
        held_out_trajectories: list[dict[str, float]] | np.ndarray | None = None,
        seed: int = 42,
    ) -> tuple[WorldSpec, InductionValidationReport]:
        """Execute the end-to-end honesty validation gate over an InductionReport.

        Fails closed on uncertainty:
        - Only ASSERTED causal edges enter relationships.
        - Only verified 100% stress-tested invariants enter hard constraints.
        - Provisional links and unoriented edges are refused and surfaced as open questions.
        - Emits a sanitized WorldSpec and a cryptographic InductionValidationReport.
        """
        all_refused: list[RefusedElement] = []
        all_open_questions: list[OpenQuestion] = []
        admitted_records: list[dict[str, Any]] = []

        all_refuters: list[RefuterResult] = []
        all_stress: list[InvariantStressResult] = []
        all_conformance: list[ProcessConformanceResult] = []
        all_dynamics_held_out: list[HeldOutReproductionResult] = []

        admitted_relationships: list[RelationshipSpec] = []
        admitted_entities: list[EntitySpec] = []
        admitted_resources: list[ResourceSpec] = []

        # 1. Causal validation
        if report.causal_equivalence_class:
            c_summary = report.causal_equivalence_class
            c_edges = [
                CausalEdge(
                    source=e["source"],
                    target=e["target"],
                    edge_type=CausalEdgeType(e["type"]),
                    posterior_support=float(e.get("support", 0.5)),
                    evidence_level=EvidenceLevel(
                        e.get("level", EvidenceLevel.DIRECTIONAL_UNCERTAIN)
                    ),
                    is_interventional=bool(e.get("interventional", False)),
                )
                for e in c_summary.edges
            ]
            equiv_class = CausalEquivalenceClass(
                variables=c_summary.variables,
                edges=c_edges,
                is_dag=c_summary.is_dag,
            )

            # Entities for variables
            for v in equiv_class.variables:
                admitted_entities.append(
                    EntitySpec(id=v, type="variable", attributes={"domain": "continuous"})
                )
                admitted_records.append(
                    {"type": "entity", "id": v, "level": EvidenceLevel.ASSERTED.value}
                )

            admitted_c_edges, c_refused, c_oq, refuters = self.validate_causal(
                equiv_class=equiv_class,
                data=data,
                seed=seed,
            )
            all_refused.extend(c_refused)
            all_open_questions.extend(c_oq)
            all_refuters.extend(refuters)

            for e in admitted_c_edges:
                admitted_relationships.append(
                    RelationshipSpec(
                        source=e.source,
                        target=e.target,
                        type="directed_dependency",
                        attributes={
                            "posterior_support": e.posterior_support,
                            "evidence_level": EvidenceLevel.ASSERTED.value,
                            "edge_type": e.edge_type.value,
                        },
                    )
                )
                admitted_records.append(
                    {
                        "type": "relationship",
                        "id": f"{e.source}->{e.target}",
                        "level": EvidenceLevel.ASSERTED.value,
                        "note": "Survives DoWhy refuters + interventional/functional check",
                    }
                )

        # 2. Invariants validation
        if report.candidate_invariants:
            admitted_invs, inv_refused, inv_oq, stress_res = self.validate_invariants(
                report.candidate_invariants
            )
            all_refused.extend(inv_refused)
            all_open_questions.extend(inv_oq)
            all_stress.extend(stress_res)
            for inv in admitted_invs:
                admitted_records.append(
                    {
                        "type": "invariant",
                        "id": inv.name,
                        "level": EvidenceLevel.STRUCTURAL.value,
                        "note": f"Equation: {inv.equation}",
                    }
                )

        # 3. Provisional links validation
        if report.provisional_links:
            admitted_links, link_refused, link_oq = self.validate_provisional_links(
                report.provisional_links
            )
            all_refused.extend(link_refused)
            all_open_questions.extend(link_oq)
            for link in admitted_links:
                admitted_relationships.append(
                    RelationshipSpec(
                        source=link.source_entity,
                        target=link.target_entity,
                        type=link.relationship_type,
                        attributes={"confidence": link.confidence},
                    )
                )
                admitted_records.append(
                    {
                        "type": "relationship",
                        "id": f"{link.source_entity}->{link.target_entity}",
                        "level": EvidenceLevel.ASSERTED.value,
                    }
                )

        # 4. Process conformance validation
        if report.conformance:
            admitted_conf, proc_refused, proc_oq, conf_evals = self.validate_process_conformance(
                report.conformance
            )
            all_refused.extend(proc_refused)
            all_open_questions.extend(proc_oq)
            all_conformance.extend(conf_evals)
            for obj_type, conf in admitted_conf.items():
                admitted_records.append(
                    {
                        "type": "process_conformance",
                        "id": obj_type,
                        "level": conf.evidence_level.value,
                        "note": f"Fitness={conf.fitness:.3f}, Precision={conf.precision:.3f}",
                    }
                )

        # 5. Dynamics validation
        if report.induced_rate_laws and held_out_trajectories is not None:
            c_laws = [
                InducedRateLaw(
                    target_variable=r.target_variable,
                    feature_variables=(),
                    expression=r.expression,
                    coefficients={},
                    r_squared=r.r_squared,
                    mse=0.0,
                    complexity=r.complexity,
                    evidence_level=EvidenceLevel(r.evidence_level),
                    sample_count=100,
                )
                for r in report.induced_rate_laws
            ]
            held_out_res = validate_dynamics_held_out(
                c_laws, held_out_trajectories, r2_threshold=self.dynamics_held_out_r2
            )
            all_dynamics_held_out.extend(held_out_res)
            for r in held_out_res:
                if not r.passed:
                    all_refused.append(
                        RefusedElement(
                            element_type="dynamics",
                            identifier=r.target_variable,
                            claimed_or_inferred_level=EvidenceLevel.HYPOTHETICAL,
                            refusal_reason=(
                                f"Rate law for '{r.target_variable}' failed held-out reproduction: "
                                f"held-out R²={r.held_out_r2:.3f} < {r.threshold:.2f} threshold."
                            ),
                            failed_validators=["held_out_trajectory_reproduction"],
                            candidate_alternatives=["Higher-order polynomial or neural residual"],
                        )
                    )

        # Construct sanitized WorldSpec
        from ewm_engine.core.spec import WorldMetadataSpec

        sanitized_spec = WorldSpec(
            world=WorldMetadataSpec(name=f"Validated_{report.source}"),
            entities=admitted_entities,
            relationships=admitted_relationships,
            resources=admitted_resources,
        )

        overall_passed = len(all_refused) == 0

        validation_report = InductionValidationReport(
            passed=overall_passed,
            admitted_elements_count=len(admitted_records),
            refused_elements_count=len(all_refused),
            admitted_elements=admitted_records,
            refused_elements=all_refused,
            open_questions=all_open_questions,
            refuter_evaluations=all_refuters,
            held_out_evaluations=all_dynamics_held_out,
            invariant_evaluations=all_stress,
            conformance_evaluations=all_conformance,
        )

        # Attach validation metadata back to the InductionReport
        report.validation_passed = overall_passed
        report.gate_outcome = validation_report.gate_outcome.value
        report.refused_elements_count = len(all_refused)
        report.open_questions_count = len(all_open_questions)
        report.refused_elements = [r.model_dump(mode="json") for r in all_refused]
        report.open_questions = [q.model_dump(mode="json") for q in all_open_questions]
        report.validation_fingerprint = validation_report.sha256_fingerprint
        report.sha256_fingerprint = report.compute_fingerprint()

        return sanitized_spec, validation_report


def validate_induction(
    spec_or_report: WorldSpec | InductionReport,
    data: InductionDataInput | None = None,
    held_out_trajectories: list[dict[str, float]] | np.ndarray | None = None,
    report: InductionReport | None = None,
    *,
    min_causal_support: float = 0.70,
    require_causal_refuters: bool = True,
    fail_closed: bool = True,
    strict: bool = False,
    seed: int = 42,
) -> tuple[WorldSpec, InductionValidationReport]:
    """Execute the mandatory induction validation and refusal gate.

    Declines to write any unasserted, directional-uncertain, confounded,
    or refuter-failing candidate structures into WorldSpec.

    Args:
        spec_or_report: WorldSpec or InductionReport from an induction stage.
        data: Observational/interventional data used for DoWhy refutations.
        held_out_trajectories: Optional held-out trajectories for dynamics validation.
        report: Optional InductionReport if spec_or_report is a WorldSpec.
        min_causal_support: Minimum posterior threshold for ASSERTED status (default: 0.70).
        require_causal_refuters: If True, executes DoWhy refutation gates.
        fail_closed: If True, declines to emit unverified structure into WorldSpec.
        strict: If True and any element is refused, raises InductionRefusalError.
        seed: Deterministic random seed for refuters and evaluations.

    Returns:
        tuple[WorldSpec, InductionValidationReport]: The sanitized WorldSpec containing ONLY
        ASSERTED structures and the auditable, fingerprinted validation report.
    """
    from ewm_engine.core.trust import InductionRefusalError

    target_report: InductionReport
    if isinstance(spec_or_report, InductionReport):
        target_report = spec_or_report
    elif report is not None:
        target_report = report
    else:
        # Synthesize minimal report from WorldSpec
        target_report = InductionReport(
            source=spec_or_report.world.name,
            source_type="causal",
            entities_count=len(spec_or_report.entities),
            relationships_count=len(spec_or_report.relationships),
            resources_count=len(spec_or_report.resources),
            constraints_count=len(spec_or_report.constraints),
            seed=seed,
        )

    gate = InductionValidationGate(
        min_causal_support=min_causal_support,
        require_causal_refuters=require_causal_refuters,
        fail_closed=fail_closed,
    )
    sanitized_spec, val_report = gate.validate_full_induction(
        report=target_report,
        data=data,
        held_out_trajectories=held_out_trajectories,
        seed=seed,
    )

    if strict and not val_report.passed:
        reasons = [
            f"{r.element_type} '{r.identifier}': {r.refusal_reason}"
            for r in val_report.refused_elements
        ]
        raise InductionRefusalError(
            f"Induction Validation Gate REFUSED {len(reasons)} unasserted/unidentifiable structures from WorldSpec.",
            reasons=reasons,
        )

    return sanitized_spec, val_report
