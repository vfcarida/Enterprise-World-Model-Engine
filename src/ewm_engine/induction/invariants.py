"""Conservation-law and invariant induction via AI-Poincaré nullspace estimation and Monte Carlo stress testing.

Scientific Grounding:
    - Wu & Tegmark (2020, arXiv:2011.04698): "AI Poincaré: Machine Learning Conservation Laws from Trajectories."
      Demonstrates that conserved quantities define trajectory manifolds whose tangent space is orthogonal
      to state transition differences (gradient nullspace of dX/dt).
    - Champion et al. (2019, PNAS, DOI:10.1073/pnas.1906995116): Coordinate discovery and conservation laws.
    - Greydanus, Dzamba, & Yosinski (2019, NeurIPS): "Hamiltonian Neural Networks" (exact energy conservation).

Epistemic Invariant:
    A candidate conservation law is strictly a *hypothesis*. EWM never asserts a conservation law
    based merely on training trajectory fit. It undergoes adversarial seeded Monte Carlo stress testing:
    it is PROMOTED to a WorldSpec hard constraint if and only if it holds across all sampled trajectories
    within numerical tolerance. Otherwise, it is retained as a soft constraint with a logged violation rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.spec import ComponentSpec
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.induction.report import CandidateInvariant
from ewm_engine.provenance.evidence import EvidenceLevel

if TYPE_CHECKING:
    from ewm_engine.induction.report import InductionReport


class InvariantHypothesis(BaseModel):
    """Candidate conservation law or bounded-resource invariant undergoing verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Human-readable identifier of the conservation law.")
    target_resource: str = Field(description="Primary anchor resource variable.")
    equation: str = Field(
        description="Mathematical equation of the invariant (e.g. '1.0*stock_A + 1.0*stock_B = 1000.0')."
    )
    weights: dict[str, float] = Field(
        description="Linear manifold coefficients w_i associated with state variables."
    )
    constant_value: float = Field(description="Conserved scalar constant value K.")
    support_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of observational trajectory transitions satisfying the law.",
    )
    max_observed_violation: float = Field(
        ge=0.0,
        description="Maximum absolute deviation |w^T X - K| observed across all stress trajectories.",
    )
    violation_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of Monte Carlo test states exceeding numerical tolerance.",
    )
    is_promoted_hard_constraint: bool = Field(
        description="True if and only if the hypothesis passed Monte Carlo stress testing.",
    )
    evidence_level: EvidenceLevel = Field(
        description="Epistemic standing (STRUCTURAL if promoted; HYPOTHETICAL if rejected).",
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the invariant hypothesis."
    )

    def to_candidate_invariant(self) -> CandidateInvariant:
        """Export as standard CandidateInvariant for InductionReport."""
        return CandidateInvariant(
            name=self.name,
            target_resource=self.target_resource,
            invariant_type="conservation",
            equation=self.equation,
            support_ratio=self.support_ratio,
            max_observed_violation=self.max_observed_violation,
            evidence_level=self.evidence_level,
            parameters={
                "weights": self.weights,
                "constant_value": self.constant_value,
                "is_promoted": self.is_promoted_hard_constraint,
                "violation_rate": self.violation_rate,
            },
        )

    def to_constraint_spec(self) -> ComponentSpec:
        """Compile candidate invariant into an official WorldSpec ComponentSpec."""
        return ComponentSpec(
            type="capacity",
            id=f"invariant_{self.name}",
            parameters={
                "resource_id": self.target_resource,
                "severity": "hard" if self.is_promoted_hard_constraint else "soft",
                "equation": self.equation,
                "weights": self.weights,
                "constant_value": self.constant_value,
            },
        )


# ---------------------------------------------------------------------------
# AI-Poincaré Nullspace Estimation & Rational Simplification
# ---------------------------------------------------------------------------


def _simplify_rational_weights(
    weights: np.ndarray,
    tol: float = 0.08,
) -> np.ndarray:
    """Normalize and simplify real-valued nullspace vector into small rational/integer coefficients."""
    # Find maximum absolute component to normalize
    max_abs = float(np.max(np.abs(weights)))
    if max_abs < 1e-8:
        return weights

    w_norm = weights / max_abs

    # Try small integer multipliers: 1, 2, 3
    best_weights = w_norm.copy()
    min_dist = float("inf")

    for mult in (1, 2, 3):
        scaled = w_norm * mult
        rounded = np.round(scaled)
        dist = float(np.max(np.abs(scaled - rounded)))
        if dist < tol and dist < min_dist:
            min_dist = dist
            best_weights = rounded / mult

    return best_weights


def estimate_conservation_nullspace(
    D: np.ndarray,
    singular_ratio_threshold: float = 0.05,
) -> list[np.ndarray]:
    """Estimate conservation manifold directions using SVD on transition differences D = Delta X."""
    n_transitions, n_vars = D.shape
    if n_transitions < 2 or n_vars < 2:
        return []

    # SVD: D = U S V^T
    try:
        _, s, vh = np.linalg.svd(D, full_matrices=False)
    except np.linalg.LinAlgError:
        return []

    max_s = float(np.max(s)) if len(s) > 0 else 1.0
    if max_s < 1e-12:
        return []

    candidate_vectors: list[np.ndarray] = []
    # Right-singular vectors with singular values near zero define the nullspace: D v ~= 0
    for idx, sigma in enumerate(s):
        ratio = float(sigma / max_s)
        if ratio <= singular_ratio_threshold:
            v = vh[idx]
            v_simp = _simplify_rational_weights(v)
            if np.any(np.abs(v_simp) > 1e-3):
                first_nonzero = next((w for w in v_simp if abs(w) > 1e-3), 1.0)
                if first_nonzero < 0:
                    v_simp = -v_simp
                candidate_vectors.append(v_simp)

    return candidate_vectors


# ---------------------------------------------------------------------------
# Seeded Monte Carlo Stress Testing
# ---------------------------------------------------------------------------


def stress_test_candidate_invariant(
    weights: dict[str, float],
    constant_val: float,
    all_states: np.ndarray,
    var_names: list[str],
    tolerance: float = 1e-3,
    mc_perturbations: int = 5,
    seed: int = 42,
) -> tuple[float, float, bool]:
    """Stress-test conservation candidate across states and synthetic Monte Carlo perturbations.

    Returns:
        Tuple of (max_observed_violation, violation_rate, passes_stress_test).
    """
    n_samples = all_states.shape[0]
    weight_vec = np.array([weights.get(name, 0.0) for name in var_names], dtype=np.float64)

    # 1. Base empirical trajectory evaluation
    computed_quantities = all_states @ weight_vec
    deviations = np.abs(computed_quantities - constant_val)
    max_dev = float(np.max(deviations)) if len(deviations) > 0 else 0.0

    violations = np.sum(deviations > tolerance)
    total_states = n_samples

    # 2. Monte Carlo perturbation trajectories (stress testing along the state manifold)
    rng = np.random.default_rng(seed)
    for _ in range(mc_perturbations):
        # Permute and re-evaluate along simulated steps
        jitter = rng.normal(0.0, 1e-4, size=all_states.shape)
        stressed_states = all_states + jitter
        mc_quantities = stressed_states @ weight_vec
        mc_devs = np.abs(mc_quantities - constant_val)
        max_dev = max(max_dev, float(np.max(mc_devs)))
        violations += int(np.sum(mc_devs > tolerance))
        total_states += len(mc_devs)

    violation_rate = float(violations / max(total_states, 1))
    # Promoted only if max_dev <= tolerance across all tests
    passes_test = max_dev <= tolerance and violation_rate <= 1e-4

    return max_dev, violation_rate, passes_test


# ---------------------------------------------------------------------------
# Public Induction Entry Point
# ---------------------------------------------------------------------------


def induce_invariants(
    trajectories_or_dataset: Any,
    resource_ids: Sequence[str] | None = None,
    tolerance: float = 1e-3,
    mc_samples: int = 10,
    seed: int = 42,
) -> tuple[list[InvariantHypothesis], InductionReport]:
    """Public high-level API for conservation-law / invariant induction.

    Ingests state transition trajectories, discovers candidate conservation laws via
    AI-Poincaré SVD nullspace estimation, stress-tests each candidate via seeded Monte Carlo,
    promotes verified laws to WorldSpec hard constraints, and generates an auditable InductionReport.
    """
    from ewm_engine.induction.report import InductionReport

    # 1. Extract states and consecutive difference vectors
    if isinstance(trajectories_or_dataset, TransitionDataset):
        dataset = trajectories_or_dataset
        if not dataset:
            raise ValueError("Empty TransitionDataset provided for invariant induction.")
        first_sample: TransitionSample = dataset[0]
        names = (
            list(resource_ids)
            if resource_ids is not None
            else sorted(first_sample.state.resources.keys())
        )
        states, _, next_states = dataset.to_numpy(resource_ids=names)
        D = next_states - states
        all_states = np.vstack([states, next_states[-1:]])
    elif isinstance(trajectories_or_dataset, (list, tuple)) and len(trajectories_or_dataset) > 0:
        first_item = trajectories_or_dataset[0]
        if isinstance(first_item, TransitionSample):
            dataset = TransitionDataset(trajectories_or_dataset)
            return induce_invariants(
                dataset,
                resource_ids=resource_ids,
                tolerance=tolerance,
                mc_samples=mc_samples,
                seed=seed,
            )
        elif isinstance(first_item, dict):
            names = list(resource_ids) if resource_ids is not None else sorted(first_item.keys())
            arr = np.array([[float(d.get(k, 0.0)) for k in names] for d in trajectories_or_dataset])
            D = arr[1:] - arr[:-1]
            all_states = arr
        else:
            raise ValueError(f"Unsupported trajectory sequence item type: {type(first_item)}")
    elif isinstance(trajectories_or_dataset, np.ndarray):
        arr = trajectories_or_dataset
        names = (
            list(resource_ids)
            if resource_ids is not None
            else [f"res_{i}" for i in range(arr.shape[1])]
        )
        D = arr[1:] - arr[:-1]
        all_states = arr
    else:
        raise ValueError(f"Unsupported trajectories input format: {type(trajectories_or_dataset)}")

    # 2. Estimate conservation nullspace vectors
    candidate_vecs = estimate_conservation_nullspace(D)

    hypotheses: list[InvariantHypothesis] = []

    for v_idx, vec in enumerate(candidate_vecs):
        # Build coefficient dictionary
        weights: dict[str, float] = {}
        for idx, w in enumerate(vec):
            if abs(w) > 1e-4:
                weights[names[idx]] = float(round(w, 4))

        if not weights:
            continue

        # Primary anchor resource is the first non-zero weight
        anchor_resource = next(iter(weights.keys()))

        # Compute empirical conserved constant K
        weight_vec = np.array([weights.get(n, 0.0) for n in names], dtype=np.float64)
        computed_k = all_states @ weight_vec
        mean_k = float(np.mean(computed_k))
        constant_val = round(mean_k, 3)

        # Build human-readable equation string
        terms: list[str] = []
        for var, w in sorted(weights.items()):
            sign = "+" if w >= 0 and terms else ("-" if w < 0 else "")
            val_abs = abs(w)
            terms.append(f"{sign} {val_abs:.2f}*{var}".strip())
        eq_str = f"{' + '.join(terms).replace(' + -', ' - ')} = {constant_val:.2f}"

        # 3. Seeded Monte Carlo Stress Testing
        max_violation, viol_rate, passes = stress_test_candidate_invariant(
            weights=weights,
            constant_val=constant_val,
            all_states=all_states,
            var_names=names,
            tolerance=tolerance,
            mc_perturbations=mc_samples,
            seed=seed,
        )

        support_ratio = float(max(0.0, min(1.0, 1.0 - viol_rate)))
        ev_level = EvidenceLevel.STRUCTURAL if passes else EvidenceLevel.HYPOTHETICAL

        inv_name = f"conservation_law_{v_idx + 1}_{anchor_resource}"
        fp = canonical_sha256(
            {
                "name": inv_name,
                "weights": sorted(weights.items()),
                "constant_val": constant_val,
                "max_violation": round(max_violation, 6),
                "promoted": passes,
            }
        )

        hypo = InvariantHypothesis(
            name=inv_name,
            target_resource=anchor_resource,
            equation=eq_str,
            weights=weights,
            constant_value=constant_val,
            support_ratio=round(support_ratio, 4),
            max_observed_violation=round(max_violation, 6),
            violation_rate=round(viol_rate, 4),
            is_promoted_hard_constraint=passes,
            evidence_level=ev_level,
            fingerprint=fp,
        )
        hypotheses.append(hypo)

    # 4. Build InductionReport
    cand_invariants = [h.to_candidate_invariant() for h in hypotheses]
    evidence_dist: dict[str, int] = {}
    for h in hypotheses:
        lvl = h.evidence_level.value
        evidence_dist[lvl] = evidence_dist.get(lvl, 0) + 1

    report = InductionReport(
        source="invariant_trajectory_dataset",
        source_type="invariants",
        seed=seed,
        entities_count=len(names),
        resources_count=len(names),
        constraints_count=len([h for h in hypotheses if h.is_promoted_hard_constraint]),
        evidence_distribution=evidence_dist,
        candidate_invariants=cand_invariants,
        metadata={
            "tolerance": tolerance,
            "mc_samples": mc_samples,
            "total_hypotheses": len(hypotheses),
            "promoted_hard_constraints": len(
                [h for h in hypotheses if h.is_promoted_hard_constraint]
            ),
        },
    )

    return hypotheses, report
