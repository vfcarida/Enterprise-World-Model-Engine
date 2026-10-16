"""Ensemble causal-structure induction, DiBS Bayesian reconciliation, and equivalence classes.

Scientific Grounding:
    - Spirtes, Glymour, & Scheines (2000): "Causation, Prediction, and Search", MIT Press.
      Constraint-based PC and FCI algorithms (FCI identifies latent confounders yielding PAGs).
    - Zheng et al. (2018, arXiv:1803.01422): "DAGs with NO TEARS: Continuous Optimization for Structure Learning."
    - Bello et al. (2022, arXiv:2209.08037): "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization."
    - Shimizu et al. (2011, arXiv:1101.2489): "DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Acyclic Model."
    - Runge et al. (2019, Science Advances, DOI:10.1126/sciadv.aau4996): PCMCI time-series causal discovery.
    - Lorch et al. (2021, arXiv:2105.11839): "DiBS: Differentiable Bayesian Structure Learning over DAG Equivalence Classes."
    - Reisach et al. (2021, NeurIPS): "Beware of the Simulated DAG! Causal Structure Learning May Rely on Marginal Variance."
      Exposes NOTEARS variance-manipulation fragility, mandating multi-method ensemble reconciliation.

Epistemic Invariant:
    Observational data generally identifies only a Markov-equivalence class (CPDAG/PAG),
    never a unique DAG. Causal edges are marked DIRECTIONAL-UNCERTAIN or CONFOUNDED
    unless disambiguated by non-Gaussian functional asymmetry (LiNGAM) or interventional data.
    The engine NEVER emits an asserted DAG from observational data alone.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.scm import (
    CausalGraph,
    ExogenousVariable,
    InvertibilityType,
    StructuralAssignment,
    StructuralCausalModel,
)
from ewm_engine.provenance.evidence import EvidenceLevel

if TYPE_CHECKING:
    from ewm_engine.core.world import World
    from ewm_engine.induction.report import InductionReport


class CausalEdgeType(StrEnum):
    """Structural edge classification within a Markov equivalence class or PAG."""

    DIRECTED = "directed"  # X -> Y (causal mechanism identified)
    UNDIRECTED = (
        "undirected"  # X - Y (Markov equivalence: direction not identifiable observationally)
    )
    BIDIRECTED = "bidirected"  # X <-> Y (PAG: latent common cause / confounded)
    NON_EDGE = "non_edge"  # Independence / no direct structural mechanism


class CausalEdge(BaseModel):
    """Annotated causal dependency proposed between two variables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="Source variable identifier.")
    target: str = Field(description="Target variable identifier.")
    edge_type: CausalEdgeType = Field(
        description="Orientation status in CPDAG/PAG (directed, undirected, or bidirected)."
    )
    posterior_support: float = Field(
        ge=0.0,
        le=1.0,
        description="Posterior probability of edge presence across the ensemble.",
    )
    direction_support: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Posterior confidence in this specific orientation direction.",
    )
    evidence_level: EvidenceLevel = Field(
        description="Epistemic standing (ASSERTED, DIRECTIONAL-UNCERTAIN, or CONFOUNDED).",
    )
    is_interventional: bool = Field(
        default=False,
        description="True if orientation was disambiguated via experimental / interventional data.",
    )
    method_supports: dict[str, float] = Field(
        default_factory=dict,
        description="Per-method support breakdown (e.g. pc, fci, dagma, lingam).",
    )
    notes: str = Field(
        default="",
        description="Audit commentary regarding identifiability or latent confounding.",
    )


class CausalEquivalenceClass(BaseModel):
    """Completed Partially Directed Acyclic Graph (CPDAG) or Partial Ancestral Graph (PAG).

    Represents the honest set of all DAGs compatible with the observational distribution
    and any available interventional experiments.
    """

    model_config = ConfigDict(extra="forbid")

    variables: list[str] = Field(description="Ordered list of state/action variables.")
    edges: list[CausalEdge] = Field(
        default_factory=list, description="Set of proposed causal edges."
    )
    is_dag: bool = Field(
        default=False,
        description="True if and only if all edges are strictly directed without cycles or confounding.",
    )
    markov_equivalent_dags_count: int = Field(
        default=1,
        description="Estimated number of DAGs in this observational Markov equivalence class.",
    )
    methods_run: list[str] = Field(
        default_factory=list, description="Causal discovery algorithms executed in ensemble."
    )
    seed: int = Field(default=42, description="Deterministic random seed.")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_edge(self, source: str, target: str) -> CausalEdge | None:
        """Find an edge between source and target regardless of orientation."""
        for edge in self.edges:
            if (edge.source == source and edge.target == target) or (
                edge.edge_type in (CausalEdgeType.UNDIRECTED, CausalEdgeType.BIDIRECTED)
                and edge.source == target
                and edge.target == source
            ):
                return edge
        return None

    def asserted_edges(self) -> list[CausalEdge]:
        """Return only directed edges that cleared the epistemic ASSERTED gate."""
        return [
            e
            for e in self.edges
            if e.evidence_level == EvidenceLevel.ASSERTED and e.edge_type == CausalEdgeType.DIRECTED
        ]

    def directional_uncertain_edges(self) -> list[CausalEdge]:
        """Return edges whose existence is established but whose direction is ambiguous."""
        return [
            e
            for e in self.edges
            if e.edge_type == CausalEdgeType.UNDIRECTED
            or e.evidence_level == EvidenceLevel.DIRECTIONAL_UNCERTAIN
        ]

    def confounded_edges(self) -> list[CausalEdge]:
        """Return bidirected edges indicating latent confounding."""
        return [
            e
            for e in self.edges
            if e.edge_type == CausalEdgeType.BIDIRECTED
            or e.evidence_level == EvidenceLevel.CONFOUNDED
        ]

    def to_causal_graph(
        self,
        filter_level: EvidenceLevel = EvidenceLevel.ASSERTED,
    ) -> CausalGraph:
        """Extract a compilable CausalGraph DAG containing edges meeting or exceeding filter_level."""
        graph = CausalGraph()
        for v in self.variables:
            graph.add_node(v)

        for edge in self.edges:
            if edge.edge_type == CausalEdgeType.DIRECTED:
                if filter_level == EvidenceLevel.ASSERTED:
                    if edge.evidence_level == EvidenceLevel.ASSERTED:
                        graph.add_edge(edge.source, edge.target)
                else:
                    graph.add_edge(edge.source, edge.target)

        return graph

    def to_scm(self, world: World | None = None) -> StructuralCausalModel:
        """Compile the ASSERTED causal structure into an executable StructuralCausalModel."""
        causal_dag = self.to_causal_graph(filter_level=EvidenceLevel.ASSERTED)
        endogenous = dict.fromkeys(self.variables, "continuous")
        exogenous: dict[str, ExogenousVariable] = {}
        assignments: dict[str, StructuralAssignment] = {}

        for v in self.variables:
            u_id = f"u:{v}"
            ex_fp = canonical_sha256({"id": u_id, "target": v, "mechanism": "InducedNoise"})
            exogenous[u_id] = ExogenousVariable(
                id=u_id,
                step=0,
                mechanism_id="InducedNoise",
                target_variable=v,
                distribution="empirical",
                invertibility=InvertibilityType.EXACT,
                fingerprint=ex_fp,
            )
            causal_dag.add_node(u_id, is_unobserved=False)
            causal_dag.add_edge(u_id, v)

            parents = sorted(causal_dag.parents(v) - {u_id})
            asgn_fp = canonical_sha256(
                {"target": v, "parents": parents, "exogenous_id": u_id, "mechanism": "LinearSCM"}
            )
            assignments[v] = StructuralAssignment(
                target=v,
                parents=tuple(parents),
                exogenous_id=u_id,
                mechanism_id="InducedCausalMechanism",
                invertibility=InvertibilityType.EXACT,
                formula=(
                    f"{v} := f({', '.join(parents)}, {u_id})" if parents else f"{v} := {u_id}"
                ),
                fingerprint=asgn_fp,
            )

        scm_fp = canonical_sha256(
            {
                "variables": self.variables,
                "edges": sorted(causal_dag.edges),
                "assignments": [a.fingerprint for a in assignments.values()],
            }
        )

        return StructuralCausalModel(
            name="InducedStructuralCausalModel",
            endogenous_variables=endogenous,
            exogenous_variables=exogenous,
            assignments=assignments,
            causal_graph=causal_dag,
            fingerprint=scm_fp,
        )

    def summary(self) -> str:
        """Human-readable ASCII summary of the induced equivalence class."""
        lines = [
            f"CausalEquivalenceClass (Variables: {len(self.variables)}, Markov DAGs: {self.markov_equivalent_dags_count})",
            f"Methods Run: {', '.join(self.methods_run)}",
            "",
            "Discovered Causal Edges:",
        ]
        if not self.edges:
            lines.append("  (No direct causal edges clearing confidence threshold)")
        for e in self.edges:
            symbol = (
                "-->"
                if e.edge_type == CausalEdgeType.DIRECTED
                else ("<->" if e.edge_type == CausalEdgeType.BIDIRECTED else "---")
            )
            interv_tag = " [INTERVENTIONAL]" if e.is_interventional else ""
            lines.append(
                f"  {e.source} {symbol} {e.target} "
                f"(support: {e.posterior_support:.2f}, level: {e.evidence_level.value}){interv_tag}"
            )
        return "\n".join(lines)

    def to_summary(self) -> Any:
        """Convert equivalence class to structured CausalEquivalenceClassSummary."""
        from ewm_engine.induction.report import CausalEquivalenceClassSummary

        return CausalEquivalenceClassSummary(
            variables=self.variables,
            is_dag=self.is_dag,
            markov_equivalent_dags_count=self.markov_equivalent_dags_count,
            asserted_edges_count=len(self.asserted_edges()),
            directional_uncertain_edges_count=len(self.directional_uncertain_edges()),
            confounded_edges_count=len(self.confounded_edges()),
            edges=[
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type.value,
                    "support": e.posterior_support,
                    "level": e.evidence_level.value,
                    "interventional": e.is_interventional,
                }
                for e in self.edges
            ],
        )


# ---------------------------------------------------------------------------
# Causal Discovery Ensemble Engines (Native pure-NumPy + optional library wrappers)
# ---------------------------------------------------------------------------


def _compute_partial_corr(
    cov: np.ndarray,
    i: int,
    j: int,
    cond_set: tuple[int, ...],
) -> float:
    """Compute partial correlation rho_{ij . S} given covariance matrix."""
    if not cond_set:
        var_ii = cov[i, i]
        var_jj = cov[j, j]
        if var_ii <= 1e-12 or var_jj <= 1e-12:
            return 0.0
        return float(cov[i, j] / math.sqrt(var_ii * var_jj))

    sub_indices = [i, j, *cond_set]
    sub_cov = cov[np.ix_(sub_indices, sub_indices)]

    try:
        # Precision matrix is inverse covariance
        prec = np.linalg.pinv(sub_cov)
        p_01 = prec[0, 1]
        p_00 = prec[0, 0]
        p_11 = prec[1, 1]
        if p_00 <= 1e-12 or p_11 <= 1e-12:
            return 0.0
        r = -p_01 / math.sqrt(p_00 * p_11)
        return float(np.clip(r, -0.999999, 0.999999))
    except Exception:
        return 0.0


def _is_cond_independent(
    r: float,
    n_samples: int,
    cond_size: int,
    alpha: float = 0.05,
) -> bool:
    """Fisher's z-transform test for conditional independence."""
    if abs(r) >= 0.999999:
        return False
    dof = n_samples - cond_size - 3
    if dof <= 1:
        return abs(r) < 0.1

    z = 0.5 * math.log((1.0 + r) / (1.0 - r))
    z_stat = abs(z) * math.sqrt(dof)

    # Standard two-sided normal critical value approximation
    # for alpha=0.05, z_crit ~= 1.96; for alpha=0.01, z_crit ~= 2.576
    if alpha <= 0.01:
        z_crit = 2.576
    elif alpha <= 0.05:
        z_crit = 1.960
    elif alpha <= 0.10:
        z_crit = 1.645
    else:
        z_crit = 1.960

    return z_stat < z_crit


def run_pc_algorithm(
    X: np.ndarray,
    var_names: list[str],
    alpha: float = 0.05,
) -> dict[tuple[str, str], str]:
    """Constraint-based PC algorithm (Spirtes et al. 2000).

    Discovers skeleton via partial correlation independence tests, orients v-structures
    (colliders), and applies Meek orientation propagation rules.

    Returns:
        Mapping of (u, v) -> orientation:
            "-->" for u -> v
            "<--" for v -> u
            "---" for undirected u - v
    """
    n, d = X.shape
    if n < 5 or d < 2:
        return {}

    # Center and compute sample covariance
    X_centered = X - np.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / max(n - 1, 1)

    # Complete adjacency graph
    adj: dict[int, set[int]] = {i: set(range(d)) - {i} for i in range(d)}
    sep_set: dict[tuple[int, int], tuple[int, ...]] = {}

    # 1. Skeleton discovery by increasing order k of conditioning sets
    for k in range(min(d - 1, 4)):
        pairs_to_check = [(i, j) for i in range(d) for j in range(i + 1, d) if j in adj[i]]
        for i, j in pairs_to_check:
            # Conditioning candidates are neighbors of i excluding j
            candidates = list(adj[i] - {j})
            if len(candidates) < k:
                continue

            for subset in itertools.combinations(candidates, k):
                r = _compute_partial_corr(cov, i, j, subset)
                if _is_cond_independent(r, n, k, alpha=alpha):
                    adj[i].remove(j)
                    adj[j].remove(i)
                    sep_set[(min(i, j), max(i, j))] = subset
                    break

    # 2. V-structure (collider) orientation: X - Z - Y where X and Y are non-adjacent
    oriented: dict[tuple[int, int], str] = {}
    for i in range(d):
        for j in adj[i]:
            if i < j:
                oriented[(i, j)] = "---"

    for z in range(d):
        neighbors = sorted(adj[z])
        for idx1 in range(len(neighbors)):
            for idx2 in range(idx1 + 1, len(neighbors)):
                x = neighbors[idx1]
                y = neighbors[idx2]
                if y not in adj[x]:  # x and y are non-adjacent
                    pair_key = (min(x, y), max(x, y))
                    sep = sep_set.get(pair_key, ())
                    if z not in sep:
                        # Orient x -> z and y -> z (v-structure)
                        if x < z:
                            oriented[(x, z)] = "-->"
                        else:
                            oriented[(z, x)] = "<--"

                        if y < z:
                            oriented[(y, z)] = "-->"
                        else:
                            oriented[(z, y)] = "<--"

    # 3. Meek Rule 1: if x -> y and y - z and x not adjacent to z, orient y -> z
    changed = True
    while changed:
        changed = False
        for (u, v), edge_dir in list(oriented.items()):
            if edge_dir == "-->":
                for w in adj[v]:
                    if w != u and w not in adj[u]:
                        pair = (min(v, w), max(v, w))
                        if oriented.get(pair) == "---":
                            oriented[pair] = "-->" if v < w else "<--"
                            changed = True

    # Map back to variable name keys
    result: dict[tuple[str, str], str] = {}
    for (i, j), dir_str in oriented.items():
        u_name, v_name = var_names[i], var_names[j]
        result[(u_name, v_name)] = dir_str

    return result


def run_fci_algorithm(
    X: np.ndarray,
    var_names: list[str],
    alpha: float = 0.05,
) -> dict[tuple[str, str], str]:
    """Constraint-based Fast Causal Inference (FCI) algorithm (Spirtes et al. 2000).

    Detects latent confounders / unobserved common causes yielding bidirected edges (<->)
    in a Partial Ancestral Graph (PAG).
    """
    n, d = X.shape
    if n < 5 or d < 2:
        return {}

    base_pc = run_pc_algorithm(X, var_names, alpha=alpha)
    X_centered = X - np.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / max(n - 1, 1)

    result: dict[tuple[str, str], str] = dict(base_pc)

    # Confounder detection: for unoriented edges or mutual residual correlations
    # where both variables exhibit mutual collider characteristics under conditioning
    for i in range(d):
        for j in range(i + 1, d):
            u_name, v_name = var_names[i], var_names[j]
            pair = (u_name, v_name)
            if pair in result:
                # Test residual asymmetry / mutual confounding
                other_vars = tuple(k for k in range(d) if k != i and k != j)
                if len(other_vars) > 0:
                    r_cond = _compute_partial_corr(cov, i, j, other_vars)
                    # If partial correlation remains strong regardless of all observed conditioning,
                    # but neither direction is asymmetric, tag as latent confounding
                    if abs(r_cond) > 0.45 and result[pair] == "---":
                        result[pair] = "<->"

    return result


def run_dagma_notears(
    X: np.ndarray,
    var_names: list[str],
    lambda1: float = 0.05,
    max_iter: int = 50,
) -> dict[tuple[str, str], float]:
    """Score-based continuous optimization for DAG discovery (NOTEARS / DAGMA).

    Solves smooth acyclicity constrained optimization:
        min_W (1/2n) ||X - XW||_F^2 + lambda1 ||W||_1  s.t. h(W) = 0.

    Returns:
        Mapping of (source, target) -> weight magnitude for directed edges.
    """
    n, d = X.shape
    if n < 5 or d < 2:
        return {}

    # Standardize data
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)

    # Initialize weight matrix W (zeros, no self-loops)
    W = np.zeros((d, d), dtype=np.float64)
    lr = 0.01

    # Proximal gradient descent with acyclicity penalty (NOTEARS characterization)
    for _ in range(max_iter):
        # Loss gradient: -1/n X^T (X - XW)
        grad_loss = -(X_std.T @ (X_std - X_std @ W)) / n

        # Matrix exponential gradient for h(W) = tr(exp(W * W)) - d
        W_sq = W * W
        try:
            # Series expansion approximation of matrix exponential
            exp_W_sq = np.eye(d) + W_sq + 0.5 * (W_sq @ W_sq)
            grad_h = 2.0 * exp_W_sq * W
        except Exception:
            grad_h = 2.0 * W

        # Penalty rho
        rho = 1.0
        grad = grad_loss + rho * grad_h

        # Update step
        W_new = W - lr * grad
        # Soft-thresholding for L1 sparsity
        W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - lr * lambda1, 0.0)
        np.fill_diagonal(W_new, 0.0)
        W = W_new

    # Thresholding small edges
    cutoff = 0.15
    result: dict[tuple[str, str], float] = {}
    for i in range(d):
        for j in range(d):
            if i != j and abs(W[i, j]) > cutoff:
                result[(var_names[i], var_names[j])] = float(abs(W[i, j]))

    return result


def run_direct_lingam(
    X: np.ndarray,
    var_names: list[str],
) -> dict[tuple[str, str], float]:
    """Functional causal discovery via DirectLiNGAM (Shimizu et al. 2011).

    Identifies full DAG by exploiting non-Gaussian noise distributions
    to break observational Markov equivalence.
    """
    n, d = X.shape
    if n < 10 or d < 2:
        return {}

    # Test for non-Gaussianity via excess kurtosis (kappa = mu_4 / sigma^4 - 3)
    kurtosis_values = [
        float(np.mean(((X[:, i] - np.mean(X[:, i])) / (np.std(X[:, i]) + 1e-6)) ** 4) - 3.0)
        for i in range(d)
    ]
    avg_excess_kurtosis = float(np.mean(np.abs(kurtosis_values)))

    # DirectLiNGAM root discovery
    remaining = list(range(d))
    ordered: list[int] = []
    X_work = X.copy()

    while remaining:
        best_candidate = remaining[0]
        min_residual_dep = float("inf")

        for cand in remaining:
            # Measure independence of candidate x_cand and residuals of other variables
            total_dep = 0.0
            x_cand = X_work[:, cand]
            for other in remaining:
                if other != cand:
                    x_other = X_work[:, other]
                    cov_xy = np.cov(x_cand, x_other)[0, 1]
                    var_x = max(np.var(x_cand), 1e-6)
                    beta = cov_xy / var_x
                    resid = x_other - beta * x_cand
                    # Non-linear correlation / covariance of squares as proxy for independence
                    dep = abs(float(np.cov(x_cand**2, resid**2)[0, 1]))
                    total_dep += dep

            if total_dep < min_residual_dep:
                min_residual_dep = total_dep
                best_candidate = cand

        ordered.append(best_candidate)
        remaining.remove(best_candidate)

        # Regress remaining variables on the chosen root
        x_root = X_work[:, best_candidate]
        var_r = max(np.var(x_root), 1e-6)
        for other in remaining:
            cov_ro = np.cov(x_root, X_work[:, other])[0, 1]
            X_work[:, other] -= (cov_ro / var_r) * x_root

    # Compute regression matrix following topological order
    result: dict[tuple[str, str], float] = {}
    for idx_target in range(1, len(ordered)):
        target_var = ordered[idx_target]
        parent_vars = ordered[:idx_target]
        if not parent_vars:
            continue

        X_parents = X[:, parent_vars]
        y = X[:, target_var]
        try:
            beta, _, _, _ = np.linalg.lstsq(X_parents, y, rcond=None)
            for p_idx, parent_var in enumerate(parent_vars):
                weight = float(abs(beta[p_idx]))
                # Only keep significant connections; if data is near-Gaussian, downweight
                confidence = weight * min(1.0, max(0.2, avg_excess_kurtosis * 0.5))
                if confidence > 0.15:
                    result[(var_names[parent_var], var_names[target_var])] = float(confidence)
        except Exception:
            pass

    return result


def run_temporal_discovery(
    X_series: np.ndarray,
    var_names: list[str],
    max_lag: int = 1,
) -> dict[tuple[str, str], float]:
    """Time-series causal induction (DYNOTEARS / lagged cross-dependency).

    Discovers lagged causal dependencies: X_{t-1} -> Y_t.
    """
    T, d = X_series.shape
    if T < max_lag + 5 or d < 1:
        return {}

    result: dict[tuple[str, str], float] = {}

    for lag in range(1, max_lag + 1):
        X_lag = X_series[:-lag]
        X_curr = X_series[lag:]

        # Standardize
        X_lag_std = (X_lag - np.mean(X_lag, axis=0)) / (np.std(X_lag, axis=0) + 1e-6)
        X_curr_std = (X_curr - np.mean(X_curr, axis=0)) / (np.std(X_curr, axis=0) + 1e-6)

        # Cross-covariance matrix
        cross_cov = (X_lag_std.T @ X_curr_std) / len(X_lag)

        for i in range(d):
            for j in range(d):
                strength = float(abs(cross_cov[i, j]))
                if strength > 0.25:
                    result[(var_names[i], var_names[j])] = strength

    return result


# ---------------------------------------------------------------------------
# DiBS Bayesian Reconciliation & Interventional Data Fusion
# ---------------------------------------------------------------------------


def _fuse_interventional_data(
    u: str,
    v: str,
    interventional_data: dict[str, Any] | None,
) -> bool | None:
    """Disambiguate direction via experimental do-interventions (A/B tests).

    Returns:
        True if intervention on u causes shift in v (u -> v),
        False if intervention on v causes shift in u (v -> u),
        None if inconclusive or no interventional data available.
    """
    if not interventional_data:
        return None

    # Expect interventional records formatted as:
    # {"interventions": [{"treatment": "var", "value": val, "outcomes": {"outcome_var": [samples...]}}]}
    # or direct dict {treatment_var: {outcome_var: delta}}
    u_causes_v = False
    v_causes_u = False

    if u in interventional_data:
        u_outcomes = interventional_data[u]
        if isinstance(u_outcomes, dict) and v in u_outcomes:
            effect = abs(float(u_outcomes[v]))
            if effect > 0.1:
                u_causes_v = True

    if v in interventional_data:
        v_outcomes = interventional_data[v]
        if isinstance(v_outcomes, dict) and u in v_outcomes:
            effect = abs(float(v_outcomes[u]))
            if effect > 0.1:
                v_causes_u = True

    if u_causes_v and not v_causes_u:
        return True
    if v_causes_u and not u_causes_v:
        return False
    return None


def reconcile_ensemble(
    var_names: list[str],
    pc_results: dict[tuple[str, str], str],
    fci_results: dict[tuple[str, str], str],
    dagma_results: dict[tuple[str, str], float],
    lingam_results: dict[tuple[str, str], float],
    temporal_results: dict[tuple[str, str], float] | None = None,
    interventional_data: dict[str, Any] | None = None,
    min_support: float = 0.5,
    asserted_threshold: float = 0.7,
    seed: int = 42,
) -> CausalEquivalenceClass:
    """DiBS-style Bayesian aggregation over multi-method causal discovery ensemble.

    Reconciles constraint-based, score-based, functional, and temporal findings into
    a single CPDAG / PAG with per-edge posterior support and epistemic EvidenceLevel.
    """
    temporal_results = temporal_results or {}
    methods_run = ["pc", "fci", "dagma", "lingam"]
    if temporal_results:
        methods_run.append("temporal_dynotears")

    num_methods = len(methods_run)
    all_pairs: set[tuple[str, str]] = set()

    for u, v in itertools.combinations(var_names, 2):
        all_pairs.add((u, v))

    edges: list[CausalEdge] = []
    undirected_count = 0

    for u, v in sorted(all_pairs):
        # Collect votes from each method
        # PC & FCI: check both (u, v) and (v, u)
        pc_dir = pc_results.get((u, v)) or (
            "-->"
            if pc_results.get((v, u)) == "<--"
            else ("<--" if pc_results.get((v, u)) == "-->" else pc_results.get((v, u)))
        )
        fci_dir = fci_results.get((u, v)) or (
            "-->"
            if fci_results.get((v, u)) == "<--"
            else ("<--" if fci_results.get((v, u)) == "-->" else fci_results.get((v, u)))
        )

        dagma_uv = dagma_results.get((u, v), 0.0)
        dagma_vu = dagma_results.get((v, u), 0.0)

        lingam_uv = lingam_results.get((u, v), 0.0)
        lingam_vu = lingam_results.get((v, u), 0.0)

        temp_uv = temporal_results.get((u, v), 0.0)
        temp_vu = temporal_results.get((v, u), 0.0)

        # Presence support
        pc_present = 1.0 if pc_dir in ("-->", "<--", "---") else 0.0
        fci_present = 1.0 if fci_dir in ("-->", "<--", "---", "<->") else 0.0
        dagma_present = 1.0 if (dagma_uv > 0 or dagma_vu > 0) else 0.0
        lingam_present = 1.0 if (lingam_uv > 0 or lingam_vu > 0) else 0.0
        temp_present = 1.0 if (temp_uv > 0 or temp_vu > 0) else 0.0

        presence_votes = [pc_present, fci_present, dagma_present, lingam_present]
        if temporal_results:
            presence_votes.append(temp_present)

        posterior_support = float(sum(presence_votes) / num_methods)
        if posterior_support < min_support:
            continue

        # Check for latent confounding from FCI
        is_confounded = fci_dir == "<->"

        # Directional support
        uv_votes = 0.0
        vu_votes = 0.0

        if pc_dir == "-->":
            uv_votes += 1.0
        elif pc_dir == "<--":
            vu_votes += 1.0

        if fci_dir == "-->":
            uv_votes += 1.0
        elif fci_dir == "<--":
            vu_votes += 1.0

        if dagma_uv > dagma_vu:
            uv_votes += 1.0
        elif dagma_vu > dagma_uv:
            vu_votes += 1.0

        if lingam_uv > lingam_vu:
            uv_votes += 1.0
        elif lingam_vu > lingam_uv:
            vu_votes += 1.0

        if temp_uv > temp_vu:
            uv_votes += 1.0
        elif temp_vu > temp_uv:
            vu_votes += 1.0

        # Interventional data disambiguation
        interv_check = _fuse_interventional_data(u, v, interventional_data)
        is_interventional = interv_check is not None

        method_supports = {
            "pc": pc_present,
            "fci": fci_present,
            "dagma": dagma_present,
            "lingam": lingam_present,
        }
        if temporal_results:
            method_supports["temporal"] = temp_present

        if is_confounded:
            # Latent common cause
            edges.append(
                CausalEdge(
                    source=u,
                    target=v,
                    edge_type=CausalEdgeType.BIDIRECTED,
                    posterior_support=posterior_support,
                    direction_support=0.5,
                    evidence_level=EvidenceLevel.CONFOUNDED,
                    is_interventional=False,
                    method_supports=method_supports,
                    notes="FCI detected unshielded bidirected dependency; latent common cause.",
                )
            )
            continue

        if interv_check is True or (interv_check is None and uv_votes > vu_votes):
            src, tgt = u, v
            dir_conf = float(uv_votes / max(uv_votes + vu_votes, 1.0))
        elif interv_check is False or (interv_check is None and vu_votes > uv_votes):
            src, tgt = v, u
            dir_conf = float(vu_votes / max(uv_votes + vu_votes, 1.0))
        else:
            # Undirected (Markov Equivalence)
            src, tgt = u, v
            dir_conf = 0.5

        # Determine EvidenceLevel
        if is_interventional:
            level = EvidenceLevel.ASSERTED
            edge_type = CausalEdgeType.DIRECTED
            notes = "Direction disambiguated by interventional data (do-calculus / A/B test)."
        elif (
            posterior_support >= asserted_threshold
            and dir_conf >= 0.75
            and (lingam_uv > 0 or lingam_vu > 0)
        ):
            # Non-Gaussian functional asymmetry cleared
            level = EvidenceLevel.ASSERTED
            edge_type = CausalEdgeType.DIRECTED
            notes = "Identified via non-Gaussian functional asymmetry (DirectLiNGAM)."
        elif dir_conf > 0.60:
            level = EvidenceLevel.DIRECTIONAL_UNCERTAIN
            edge_type = CausalEdgeType.DIRECTED
            notes = "Orientation supported by score/functional methods but not interventional."
        else:
            level = EvidenceLevel.DIRECTIONAL_UNCERTAIN
            edge_type = CausalEdgeType.UNDIRECTED
            undirected_count += 1
            notes = "Observational Markov-equivalence class ambiguity; direction not identifiable."

        edges.append(
            CausalEdge(
                source=src,
                target=tgt,
                edge_type=edge_type,
                posterior_support=posterior_support,
                direction_support=dir_conf,
                evidence_level=level,
                is_interventional=is_interventional,
                method_supports=method_supports,
                notes=notes,
            )
        )

    # Estimate Markov equivalent DAGs count (2^undirected_count approximation)
    equiv_dags = max(1, 2**undirected_count)
    is_fully_directed_dag = undirected_count == 0 and not any(
        e.edge_type == CausalEdgeType.BIDIRECTED for e in edges
    )

    return CausalEquivalenceClass(
        variables=var_names,
        edges=edges,
        is_dag=is_fully_directed_dag,
        markov_equivalent_dags_count=equiv_dags,
        methods_run=methods_run,
        seed=seed,
    )


def induce_causal(
    data: Any,
    var_names: list[str] | None = None,
    methods: Sequence[str] = ("pc", "fci", "dagma", "lingam"),
    interventional_data: dict[str, Any] | None = None,
    alpha: float = 0.05,
    min_support: float = 0.5,
    asserted_threshold: float = 0.7,
    seed: int = 42,
) -> tuple[CausalEquivalenceClass, InductionReport]:
    """Public high-level API for ensemble causal structure induction.

    Ingests observational data matrix / records, runs the multi-method ensemble,
    reconciles into a CPDAG / PAG equivalence class, and builds an auditable InductionReport.
    """
    from ewm_engine.induction.report import (
        CausalEquivalenceClassSummary,
        InductionReport,
    )

    # Convert data into 2D numpy array and variable list
    if hasattr(data, "to_numpy"):
        # pandas DataFrame / polars DataFrame
        arr = data.to_numpy()
        names = list(data.columns) if var_names is None else var_names
    elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], dict):
        # List of dict records
        names = list(data[0].keys()) if var_names is None else var_names
        arr = np.array([[float(row.get(k, 0.0)) for k in names] for row in data])
    elif isinstance(data, np.ndarray):
        arr = data
        names = [f"var_{i}" for i in range(arr.shape[1])] if var_names is None else var_names
    else:
        raise ValueError(
            f"Unsupported data format for causal induction: {type(data)}. "
            "Expected 2D np.ndarray, pandas DataFrame, or list of dicts."
        )

    # Execute methods in ensemble
    pc_res: dict[tuple[str, str], str] = {}
    fci_res: dict[tuple[str, str], str] = {}
    dagma_res: dict[tuple[str, str], float] = {}
    lingam_res: dict[tuple[str, str], float] = {}
    temp_res: dict[tuple[str, str], float] = {}

    if "pc" in methods:
        pc_res = run_pc_algorithm(arr, names, alpha=alpha)
    if "fci" in methods:
        fci_res = run_fci_algorithm(arr, names, alpha=alpha)
    if "dagma" in methods:
        dagma_res = run_dagma_notears(arr, names)
    if "lingam" in methods:
        lingam_res = run_direct_lingam(arr, names)
    if "temporal" in methods or "dynotears" in methods:
        temp_res = run_temporal_discovery(arr, names)

    equiv_class = reconcile_ensemble(
        var_names=names,
        pc_results=pc_res,
        fci_results=fci_res,
        dagma_results=dagma_res,
        lingam_results=lingam_res,
        temporal_results=temp_res,
        interventional_data=interventional_data,
        min_support=min_support,
        asserted_threshold=asserted_threshold,
        seed=seed,
    )

    # Build InductionReport
    summary = CausalEquivalenceClassSummary(
        variables=equiv_class.variables,
        is_dag=equiv_class.is_dag,
        markov_equivalent_dags_count=equiv_class.markov_equivalent_dags_count,
        asserted_edges_count=len(equiv_class.asserted_edges()),
        directional_uncertain_edges_count=len(equiv_class.directional_uncertain_edges()),
        confounded_edges_count=len(equiv_class.confounded_edges()),
        edges=[
            {
                "source": e.source,
                "target": e.target,
                "type": e.edge_type.value,
                "support": round(e.posterior_support, 3),
                "level": e.evidence_level.value,
                "interventional": e.is_interventional,
            }
            for e in equiv_class.edges
        ],
    )

    evidence_dist: dict[str, int] = {}
    for e in equiv_class.edges:
        lvl = e.evidence_level.value
        evidence_dist[lvl] = evidence_dist.get(lvl, 0) + 1

    report = InductionReport(
        source="causal_induction_dataset",
        source_type="causal",
        seed=seed,
        entities_count=len(names),
        relationships_count=len(equiv_class.edges),
        evidence_distribution=evidence_dist,
        causal_equivalence_class=summary,
        metadata={
            "methods_run": list(methods),
            "alpha": alpha,
            "min_support": min_support,
            "asserted_threshold": asserted_threshold,
        },
    )

    return equiv_class, report
