"""Dynamic Causality-Aware Root Cause Analysis (Dynamic RCA) via eIRWR & DynaCausal (E7).

Models microservice incident cascades, upstream retry storms, and anomaly-conditioned
resilience degradation using sparse row-normalized transition matrices (`scipy.sparse`)
and Enhanced Iterative Random Walk with Restart (eIRWR).

Mathematical Formulation & Hyperparameters (E7 Requirements 7, 8, 12)
---------------------------------------------------------------------
Given a microservice call graph $\\mathcal{G} = (V, E)$ with $N = |V|$ services where a
directed edge $(u, v) \\in E$ denotes service $u$ (caller) invoking service $v$ (callee):

1. **Base Call Adjacency Matrix ($W$)**:
   $W \\in \\mathbb{R}_{\\ge 0}^{N \\times N}$ (`scipy.sparse.csr_matrix`) encodes direct
   call dependencies $W_{uv} > 0$ for $(u, v) \\in E$.

2. **Anomaly-Conditioned Resilience ($\\beta$, $R_{\\text{base}}$)**:
   At iteration $t$, let $\\hat{s}_i^{(t)} = s_i^{(t)} / (\\max_j s_j^{(t)} + \\epsilon) \\in [0, 1]$
   be the normalized stress score of service $i$. Service $i$'s dynamic resilience degrades
   exponentially with stress:
       $$R_i^{(t)} = R_{\\text{base}} \\cdot \\exp\\!\\left(-\\beta \\, \\hat{s}_i^{(t)}\\right)$$
   where:
   - $\\beta > 0$ (`beta`): **Anomaly sensitivity parameter** governing how rapidly stress
     erodes node resilience and accelerates fault propagation.
   - $R_{\\text{base}} \\in (0, 1]$ (`r_base`): **Baseline healthy resilience** of an unstressed
     microservice ($\\hat{s}_i = 0 \\implies R_i = R_{\\text{base}}$).

3. **Backward Retry-Storm Edges ($\\rho$) & Self-Loop Accumulation**:
   When a downstream callee $v$ degrades, caller $u$ experiences timeouts and triggers retry
   storms, propagating stress upstream along the reversed edge $v \\to u$. Controlled by:
   - $\\rho \\in (0, 1]$ (`rho`): **Backward discount factor** weighting upstream retry-storm
     edges $W_{\\text{bwd}} = \\rho \\, W^{\\top}$ relative to forward call edges.
   - Self-loop retention $D_{ii}^{(t)} = (1 - R_i^{(t)}) \\cdot (1 + \\hat{s}_i^{(t)})$ accumulates
     probability density at highly stressed origin nodes.

4. **Strict Row-Normalization & eIRWR Propagation**:
   The combined sparse transition matrix $P^{(t)} = D_{\\text{row}}^{-1} \\tilde{P}^{(t)}$ is
   strictly row-normalized ($\\sum_j P_{ij}^{(t)} = 1$ for every row $i$), and the eIRWR
   state vector updates over at most `max_iterations <= 100` steps via:
       $$s^{(t+1)} = (1 - \\alpha) \\, \\left(s^{(t)} P^{(t)}\\right) + \\alpha \\, v^{(0)}$$
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Final, List, Optional, Tuple, Union

import numpy as np

from ewm_engine.core.types import EngineError

if TYPE_CHECKING:
    import networkx as nx
    import scipy.sparse as sp

MAX_EIRWR_ITERATIONS: Final[int] = 100


class DynamicRCAError(EngineError):
    """Raised when Dynamic RCA graph inputs, anomaly vectors, or iteration bounds are invalid."""


def _import_scipy_sparse() -> Any:
    """Lazily import `scipy.sparse` isolated to the SWE domain package (E7 Req 13)."""
    import scipy.sparse as sp

    return sp


def _import_networkx() -> Any:
    """Lazily import `networkx`."""
    import networkx as nx

    return nx


@dataclass(frozen=True)
class DynamicRCAResult:
    """Output of the Dynamic RCA (eIRWR + DynaCausal) microservice fault propagation simulator.

    Attributes:
        nodes: Ordered tuple of microservice node identifiers corresponding to matrix indices.
        forward_adjacency: Dense $N \\times N$ representation of the forward call matrix $W$.
        backward_adjacency: Dense $N \\times N$ representation of the backward retry matrix $\\rho W^{\\top}$.
        final_transition_matrix: Strictly row-normalized $N \\times N$ transition matrix $P^{(T)}$ at termination.
        timeline_scores: Tuple of anomaly score dictionaries across all simulated steps $t = 0, \\dots, T$.
        timeline_resilience: Tuple of node resilience dictionaries $R_i^{(t)} = R_{\\text{base}} \\exp(-\\beta \\hat{s}_i^{(t)})$.
        final_scores: Final converged eIRWR anomaly/root-cause probability distribution at step $T$.
        ranked_root_causes: Tuple of `(node_id, score)` sorted in descending order of root-cause probability.
        identified_origin: Top-1 identified root-cause origin microservice.
        static_pagerank_scores: Baseline static PageRank score dictionary for comparison.
        static_pagerank_top1: Top-1 node returned by the static PageRank baseline.
        iterations_run: Number of eIRWR power iterations executed ($\\le 100$).
        converged: True if $\\|s^{(t+1)} - s^{(t)}\\|_1 < \text{tol}$ within `max_iterations`.
    """

    nodes: Tuple[str, ...]
    forward_adjacency: List[List[float]]
    backward_adjacency: List[List[float]]
    final_transition_matrix: List[List[float]]
    timeline_scores: Tuple[Dict[str, float], ...]
    timeline_resilience: Tuple[Dict[str, float], ...]
    final_scores: Dict[str, float]
    ranked_root_causes: Tuple[Tuple[str, float], ...]
    identified_origin: str
    static_pagerank_scores: Dict[str, float]
    static_pagerank_top1: str
    iterations_run: int
    converged: bool


class DynamicRCASimulator:
    """Dynamic causality-aware microservice fault propagation and Root Cause Analysis simulator (E7).

    Combines **DynaCausal** anomaly-conditioned resilience:
        $$R_i^{(t)} = R_{\\text{base}} \\cdot \\exp\\!\\left(-\\beta \\, \\hat{s}_i^{(t)}\\right)$$
    with **Enhanced Iterative Random Walk with Restart (eIRWR)** over sparse row-normalized
    transition matrices (`scipy.sparse.csr_matrix`) incorporating backward retry-storm edges
    weighted by discount factor $\\rho$.

    Parameters
    ----------
    call_graph : networkx.DiGraph or Mapping[str, Sequence[str]]
        Microservice call graph where directed edge `(u, v)` means caller `u` calls callee `v`.
    r_base : float, default=0.85
        Baseline healthy resilience $R_{\\text{base}} \\in (0, 1]$ when node stress $\\hat{s}_i = 0$.
    beta : float, default=2.5
        Sensitivity parameter $\\beta > 0$ controlling exponential resilience decay
        $R_i = R_{\\text{base}} \\exp(-\\beta \\hat{s}_i)$.
    rho : float, default=0.75
        Backward discount factor $\\rho \\in (0, 1]$ scaling upstream retry-storm transitions
        from stressed callees back to callers ($v \\to u$).
    forward_weight : float, default=0.15
        Relative weight $\\gamma_{\\text{fwd}} \\ge 0$ for forward downstream call propagation ($u \\to v$).
    restart_alpha : float, default=0.30
        eIRWR restart probability $\\alpha \\in (0, 1)$ anchoring the walk to the initial anomaly vector.
    max_iterations : int, default=50
        Maximum eIRWR power iterations (operational safeguard: strictly capped at `<= 100`).
    tol : float, default=1e-6
        L1 convergence threshold $\\|s^{(t+1)} - s^{(t)}\\|_1 < \\text{tol}$.
    """

    def __init__(
        self,
        call_graph: Union[nx.DiGraph, Mapping[str, Sequence[str]]],
        *,
        r_base: float = 0.85,
        beta: float = 2.5,
        rho: float = 0.75,
        forward_weight: float = 0.15,
        restart_alpha: float = 0.30,
        max_iterations: int = 50,
        tol: float = 1e-6,
    ) -> None:
        if not (0.0 < r_base <= 1.0):
            raise DynamicRCAError(f"r_base must be in (0, 1], got {r_base}.")
        if beta <= 0.0:
            raise DynamicRCAError(f"Sensitivity parameter beta must be > 0, got {beta}.")
        if not (0.0 < rho <= 1.0):
            raise DynamicRCAError(f"Backward discount factor rho must be in (0, 1], got {rho}.")
        if forward_weight < 0.0:
            raise DynamicRCAError(f"forward_weight must be >= 0, got {forward_weight}.")
        if not (0.0 < restart_alpha < 1.0):
            raise DynamicRCAError(f"restart_alpha must be in (0, 1), got {restart_alpha}.")
        if max_iterations < 1 or max_iterations > MAX_EIRWR_ITERATIONS:
            raise DynamicRCAError(
                f"Operational safeguard: max_iterations must be in [1, {MAX_EIRWR_ITERATIONS}], "
                f"got {max_iterations}."
            )

        self.r_base: float = float(r_base)
        self.beta: float = float(beta)
        self.rho: float = float(rho)
        self.forward_weight: float = float(forward_weight)
        self.restart_alpha: float = float(restart_alpha)
        self.max_iterations: int = int(max_iterations)
        self.tol: float = float(tol)

        self._nodes, self._w_fwd, self._w_bwd = self._build_sparse_adjacency(call_graph)

    @property
    def nodes(self) -> Tuple[str, ...]:
        """Return ordered tuple of microservice node names."""
        return self._nodes

    @property
    def adjacency_matrix(self) -> sp.csr_matrix:
        """Return the sparse forward microservice call-graph adjacency matrix $W$ (`scipy.sparse.csr_matrix`)."""
        return self._w_fwd.copy()

    @property
    def backward_adjacency_matrix(self) -> sp.csr_matrix:
        """Return the sparse backward retry-storm adjacency matrix $\\rho W^{\\top}$ (`scipy.sparse.csr_matrix`)."""
        return self._w_bwd.copy()

    def _build_sparse_adjacency(
        self,
        call_graph: Union[nx.DiGraph, Mapping[str, Sequence[str]]],
    ) -> Tuple[Tuple[str, ...], sp.csr_matrix, sp.csr_matrix]:
        """Construct sparse CSR forward ($W$) and backward ($\\rho W^{\\top}$) adjacency matrices."""
        sp = _import_scipy_sparse()
        nx = _import_networkx()

        if isinstance(call_graph, nx.DiGraph):
            if call_graph.number_of_nodes() == 0:
                raise DynamicRCAError("Microservice call_graph must contain at least one node.")
            ordered_nodes = tuple(str(n) for n in call_graph.nodes())
            node_idx = {name: i for i, name in enumerate(ordered_nodes)}
            rows: List[int] = []
            cols: List[int] = []
            vals: List[float] = []
            for u, v, data in call_graph.edges(data=True):
                u_str, v_str = str(u), str(v)
                w = float(data.get("weight", 1.0)) if isinstance(data, Mapping) else 1.0
                if w < 0.0:
                    raise DynamicRCAError(
                        f"Edge weight ({u_str} -> {v_str}) must be >= 0, got {w}."
                    )
                rows.append(node_idx[u_str])
                cols.append(node_idx[v_str])
                vals.append(w)
        elif isinstance(call_graph, Mapping):
            node_set: set[str] = {str(k) for k in call_graph.keys()}
            for targets in call_graph.values():
                node_set.update(str(t) for t in targets)
            if not node_set:
                raise DynamicRCAError("Microservice call_graph mapping cannot be empty.")
            ordered_nodes = tuple(sorted(node_set))
            node_idx = {name: i for i, name in enumerate(ordered_nodes)}
            rows = []
            cols = []
            vals = []
            for u, targets in call_graph.items():
                u_i = node_idx[str(u)]
                for v in targets:
                    v_i = node_idx[str(v)]
                    rows.append(u_i)
                    cols.append(v_i)
                    vals.append(1.0)
        else:
            raise DynamicRCAError(
                f"Expected networkx.DiGraph or adjacency mapping, got {type(call_graph).__name__}."
            )

        n = len(ordered_nodes)
        w_fwd = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        w_bwd = (w_fwd.transpose() * self.rho).tocsr()
        return ordered_nodes, w_fwd, w_bwd

    def _coerce_stress_vector(
        self,
        vector: Union[Mapping[str, float], Sequence[float], np.ndarray],
        *,
        param_name: str = "stress_vector",
    ) -> np.ndarray:
        """Validate and coerce a node stress/anomaly vector into a finite 1D non-negative array."""
        n = len(self._nodes)
        if isinstance(vector, Mapping):
            s_arr = np.zeros(n, dtype=np.float64)
            for idx, node_name in enumerate(self._nodes):
                if node_name in vector:
                    try:
                        val = float(vector[node_name])
                    except (TypeError, ValueError) as exc:
                        raise DynamicRCAError(
                            f"{param_name} entry for '{node_name}' must be numeric."
                        ) from exc
                    if val < 0.0 or not np.isfinite(val):
                        raise DynamicRCAError(
                            f"{param_name} for '{node_name}' must be finite and >= 0, got {val}."
                        )
                    s_arr[idx] = val
            return s_arr

        try:
            s_arr = np.asarray(vector, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise DynamicRCAError(f"{param_name} must be a 1D numeric array or mapping.") from exc
        if len(s_arr) != n:
            raise DynamicRCAError(
                f"{param_name} length ({len(s_arr)}) does not match node count ({n})."
            )
        if np.any(s_arr < 0.0) or not np.all(np.isfinite(s_arr)):
            raise DynamicRCAError(f"{param_name} values must be finite and >= 0.")
        return s_arr

    def compute_resilience(
        self,
        stress_vector: Union[Mapping[str, float], Sequence[float], np.ndarray],
    ) -> np.ndarray:
        """Compute Anomaly-Conditioned Resilience $R_i = R_{\\text{base}} \\cdot \\exp(-\\beta \\hat{s}_i)$ (E7 Req 7).

        Args:
            stress_vector: Mapping or 1D numpy array of non-negative finite node stress/anomaly scores $s_i$.

        Returns:
            1D numpy array of node resilience values $R_i \\in (0, R_{\\text{base}}]$.
        """
        s_arr = self._coerce_stress_vector(stress_vector, param_name="stress_vector")
        max_s = float(np.max(s_arr))
        if max_s > 1e-12:
            s_hat = s_arr / max_s
        else:
            s_hat = np.zeros_like(s_arr)
        return self.r_base * np.exp(-self.beta * s_hat)

    def build_transition_matrix(
        self,
        stress_vector: Union[Mapping[str, float], Sequence[float], np.ndarray],
        initial_anomaly: Optional[Union[Mapping[str, float], Sequence[float], np.ndarray]] = None,
    ) -> sp.csr_matrix:
        """Construct the strictly row-normalized dynamic sparse transition matrix $P^{(t)}$ (E7 Req 8).

        For each row $i$:
        - Outgoing backward retry-storm edges $i \\to j$ (where $j$ calls $i$) carry weight
          $\\rho W_{ji} (1 - R_i^{(t)} + 0.05)$, modeling how a stressed callee $i$ propagates
          retry-storm pressure upstream to caller $j$.
        - Outgoing forward call edges $i \\to k$ (where $i$ calls $k$) carry weight
          $\\gamma_{\\text{fwd}} W_{ik} (1 - R_k^{(t)} + 0.05)$.
        - Self-loop $i \\to i$ carries weight $(1 - R_i^{(t)}) \\cdot (1 + 2 \\hat{s}_i^{(0)}) + R_i^{(t)} \\cdot 0.1$,
          accumulating probability density at the true fault origin while ensuring every row has
          strictly positive sum prior to exact row normalization.
        """
        sp = _import_scipy_sparse()
        n = len(self._nodes)
        s_arr = self._coerce_stress_vector(stress_vector, param_name="stress_vector")
        resilience = self.compute_resilience(s_arr)
        fragility = 1.0 - resilience  # Higher when node is stressed (R_i is low)

        if initial_anomaly is not None:
            init_arr = self._coerce_stress_vector(initial_anomaly, param_name="initial_anomaly")
            max_init = float(np.max(init_arr))
            s0_hat = (init_arr / max_init) if max_init > 1e-12 else np.zeros(n, dtype=np.float64)
        else:
            max_s = float(np.max(s_arr))
            s0_hat = (s_arr / max_s) if max_s > 1e-12 else np.zeros(n, dtype=np.float64)

        # Row-scale backward edges by source node fragility (stressed callee sheds retry load upstream)
        bwd_row_scale = sp.diags(fragility + 0.05, format="csr")
        bwd_part = bwd_row_scale @ self._w_bwd

        # Column-scale forward edges by target callee fragility (callers are drawn into stressed callees)
        fwd_col_scale = sp.diags(fragility + 0.05, format="csr")
        fwd_part = (self._w_fwd * self.forward_weight) @ fwd_col_scale

        # Anomaly-conditioned self-loops accumulate mass at high-stress / low-resilience origin nodes
        self_loop_weights = fragility * (1.5 + 2.5 * s0_hat) + resilience * 0.05
        diag_part = sp.diags(self_loop_weights, format="csr")

        p_tilde = (bwd_part + fwd_part + diag_part).tocsr()

        # Strict row-normalization: ensure sum_j P[i, j] == 1.0 for every row i
        row_sums = np.asarray(p_tilde.sum(axis=1)).reshape(-1)
        inv_row_sums = np.divide(
            1.0,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 1e-15,
        )
        p_norm = (sp.diags(inv_row_sums, format="csr") @ p_tilde).tocsr()
        return p_norm

    def compute_static_pagerank_baseline(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
    ) -> Dict[str, float]:
        """Compute a static PageRank baseline over the unconditioned call graph for comparison (E7 Req 10).

        Because static PageRank ignores backward retry-storm dynamics ($\\rho$) and
        anomaly-conditioned resilience degradation ($R_i = R_{\\text{base}} \\exp(-\\beta \\hat{s}_i)$),
        it fails to distinguish whether an upstream caller or a downstream bottleneck triggered
        a reverse retry cascade.
        """
        sp = _import_scipy_sparse()
        n = len(self._nodes)
        uniform = np.full(n, 1.0 / n, dtype=np.float64)

        # Static reverse call transition without anomaly-conditioned resilience or self-loop anchoring
        w_rev = self._w_fwd.transpose().tocsr()
        row_sums = np.asarray(w_rev.sum(axis=1)).reshape(-1)
        inv_row = np.divide(
            1.0,
            row_sums,
            out=np.zeros_like(row_sums, dtype=np.float64),
            where=row_sums > 1e-15,
        )
        p_static = (sp.diags(inv_row, format="csr") @ w_rev).tocsr()
        dangling_mask = row_sums <= 1e-15

        scores = uniform.copy()
        for _ in range(min(max_iter, MAX_EIRWR_ITERATIONS)):
            dangling_mass = float(np.sum(scores[dangling_mask]))
            next_scores = (
                damping * (scores @ p_static + dangling_mass * uniform) + (1.0 - damping) * uniform
            )
            next_scores = np.asarray(next_scores).reshape(-1)
            next_scores /= float(np.sum(next_scores))
            if float(np.sum(np.abs(next_scores - scores))) < self.tol:
                scores = next_scores
                break
            scores = next_scores

        return {self._nodes[i]: float(scores[i]) for i in range(n)}

    def simulate(
        self,
        initial_anomaly: Union[Mapping[str, float], Sequence[float], np.ndarray],
        *,
        steps: Optional[int] = None,
    ) -> DynamicRCAResult:
        r"""Run the eIRWR + DynaCausal fault propagation loop and identify the root-cause cascade.

        Args:
            initial_anomaly: Mapping of `{node_id: anomaly_magnitude}` or 1D array of length $N$.
            steps: Optional explicit number of power iterations ($\le 100$); defaults to `self.max_iterations`.

        Returns:
            `DynamicRCAResult` containing the propagation timeline, resilience trajectory,
            strictly row-normalized transition matrix, and ranked root cause origin.
        """
        n = len(self._nodes)
        max_steps = self.max_iterations if steps is None else int(steps)
        if max_steps < 1 or max_steps > MAX_EIRWR_ITERATIONS:
            raise DynamicRCAError(
                f"Operational safeguard: steps must be in [1, {MAX_EIRWR_ITERATIONS}], got {max_steps}."
            )

        if isinstance(initial_anomaly, Mapping):
            s0 = np.zeros(n, dtype=np.float64)
            for idx, node_name in enumerate(self._nodes):
                if node_name in initial_anomaly:
                    val = float(initial_anomaly[node_name])
                    if val < 0.0 or not np.isfinite(val):
                        raise DynamicRCAError(
                            f"Initial anomaly for '{node_name}' must be finite and >= 0, got {val}."
                        )
                    s0[idx] = val
        else:
            s0 = np.asarray(initial_anomaly, dtype=np.float64).reshape(-1)
            if len(s0) != n:
                raise DynamicRCAError(
                    f"Initial anomaly vector length ({len(s0)}) does not match node count ({n})."
                )
            if np.any(s0 < 0.0) or not np.all(np.isfinite(s0)):
                raise DynamicRCAError("Initial anomaly vector values must be finite and >= 0.")

        total_s0 = float(np.sum(s0))
        if total_s0 <= 1e-15:
            raise DynamicRCAError(
                "Initial anomaly vector must contain at least one positive value."
            )

        v_restart = s0 / total_s0
        current_s = v_restart.copy()

        timeline_scores: List[Dict[str, float]] = [
            {self._nodes[i]: float(current_s[i]) for i in range(n)}
        ]
        init_res = self.compute_resilience(current_s)
        timeline_resilience: List[Dict[str, float]] = [
            {self._nodes[i]: float(init_res[i]) for i in range(n)}
        ]

        p_mat = self.build_transition_matrix(current_s, initial_anomaly=s0)
        converged = False
        iters_run = 0

        for _step_idx in range(max_steps):
            iters_run += 1
            p_mat = self.build_transition_matrix(current_s, initial_anomaly=s0)

            # eIRWR step: s^{(t+1)} = (1 - alpha) * (s^{(t)} @ P^{(t)}) + alpha * v^{(0)}
            propagated = np.asarray(current_s @ p_mat).reshape(-1)
            next_s = (1.0 - self.restart_alpha) * propagated + self.restart_alpha * v_restart
            next_s_sum = float(np.sum(next_s))
            if next_s_sum > 1e-15:
                next_s = next_s / next_s_sum

            res_t = self.compute_resilience(next_s)
            timeline_scores.append({self._nodes[i]: float(next_s[i]) for i in range(n)})
            timeline_resilience.append({self._nodes[i]: float(res_t[i]) for i in range(n)})

            l1_diff = float(np.sum(np.abs(next_s - current_s)))
            current_s = next_s
            if l1_diff < self.tol:
                converged = True
                break

        final_scores = {self._nodes[i]: float(current_s[i]) for i in range(n)}
        ranked = tuple(sorted(final_scores.items(), key=lambda kv: (-kv[1], kv[0])))
        identified_origin = ranked[0][0]

        static_pr = self.compute_static_pagerank_baseline()
        static_top1 = sorted(static_pr.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

        return DynamicRCAResult(
            nodes=self._nodes,
            forward_adjacency=self._w_fwd.toarray().tolist(),
            backward_adjacency=self._w_bwd.toarray().tolist(),
            final_transition_matrix=p_mat.toarray().tolist(),
            timeline_scores=tuple(timeline_scores),
            timeline_resilience=tuple(timeline_resilience),
            final_scores=final_scores,
            ranked_root_causes=ranked,
            identified_origin=identified_origin,
            static_pagerank_scores=static_pr,
            static_pagerank_top1=static_top1,
            iterations_run=iters_run,
            converged=converged,
        )


__all__ = [
    "MAX_EIRWR_ITERATIONS",
    "DynamicRCAError",
    "DynamicRCAResult",
    "DynamicRCASimulator",
]
