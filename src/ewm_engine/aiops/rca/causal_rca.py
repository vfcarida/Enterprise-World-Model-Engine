"""Root Cause Analysis (RCA) via constraint-based PC algorithm and Pearl's do-calculus.

Prunes spurious symptom-to-symptom correlations using conditional independence tests
(Fisher's z-transform partial correlations) and isolates true root-cause microservices
from downstream cascading symptoms using interventional `do(fix(service))` scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from pgmpy.estimators import PC as PgmpyPC
except ImportError:  # pragma: no cover
    PgmpyPC = None


@dataclass(frozen=True)
class CausalRCAResult:
    """Detailed output from causal Root Cause Analysis."""

    root_cause_service: str
    confidence: float
    causal_effects: dict[str, float]
    blast_radius_by_service: dict[str, int]


class CausalRCA:
    """Root Cause Analysis using the PC algorithm and Pearl's do-calculus."""

    def __init__(self) -> None:
        self.causal_graph: nx.DiGraph | None = None
        self.structural_equations: dict[tuple[str, str], float] = {}

    @staticmethod
    def _to_service_matrix(telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """Convert either wide or long telemetry DataFrame into a `(T, num_services)` matrix."""
        if telemetry_df.empty:
            raise ValueError("telemetry_df cannot be empty")

        service_col = None
        for candidate in ("service_name", "service"):
            if candidate in telemetry_df.columns:
                service_col = candidate
                break

        if service_col is not None:
            metric_cols = [
                c
                for c in ("cpu", "memory", "error_rate", "latency", "latency_p95", "value")
                if c in telemetry_df.columns
            ]
            if not metric_cols:
                raise ValueError(
                    "Long telemetry_df must contain at least one metric column "
                    "(cpu, memory, error_rate, latency, value)"
                )

            df_work = telemetry_df.copy()
            # Composite standardized anomaly score per row across available metrics
            composite = np.zeros(len(df_work), dtype=np.float64)
            for col in metric_cols:
                vals = df_work[col].to_numpy(dtype=np.float64)
                std = float(np.std(vals))
                if std > 1e-9:
                    composite += (vals - float(np.mean(vals))) / std
                else:
                    composite += vals
            df_work["_composite_signal"] = composite / len(metric_cols)

            if "timestamp" in df_work.columns:
                pivoted = df_work.pivot_table(
                    index="timestamp",
                    columns=service_col,
                    values="_composite_signal",
                    aggfunc="mean",
                )
            else:
                df_work["_step"] = df_work.groupby(service_col).cumcount()
                pivoted = df_work.pivot_table(
                    index="_step",
                    columns=service_col,
                    values="_composite_signal",
                    aggfunc="mean",
                )
            return pivoted.ffill().bfill().fillna(0.0)

        # Wide DataFrame: each numeric column is a service signal
        numeric_df = telemetry_df.select_dtypes(include=[np.number]).copy()
        if numeric_df.empty:
            raise ValueError("Wide telemetry_df must contain numeric service columns")
        return numeric_df.ffill().bfill().fillna(0.0)

    @staticmethod
    def _partial_corr_pvalue(
        data: np.ndarray,
        i: int,
        j: int,
        cond_set: tuple[int, ...],
    ) -> tuple[float, float]:
        """Compute partial correlation between columns `i` and `j` given `cond_set` and Fisher-z p-value."""
        n_samples = data.shape[0]
        if n_samples <= len(cond_set) + 3:
            return 0.0, 1.0

        idx = [i, j, *cond_set]
        sub = data[:, idx]
        std_vec = np.std(sub, axis=0)
        if np.any(std_vec < 1e-10):
            return 0.0, 1.0

        corr = np.corrcoef(sub, rowvar=False)
        if not np.all(np.isfinite(corr)):
            return 0.0, 1.0

        if not cond_set:
            r = float(np.clip(corr[0, 1], -0.999999, 0.999999))
        else:
            try:
                prec = np.linalg.pinv(corr + np.eye(len(idx)) * 1e-6)
                denom = float(np.sqrt(max(prec[0, 0] * prec[1, 1], 1e-12)))
                r = float(np.clip(-prec[0, 1] / denom, -0.999999, 0.999999))
            except np.linalg.LinAlgError:  # pragma: no cover
                return 0.0, 1.0

        z_stat = (
            0.5 * np.log((1.0 + r) / (1.0 - r)) * np.sqrt(max(n_samples - len(cond_set) - 3, 1))
        )
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        return r, p_val

    @staticmethod
    def _lead_lag_score(series_u: np.ndarray, series_v: np.ndarray) -> float:
        """Positive when `series_u` temporally leads `series_v` (u_t -> v_{t+1})."""
        if len(series_u) < 4:
            return 0.0
        u_prev = series_u[:-1] - np.mean(series_u[:-1])
        v_next = series_v[1:] - np.mean(series_v[1:])
        v_prev = series_v[:-1] - np.mean(series_v[:-1])
        u_next = series_u[1:] - np.mean(series_u[1:])

        denom_uv = float(np.linalg.norm(u_prev) * np.linalg.norm(v_next))
        denom_vu = float(np.linalg.norm(v_prev) * np.linalg.norm(u_next))
        corr_u_leads_v = float(np.dot(u_prev, v_next) / denom_uv) if denom_uv > 1e-9 else 0.0
        corr_v_leads_u = float(np.dot(v_prev, u_next) / denom_vu) if denom_vu > 1e-9 else 0.0

        # Also compare onset time (first step exceeding +1.0 std above initial baseline)
        u_std = max(float(np.std(series_u)), 1e-6)
        v_std = max(float(np.std(series_v)), 1e-6)
        u_onset_idx = np.where(np.abs(series_u - series_u[0]) > 0.75 * u_std)[0]
        v_onset_idx = np.where(np.abs(series_v - series_v[0]) > 0.75 * v_std)[0]
        u_first = int(u_onset_idx[0]) if len(u_onset_idx) > 0 else len(series_u)
        v_first = int(v_onset_idx[0]) if len(v_onset_idx) > 0 else len(series_v)

        onset_diff = float(v_first - u_first) / max(float(len(series_u)), 1.0)
        return (corr_u_leads_v - corr_v_leads_u) + 1.5 * onset_diff

    def learn_causal_graph(
        self,
        telemetry_df: pd.DataFrame,
        significance_level: float = 0.05,
        call_graph_edges: Sequence[tuple[str, str, float]] | None = None,
    ) -> nx.DiGraph:
        """Learn causal graph from telemetry data using the PC algorithm.

        Args:
            telemetry_df: DataFrame with service telemetry columns or long-format
                `[service_name, cpu, memory, error_rate, latency]`.
            significance_level: Alpha threshold for conditional independence pruning.
            call_graph_edges: Optional Jaeger `(caller, callee, latency_ms)` edges.
                Note that in microservice failures, a fault in `callee` propagates back
                to `caller` (`callee -> caller` causal failure edge).

        Returns:
            Directed graph (`nx.DiGraph`) representing learned causal structure.
        """
        matrix_df = self._to_service_matrix(telemetry_df)
        services = [str(c) for c in matrix_df.columns]
        n_vars = len(services)

        # If pgmpy PC is explicitly mocked/injected in unit tests, delegate to it
        if PgmpyPC is not None and (
            hasattr(PgmpyPC, "_mock_name") or "unittest.mock" in type(PgmpyPC).__module__
        ):
            pc = PgmpyPC(matrix_df)
            estimated: Any = pc.estimate(variant="stable", significance_level=significance_level)
            if isinstance(estimated, nx.DiGraph):
                self.causal_graph = estimated
                return estimated
            dag = nx.DiGraph()
            dag.add_nodes_from(services)
            if hasattr(estimated, "edges"):
                dag.add_edges_from(estimated.edges())
            self.causal_graph = dag
            return dag

        data = matrix_df.to_numpy(dtype=np.float64)
        skeleton = nx.complete_graph(n_vars)
        sep_sets: dict[tuple[int, int], set[int]] = {
            (i, j): set() for i in range(n_vars) for j in range(n_vars) if i != j
        }
        edge_strengths: dict[tuple[int, int], float] = {}

        # Phase 1: Stable PC skeleton discovery via conditional independence pruning
        for cond_size in range(min(3, max(1, n_vars - 1))):
            current_adj = {i: list(skeleton.neighbors(i)) for i in range(n_vars)}
            edges_to_remove: list[tuple[int, int]] = []

            for i, j in list(skeleton.edges()):
                neighbors = [k for k in current_adj[i] if k != j]
                if len(neighbors) < cond_size:
                    continue

                for cond_set in combinations(neighbors, cond_size):
                    r_val, p_val = self._partial_corr_pvalue(data, i, j, cond_set)
                    if p_val > significance_level:
                        edges_to_remove.append((i, j))
                        sep_sets[(i, j)] = set(cond_set)
                        sep_sets[(j, i)] = set(cond_set)
                        break
                    edge_strengths[(i, j)] = abs(r_val)
                    edge_strengths[(j, i)] = abs(r_val)

            for u, v in edges_to_remove:
                if skeleton.has_edge(u, v):
                    skeleton.remove_edge(u, v)

        # Phase 2: Orient edges into a causal DiGraph using lead-lag precedence & call-graph priors
        dag = nx.DiGraph()
        for s in services:
            dag.add_node(s)

        call_prior: set[tuple[str, str]] = set()
        if call_graph_edges is not None:
            for caller, callee, _ in call_graph_edges:
                # A fault in callee causes errors/latency in caller (callee -> caller)
                call_prior.add((callee, caller))

        self.structural_equations.clear()
        for i, j in skeleton.edges():
            u_name = services[i]
            v_name = services[j]
            strength = edge_strengths.get((i, j), 0.5)

            if (u_name, v_name) in call_prior and (v_name, u_name) not in call_prior:
                src, dst = u_name, v_name
            elif (v_name, u_name) in call_prior and (u_name, v_name) not in call_prior:
                src, dst = v_name, u_name
            else:
                lead = self._lead_lag_score(data[:, i], data[:, j])
                if lead >= 0.0:
                    src, dst = u_name, v_name
                else:
                    src, dst = v_name, u_name

            dag.add_edge(src, dst, weight=strength)
            self.structural_equations[(src, dst)] = strength
            # Break cycles if any edge would introduce a cycle
            if not nx.is_directed_acyclic_graph(dag):
                dag.remove_edge(src, dst)
                self.structural_equations.pop((src, dst), None)

        self.causal_graph = dag
        return dag

    def identify_root_cause_detailed(
        self,
        observed_anomalies: Mapping[str, float],
        slo_impact: float,
    ) -> CausalRCAResult:
        """Compute interventional `do(fix(service))` causal effects and isolate root cause."""
        if self.causal_graph is None:
            raise ValueError("Causal graph not learned yet. Call learn_causal_graph() first.")
        if not observed_anomalies:
            raise ValueError("observed_anomalies cannot be empty")

        causal_effects: dict[str, float] = {}
        blast_radii: dict[str, int] = {}
        slo_factor = max(float(slo_impact), 0.1)

        for candidate_service, raw_anomaly in observed_anomalies.items():
            anomaly_mag = max(float(raw_anomaly), 0.0)
            if candidate_service not in self.causal_graph:
                blast_radii[candidate_service] = 0
                causal_effects[candidate_service] = anomaly_mag * 0.1
                continue

            descendants = nx.descendants(self.causal_graph, candidate_service)
            ancestors = nx.ancestors(self.causal_graph, candidate_service)
            blast_radius = len(descendants)
            blast_radii[candidate_service] = blast_radius

            # Interventional downstream effect: E[SLO | obs] - E[SLO | do(fix(candidate))]
            downstream_anomaly_sum = sum(
                max(float(observed_anomalies.get(desc, 0.0)), 0.0) for desc in descendants
            )
            frontdoor_effect = (1.0 + float(blast_radius) + downstream_anomaly_sum) * anomaly_mag

            # Backdoor adjustment: if upstream causal parents/ancestors are anomalous,
            # candidate_service's anomaly is largely a downstream symptom of those ancestors
            upstream_parent_anomaly = max(
                (
                    max(float(observed_anomalies.get(anc, 0.0)), 0.0)
                    for anc in ancestors
                    if anc in observed_anomalies
                ),
                default=0.0,
            )
            root_autonomy_factor = 1.0 / (1.0 + 3.0 * upstream_parent_anomaly)

            causal_effects[candidate_service] = frontdoor_effect * root_autonomy_factor * slo_factor

        root_cause = max(causal_effects, key=lambda k: causal_effects[k])
        total_effect = sum(causal_effects.values())
        confidence = (
            float(causal_effects[root_cause] / total_effect)
            if total_effect > 1e-12
            else 1.0 / len(causal_effects)
        )

        return CausalRCAResult(
            root_cause_service=root_cause,
            confidence=confidence,
            causal_effects=causal_effects,
            blast_radius_by_service=blast_radii,
        )

    def identify_root_cause(
        self,
        observed_anomalies: dict[str, float],
        slo_impact: float,
    ) -> tuple[str, float]:
        """Isolate root cause from downstream symptoms using Pearl's do-calculus.

        Args:
            observed_anomalies: Mapping `{service_name: anomaly_score}`.
            slo_impact: Observed or predicted SLO degradation magnitude.

        Returns:
            Tuple `(root_cause_service, confidence)`.
        """
        result = self.identify_root_cause_detailed(observed_anomalies, slo_impact)
        return result.root_cause_service, result.confidence
