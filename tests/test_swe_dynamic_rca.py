"""Verification suite for Prompt E7: Software Engineering Application Evolution - Dynamic RCA.

Verifies:
1. Sparse adjacency matrix $W$ (`scipy.sparse.csr_matrix`) initialization and backward
   retry-storm transition edges ($\\rho W^{\\top}$).
2. Anomaly-Conditioned Resilience $R_i = R_{\\text{base}} \\cdot \\exp(-\\beta \\hat{s}_i)$.
3. Strict row-normalization of the combined dynamic transition matrix $P^{(t)}$ at every step.
4. Chain topology $A \\rightarrow B \\rightarrow C$ with a severe latency fault injected at $C$:
   confirming stress propagates upstream to $B$ and $A$ via backward edges while identifying
   $C$ as the root cause origin.
5. Root-cause localization accuracy across synthetic topologies surpassing the static
   PageRank baseline.
6. Operational safeguards capping power iterations at `MAX_EIRWR_ITERATIONS = 100` and
   docstring documentation of $\\beta$ and $\\rho$.
"""

from __future__ import annotations

import inspect

import networkx as nx
import numpy as np
import pytest

import ewm_engine.domains.swe.dynamic_rca as rca_mod
from ewm_engine.core.types import EngineError
from ewm_engine.domains.software_engineering import (
    MAX_EIRWR_ITERATIONS,
    DynamicRCAError,
    DynamicRCASimulator,
)


def test_chain_topology_upstream_fault_propagation_and_row_normalization() -> None:
    """Chain A -> B -> C with fault at C propagates stress upstream to B and A and identifies C."""
    dag = nx.DiGraph()
    dag.add_nodes_from(["A", "B", "C"])
    dag.add_edges_from([("A", "B"), ("B", "C")])

    sim = DynamicRCASimulator(
        dag,
        r_base=0.85,
        beta=2.5,
        rho=0.75,
        forward_weight=0.15,
        restart_alpha=0.30,
        max_iterations=50,
    )
    assert sim.nodes == ("A", "B", "C")

    # 1. Verify forward adjacency W and backward retry adjacency rho * W^T
    w_fwd = sim.adjacency_matrix.toarray()
    w_bwd = sim.backward_adjacency_matrix.toarray()
    expected_fwd = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    expected_bwd = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [0.0, 0.75, 0.0],
        ]
    )
    assert np.allclose(w_fwd, expected_fwd)
    assert np.allclose(w_bwd, expected_bwd)

    # 2. Verify Anomaly-Conditioned Resilience R_i = R_base * exp(-beta * s_hat_i)
    res_vec = sim.compute_resilience(np.array([0.0, 0.0, 1.0]))
    assert res_vec[0] == pytest.approx(0.85)
    assert res_vec[1] == pytest.approx(0.85)
    assert res_vec[2] == pytest.approx(0.85 * np.exp(-2.5))

    # 3. Verify strict row-normalization of the combined transition matrix P^(t)
    p_mat = sim.build_transition_matrix(np.array([0.0, 0.0, 1.0])).toarray()
    row_sums = np.sum(p_mat, axis=1)
    assert np.allclose(row_sums, np.ones(3), atol=1e-12)
    assert np.all(p_mat >= 0.0)

    # 4. Simulate severe latency fault injected at C
    result = sim.simulate({"A": 0.0, "B": 0.0, "C": 10.0})
    assert result.converged is True
    assert 2 <= result.iterations_run <= MAX_EIRWR_ITERATIONS

    # Step 1: fault at C propagates upstream to caller B
    step1_scores = result.timeline_scores[1]
    assert step1_scores["B"] > 0.0
    assert step1_scores["C"] > step1_scores["B"]

    # Step 2 & final: stress cascades further upstream from B to caller A
    assert result.final_scores["B"] > 0.0
    assert result.final_scores["A"] > 0.0
    assert result.final_scores["C"] > result.final_scores["B"] > result.final_scores["A"] > 0.0

    # Origin C is accurately identified as Top-1 root cause
    assert result.identified_origin == "C"
    assert result.ranked_root_causes[0][0] == "C"

    # Final transition matrix is strictly row-normalized
    final_p = np.asarray(result.final_transition_matrix)
    assert np.allclose(np.sum(final_p, axis=1), np.ones(3), atol=1e-12)


def test_dynamic_rca_surpasses_static_pagerank_baseline_across_topologies() -> None:
    """Verify eIRWR + DynaCausal surpasses static PageRank in root cause localization accuracy."""
    # Case 1: 3-service chain A -> B -> C where fault originates at C and cascades upstream
    chain_dag = nx.DiGraph([("A", "B"), ("B", "C")])
    chain_sim = DynamicRCASimulator(chain_dag, r_base=0.85, beta=2.5, rho=0.75)
    # Even when upstream services A and B show secondary symptoms from retry storms:
    chain_res = chain_sim.simulate({"A": 0.15, "B": 0.35, "C": 1.0})
    assert chain_res.identified_origin == "C"
    assert chain_res.static_pagerank_top1 == "A"  # Static reverse PageRank drifts to root caller A!

    # Case 2: 5-service microservice mesh:
    # Gateway -> Auth, Gateway -> Order, Order -> Payment, Order -> Database, Auth -> Database
    mesh_dag = nx.DiGraph(
        [
            ("Gateway", "Auth"),
            ("Gateway", "Order"),
            ("Order", "Payment"),
            ("Order", "Database"),
            ("Auth", "Database"),
        ]
    )
    mesh_sim = DynamicRCASimulator(mesh_dag, r_base=0.85, beta=3.0, rho=0.80)
    # Inject latency fault at Payment service with secondary retry symptoms on Order and Gateway
    mesh_res = mesh_sim.simulate(
        {
            "Gateway": 0.20,
            "Auth": 0.02,
            "Order": 0.45,
            "Payment": 1.00,
            "Database": 0.05,
        }
    )
    assert mesh_res.identified_origin == "Payment"
    assert mesh_res.static_pagerank_top1 != "Payment"


def test_operational_safeguards_and_hyperparameter_documentation() -> None:
    """Verify max_iterations <= 100 safeguard, error hierarchy, and beta/rho docstrings."""
    assert issubclass(DynamicRCAError, EngineError)

    mod_doc = inspect.getdoc(rca_mod) or ""
    cls_doc = inspect.getdoc(DynamicRCASimulator) or ""
    assert "beta" in mod_doc and "rho" in mod_doc
    assert "beta" in cls_doc and "rho" in cls_doc

    dag = nx.DiGraph([("A", "B"), ("B", "C")])

    # Exceeding MAX_EIRWR_ITERATIONS (100) raises DynamicRCAError
    with pytest.raises(DynamicRCAError, match="max_iterations"):
        DynamicRCASimulator(dag, max_iterations=101)

    with pytest.raises(DynamicRCAError, match="max_iterations"):
        DynamicRCASimulator(dag, max_iterations=0)

    sim = DynamicRCASimulator({"A": ["B"], "B": ["C"]})
    with pytest.raises(DynamicRCAError, match="steps"):
        sim.simulate({"A": 0.0, "B": 0.0, "C": 1.0}, steps=150)

    with pytest.raises(DynamicRCAError, match="positive value"):
        sim.simulate({"A": 0.0, "B": 0.0, "C": 0.0})
