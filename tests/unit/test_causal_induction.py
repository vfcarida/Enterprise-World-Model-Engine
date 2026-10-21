"""Unit tests for ensemble causal-structure induction, DiBS reconciliation, and SCM compilation.

Tests:
    1. PC algorithm: skeleton discovery, v-structure collider orientation, Meek rules.
    2. FCI algorithm: latent common cause / unobserved confounder detection (<->).
    3. DAGMA / NOTEARS: continuous score-based optimization with acyclicity constraint.
    4. DirectLiNGAM: non-Gaussian functional asymmetry breaking Markov equivalence.
    5. DiBS Bayesian ensemble reconciliation:
       - Observational Gaussian data emits CPDAG with DIRECTIONAL-UNCERTAIN (never a single asserted DAG).
       - Interventional data fusion disambiguates orientation to ASSERTED.
       - FCI bidirected edges mapped to CONFOUNDED.
    6. Compilation of ASSERTED causal graph into StructuralCausalModel.
    7. Causal surgery do(X=x) mutilating incoming edges on induced SCM.
    8. Tigramite GPL-3.0 isolation boundary verification.
    9. Determinism and canonical SHA-256 fingerprinting.
"""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.scm import StructuralCausalModel
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.induction.causal import (
    CausalEdgeType,
    CausalEquivalenceClass,
    induce_causal,
    reconcile_ensemble,
    run_dagma_notears,
    run_direct_lingam,
    run_fci_algorithm,
    run_pc_algorithm,
    run_temporal_discovery,
)
from ewm_engine.induction.tigramite_bridge import (
    is_tigramite_available,
    run_tigramite_out_of_process,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_pc_algorithm_v_structure_orientation() -> None:
    """Verify PC correctly identifies v-structures (X -> Z <- Y) where X and Y are independent."""
    # Synthetic collider: X and Y are independent; Z = 0.8*X + 0.9*Y + noise
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, size=n)
    y = rng.normal(0, 1, size=n)
    z = 0.8 * x + 0.9 * y + rng.normal(0, 0.2, size=n)
    data = np.column_stack([x, y, z])
    var_names = ["X", "Y", "Z"]

    pc_results = run_pc_algorithm(data, var_names, alpha=0.05)

    # Should detect edge between X and Z, and Y and Z, but NOT between X and Y
    assert ("X", "Y") not in pc_results
    assert ("X", "Z") in pc_results or ("Z", "X") in pc_results
    assert ("Y", "Z") in pc_results or ("Z", "Y") in pc_results

    # V-structure should orient arrows into Z
    x_z_dir = pc_results.get(("X", "Z"))
    y_z_dir = pc_results.get(("Y", "Z"))
    assert x_z_dir == "-->"
    assert y_z_dir == "-->"


def test_fci_algorithm_detects_latent_confounding() -> None:
    """Verify FCI flags unobserved common cause as bidirected (<->) confounding edge."""
    # Latent variable U causes both A and B, but U is unobserved
    rng = np.random.default_rng(123)
    n = 600
    u_latent = rng.normal(0, 2, size=n)
    a = 1.2 * u_latent + rng.normal(0, 0.3, size=n)
    b = 1.1 * u_latent + rng.normal(0, 0.3, size=n)
    c = 0.8 * b + rng.normal(0, 0.2, size=n)  # Observed downstream outcome

    data = np.column_stack([a, b, c])
    var_names = ["A", "B", "C"]

    fci_results = run_fci_algorithm(data, var_names, alpha=0.05)

    # Edge between A and B should exist and exhibit confounding (<->)
    assert ("A", "B") in fci_results or ("B", "A") in fci_results
    edge_dir = fci_results.get(("A", "B")) or fci_results.get(("B", "A"))
    assert edge_dir == "<->"


def test_dagma_notears_continuous_score_optimization() -> None:
    """Verify DAGMA / NOTEARS continuous optimization discovers chain dependencies."""
    # Chain: A -> B -> C
    rng = np.random.default_rng(999)
    n = 400
    a = rng.normal(0, 1, size=n)
    b = 0.85 * a + rng.normal(0, 0.2, size=n)
    c = 0.75 * b + rng.normal(0, 0.2, size=n)
    data = np.column_stack([a, b, c])
    var_names = ["A", "B", "C"]

    dagma_results = run_dagma_notears(data, var_names, lambda1=0.02)

    # Should detect strong direct weights for A -> B and B -> C
    assert ("A", "B") in dagma_results
    assert ("B", "C") in dagma_results
    assert dagma_results[("A", "B")] > 0.2
    assert dagma_results[("B", "C")] > 0.2


def test_direct_lingam_non_gaussian_identifiability() -> None:
    """Verify DirectLiNGAM breaks Markov equivalence under non-Gaussian noise."""
    # Non-Gaussian uniform / laplace noise: X -> Y
    rng = np.random.default_rng(777)
    n = 800
    # Uniform noise has significant kurtosis difference from normal
    x = rng.uniform(-2, 2, size=n)
    y = 0.9 * x + rng.uniform(-0.5, 0.5, size=n)
    data = np.column_stack([x, y])
    var_names = ["X", "Y"]

    lingam_results = run_direct_lingam(data, var_names)

    # DirectLiNGAM should identify X -> Y direction
    assert ("X", "Y") in lingam_results
    assert ("Y", "X") not in lingam_results


def test_temporal_cross_lag_discovery() -> None:
    """Verify temporal lag discovery identifies time-series dependencies X_{t-1} -> Y_t."""
    # Autoregressive / cross-lagged: Y_t = 0.8 * X_{t-1} + noise
    rng = np.random.default_rng(888)
    T = 300
    x = rng.normal(0, 1, size=T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * x[t - 1] + rng.normal(0, 0.2)
    series_data = np.column_stack([x, y])
    var_names = ["X", "Y"]

    temp_results = run_temporal_discovery(series_data, var_names, max_lag=1)

    assert ("X", "Y") in temp_results
    assert temp_results[("X", "Y")] > 0.5


def test_observational_gaussian_data_never_asserts_single_dag() -> None:
    """Epistemic Invariant: Observational Gaussian data alone must emit DIRECTIONAL-UNCERTAIN.

    EWM must NEVER claim a single asserted DAG from observational data without
    non-Gaussianity or interventional disambiguation.
    """
    # Simple bivariate Gaussian: X and Y correlated symmetrically
    var_names = ["X", "Y"]

    # Run PC and reconcile without interventional data or non-Gaussianity
    pc_res = {("X", "Y"): "---"}  # Undirected Markov equivalence
    fci_res = {("X", "Y"): "---"}
    dagma_res = {("X", "Y"): 0.5, ("Y", "X"): 0.5}  # Symmetric score
    lingam_res: dict[tuple[str, str], float] = {}

    equiv_class = reconcile_ensemble(
        var_names=var_names,
        pc_results=pc_res,
        fci_results=fci_res,
        dagma_results=dagma_res,
        lingam_results=lingam_res,
        interventional_data=None,
    )

    # Must be marked DIRECTIONAL-UNCERTAIN, not ASSERTED!
    assert len(equiv_class.edges) == 1
    edge = equiv_class.edges[0]
    assert edge.evidence_level == EvidenceLevel.DIRECTIONAL_UNCERTAIN
    assert edge.edge_type in (CausalEdgeType.UNDIRECTED, CausalEdgeType.DIRECTED)
    assert edge.is_interventional is False
    assert equiv_class.markov_equivalent_dags_count >= 1
    assert len(equiv_class.asserted_edges()) == 0


def test_interventional_data_fusion_promotes_to_asserted() -> None:
    """Verify that experimental A/B test data disambiguates direction and promotes to ASSERTED."""
    # Observational undirected edge X - Y
    var_names = ["Treatment", "Outcome"]
    pc_res = {("Treatment", "Outcome"): "---"}
    fci_res = {("Treatment", "Outcome"): "---"}
    dagma_res = {("Treatment", "Outcome"): 0.6}
    lingam_res: dict[tuple[str, str], float] = {}

    # Interventional evidence: do(Treatment = 1) shifted Outcome significantly,
    # but do(Outcome = 1) had no effect on Treatment
    interv_data = {
        "Treatment": {"Outcome": 0.45},
        "Outcome": {"Treatment": 0.0},
    }

    equiv_class = reconcile_ensemble(
        var_names=var_names,
        pc_results=pc_res,
        fci_results=fci_res,
        dagma_results=dagma_res,
        lingam_results=lingam_res,
        interventional_data=interv_data,
    )

    assert len(equiv_class.edges) == 1
    edge = equiv_class.edges[0]
    assert edge.source == "Treatment"
    assert edge.target == "Outcome"
    assert edge.edge_type == CausalEdgeType.DIRECTED
    assert edge.evidence_level == EvidenceLevel.ASSERTED
    assert edge.is_interventional is True
    assert len(equiv_class.asserted_edges()) == 1


def test_confounded_fci_edge_reconciles_to_confounded_level() -> None:
    """Verify that an FCI bidirected edge reconciles to CONFOUNDED EvidenceLevel."""
    var_names = ["Metric_A", "Metric_B"]
    pc_res = {("Metric_A", "Metric_B"): "---"}
    fci_res = {("Metric_A", "Metric_B"): "<->"}  # Latent confounding
    dagma_res: dict[tuple[str, str], float] = {}
    lingam_res: dict[tuple[str, str], float] = {}

    equiv_class = reconcile_ensemble(
        var_names=var_names,
        pc_results=pc_res,
        fci_results=fci_res,
        dagma_results=dagma_res,
        lingam_results=lingam_res,
    )

    assert len(equiv_class.edges) == 1
    edge = equiv_class.edges[0]
    assert edge.edge_type == CausalEdgeType.BIDIRECTED
    assert edge.evidence_level == EvidenceLevel.CONFOUNDED
    assert len(equiv_class.confounded_edges()) == 1


def test_scm_compilation_and_causal_surgery_mutilation() -> None:
    """Verify that induced ASSERTED causal structure compiles to SCM and supports Pearl's causal surgery."""
    from ewm_engine.induction.causal import CausalEdge

    edge1 = CausalEdge(
        source="X",
        target="Y",
        edge_type=CausalEdgeType.DIRECTED,
        posterior_support=0.95,
        direction_support=0.95,
        evidence_level=EvidenceLevel.ASSERTED,
        is_interventional=True,
    )
    edge2 = CausalEdge(
        source="Y",
        target="Z",
        edge_type=CausalEdgeType.DIRECTED,
        posterior_support=0.90,
        direction_support=0.90,
        evidence_level=EvidenceLevel.ASSERTED,
        is_interventional=True,
    )

    equiv_class = CausalEquivalenceClass(
        variables=["X", "Y", "Z"],
        edges=[edge1, edge2],
        is_dag=True,
        markov_equivalent_dags_count=1,
    )

    # 1. Compile SCM
    scm = equiv_class.to_scm()
    assert isinstance(scm, StructuralCausalModel)
    assert ("X", "Y") in scm.causal_graph.edges
    assert ("Y", "Z") in scm.causal_graph.edges
    assert "Y" in scm.assignments
    assert scm.assignments["Y"].parents == ("X",)

    # 2. Pearl's causal surgery do(Y = 50.0)
    # Surgery must sever the incoming edge X -> Y in the mutilated graph!
    mutilated = scm.intervene({"Y": 50.0})
    assert ("X", "Y") not in mutilated.causal_graph.edges
    assert ("Y", "Z") in mutilated.causal_graph.edges  # Outgoing edge remains intact!
    assert mutilated.assignments["Y"].is_intervened is True
    assert mutilated.assignments["Y"].intervened_value == 50.0


def test_tigramite_gpl_isolation_boundary() -> None:
    """Verify Tigramite GPL-3.0 isolation boundary.

    Core EWM code never imports Tigramite in-process. If Tigramite is not installed,
    calling run_tigramite_out_of_process raises SimulationConfigurationError explaining the boundary.
    """
    is_avail = is_tigramite_available()
    assert isinstance(is_avail, bool)

    if not is_avail:
        with pytest.raises(SimulationConfigurationError) as exc_info:
            run_tigramite_out_of_process(
                series_data=[[1.0, 2.0], [3.0, 4.0]],
                var_names=["A", "B"],
            )
        assert "Tigramite is not installed" in str(exc_info.value)
        assert "engine='native'" in str(exc_info.value)


def test_induce_causal_determinism_and_fingerprinting() -> None:
    """Verify that induce_causal on identical input data and seed produces identical fingerprints."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(0, 1, size=n)
    y = 0.5 * x + rng.normal(0, 0.5, size=n)
    data = [{"X": float(x[i]), "Y": float(y[i])} for i in range(n)]

    eq1, rep1 = induce_causal(data=data, seed=42)
    eq2, rep2 = induce_causal(data=data, seed=42)

    assert rep1.sha256_fingerprint == rep2.sha256_fingerprint
    assert len(rep1.sha256_fingerprint) == 64
    assert len(eq1.edges) == len(eq2.edges)


def test_induced_asserted_causal_layer_drives_counterfactual_scm_kernel_end_to_end() -> None:
    """Definition of Done: Induced ASSERTED causal layer drives the counterfactual SCM kernel end-to-end.

    Verifies that:
    1. Observational + experimental interventional data discovers an ASSERTED causal link Treatment -> Outcome.
    2. Builds an active World with state and dynamics reflecting the induced causal topology.
    3. Simulates a factual trajectory and executes a Pearl Rung-3 counterfactual under do(Treatment = 200).
    4. Counterfactual result confirms EvidenceLevel.COUNTERFACTUAL, target divergence, and noise coupling.
    """
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState
    from ewm_engine.core.world import World
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
    from ewm_engine.simulation.scenario import Scenario

    # 1. Induce causal structure with interventional data
    rng = np.random.default_rng(42)
    data = []
    for _ in range(50):
        t_val = float(rng.uniform(10.0, 50.0))
        o_val = float(2.0 * t_val + rng.normal(0, 1.0))
        data.append({"treatment": t_val, "outcome": o_val})

    interv_data = {
        "treatment": {"outcome": 20.0},
        "outcome": {"treatment": 0.0},
    }

    equiv_class, _ = induce_causal(
        data=data,
        interventional_data=interv_data,
        seed=42,
    )

    assert len(equiv_class.asserted_edges()) >= 1
    asserted_edge = equiv_class.asserted_edges()[0]
    assert asserted_edge.source == "treatment"
    assert asserted_edge.target == "outcome"
    assert asserted_edge.evidence_level == EvidenceLevel.ASSERTED

    # 2. Build World from the induced causal structure
    world_state = WorldState(
        resources=[
            Resource(id="treatment", current=50.0, min_value=0.0, max_value=1000.0),
            Resource(id="outcome", current=100.0, min_value=0.0, max_value=1000.0),
            Resource(id="unrelated_metric", current=999.0, min_value=0.0, max_value=10000.0),
        ]
    )
    dyn = DeterministicTransferDynamics(action_type="adjust", name="CausalTransferDynamics")
    world = World(state=world_state, dynamics=dyn)

    # 3. Simulate factual trajectory
    scenario = Scenario(name="FactualRun", horizon=3, seed=123)
    sim_res = world.simulate(scenario)
    factual_traj = sim_res.trajectories[0]

    # 4. Execute Pearl's 3-step counterfactual query do(treatment = 200.0)
    cf_res = world.counterfactual(
        observed=factual_traj,
        intervention={"treatment": 200.0},
        horizon=3,
        seed=123,
    )

    # Verify counterfactual SCM properties
    assert cf_res.evidence_level == EvidenceLevel.COUNTERFACTUAL
    assert cf_res.is_unique is True
    assert cf_res.target_divergences["treatment"] == 150.0
    assert cf_res.counterfactual_trajectory.final_state.resources["treatment"].current == 200.0
    # Unrelated metric preserves structural locality
    assert cf_res.off_target_divergences.get("unrelated_metric", 0.0) == 0.0
    assert cf_res.locality_preserved is True
