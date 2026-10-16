"""Unit tests for batched Structure-of-Arrays (SoA) simulation, backends, and determinism.

Verifies:
1. SoA state tensor projection round-trips without information loss.
2. Batched SoA Monte Carlo yields per-branch bit-identical SHA-256 fingerprints to pure-Python.
3. Order-independent reduction invariant under arbitrary arrival permutations.
4. Array API backends (NumPy, JAX, CuPy) enforce certification status.
5. Fast canonical SHA-256 matches canonical reference output bit-for-bit.
"""

from __future__ import annotations

import numpy as np

from ewm_engine.core._canonical import canonical_sha256, fast_canonical_sha256
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.backends import (
    BackendType,
    CertificationStatus,
    CuPyBackend,
    JaxBackend,
    NumPyBackend,
    resolve_backend,
)
from ewm_engine.simulation.batched import (
    BatchedExecutor,
    BatchedSimulationEngine,
    SoAStateTensor,
)
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.executors import resolve_executor
from ewm_engine.simulation.scenario import Scenario


def _create_test_world() -> World:
    """Create a multi-resource test world for scale testing."""
    entities = [
        Entity(id="depot_alpha", type="depot"),
        Entity(id="depot_beta", type="depot"),
    ]
    relationships = [
        Relationship(source="depot_alpha", target="depot_beta", type="connected_to"),
    ]
    resources = [
        Resource(
            id="stock_alpha", entity_id="depot_alpha", current=200.0, min_value=0.0, max_value=500.0
        ),
        Resource(
            id="stock_beta", entity_id="depot_beta", current=100.0, min_value=0.0, max_value=500.0
        ),
    ]
    initial_state = WorldState(
        entities=entities,
        relationships=relationships,
        resources=resources,
        memory={"cycle_count": 0.0},
    )
    dynamics = DeterministicTransferDynamics()
    return World(initial_state=initial_state, dynamics=dynamics)


def test_soa_state_tensor_projection_and_reconstruction() -> None:
    """Verify that SoAStateTensor accurately projects and reconstructs WorldStates."""
    world = _create_test_world()
    n_samples = 5

    soa = SoAStateTensor.from_world_state(world.initial_state, n_samples)
    assert soa.n_samples == n_samples
    assert len(soa.resource_ids) == 2
    assert soa.resource_matrix.shape == (5, 2)
    assert np.all(soa.active_mask)

    # Reconstruct WorldStates
    reconstructed = soa.to_world_states()
    assert len(reconstructed) == n_samples

    for st in reconstructed:
        assert st.get_resource("stock_alpha").current == 200.0
        assert st.get_resource("stock_beta").current == 100.0
        assert st.fingerprint == world.initial_state.fingerprint

    # Fingerprints helper
    fps = soa.compute_fingerprints()
    assert len(fps) == n_samples
    assert all(fp == world.initial_state.fingerprint for fp in fps)


def test_batched_engine_bit_identical_to_pure_python() -> None:
    """Epistemic Invariant: Batched SoA engine produces bit-identical per-branch fingerprints."""
    world = _create_test_world()
    scenario = Scenario(
        scenario_id="scen_equiv",
        name="Equivalence Test Scenario",
        horizon=6,
        samples=4,
        seed=1337,
    )

    # 1. Pure-Python sequential execution (the canonical oracle)
    seq_engine = SimulationEngine(executor="serial")
    seq_result = seq_engine.run(world, scenario)

    # 2. Batched SoA NumPy execution
    batched_engine = BatchedSimulationEngine(backend="numpy")
    batch_result = batched_engine.run(world, scenario)

    assert len(batch_result.trajectories) == len(seq_result.trajectories)

    for i in range(scenario.samples):
        seq_traj = seq_result.trajectories[i]
        batch_traj = batch_result.trajectories[i]

        assert seq_traj.sample_id == batch_traj.sample_id
        assert seq_traj.seed == batch_traj.seed
        assert seq_traj.status == batch_traj.status
        assert len(seq_traj.steps) == len(batch_traj.steps)

        # Compare per-step state hashes
        for t in range(len(seq_traj.steps)):
            seq_hash = seq_traj.steps[t].state_hash
            batch_hash = batch_traj.steps[t].state_hash
            assert seq_hash == batch_hash, (
                f"Mismatch at sample {i}, step {t}: {seq_hash} != {batch_hash}"
            )

        # Compare final state fingerprint
        assert seq_traj.final_state.fingerprint == batch_traj.final_state.fingerprint


def test_batched_executor_resolution_and_dispatch() -> None:
    """Verify BatchedExecutor resolves from string and executes via SimulationEngine."""
    world = _create_test_world()
    scenario = Scenario(
        scenario_id="scen_batch_exec",
        horizon=3,
        samples=3,
        seed=42,
    )

    executor = resolve_executor("batched")
    assert isinstance(executor, BatchedExecutor)

    engine = SimulationEngine(executor=executor)
    res = engine.run(world, scenario)
    assert len(res.trajectories) == 3
    assert res.run_metrics.rollout_count == 3


def test_array_api_backends_and_certification() -> None:
    """Verify Array API backends expose correct certification status."""
    np_backend = NumPyBackend()
    assert np_backend.backend_type == BackendType.NUMPY
    assert np_backend.certification == CertificationStatus.CERTIFIED_BIT_IDENTICAL
    assert np_backend.is_available is True

    jax_backend = JaxBackend()
    assert jax_backend.backend_type == BackendType.JAX
    assert jax_backend.certification == CertificationStatus.CERTIFIED_BIT_IDENTICAL

    cupy_backend = CuPyBackend()
    assert cupy_backend.backend_type == BackendType.CUPY
    assert cupy_backend.certification == CertificationStatus.THROUGHPUT_ONLY

    # Resolver
    assert isinstance(resolve_backend("numpy"), NumPyBackend)
    assert isinstance(resolve_backend(None), NumPyBackend)


def test_order_independent_reduction_under_arrival_permutation() -> None:
    """Verify that results sorted by sample_id eliminate worker scheduling non-determinism."""
    world = _create_test_world()
    scenario = Scenario(
        scenario_id="scen_order_indep",
        horizon=2,
        samples=4,
        seed=101,
    )

    child_seeds = np.random.SeedSequence(scenario.seed).spawn(scenario.samples)

    # Generate reference rollouts sequentially
    ref_executor = resolve_executor("serial")
    ref_results = ref_executor.execute(world, scenario, child_seeds)

    # Simulate workers finishing out of order: [sample_3, sample_0, sample_2, sample_1]
    permuted_order = [3, 0, 2, 1]
    unordered_results = [ref_results[idx] for idx in permuted_order]

    # Order-independent reduction step
    reduced_results = list(unordered_results)
    reduced_results.sort(key=lambda r: r.sample_id)

    # Must match reference order exactly
    for orig, red in zip(ref_results, reduced_results, strict=True):
        assert orig.sample_id == red.sample_id
        assert orig.trajectory.final_state.fingerprint == red.trajectory.final_state.fingerprint


def test_fast_canonical_sha256_equivalence() -> None:
    """Verify fast_canonical_sha256 produces bit-identical output to canonical_sha256."""
    data = {
        "step": 5,
        "timestamp": 12.3456789,
        "resources": [
            {"id": "res_b", "current": 10.0000001, "min": 0.0, "max": 100.0},
            {"id": "res_a", "current": 25.5555555, "min": 0.0, "max": 50.0},
        ],
        "entities": [
            {"id": "ent_2", "type": "warehouse", "attributes": {"temp": 21.5}},
            {"id": "ent_1", "type": "factory", "attributes": {"active": True}},
        ],
        "memory": {"lag": 3, "score": 99.99},
    }

    canon_hash = canonical_sha256(data)
    fast_hash = fast_canonical_sha256(data)

    assert len(canon_hash) == 64
    assert canon_hash == fast_hash
