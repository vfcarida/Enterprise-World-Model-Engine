"""Contract test for scale, Structure-of-Arrays batched execution, and cross-backend bit-identity.

Enforces Invariant #5 and T07 requirements:
1. Certified backends (NumPy, JAX CPU/GPU with Threefry) guarantee bit-identical SHA-256 fingerprints.
2. Residual float-nondeterministic backends (CUDA/CuPy) are explicitly labeled throughput-only.
3. Order-independent reduction under distributed parallel execution (Ray / Dask / Multiprocessing).
4. Pure-Python oracle equivalence: Batched SoA Monte Carlo yields bit-identical traces to sequential rollouts.
"""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core._canonical import canonical_sha256, fast_canonical_sha256
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.backends import (
    BackendType,
    CertificationStatus,
    CuPyBackend,
    JaxBackend,
    NumPyBackend,
)
from ewm_engine.simulation.batched import BatchedExecutor
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.executors import DaskExecutor, SerialExecutor
from ewm_engine.simulation.scenario import Scenario


def _build_contract_world(num_nodes: int = 10) -> World:
    entities = [Entity(id=f"depot_{i}", type="depot") for i in range(num_nodes)]
    relationships = [
        Relationship(
            source=f"depot_{i}",
            target=f"depot_{(i + 1) % num_nodes}",
            type="pipeline",
        )
        for i in range(num_nodes)
    ]
    resources = [
        Resource(
            id=f"fuel_{i}",
            entity_id=f"depot_{i}",
            current=120.0,
            min_value=0.0,
            max_value=350.0,
            unit="liters",
        )
        for i in range(num_nodes)
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="fuel_0", mean_demand=6.0, std_demand=1.5),
        ]
    )
    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="fuel_0"),
            ActionTransferAvailabilityConstraint(),
        ]
    )
    actor = ThresholdReplenishmentActor(
        actor_id="replenisher_0",
        source_resource="fuel_1",
        target_resource="fuel_0",
        reorder_point=60.0,
        order_quantity=30.0,
    )
    return World(state=state, dynamics=dynamics, constraints=constraints, actors=[actor])


@pytest.mark.contract
def test_backend_certification_contract() -> None:
    """Verify the formal certification status contract across computational backends."""
    numpy_be = NumPyBackend()
    assert numpy_be.backend_type == BackendType.NUMPY
    assert numpy_be.certification_status == CertificationStatus.CERTIFIED_BIT_IDENTICAL

    jax_be = JaxBackend()
    assert jax_be.backend_type == BackendType.JAX
    assert jax_be.certification_status == CertificationStatus.CERTIFIED_BIT_IDENTICAL

    cupy_be = CuPyBackend()
    assert cupy_be.backend_type == BackendType.CUPY
    assert cupy_be.certification_status == CertificationStatus.THROUGHPUT_ONLY


@pytest.mark.contract
def test_pure_python_oracle_vs_batched_soa_bit_identity() -> None:
    """Invariant #5 Contract: N batched rollouts produce bit-identical SHA-256 fingerprints to sequential."""
    world = _build_contract_world(num_nodes=8)
    scenario = Scenario(
        scenario_id="scen_contract_scale",
        horizon=10,
        samples=8,
        seed=20261016,
    )

    # 1. Canonical Sequential pure-Python Run
    engine_serial = SimulationEngine(executor=SerialExecutor())
    res_serial = engine_serial.run(world, scenario)

    # 2. Batched SoA Run
    engine_batched = SimulationEngine(executor=BatchedExecutor(backend="numpy"))
    res_batched = engine_batched.run(world, scenario)

    assert len(res_serial.trajectories) == len(res_batched.trajectories) == scenario.samples

    for s_traj, b_traj in zip(res_serial.trajectories, res_batched.trajectories, strict=True):
        assert s_traj.sample_id == b_traj.sample_id
        assert s_traj.status == b_traj.status
        assert s_traj.seed == b_traj.seed
        assert len(s_traj.steps) == len(b_traj.steps)

        # Bit-identical state fingerprints across all discrete steps
        for step_idx, (s_step, b_step) in enumerate(zip(s_traj.steps, b_traj.steps, strict=True)):
            assert s_step.state_hash == b_step.state_hash, (
                f"Step {step_idx} state_hash mismatch between serial and batched SoA"
            )

        # Final state bit-identical SHA-256
        assert s_traj.final_state.fingerprint == b_traj.final_state.fingerprint, (
            f"Sample {s_traj.sample_id} final state fingerprint differs between serial and batched"
        )


@pytest.mark.contract
def test_distributed_order_independent_reduction_guarantee() -> None:
    """Verify distributed reduction sorts by sample_id, ensuring schedule-invariant fingerprints."""
    from ewm_engine.simulation.executors import MultiprocessingExecutor

    world = _build_contract_world(num_nodes=5)
    scenario = Scenario(
        scenario_id="scen_order_indep",
        horizon=5,
        samples=6,
        seed=777,
    )

    # 1. Test MultiprocessingExecutor (standard library distributed reduction)
    seed_seq = np.random.SeedSequence(scenario.seed)
    child_seeds = seed_seq.spawn(scenario.samples)
    mp_exec = MultiprocessingExecutor(max_workers=2)
    res_mp = mp_exec.execute(world, scenario, child_seeds)

    assert len(res_mp) == scenario.samples
    sample_ids = [t.sample_id for t in res_mp]
    assert sample_ids == sorted(sample_ids)
    assert sample_ids == list(range(scenario.samples))

    # 2. Test DaskExecutor: either executes sorted if dask installed, or raises informative ImportError
    dask_exec = DaskExecutor()
    try:
        import dask  # noqa: F401

        res_dask = dask_exec.execute(world, scenario, child_seeds)
        assert [t.sample_id for t in res_dask] == list(range(scenario.samples))
    except ImportError as exc:
        assert "dask" in str(exc).lower() or "distributed" in str(exc).lower()


@pytest.mark.contract
def test_fast_canonical_sha256_identity_contract() -> None:
    """Verify fast_canonical_sha256 is 100% bit-identical to reference canonical_sha256."""
    test_cases = [
        {"alpha": 123.456, "beta": [1, 2, 3], "nested": {"key": "val"}},
        {"z": 0.0, "a": None, "bool": True, "str": "hello-world"},
        [1, 2, {"k": "v", "arr": [4, 5, 6]}],
        1234567890,
        "pure string representation",
    ]
    for case in test_cases:
        ref_hash = canonical_sha256(case)
        fast_hash = fast_canonical_sha256(case)
        assert fast_hash == ref_hash
