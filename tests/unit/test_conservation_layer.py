"""Unit tests for the architectural conservation projection layer and honesty reporting."""

from __future__ import annotations

import numpy as np
import pytest

from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.deterministic import DeterministicDynamics, DeterministicTransferDynamics
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.experimental.conservation import (
    ConservationProjectionLayer,
    LinearConservationLaw,
    PortHamiltonianConservationLayer,
    evaluate_conservation_honesty,
)
from ewm_engine.experimental.grey_box import CallableResidualEstimator, GreyBoxDynamics


def test_linear_conservation_laws(sample_world_state: WorldState) -> None:
    """Test LinearConservationLaw evaluation for zero-sum and double-entry invariants."""
    zero_sum_law = LinearConservationLaw.zero_sum(
        resource_ids=["stock_wh1", "stock_wh2"],
        name="InventoryMassConservation",
    )

    # Valid conservative transition: wh1 -10, wh2 +10
    state_valid = sample_world_state.update_resource("stock_wh1", delta=-10.0)
    state_valid = state_valid.update_resource("stock_wh2", delta=10.0)
    assert zero_sum_law.evaluate_violation(sample_world_state, state_valid) == pytest.approx(0.0)
    assert zero_sum_law.is_satisfied(sample_world_state, state_valid)

    # Violating transition: wh1 -10, wh2 +5 (5 units evaporated)
    state_invalid = sample_world_state.update_resource("stock_wh1", delta=-10.0)
    state_invalid = state_invalid.update_resource("stock_wh2", delta=5.0)
    viol = zero_sum_law.evaluate_violation(sample_world_state, state_invalid)
    assert viol == pytest.approx(5.0)
    assert not zero_sum_law.is_satisfied(sample_world_state, state_invalid)


def test_conservation_projection_layer_adversarial_nullspace_projection(
    sample_world_state: WorldState,
) -> None:
    """Test KKT-hPINN nullspace projection layer zeroes out adversarial unconstrained deltas."""
    law = LinearConservationLaw.zero_sum(resource_ids=["stock_wh1", "stock_wh2"])
    layer = ConservationProjectionLayer(laws=[law], enforce_capacity_bounds=False)

    rng = np.random.default_rng(1234)
    # Test 50 adversarial unconstrained deltas
    for _ in range(50):
        d1 = float(rng.uniform(-50.0, 50.0))
        d2 = float(rng.uniform(-50.0, 50.0))
        raw_deltas = {"stock_wh1": d1, "stock_wh2": d2}

        projected = layer.project_deltas(sample_world_state, raw_deltas)

        # Invariant: projected delta sum must be structurally zero
        sum_proj = projected["stock_wh1"] + projected["stock_wh2"]
        assert sum_proj == pytest.approx(0.0, abs=1e-12)


def test_conservation_projection_layer_boundary_scaling(sample_world_state: WorldState) -> None:
    """Test that boundary scaling prevents capacity/non-negativity violations while preserving sum=0."""
    law = LinearConservationLaw.zero_sum(resource_ids=["stock_wh1", "stock_wh2"])
    layer = ConservationProjectionLayer(laws=[law], enforce_capacity_bounds=True)

    # Initial: wh1=100 (min=0, max=200), wh2=50 (min=0, max=200)
    # Propose massive delta that would push wh1 to negative: raw = {"stock_wh1": -200.0, "stock_wh2": 200.0}
    raw_deltas = {"stock_wh1": -200.0, "stock_wh2": 200.0}
    projected = layer.project_deltas(sample_world_state, raw_deltas)

    # Max available wh1 is 100.0. The projection layer scales alpha to 0.5: delta_wh1=-100, delta_wh2=+100
    assert projected["stock_wh1"] == pytest.approx(-100.0)
    assert projected["stock_wh2"] == pytest.approx(100.0)
    assert projected["stock_wh1"] + projected["stock_wh2"] == pytest.approx(0.0, abs=1e-12)


def test_port_hamiltonian_layer_skew_symmetry_and_stability() -> None:
    """Test Port-Hamiltonian layer enforces J = -J^T and non-negative dissipation."""
    ph = PortHamiltonianConservationLayer(
        resource_ids=["stock_wh1", "stock_wh2", "cash"],
        dissipation_rate=0.05,
    )

    J = ph.skew_symmetric_matrix
    R = ph.dissipation_matrix

    # Skew-symmetry: J + J^T == 0
    assert np.allclose(J + J.T, 0.0)

    # Dissipation: R is positive semi-definite
    assert np.all(np.linalg.eigvals(R) >= 0.0)

    # Energy change: dH/dt = x^T (J - R) x = x^T J x - x^T R x = 0 - x^T R x <= 0
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    flow = ph.compute_flow(x)
    energy_rate = float(x @ flow)
    assert energy_rate <= 1e-12


def test_conservation_honesty_report_structural_zero_vs_unconstrained(
    sample_world_state: WorldState,
) -> None:
    """Test honesty report confirms structural-zero violation rate vs unconstrained baseline."""
    law = LinearConservationLaw.zero_sum(resource_ids=["stock_wh1", "stock_wh2"])
    projection_layer = ConservationProjectionLayer(laws=[law])

    known_model = DeterministicTransferDynamics()

    # Unconstrained adversarial estimator that injects hallucinated inventory
    hallucinating_estimator = CallableResidualEstimator(
        fn=lambda s, a, e, next_s: {"stock_wh1": 15.0, "stock_wh2": 15.0}
    )

    # Unconstrained baseline: applies deltas without projection layer
    unconstrained_model = DeterministicDynamics(
        name="UnconstrainedBaseline",
        transition_fn=lambda s, a, e: s.update_resource("stock_wh1", delta=15.0).update_resource(
            "stock_wh2", delta=15.0
        ),
    )

    # Grey-box with architectural conservation layer
    conserved_grey_box = GreyBoxDynamics(
        known_model=known_model,
        residual_estimator=hallucinating_estimator,
        conservation_layer=projection_layer,
        name="ConservedGreyBox",
    )

    # Build evaluation dataset
    dataset = TransitionDataset()
    for _ in range(20):
        dataset.add(
            TransitionSample(
                state=sample_world_state,
                actions=(),
                events=(),
                next_state=sample_world_state,
            )
        )

    # Evaluate honesty
    report = evaluate_conservation_honesty(
        model=conserved_grey_box,
        dataset=dataset,
        laws=[law],
        unconstrained_baseline=unconstrained_model,
    )

    # The architectural grey-box MUST report structural-zero violation rate
    assert report.is_structurally_conserved is True
    assert report.violation_rate == 0.0
    assert report.violations_detected == 0
    assert report.max_absolute_violation == pytest.approx(0.0, abs=1e-10)

    # Contrast with unconstrained baseline: reports non-zero violation rate
    assert report.unconstrained_comparison is not None
    assert report.unconstrained_comparison["violation_rate"] == 1.0
    assert report.unconstrained_comparison["max_absolute_violation"] == pytest.approx(30.0)
