"""Unit tests for conservation-law and invariant induction via AI-Poincaré nullspace estimation.

Tests:
    1. AI-Poincaré nullspace estimation on state transition differences.
    2. Simplification of real-valued nullspace eigenvectors into small rational weights.
    3. Positive Control: Strict inventory conservation (Stock_A + Stock_B = Const) across
       transfers passes Monte Carlo stress testing and is PROMOTED to a WorldSpec hard constraint.
    4. Negative Control: Leaky/unconserved system fails Monte Carlo stress testing and is
       REJECTED from hard constraint promotion, logged with violation rate and EvidenceLevel.HYPOTHETICAL.
    5. Conversion of InvariantHypothesis to WorldSpec ConstraintSpec.
    6. Determinism and InductionReport fingerprinting.
"""

from __future__ import annotations

import numpy as np

from ewm_engine.induction.invariants import (
    _simplify_rational_weights,
    estimate_conservation_nullspace,
    induce_invariants,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_rational_weight_simplification() -> None:
    """Verify that noisy real-valued eigenvectors round cleanly to rational weights [1.0, 1.0]."""
    noisy_weights = np.array([0.7071, 0.7069])  # Normalized 1:1 vector
    simplified = _simplify_rational_weights(noisy_weights)

    np.testing.assert_allclose(simplified, [1.0, 1.0], atol=1e-3)


def test_ai_poincare_nullspace_estimation() -> None:
    """Verify SVD nullspace identifies conserved direction where Delta X1 + Delta X2 = 0."""
    # Transitions where X1 increases by Delta and X2 decreases by Delta exactly
    rng = np.random.default_rng(42)
    T = 50
    deltas = rng.uniform(-10.0, 10.0, size=T)
    D = np.column_stack([deltas, -deltas])  # D * [1, 1]^T = 0

    null_vectors = estimate_conservation_nullspace(D)

    assert len(null_vectors) >= 1
    # Check that null vector has weights in ratio 1:1
    v = null_vectors[0]
    assert abs(v[0] - v[1]) < 0.05 or abs(v[0] + v[1]) < 0.05


def test_positive_control_strict_conservation_promoted_to_hard_constraint() -> None:
    """Positive Control: Conserved transfer (Stock_A + Stock_B = 1000.0) holds under MC stress.

    Must be PROMOTED to a WorldSpec hard constraint with EvidenceLevel.STRUCTURAL.
    """
    rng = np.random.default_rng(101)
    T = 60
    total_stock = 1000.0
    stock_a = 500.0
    records = []

    for _ in range(T):
        transfer = rng.uniform(-20.0, 20.0)
        stock_a = np.clip(stock_a + transfer, 100.0, 900.0)
        stock_b = total_stock - stock_a
        records.append({"wh_a": stock_a, "wh_b": stock_b})

    hypotheses, _ = induce_invariants(
        trajectories_or_dataset=records,
        tolerance=1e-3,
        mc_samples=10,
        seed=101,
    )

    assert len(hypotheses) >= 1
    # Find the conservation hypothesis linking wh_a and wh_b
    hypo = next(h for h in hypotheses if "wh_a" in h.weights and "wh_b" in h.weights)

    # Must be promoted to hard constraint
    assert hypo.is_promoted_hard_constraint is True
    assert hypo.evidence_level == EvidenceLevel.STRUCTURAL
    assert hypo.max_observed_violation <= 1e-3
    assert hypo.violation_rate <= 1e-4
    assert abs(hypo.constant_value - 1000.0) < 1.0

    # ConstraintSpec must have severity == hard
    constraint = hypo.to_constraint_spec()
    assert constraint.parameters.get("severity") == "hard"


def test_negative_control_leaky_system_rejected_from_hard_constraint() -> None:
    """Negative Control: Leaky bucket system with unmetered loss violates conservation.

    Must be REJECTED from hard constraint promotion and retained as soft constraint / HYPOTHETICAL.
    """
    rng = np.random.default_rng(202)
    T = 60
    stock_a = 500.0
    stock_b = 500.0
    records = []

    for _ in range(T):
        transfer = rng.uniform(5.0, 15.0)
        leak = rng.uniform(2.0, 6.0)  # Unmetered leak breaking conservation!
        stock_a -= transfer
        stock_b += transfer - leak
        records.append({"wh_a": stock_a, "wh_b": stock_b})

    hypotheses, _ = induce_invariants(
        trajectories_or_dataset=records,
        tolerance=1e-3,
        mc_samples=10,
        seed=202,
    )

    for hypo in hypotheses:
        # Every hypothesis must fail the hard-constraint promotion gate!
        assert hypo.is_promoted_hard_constraint is False
        assert hypo.evidence_level in (EvidenceLevel.HYPOTHETICAL, EvidenceLevel.PREDICTIVE)
        assert hypo.max_observed_violation > 1e-3


def test_candidate_invariant_induction_report_fingerprint_determinism() -> None:
    """Verify that invariant induction produces identical SHA-256 fingerprints across runs."""
    records = [
        {"stock_1": 100.0, "stock_2": 200.0},
        {"stock_1": 110.0, "stock_2": 190.0},
        {"stock_1": 90.0, "stock_2": 210.0},
        {"stock_1": 105.0, "stock_2": 195.0},
    ]

    h1, rep1 = induce_invariants(records, seed=42)
    h2, rep2 = induce_invariants(records, seed=42)

    assert rep1.sha256_fingerprint == rep2.sha256_fingerprint
    assert len(rep1.sha256_fingerprint) == 64
    assert len(h1) == len(h2)
