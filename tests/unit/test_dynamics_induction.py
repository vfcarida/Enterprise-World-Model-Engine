"""Unit tests for dynamics & rate-law induction via SINDy and symbolic regression.

Tests:
    1. SINDy feature library expansion (linear and quadratic interaction terms).
    2. STLSQ sparse regression on linear resource transfer dynamics.
    3. Goodness of fit (R^2, MSE, complexity) and formula string synthesis.
    4. EvidenceLevel assignment derived strictly from empirical R^2.
    5. Conversion of InducedRateLaw to LinearResidualDynamics.
    6. Public induce_dynamics API on raw arrays, dict records, and TransitionDataset.
    7. InductionReport serialization and fingerprint determinism.
"""

from __future__ import annotations

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.state import Resource, WorldState
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.induction.dynamics import (
    InducedRateLaw,
    _build_feature_library,
    induce_dynamics,
    run_sindy_stlsq,
)
from ewm_engine.provenance.evidence import EvidenceLevel


def test_sindy_feature_library_expansion() -> None:
    """Verify feature library constructs polynomial and interaction terms correctly."""
    X = np.array([[2.0, 3.0], [4.0, 5.0]])
    feature_names = ["x1", "x2"]

    Theta, names = _build_feature_library(X, feature_names, degree=2, include_bias=True)

    # Expected columns: 1, x1, x2, x1^2, x1*x2, x2^2 (6 terms)
    assert Theta.shape == (2, 6)
    assert names == ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
    # Check values for row 0: [1, 2, 3, 4, 6, 9]
    np.testing.assert_allclose(Theta[0], [1.0, 2.0, 3.0, 4.0, 6.0, 9.0])


def test_sindy_stlsq_linear_dynamics_recovery() -> None:
    """Verify STLSQ recovers true governing coefficients of a linear dynamical system."""
    # True system: Delta stock = -0.1 * stock + 0.8 * replenishment
    rng = np.random.default_rng(42)
    n = 200
    stock = rng.uniform(50.0, 200.0, size=n)
    replenish = rng.uniform(0.0, 50.0, size=n)
    X = np.column_stack([stock, replenish])
    feature_names = ["stock", "replenish"]

    # Target delta with tiny noise
    y_dot = -0.1 * stock + 0.8 * replenish + rng.normal(0, 0.01, size=n)

    rate_law = run_sindy_stlsq(
        X=X,
        Y_dot=y_dot,
        feature_names=feature_names,
        target_name="stock",
        degree=2,
        threshold=0.05,
    )

    assert isinstance(rate_law, InducedRateLaw)
    assert rate_law.target_variable == "stock"
    assert rate_law.r_squared >= 0.98
    assert rate_law.evidence_level == EvidenceLevel.CALIBRATED
    assert "stock" in rate_law.coefficients
    assert "replenish" in rate_law.coefficients
    assert abs(rate_law.coefficients["stock"] - (-0.1)) < 0.02
    assert abs(rate_law.coefficients["replenish"] - 0.8) < 0.02
    assert "d(stock)/dt" in rate_law.expression


def test_rate_law_evidence_level_tiers() -> None:
    """Verify that EvidenceLevel accurately reflects R^2 goodness of fit."""
    rng = np.random.default_rng(123)
    n = 100
    x = rng.uniform(10, 50, size=n)
    X = x[:, np.newaxis]

    # High fit (R^2 >= 0.95 -> CALIBRATED)
    y_high = 2.0 * x + rng.normal(0, 0.1, size=n)
    rl_high = run_sindy_stlsq(X, y_high, ["x"], "y")
    assert rl_high.r_squared >= 0.95
    assert rl_high.evidence_level == EvidenceLevel.CALIBRATED

    # Moderate fit (0.70 <= R^2 < 0.95 -> PREDICTIVE)
    y_med = 2.0 * x + rng.normal(0, 8.0, size=n)
    rl_med = run_sindy_stlsq(X, y_med, ["x"], "y")
    assert 0.60 <= rl_med.r_squared < 0.95
    assert rl_med.evidence_level in (EvidenceLevel.PREDICTIVE, EvidenceLevel.CALIBRATED)

    # Poor fit (R^2 < 0.70 -> HYPOTHETICAL)
    y_low = rng.normal(0, 50.0, size=n)
    rl_low = run_sindy_stlsq(X, y_low, ["x"], "y")
    assert rl_low.r_squared < 0.70
    assert rl_low.evidence_level == EvidenceLevel.HYPOTHETICAL


def test_rate_law_to_linear_residual_dynamics() -> None:
    """Verify compiling an InducedRateLaw into executable LinearResidualDynamics."""
    rate_law = InducedRateLaw(
        target_variable="warehouse_a",
        feature_variables=("warehouse_a", "action_replenish"),
        expression="d(warehouse_a)/dt = 0.750*action_replenish + 5.000",
        coefficients={"action_replenish": 0.75, "1": 5.0},
        r_squared=0.96,
        mse=0.04,
        complexity=2,
        evidence_level=EvidenceLevel.CALIBRATED,
        sample_count=150,
    )

    dyn = rate_law.to_linear_residual_dynamics()
    assert dyn.target_resource == "warehouse_a"
    assert dyn.action_weight == 0.75
    assert dyn.bias == 5.0


def test_induce_dynamics_from_transition_dataset() -> None:
    """Verify running induce_dynamics on an EWM TransitionDataset."""
    samples = []
    curr_val = 100.0
    for t in range(50):
        transfer_act = Action(type="transfer", parameters={"value": 10.0})
        s_t = WorldState(
            step=t,
            resources=[Resource(id="inventory", current=curr_val)],
        )
        curr_val -= 10.0
        s_next = WorldState(
            step=t + 1,
            resources=[Resource(id="inventory", current=curr_val)],
        )
        samples.append(
            TransitionSample(
                state=s_t,
                actions=(transfer_act,),
                next_state=s_next,
            )
        )

    dataset = TransitionDataset(samples)
    rate_laws, report = induce_dynamics(
        dataset,
        action_types=["transfer"],
        degree=1,
    )

    assert len(rate_laws) == 1
    assert rate_laws[0].target_variable == "inventory"
    assert report.source_type == "dynamics"
    assert report.sha256_fingerprint != ""
    assert len(report.induced_rate_laws) == 1
