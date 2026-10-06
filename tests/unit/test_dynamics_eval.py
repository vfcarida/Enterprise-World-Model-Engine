"""Unit tests for the learned dynamics evaluation harness and consistency gate."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ewm_engine.core import Action, ExogenousEvent, Resource, WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.dynamics.learned import LinearResidualDynamics, TransitionDataset, TransitionSample
from ewm_engine.experimental.dynamics_eval import (
    evaluate_dynamics_model,
    evaluate_interventional_shift,
    evaluate_one_step,
    evaluate_stochastic_calibration,
    symexp,
    symlog,
)
from ewm_engine.provenance.evidence import EvidenceLevel

# ---------------------------------------------------------------------------
# Test Mocks
# ---------------------------------------------------------------------------


class MockViolatingDynamics(DynamicsModel):
    """Dynamics model that produces non-negative and bound-violating states."""

    def __init__(self, mode: str = "bounds") -> None:
        self.mode = mode
        self.name = f"MockViolatingDynamics_{mode}"

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        if self.mode == "bounds":
            # Force current well above max_value (500)
            next_state = state.update_resource("cash", delta=1000.0, clamp=False)
        elif self.mode == "non_negativity":
            # Force current negative below 0
            next_state = state.update_resource("cash", delta=-500.0, clamp=False)
        elif self.mode == "conservation":
            # Cash increases by 50, but debt remains unchanged (breaches zero-sum balance)
            next_state = state.update_resource("cash", delta=50.0, clamp=True)
        else:
            next_state = state

        return TransitionResult(
            next_state=next_state,
            applied_changes={"mode": self.mode},
            evidence_level=EvidenceLevel.PREDICTIVE,
            model_name=self.name,
        )


class MockStochasticDynamics(DynamicsModel):
    """Dynamics model that adds Gaussian noise to transitions."""

    def __init__(self, mean_delta: float = 10.0, std: float = 2.0) -> None:
        self.mean_delta = mean_delta
        self.std = std
        self.name = "MockStochasticDynamics"

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        noise = float(rng.normal(self.mean_delta, self.std))
        next_state = state.update_resource("cash", delta=noise, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"delta": noise},
            evidence_level=EvidenceLevel.PREDICTIVE,
            model_name=self.name,
        )


# ---------------------------------------------------------------------------
# Symlog & Symexp Tests
# ---------------------------------------------------------------------------


def test_symlog_symexp_properties() -> None:
    """Test mathematical properties of DreamerV3 symlog and symexp functions."""
    # 1. Zero preservation
    assert symlog(0.0) == 0.0
    assert symexp(0.0) == 0.0

    # 2. Invertibility across wide ranges
    test_values = np.array([-1e6, -100.0, -1.0, 0.0, 0.05, 1.0, 50.0, 1e7], dtype=float)
    compressed = symlog(test_values)
    reconstructed = symexp(compressed)
    np.testing.assert_allclose(test_values, reconstructed, rtol=1e-5, atol=1e-5)

    # 3. Scale compression
    assert symlog(1e6) < 15.0
    assert symlog(-1e6) > -15.0


# ---------------------------------------------------------------------------
# Consistency Gate & Invariant Tests
# ---------------------------------------------------------------------------


def test_invariant_gate_detects_bounds_violation() -> None:
    """Consistency gate must flag models that violate resource [min, max] bounds."""
    s1 = WorldState(
        resources={"cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)}
    )
    s2 = WorldState(
        resources={"cash": Resource(id="cash", current=110.0, min_value=0.0, max_value=500.0)}
    )
    act = Action(id="a1", type="transfer", parameters={"value": 10.0})
    dataset = TransitionDataset([TransitionSample(state=s1, actions=(act,), next_state=s2)] * 5)

    violating_model = MockViolatingDynamics(mode="bounds")
    _err_res, inv_res = evaluate_one_step(violating_model, dataset)

    assert not inv_res.passed
    assert inv_res.bounds_violations > 0
    assert "cash" in inv_res.violating_resources


def test_invariant_gate_detects_non_negativity_violation() -> None:
    """Consistency gate must flag models that violate declared non-negativity."""
    s1 = WorldState(
        resources={"cash": Resource(id="cash", current=50.0, min_value=0.0, max_value=500.0)}
    )
    s2 = WorldState(
        resources={"cash": Resource(id="cash", current=40.0, min_value=0.0, max_value=500.0)}
    )
    act = Action(id="a1", type="spend", parameters={"value": 10.0})
    dataset = TransitionDataset([TransitionSample(state=s1, actions=(act,), next_state=s2)] * 5)

    violating_model = MockViolatingDynamics(mode="non_negativity")
    _err_res, inv_res = evaluate_one_step(violating_model, dataset)

    assert not inv_res.passed
    assert inv_res.non_negativity_violations > 0
    assert "cash" in inv_res.violating_resources


def test_invariant_gate_detects_conservation_violation() -> None:
    """Consistency gate must detect when coupled conservation balances are violated."""
    # Account balance: cash + savings must remain constant
    s1 = WorldState(
        resources={
            "cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=1000.0),
            "savings": Resource(id="savings", current=200.0, min_value=0.0, max_value=1000.0),
        }
    )
    s2 = WorldState(
        resources={
            "cash": Resource(id="cash", current=150.0, min_value=0.0, max_value=1000.0),
            "savings": Resource(id="savings", current=150.0, min_value=0.0, max_value=1000.0),
        }
    )
    act = Action(id="a1", type="transfer", parameters={"value": 50.0})
    dataset = TransitionDataset([TransitionSample(state=s1, actions=(act,), next_state=s2)] * 5)

    # Model modifies cash without adjusting savings
    violating_model = MockViolatingDynamics(mode="conservation")
    _err_res, inv_res = evaluate_one_step(
        violating_model,
        dataset,
        conservation_groups=[("cash", "savings")],
    )

    assert not inv_res.passed
    assert inv_res.conservation_violations > 0


def test_evaluate_dynamics_model_fails_on_invariant_even_if_error_zero() -> None:
    """Overall evaluation report must set overall_valid=False if invariants fail."""
    s1 = WorldState(
        resources={"cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)}
    )
    s2 = WorldState(
        resources={"cash": Resource(id="cash", current=1100.0, min_value=0.0, max_value=500.0)}
    )
    act = Action(id="a1", type="transfer", parameters={"value": 10.0})
    # Synthetic dataset where next_state itself breaches max_value
    dataset = TransitionDataset([TransitionSample(state=s1, actions=(act,), next_state=s2)] * 5)

    violating_model = MockViolatingDynamics(mode="bounds")
    report = evaluate_dynamics_model(violating_model, dataset)

    assert not report.invariants.passed
    assert not report.overall_valid


# ---------------------------------------------------------------------------
# Calibration & Stochastic Evaluation Tests
# ---------------------------------------------------------------------------


def test_evaluate_stochastic_calibration() -> None:
    """Calibration must compute CRPS and interval coverage using uncertainty tools."""
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(20):
        s1 = WorldState(
            resources={"cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)}
        )
        actual_delta = float(rng.normal(10.0, 2.0))
        s2 = WorldState(
            resources={
                "cash": Resource(
                    id="cash", current=100.0 + actual_delta, min_value=0.0, max_value=500.0
                )
            }
        )
        act = Action(id="a1", type="step", parameters={"value": 1.0})
        samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    dataset = TransitionDataset(samples)
    stochastic_model = MockStochasticDynamics(mean_delta=10.0, std=2.0)

    calib = evaluate_stochastic_calibration(stochastic_model, dataset, samples=50, seed=42)

    assert calib.crps > 0.0
    assert 0.70 <= calib.coverage <= 1.0
    assert calib.is_calibrated


# ---------------------------------------------------------------------------
# Interventional Shift Detection Tests
# ---------------------------------------------------------------------------


def test_evaluate_interventional_shift_detection() -> None:
    """Harness must detect when predictive error degrades under out-of-distribution actions."""
    in_samples = []
    out_samples = []

    # In-distribution actions: value = 2.0
    for _ in range(15):
        s1 = WorldState(
            resources={"cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=1000.0)}
        )
        s2 = WorldState(
            resources={"cash": Resource(id="cash", current=104.0, min_value=0.0, max_value=1000.0)}
        )
        act = Action(id="a1", type="invest", parameters={"value": 2.0})
        in_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    # Out-of-distribution interventional actions: value = 20.0, but non-linear saturation
    for _ in range(15):
        s1 = WorldState(
            resources={"cash": Resource(id="cash", current=100.0, min_value=0.0, max_value=1000.0)}
        )
        # Saturated return: only +15 instead of expected +40
        s2 = WorldState(
            resources={"cash": Resource(id="cash", current=115.0, min_value=0.0, max_value=1000.0)}
        )
        act = Action(id="a1", type="invest", parameters={"value": 20.0})
        out_samples.append(TransitionSample(state=s1, actions=(act,), next_state=s2))

    in_ds = TransitionDataset(in_samples)
    out_ds = TransitionDataset(out_samples)

    # Train linear model on in-distribution data
    linear_model = LinearResidualDynamics(target_resource="cash", action_type="invest")
    linear_model.fit(in_ds)

    shift_res = evaluate_interventional_shift(linear_model, in_ds, out_ds)

    assert shift_res.shift_detected
    assert shift_res.interventional_mae > shift_res.in_distribution_mae
    assert shift_res.interventional_gap > 0.0
