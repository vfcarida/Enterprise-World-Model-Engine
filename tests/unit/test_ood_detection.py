"""Unit tests for Out-of-Distribution (OOD) and regime-shift detection (P09)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.experimental.ood import (
    MahalanobisOODDetector,
    OODStatus,
    SupportBoundaryOODDetector,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.trajectory import StepRecord, Trajectory


def _make_state(
    inventory: float, cash: float, active_rules: dict[str, bool] | None = None
) -> WorldState:
    return WorldState(
        resources={
            "inventory": Resource(
                id="inventory", current=inventory, min_value=0.0, max_value=500.0
            ),
            "cash": Resource(id="cash", current=cash, min_value=0.0, max_value=2000.0),
        },
        active_rules=active_rules or {},
    )


def _make_trajectory(states: list[WorldState], sample_id: int = 0) -> Trajectory:
    steps: list[StepRecord] = []
    for idx, (curr_s, next_s) in enumerate(itertools.pairwise(states)):
        rec = StepRecord(
            step=idx,
            timestamp=float(idx),
            state_hash=curr_s.state_hash,
            transition_result=TransitionResult(
                next_state=next_s,
                applied_changes={},
                evidence_level=EvidenceLevel.STRUCTURAL,
            ),
            step_metrics={},
        )
        steps.append(rec)

    return Trajectory(
        sample_id=sample_id,
        seed=42 + sample_id,
        initial_state=states[0],
        steps=steps,
        final_state=states[-1],
    )


def test_support_boundary_ood_detector_in_and_out_of_distribution() -> None:
    """SupportBoundaryOODDetector stays silent in-distribution and flags boundary violations."""
    # Training / calibration states: inventory in [10, 50], cash in [100, 500]
    calibration_states = [
        _make_state(10.0, 100.0),
        _make_state(25.0, 250.0),
        _make_state(50.0, 500.0),
    ]

    detector = SupportBoundaryOODDetector(tolerance_fraction=0.05)
    detector.fit(calibration_states)
    assert detector.is_fitted

    # In-distribution state
    state_in = _make_state(30.0, 300.0)
    assert not detector.is_ood(state_in)
    assert detector.score_state(state_in) == 0.0

    # OOD state: extreme inventory spike (120.0 vs max 50.0)
    state_out = _make_state(120.0, 300.0)
    assert detector.is_ood(state_out)
    score_out = detector.score_state(state_out)
    # Range is 40.0 (50-10), deviation is 70.0 -> score = 70 / 40 = 1.75
    assert score_out == pytest.approx(1.75)

    step_rep = detector.evaluate_step(state_out, step_idx=1)
    assert step_rep.is_ood
    assert step_rep.status == OODStatus.UNGROUNDED
    assert "inventory" in step_rep.offending_features


def test_support_boundary_ood_detector_novel_rule_shift() -> None:
    """Detects regime shift when an unobserved policy rule becomes active."""
    calibration_states = [
        _make_state(20.0, 200.0, active_rules={"standard_tax": True}),
        _make_state(30.0, 300.0, active_rules={"standard_tax": True}),
    ]

    detector = SupportBoundaryOODDetector()
    detector.fit(calibration_states)

    # State with novel rule "emergency_rationing"
    state_novel_rule = _make_state(25.0, 250.0, active_rules={"emergency_rationing": True})
    assert detector.is_ood(state_novel_rule)
    assert detector.score_state(state_novel_rule) >= 1.0


def test_trajectory_groundedness_evaluation() -> None:
    """Calculates grounded_fraction and flags first OOD step along trajectory."""
    calibration = [_make_state(i * 10.0, 200.0) for i in range(1, 6)]  # inv in [10, 50]
    detector = SupportBoundaryOODDetector(tolerance_fraction=0.05)
    detector.fit(calibration)

    # 1. Fully grounded trajectory (5 steps within [10, 50])
    grounded_states = [_make_state(20.0 + i * 5.0, 200.0) for i in range(5)]
    traj_grounded = _make_trajectory(grounded_states)
    rep_grounded = detector.evaluate_trajectory(traj_grounded)

    assert rep_grounded.grounded_fraction == 1.0
    assert rep_grounded.ood_steps == 0
    assert rep_grounded.first_ood_step is None

    # 2. Trajectory with sudden shock at step 2 (out of 4 transitions)
    shock_states = [
        _make_state(20.0, 200.0),  # step 0 next
        _make_state(30.0, 200.0),  # step 1 next
        _make_state(100.0, 200.0),  # step 2 next (shock!)
        _make_state(110.0, 200.0),  # step 3 next (still shock)
    ]
    traj_shock = _make_trajectory(shock_states)
    rep_shock = detector.evaluate_trajectory(traj_shock)

    # 3 transitions: 1 in-distribution, 2 out-of-distribution
    assert rep_shock.total_steps == 3
    assert rep_shock.ood_steps == 2
    assert rep_shock.first_ood_step == 1
    assert rep_shock.grounded_fraction == pytest.approx(1.0 / 3.0)


def test_trace_annotation_does_not_modify_evidence_level() -> None:
    """Trace annotation adds diagnostic metadata while leaving EvidenceLevel untouched."""
    calibration = [_make_state(10.0, 100.0), _make_state(50.0, 200.0)]
    detector = SupportBoundaryOODDetector()
    detector.fit(calibration)

    shock_states = [_make_state(20.0, 150.0), _make_state(150.0, 150.0)]
    traj = _make_trajectory(shock_states)
    rep = detector.evaluate_trajectory(traj)

    trace = SystemicTrace()
    # Add a base structural node
    trace.add_node(
        "base_transition",
        step=0,
        category="transition",
        label="Transition",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )

    detector.annotate_trace(trace, rep)

    # Base node remains STRUCTURAL
    assert trace.nodes["base_transition"].evidence_level == EvidenceLevel.STRUCTURAL

    # OOD node added with diagnostic details
    assert "ood_step_0" in trace.nodes
    ood_node = trace.nodes["ood_step_0"]
    assert ood_node.category == "ood_regime_shift"
    assert "inventory" in ood_node.details["offending_features"]


def test_mahalanobis_ood_detector_correlation_anomalies() -> None:
    """MahalanobisOODDetector detects correlation structure breakdown even if marginals look valid."""
    # Training data: cash and inventory are strongly positively correlated (y = 2x)
    rng = np.random.default_rng(42)
    calib_states: list[WorldState] = []
    for _ in range(50):
        inv = float(rng.uniform(20.0, 40.0))
        cash = float(2.0 * inv + rng.normal(0.0, 1.0))
        calib_states.append(_make_state(inv, cash))

    detector = MahalanobisOODDetector(threshold_quantile=0.99)
    detector.fit(calib_states)

    # In-distribution point conforming to y = 2x
    s_in = _make_state(30.0, 60.0)
    assert not detector.is_ood(s_in)
    assert detector.score_state(s_in) < 1.0

    # Anomalous point: inventory=35 (in range), cash=10 (in range [40, 80]), but correlation broken!
    s_anom = _make_state(35.0, 10.0)
    assert detector.is_ood(s_anom)
    assert detector.score_state(s_anom) > 1.0
