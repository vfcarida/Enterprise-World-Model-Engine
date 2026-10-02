"""Contract test locking the public Stable API surface of EWM Engine."""

from __future__ import annotations

import pytest

import ewm_engine

# NOTE: updated by M1 to match spec Stable list
EXPECTED_STABLE_ALL: set[str] = {
    "Action",
    "Constraint",
    "ConstraintPhase",
    "ConstraintResult",
    "ConstraintSeverity",
    "DynamicsModel",
    "Entity",
    "EvidenceLevel",
    "ExogenousEvent",
    "Provenance",
    "Relationship",
    "Resource",
    "Scenario",
    "SimulationEngine",
    "SimulationResult",
    "TraceEdge",
    "Trajectory",
    "TrajectoryStatus",
    "TransitionResult",
    "World",
    "WorldState",
    "__version__",
    "compare_scenarios",
}


@pytest.mark.contract
def test_public_api_surface_matches_stable_contract() -> None:
    """AC-001: Assert that ewm_engine.__all__ exactly matches the declared Stable surface."""
    actual_all = set(ewm_engine.__all__)
    missing = EXPECTED_STABLE_ALL - actual_all
    unexpected = actual_all - EXPECTED_STABLE_ALL

    assert not missing, f"Missing symbols from ewm_engine.__all__: {sorted(missing)}"
    assert not unexpected, f"Unexpected symbols in ewm_engine.__all__: {sorted(unexpected)}"
    assert actual_all == EXPECTED_STABLE_ALL


@pytest.mark.contract
def test_all_symbols_importable_from_package_root() -> None:
    """AC-001: Assert that every symbol in __all__ is an accessible attribute of ewm_engine."""
    for symbol_name in ewm_engine.__all__:
        assert hasattr(ewm_engine, symbol_name), (
            f"Symbol '{symbol_name}' listed in __all__ but not found on package root."
        )
        obj = getattr(ewm_engine, symbol_name)
        assert obj is not None, f"Symbol '{symbol_name}' resolved to None."


@pytest.mark.contract
def test_experimental_namespace_is_distinct_from_stable() -> None:
    """G21: Assert that experimental features are accessible under ewm_engine.experimental."""
    import ewm_engine.experimental as experimental

    expected_experimental = {
        "Intervention",
        "LearnedDynamics",
        "LinearResidualDynamics",
        "MPCDecisionRecord",
        "RecedingHorizonSimulator",
        "TransitionDataset",
        "TransitionSample",
    }

    actual_experimental = set(experimental.__all__)
    for exp_sym in expected_experimental:
        assert exp_sym in actual_experimental, f"Expected '{exp_sym}' in ewm_engine.experimental."
        assert hasattr(experimental, exp_sym)
        assert getattr(experimental, exp_sym) is not None
