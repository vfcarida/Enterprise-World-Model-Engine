"""Public API compatibility and contract gate tests (AC-024).

Asserts that no unintended drift occurs to the Stable public surface,
callable method signatures, or serialized schema contracts across 1.x without
an approved API Change Proposal (ACP).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import ewm_engine
from ewm_engine import (
    SimulationEngine,
    Trajectory,
    World,
    compare_scenarios,
)
from scripts.generate_schemas import SCHEMA_REGISTRY, SCHEMAS_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Committed v1.0.0 Stable Public Surface Baseline
LOCKED_V1_STABLE_SURFACE: frozenset[str] = frozenset(
    {
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
        "compare_scenarios",
        "__version__",
    }
)


@pytest.mark.contract
def test_public_api_surface_drift_gate() -> None:
    """AC-024: Fail if any symbol is added, removed, or renamed from __all__ without an API Change Proposal."""
    current_surface = frozenset(ewm_engine.__all__)

    added_symbols = current_surface - LOCKED_V1_STABLE_SURFACE
    removed_symbols = LOCKED_V1_STABLE_SURFACE - current_surface

    assert not removed_symbols, (
        f"CRITICAL API DRIFT: Symbols removed from Stable public API: {sorted(removed_symbols)}. "
        f"Removals violate SemVer 1.x and require an approved API Change Proposal and major release."
    )

    assert not added_symbols, (
        f"API DRIFT DETECTED: New symbols added to root ewm_engine.__all__: {sorted(added_symbols)}. "
        f"New Stable symbols must be reviewed under .github/API_CHANGE_PROPOSAL.md and added to "
        f"LOCKED_V1_STABLE_SURFACE."
    )


@pytest.mark.contract
def test_core_engine_method_signatures_have_no_breaking_drift() -> None:
    """AC-024: Ensure key entrypoints maintain backwards-compatible parameter signatures."""
    # 1. World.branch() takes no required arguments
    branch_sig = inspect.signature(World.branch)
    for name, param in branch_sig.parameters.items():
        if name != "self":
            assert param.default != inspect.Parameter.empty, (
                f"World.branch has new required parameter '{name}'"
            )

    # 2. SimulationEngine.run(world, scenario)
    run_sig = inspect.signature(SimulationEngine.run)
    run_params = list(run_sig.parameters.keys())
    assert "world" in run_params
    assert "scenario" in run_params

    # 3. compare_scenarios signature
    comp_sig = inspect.signature(compare_scenarios)
    comp_params = list(comp_sig.parameters.keys())
    assert "baseline" in comp_params
    assert "candidates" in comp_params

    # 4. Trajectory.finalize()
    finalize_sig = inspect.signature(Trajectory.finalize)
    assert len(finalize_sig.parameters) == 1  # only self


@pytest.mark.contract
def test_schema_versions_locked_at_v1() -> None:
    """AC-024: All committed schemas must remain locked to schema_version 1.0.0."""
    import json

    for filename in SCHEMA_REGISTRY:
        schema_path = SCHEMAS_DIR / filename
        assert schema_path.exists(), f"Schema file {filename} missing"
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        props = data.get("properties", {})
        assert "schema_version" in props, f"schema_version missing in {filename}"
        assert props["schema_version"].get("const") == "1.0.0", (
            f"schema_version in {filename} drifted from 1.0.0"
        )
