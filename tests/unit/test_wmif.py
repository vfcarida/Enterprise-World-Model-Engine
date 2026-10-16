"""Unit tests for World Model Interchange Format (WMIF) container standard."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.exceptions import SerializationError, SimulationConfigurationError
from ewm_engine.serialization.wmif import (
    PKG_CAUSAL,
    WMIFPackage,
    export_wmif,
    import_wmif,
    verify_wmif_conformance,
)
from ewm_engine.simulation.scenario import Scenario


def _create_sample_world() -> World:
    entities = [
        Entity(id="plant_1", type="factory"),
        Entity(id="store_1", type="retail"),
    ]
    relationships = [
        Relationship(source="plant_1", target="store_1", type="supplies"),
    ]
    resources = [
        Resource(
            id="inventory_p1", entity_id="plant_1", current=500.0, min_value=0.0, max_value=2000.0
        ),
        Resource(
            id="inventory_s1", entity_id="store_1", current=150.0, min_value=0.0, max_value=1000.0
        ),
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = DeterministicTransferDynamics()
    constraints = ConstraintRegistry([ResourceCapacityConstraint(resource_id="inventory_s1")])
    return World(state=state, dynamics=dynamics, constraints=constraints)


def test_wmif_export_import_roundtrip_bit_identity(tmp_path: Path) -> None:
    """Verify that exporting to .wmif and importing back produces bit-identical simulation fingerprints."""
    world = _create_sample_world()
    scenario = Scenario(
        scenario_id="scen_roundtrip",
        horizon=5,
        samples=4,
        seed=12345,
    )

    out_file = tmp_path / "sample_model.wmif"
    exported_path = export_wmif(
        world=world,
        output_path=out_file,
        scenarios=[scenario],
        packages=[PKG_CAUSAL],
        model_name="Sample Logistics World",
        description="Test roundtrip WMIF container",
    )

    assert exported_path.exists()
    assert zipfile.is_zipfile(exported_path)

    # Import back
    imported_world, manifest = import_wmif(exported_path)
    assert manifest.model_name == "Sample Logistics World"
    assert manifest.level == 1
    assert manifest.status == "Proposed/Experimental"
    assert len(manifest.packages) == 1
    assert manifest.packages[0].name == "PkgCausal"
    assert len(manifest.reference_results) == 1

    # Simulate both worlds and compare fingerprints
    res_orig = world.simulate(scenario=scenario)
    res_imp = imported_world.simulate(scenario=scenario)

    assert res_imp.fingerprint == res_orig.fingerprint
    for t_orig, t_imp in zip(res_orig.trajectories, res_imp.trajectories, strict=True):
        assert t_imp.final_state.fingerprint == t_orig.final_state.fingerprint


def test_verify_wmif_conformance(tmp_path: Path) -> None:
    """Verify automated conformance verification of reference results inside a .wmif container."""
    world = _create_sample_world()
    scenario = Scenario(
        scenario_id="scen_conformance",
        horizon=4,
        samples=2,
        seed=9999,
    )

    out_file = tmp_path / "conformance_test.wmif"
    export_wmif(world=world, output_path=out_file, scenarios=[scenario])

    report = verify_wmif_conformance(out_file)
    assert report["conformance"] == "pass"
    assert report["passed"] is True
    assert len(report["scenarios"]) == 1
    assert report["scenarios"][0]["passed"] is True


def test_wmif_unsupported_required_package_fails_closed(tmp_path: Path) -> None:
    """Verify SBML-style fail-closed rule: unknown required packages raise SimulationConfigurationError."""
    world = _create_sample_world()
    unknown_req_pkg = WMIFPackage(
        name="PkgQuantumState",
        version="1.0.0",
        required=True,
        description="Fictional quantum state package",
    )

    out_file = tmp_path / "unknown_pkg.wmif"
    export_wmif(world=world, output_path=out_file, packages=[unknown_req_pkg])

    with pytest.raises(
        SimulationConfigurationError, match="requires unsupported package 'PkgQuantumState'"
    ):
        import_wmif(out_file)


def test_wmif_security_rejects_path_traversal(tmp_path: Path) -> None:
    """Verify zip slip path traversal vulnerabilities are strictly caught and rejected."""
    malicious_zip = tmp_path / "malicious.wmif"
    with zipfile.ZipFile(malicious_zip, "w") as zf:
        zf.writestr("../evil.txt", "payload")
        zf.writestr("modelDescription.json", "{}")

    with pytest.raises(SerializationError, match="Insecure archive entry rejected"):
        import_wmif(malicious_zip)
