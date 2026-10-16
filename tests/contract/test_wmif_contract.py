"""Contract tests for World Model Interchange Format (WMIF) and Conformance Suite (T08).

Enforces:
1. WMIF manifest conforms strictly to Draft 2020-12 schema contract.
2. Reference Worlds produce exact, immutable golden fingerprints for certified seeds.
3. The standalone Conformance Suite runner passes on latest cases and fails closed on negative mutations.
4. WMIF container export-import round-trip preserves bit-identical simulation fingerprints.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conformance.reference_worlds import (
    create_closed_form_stochastic_world,
    create_constraint_stress_world,
    create_seed_trivial_world,
)
from conformance.runner import TESTS_DIR, run_suite
from ewm_engine.serialization.wmif import (
    WMIFManifest,
    export_wmif,
    import_wmif,
    verify_wmif_conformance,
)


@pytest.mark.contract
def test_wmif_manifest_schema_contract() -> None:
    """Verify WMIFManifest adheres to Draft 2020-12 schema standard."""
    schema = WMIFManifest.model_json_schema()
    assert "properties" in schema
    assert "schema_version" in schema["properties"]
    assert "format" in schema["properties"]
    assert "level" in schema["properties"]
    assert "packages" in schema["properties"]
    assert "reference_results" in schema["properties"]


@pytest.mark.contract
def test_reference_worlds_golden_fingerprints_contract() -> None:
    """Invariant #5 Contract: Reference Worlds must reproduce exact golden SHA-256 fingerprints."""
    # 1. Reference World 1: Seed-only Trivial World
    w1, s1 = create_seed_trivial_world()
    r1 = w1.simulate(s1)
    assert r1.fingerprint == "5dbb3db473211cc725a549e714449a2ac43c7d563e301470a70447f323e72dda"
    assert [t.final_state.fingerprint for t in r1.trajectories] == [
        "78e3fff97818dbe1422fbeb1a09a263edc2fbeb2076937ac228e673122675274"
    ]
    assert r1.trajectories[0].status.value == "completed"

    # 2. Reference World 2: Closed-form Stochastic Process
    w2, s2 = create_closed_form_stochastic_world()
    r2 = w2.simulate(s2)
    assert r2.fingerprint == "7a811e8a2a765dad0299b632007a4bd03bcf0baee95eafd1745e9669b09d3ee2"
    assert [t.final_state.fingerprint for t in r2.trajectories] == [
        "9f0e71cbd1952535a9928f545781874d461a55b6401a85bf42ca6d478acf7ea7",
        "f48c3058f510cb4b4f09edbacd612b44908563462d22172b8a3896d1089bbe3b",
        "1ab7f46a83f860e131773f9605b99e281d31822601b5d21ffd3a5ea5652164ca",
    ]
    assert r2.trajectories[0].status.value == "completed"

    # 3. Reference World 3: Constraint Saturation Invalidation
    w3, s3 = create_constraint_stress_world()
    r3 = w3.simulate(s3)
    assert r3.fingerprint == "c51a1b7ed371c3745967046cf716dd60a38e19c2affc9e26d5eae4d2c57830a0"
    assert r3.trajectories[0].status.value == "invalid"
    assert r3.trajectories[0].total_violations == 1


@pytest.mark.contract
def test_conformance_suite_runner_contract() -> None:
    """Verify standalone conformance suite runner executes and passes all latest fixtures."""
    results, all_passed = run_suite(suite_dir=TESTS_DIR / "latest")
    assert all_passed is True
    assert len(results) >= 3
    assert all(r["passed"] for r in results)

    # Assert negative mutation testing catches divergences
    _, neg_passed = run_suite(suite_dir=TESTS_DIR / "latest", test_negative=True)
    assert neg_passed is False, "Negative mutation test must fail when fingerprints diverge"


@pytest.mark.contract
def test_wmif_container_roundtrip_contract(tmp_path: Path) -> None:
    """Verify WMIF container export-import roundtrip produces bit-exact simulation fingerprints."""
    world, scenario = create_seed_trivial_world()
    wmif_file = tmp_path / "seed_trivial.wmif"

    export_wmif(world=world, output_path=wmif_file, scenarios=[scenario])
    conformance = verify_wmif_conformance(wmif_file)
    assert conformance["conformance"] == "pass"
    assert conformance["passed"] is True

    imported_world, manifest = import_wmif(wmif_file)
    assert manifest.format == "WMIF/1.0"
    assert manifest.level == 1
    assert manifest.status == "Proposed/Experimental"

    res_orig = world.simulate(scenario)
    res_imp = imported_world.simulate(scenario)
    assert res_imp.fingerprint == res_orig.fingerprint
