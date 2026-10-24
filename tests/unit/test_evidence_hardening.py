"""Unit and contract tests for Evidence & Reproducibility Hardening.

Verifies:
1. Section A: Claim-to-Evidence Registry completeness (ADR-053 capabilities, ADR-054 families A-L,
   EXTERNALLY_REPRODUCED == 0, and Section 1 / Section 2 / README count synchronization).
2. Section B: Reproduction manifest coverage of ADR-053/054, flagship, comparative ladder, validity limits, backend versions,
   seeds, tolerances, canonical artifact SHA-256 digests, and dirty-worktree certification rules.
3. Section C: Artifact -> Docs pipeline and stale-artifact / hand-pasted-number detection.
4. Section D: Automated evidence-tier downgrade safety guard (including negative mutation tests).
5. Section E: Native card (ModelCard, ScenarioCard, DatasetCard) and benchmark result-card
   status coherence (validate_card_coherence and validate_result_card_dict).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ewm_engine.cards import (
    DatasetCard,
    ModelCard,
    ScenarioCard,
    validate_result_card_dict,
)
from ewm_engine.core._canonical import canonical_sha256
from scripts.check_release_truth import (
    REQUIRED_ADR_053_MARKERS,
    REQUIRED_ADR_054_FAMILIES,
    RESULT_CARD_MD_FILES,
    check_artifact_to_docs_pipeline,
    check_claim_evidence_downgrade_safety,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_manifest() -> dict[str, Any]:
    manifest_path = REPO_ROOT / "reproduction_manifest.json"
    return dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def _load_matrix_text() -> str:
    matrix_path = REPO_ROOT / "docs" / "scientific" / "claim-evidence-matrix.md"
    return matrix_path.read_text(encoding="utf-8")


def _resign_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(manifest)
    updated.pop("reproduction_fingerprint", None)
    updated["reproduction_fingerprint"] = canonical_sha256(updated)
    return updated


class TestSectionAClaimRegistryCompleteness:
    """Verify Claim-to-Evidence Registry completeness across ADR-053/054 and core scientific claims."""

    def test_claim_registry_passes_all_completeness_and_downgrade_checks(self) -> None:
        errors = check_claim_evidence_downgrade_safety(repo_root=REPO_ROOT)
        assert errors == [], f"Unexpected claim registry errors: {errors}"

    def test_all_adr_053_capabilities_and_adr_054_families_present(self) -> None:
        matrix_text = _load_matrix_text()
        for cap in REQUIRED_ADR_053_MARKERS:
            assert cap in matrix_text, f"Missing ADR-053 capability {cap!r}"
        for fam in REQUIRED_ADR_054_FAMILIES:
            assert fam in matrix_text, f"Missing ADR-054 family {fam!r}"


class TestSectionBReproductionManifestCoverage:
    """Verify reproduction_manifest.json captures all ADR-053/054, flagship, comparative ladder, and validity-limit evidence."""

    def test_manifest_covers_adr_053_054_layers_and_provenance(self) -> None:
        manifest = _load_manifest()
        assert manifest["verification_checks_passed"] is True
        assert manifest["certification_status"] in (
            "REPRODUCIBLE_DIRTY_WORKTREE",
            "CERTIFIED_RELEASE",
        )
        if manifest["provenance"]["git"]["dirty"]:
            assert manifest["certification_status"] == "REPRODUCIBLE_DIRTY_WORKTREE"

        env = manifest["provenance"]["environment"]
        assert "backend_versions" in env
        assert "ewm_engine" in env["backend_versions"]
        assert "numpy" in env["backend_versions"]
        assert len(env["uv_lock_sha256"]) == 64

        proto = manifest["provenance"]["reproducibility_protocol"]
        for key in (
            "enterprise_benchmark_suite",
            "fraud_cascade_flagship",
            "comparative_evaluation_ladder",
            "validity_limits_evaluation",
        ):
            assert proto["rng_seeds"][key] == 42

        adr_layer = manifest["results"]["adr_053_054_evidence_layer"]
        for key in (
            "enterprise_benchmark_suite",
            "fraud_cascade_flagship",
            "comparative_evaluation_ladder",
            "validity_limits_evaluation",
        ):
            assert key in adr_layer
            assert adr_layer[key]["status"] == "PASSED"

        assert adr_layer["enterprise_benchmark_suite"]["families_evaluated"] == 12
        assert adr_layer["enterprise_benchmark_suite"]["all_leakage_certificates_passed"] is True
        assert adr_layer["fraud_cascade_flagship"]["mechanism_recovery_all_passed"] is True
        assert adr_layer["fraud_cascade_flagship"]["bitemporal_leakage_blocked"] is True
        assert adr_layer["comparative_evaluation_ladder"]["rungs_evaluated"] == 5
        assert adr_layer["comparative_evaluation_ladder"]["negative_results_confirmed"] == 4
        assert adr_layer["validity_limits_evaluation"]["plan_self_invalidating_detected"] is True


class TestSectionCArtifactToDocsPipeline:
    """Verify artifact -> docs pipeline passes and catches stale or hand-edited numbers."""

    def test_artifact_to_docs_pipeline_passes_on_repo(self) -> None:
        errors = check_artifact_to_docs_pipeline(repo_root=REPO_ROOT)
        assert errors == [], f"Unexpected artifact-to-docs pipeline errors: {errors}"

    def test_rejects_certified_release_on_dirty_worktree(self) -> None:
        manifest = _load_manifest()
        manifest["provenance"]["git"]["dirty"] = True
        manifest["certification_status"] = "CERTIFIED_RELEASE"
        mutated = _resign_manifest(manifest)

        errors = check_artifact_to_docs_pipeline(repo_root=REPO_ROOT, manifest_override=mutated)
        assert any("Dirty working tree cannot claim CERTIFIED_RELEASE" in e for e in errors)

    def test_detects_stale_canonical_artifact_sha256(self) -> None:
        manifest = _load_manifest()
        manifest["canonical_artifacts"]["docs/results/artifacts/fraud_cascade_flagship.json"][
            "file_sha256"
        ] = "0" * 64
        mutated = _resign_manifest(manifest)

        errors = check_artifact_to_docs_pipeline(repo_root=REPO_ROOT, manifest_override=mutated)
        assert any(
            "Stale canonical artifact docs/results/artifacts/fraud_cascade_flagship.json" in e
            for e in errors
        )

    def test_detects_hand_edited_prose_number_mismatch(self) -> None:
        manifest = _load_manifest()
        manifest["results"]["adr_053_054_evidence_layer"]["fraud_cascade_flagship"][
            "myopic_decision_regret_usd"
        ] = 999999.99
        mutated = _resign_manifest(manifest)

        errors = check_artifact_to_docs_pipeline(repo_root=REPO_ROOT, manifest_override=mutated)
        assert any("docs/results/index.md missing flagship myopic regret" in e for e in errors)


class TestSectionDEvidenceTierDowngradeSafety:
    """Verify evidence-tier downgrade guard catches inflated tiers and missing/broken evidence."""

    def test_rejects_externally_reproduced_greater_than_zero(self) -> None:
        matrix_text = _load_matrix_text()
        mutated_text = matrix_text.replace(
            "| `EXTERNALLY_REPRODUCED` | Independently replicated by unaffiliated third-party researchers in a peer-reviewed replication venue. | **0** |",
            "| `EXTERNALLY_REPRODUCED` | Independently replicated by unaffiliated third-party researchers in a peer-reviewed replication venue. | **1** |",
        )
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, matrix_text_override=mutated_text
        )
        assert any("EXTERNALLY_REPRODUCED count in Section 1 must be 0" in e for e in errors)

    def test_rejects_section1_vs_section2_tier_count_mismatch(self) -> None:
        matrix_text = _load_matrix_text()
        mutated_text = matrix_text.replace(
            "| `UNIT_VERIFIED` | Verified by deterministic unit and integration tests over fixed inputs and edge cases. | 6 |",
            "| `UNIT_VERIFIED` | Verified by deterministic unit and integration tests over fixed inputs and edge cases. | 99 |",
        )
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, matrix_text_override=mutated_text
        )
        assert any("Evidence tier count mismatch for UNIT_VERIFIED" in e for e in errors)

    def test_rejects_missing_test_file_for_verified_claim(self) -> None:
        matrix_text = _load_matrix_text()
        mutated_text = matrix_text.replace(
            "tests/unit/test_dependency_evidence_nature.py",
            "tests/unit/test_non_existent_deleted_file.py",
        )
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, matrix_text_override=mutated_text
        )
        assert any("test_non_existent_deleted_file.py" in e for e in errors)

    def test_rejects_broken_pytest_k_selector(self) -> None:
        matrix_text = _load_matrix_text()
        mutated_text = matrix_text.replace(
            'pytest tests/contract/test_world_model_semantics_p02.py -k "property_7" -q',
            'pytest tests/contract/test_world_model_semantics_p02.py -k "nonexistent_test_selector_xyz" -q',
        )
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, matrix_text_override=mutated_text
        )
        assert any("nonexistent_test_selector_xyz" in e for e in errors)

    def test_rejects_benchmarked_tier_when_card_fails_or_benchmark_is_na(self) -> None:
        manifest = _load_manifest()
        manifest["results"]["scientific_cards"]["seir"]["status"] = "FAILED"
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, manifest_override=manifest
        )
        assert any("references card 'seir'" in e and "not 'PASSED'" in e for e in errors)

        matrix_text = _load_matrix_text()
        mutated_na = matrix_text.replace(
            "| `BENCHMARKED` | `src/ewm_engine/flagship/fraud_cascade.py`, `src/ewm_engine/core/bitemporal.py`, `scripts/run_fraud_cascade_flagship.py` | `tests/unit/test_fraud_cascade_flagship.py` | `docs/results/artifacts/fraud_cascade_flagship.json` |",
            "| `BENCHMARKED` | `src/ewm_engine/flagship/fraud_cascade.py`, `src/ewm_engine/core/bitemporal.py`, `scripts/run_fraud_cascade_flagship.py` | `tests/unit/test_fraud_cascade_flagship.py` | N/A |",
        )
        errors_na = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, matrix_text_override=mutated_na
        )
        assert any("tier BENCHMARKED cannot have benchmark=" in e for e in errors_na)

    def test_rejects_calibrated_tier_when_coverage_not_certified(self) -> None:
        manifest = _load_manifest()
        manifest["results"]["scientific_cards"]["calibrated_posterior"]["coverage_certified"] = (
            False
        )
        errors = check_claim_evidence_downgrade_safety(
            repo_root=REPO_ROOT, manifest_override=manifest
        )
        assert any("coverage_certified=True" in e for e in errors)


class TestSectionECardsCoherence:
    """Verify ModelCard, ScenarioCard, DatasetCard, and scientific result card status coherence."""

    def test_all_seven_scientific_result_cards_pass_validation(self) -> None:
        manifest = _load_manifest()
        sci_cards = manifest["results"]["scientific_cards"]
        assert set(sci_cards.keys()) == set(RESULT_CARD_MD_FILES.keys())
        for card_key, card_dict in sci_cards.items():
            errs = validate_result_card_dict(card_dict)
            assert errs == [], f"Card {card_key} failed validation: {errs}"

    def test_validate_result_card_dict_catches_incoherent_status_fields(self) -> None:
        bad_card = {
            "card": "seir_outbreak",
            "status": "PASSED",
            "card_verification_status": "FAILED_DRIFT",
            "physics_iq_score": 0.5,
            "max_mass_balance_drift": 1e-3,
        }
        errs = validate_result_card_dict(bad_card, strict=False)
        assert any("contradicts card_verification_status" in e for e in errs)
        assert any("physics_iq_score == 1.0" in e for e in errs)
        assert any("max_mass_balance_drift <= 1e-12" in e for e in errs)

        bad_s09 = {
            "card": "induction_validation_s09",
            "status": "PASSED",
            "card_verification_status": "PASSED_REFUSAL_GATE_VERIFIED",
            "validation_passed": True,
            "all_candidates_admitted": False,
            "gate_outcome": "REFUSED_UNIDENTIFIABLE_AS_DESIGNED",
            "status_explanation": "Explanation present.",
        }
        errs_s09 = validate_result_card_dict(bad_s09, strict=False)
        assert any(
            "validation_passed (True) != all_candidates_admitted (False)" in e for e in errs_s09
        )

    def test_native_cards_validate_coherence(self) -> None:
        valid_model_card = ModelCard(
            artifact_fingerprint="a" * 64,
            model_id="mc_flagship",
            model_name="Flagship Cascade Model",
            version="1.5.0",
            intended_use=["Decision-first comparative evaluation on synthetic cascade"],
            out_of_scope=["Production banking authorization without live calibration"],
            assumptions=["Closed customer cohort of 10,000"],
            limitations=["Controlled synthetic mechanisms"],
            metrics={"status": "PASSED", "regret_usd": 0.0},
            calibration_score=0.90,
        )
        assert valid_model_card.validate_card_coherence(strict=False) == []

        incoherent_model_card = ModelCard(
            artifact_fingerprint="b" * 64,
            maturity="production",
            model_id="mc_bad",
            model_name="Incoherent Model",
            version="1.5.0",
            intended_use=[],
            out_of_scope=[],
            assumptions=[],
            limitations=[],
            metrics={"status": "FAILED"},
            calibration_score=1.5,
        )
        mc_errs = incoherent_model_card.validate_card_coherence(strict=False)
        assert len(mc_errs) >= 3

        incoherent_scenario = ScenarioCard(
            artifact_fingerprint="c" * 64,
            maturity="production",
            scenario_id="sc_bad",
            horizon=10,
            samples=5,
            seed=42,
            assumptions=[],
            out_of_scope=[],
            metrics={"status": "FAILED"},
        )
        assert len(incoherent_scenario.validate_card_coherence(strict=False)) >= 2

        incoherent_dataset = DatasetCard(
            artifact_fingerprint="d" * 64,
            dataset_id="dc_bad",
            name="Incoherent Dataset",
            version="1.5.0",
            num_trajectories=10,
            num_steps=2,
            features=[],
            intended_use=[],
        )
        assert len(incoherent_dataset.validate_card_coherence(strict=False)) >= 3
