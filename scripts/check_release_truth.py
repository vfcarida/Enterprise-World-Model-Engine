"""Release engineering, version coherence, artifact freshness, and security control verifier.

Enforces release, artifact freshness, and security engineering invariants:
1. Version Coherence:
   `pyproject.toml (fallback_version) == CITATION.cff (version & preferred-citation.version)
   == base_version(ewm_engine.__version__)` (and `== git tag` when `--tag` is provided).
2. Canonical Artifact Non-Staleness:
   - JSON Schemas (`scripts/generate_schemas.py --check`) are up to date.
   - `benchmarks/mutation_baseline.json` and `reproduction_manifest.json` exist and parse cleanly.
3. Security Automation Verification (Verified in Workflow Code, Not Merely Documented):
   - All third-party GitHub Actions pinned to 40-char commit SHAs.
   - Top-level `permissions: contents: read` enforced across CI/Release workflows.
   - CodeQL (`codeql.yml`), OSSF Scorecard (`scorecard.yml`), Gitleaks & pip-audit (`security.yml`),
     Dependabot (`.github/dependabot.yml`), SBOMs (`cyclonedx-py` + `anchore/sbom-action`),
     OIDC trusted publishing (`pypa/gh-action-pypi-publish`), and SLSA build provenance
     (`actions/attest-build-provenance`) are present and active in workflow definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def _base_semver(version_str: str) -> str:
    """Extract major.minor.patch base version from a PEP 440 version string."""
    match = re.match(r"^v?(\d+\.\d+\.\d+)", version_str.strip())
    if not match:
        raise ValueError(f"Cannot parse base semantic version from {version_str!r}")
    return match.group(1)


def _next_patch_semver(version_str: str) -> str:
    """Return major.minor.(patch+1) for hatch-vcs post-tag .dev version comparison."""
    parts = [int(x) for x in _base_semver(version_str).split(".")]
    return f"{parts[0]}.{parts[1]}.{parts[2] + 1}"


def check_version_coherence(tag: str | None = None) -> list[str]:
    """Verify that package, fallback, citation, and optional tag versions agree."""
    import ewm_engine

    errors: list[str] = []

    pyproject_data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    fallback_version = (
        pyproject_data.get("tool", {})
        .get("hatch", {})
        .get("version", {})
        .get("raw-options", {})
        .get("fallback_version")
    )
    if not fallback_version:
        errors.append("Missing [tool.hatch.version.raw-options].fallback_version in pyproject.toml")
        fallback_version = "0.0.0"

    citation_data = yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    cff_version = str(citation_data.get("version", ""))
    pref_version = str(citation_data.get("preferred-citation", {}).get("version", ""))
    cff_date = str(citation_data.get("date-released", ""))

    changelog_text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    changelog_date_match = re.search(
        rf"^##\s+\[{re.escape(fallback_version)}\]\s+-\s+(\d{{4}}-\d{{2}}-\d{{2}})",
        changelog_text,
        flags=re.MULTILINE,
    )
    if not changelog_date_match:
        errors.append(
            f"Missing release header '## [{fallback_version}] - YYYY-MM-DD' in CHANGELOG.md"
        )
    else:
        changelog_date = changelog_date_match.group(1)
        if cff_date != changelog_date:
            errors.append(
                f"Release date mismatch for {fallback_version}: CHANGELOG.md ({changelog_date}) != CITATION.cff date-released ({cff_date})"
            )

    init_text = (REPO_ROOT / "src" / "ewm_engine" / "__init__.py").read_text(encoding="utf-8")
    init_fallback_match = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    init_fallback = init_fallback_match.group(1) if init_fallback_match else ""

    runtime_version = str(ewm_engine.__version__)
    runtime_base = _base_semver(runtime_version)

    if fallback_version != cff_version:
        errors.append(
            f"Version mismatch: pyproject.toml fallback_version ({fallback_version}) != CITATION.cff version ({cff_version})"
        )
    if fallback_version != pref_version:
        errors.append(
            f"Version mismatch: pyproject.toml fallback_version ({fallback_version}) != CITATION.cff preferred-citation.version ({pref_version})"
        )
    if init_fallback and init_fallback != fallback_version:
        errors.append(
            f"Version mismatch: src/ewm_engine/__init__.py fallback ({init_fallback}) != pyproject.toml fallback_version ({fallback_version})"
        )

    allowed_bases = {fallback_version}
    if ".dev" in runtime_version:
        allowed_bases.add(_next_patch_semver(fallback_version))
    if runtime_base not in allowed_bases:
        errors.append(
            f"Version mismatch: ewm_engine.__version__ base ({runtime_base} from {runtime_version}) not coherent with fallback_version ({fallback_version})"
        )

    if tag is not None:
        expected = tag.lstrip("v")
        if runtime_version != expected:
            errors.append(
                f"Release tag mismatch: git tag ({tag}) != ewm_engine.__version__ ({runtime_version})"
            )
        if ".dev" in runtime_version or "+" in runtime_version:
            errors.append(
                f"Release version ({runtime_version}) contains '.dev' or '+' local segment."
            )
        if cff_version != expected:
            errors.append(
                f"Release tag mismatch: git tag ({tag}) != CITATION.cff version ({cff_version})"
            )
        if fallback_version != expected:
            errors.append(
                f"Release tag mismatch: git tag ({tag}) != pyproject.toml fallback_version ({fallback_version})"
            )

    return errors


def check_canonical_artifacts() -> list[str]:
    """Verify that committed schemas and canonical scientific artifacts are non-stale."""
    errors: list[str] = []

    # 1. Verify JSON schemas are non-stale
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_schemas.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        errors.append(f"Canonical JSON schemas are stale:\n{res.stdout}\n{res.stderr}")

    # 2. Verify mutation baseline artifact exists and is valid
    mut_baseline = REPO_ROOT / "benchmarks" / "mutation_baseline.json"
    if not mut_baseline.exists():
        errors.append("Missing canonical benchmarks/mutation_baseline.json artifact.")
    else:
        try:
            data = json.loads(mut_baseline.read_text(encoding="utf-8"))
            if data.get("overall", {}).get("mutation_score_pct", 0.0) < 90.0:
                errors.append("Mutation baseline overall score is below 90.0%.")
        except Exception as exc:
            errors.append(f"Invalid benchmarks/mutation_baseline.json: {exc}")

    # 3. Verify reproduction_manifest.json exists and is valid JSON
    repro_manifest = REPO_ROOT / "reproduction_manifest.json"
    if not repro_manifest.exists():
        errors.append("Missing canonical reproduction_manifest.json artifact.")
    else:
        try:
            manifest = json.loads(repro_manifest.read_text(encoding="utf-8"))
            if (
                "pillars" not in manifest
                and "cards" not in manifest
                and "artifacts" not in manifest
            ):
                # Check basic non-empty structure
                if not isinstance(manifest, dict) or not manifest:
                    errors.append("reproduction_manifest.json is empty.")
            ind_card = (
                manifest.get("results", {})
                .get("scientific_cards", {})
                .get("induction_validation", {})
            )
            if isinstance(ind_card, dict) and ind_card:
                for req_key in (
                    "validation_passed",
                    "all_candidates_admitted",
                    "gate_outcome",
                    "card_verification_status",
                    "status_explanation",
                ):
                    if req_key not in ind_card:
                        errors.append(
                            f"reproduction_manifest.json induction_validation missing disambiguating field {req_key!r}."
                        )
        except Exception as exc:
            errors.append(f"Invalid reproduction_manifest.json: {exc}")

    # 3b. Verify newsfragments coverage for ADR-053 and ADR-054
    news_dir = REPO_ROOT / "newsfragments"
    if not news_dir.exists():
        errors.append("Missing newsfragments/ directory.")
    else:
        all_fragments_text = "\n".join(
            p.read_text(encoding="utf-8")
            for p in sorted(news_dir.glob("*.md"))
            if p.name != "README.md"
        )
        for adr_id in ("ADR-053", "ADR-054"):
            if adr_id not in all_fragments_text:
                errors.append(
                    f"Missing Towncrier newsfragment covering {adr_id} in newsfragments/."
                )

    # 4. Verify docs/scientific/claim-evidence-matrix.md exists and all referenced repo paths exist
    claim_matrix = REPO_ROOT / "docs" / "scientific" / "claim-evidence-matrix.md"
    if not claim_matrix.exists():
        errors.append("Missing canonical docs/scientific/claim-evidence-matrix.md artifact.")
    else:
        matrix_text = claim_matrix.read_text(encoding="utf-8")
        # Check relative markdown links in claim-evidence-matrix.md
        for link_match in re.finditer(r"\]\(([^)#]+)\)", matrix_text):
            rel_target = link_match.group(1).strip()
            if rel_target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (claim_matrix.parent / rel_target).resolve()
            if not resolved.exists():
                errors.append(
                    f"Broken link in claim-evidence-matrix.md: {rel_target!r} -> {resolved}"
                )
        # Check inline repo paths like src/..., tests/..., scripts/..., benchmarks/..., docs/...
        for path_match in re.finditer(
            r"`((?:src|tests|scripts|benchmarks|docs|conformance)/[A-Za-z0-9_./\-]+)`",
            matrix_text,
        ):
            repo_rel = path_match.group(1).rstrip("/")
            if not (REPO_ROOT / repo_rel).exists():
                errors.append(
                    f"Non-existent repository path in claim-evidence-matrix.md: {repo_rel!r}"
                )

        # 4b. Anti-renaming & demarcation enforcement on claim-evidence-matrix.md
        demarcation_doc = REPO_ROOT / "docs" / "explanation" / "what-is-an-ewm.md"
        if not demarcation_doc.exists():
            errors.append(
                "Missing scientific demarcation artifact docs/explanation/what-is-an-ewm.md."
            )
        else:
            demarc_text = demarcation_doc.read_text(encoding="utf-8")
            for required_marker in (
                "Working Definition",
                "The Two Guiding Research Questions",
                "Positioning Against Mature Neighbors",
                "When You Do NOT Need an EWM",
                "DependencyEvidenceNature",
                "STATISTICAL_ASSOCIATION",
            ):
                if required_marker not in demarc_text:
                    errors.append(
                        f"Demarcation doc docs/explanation/what-is-an-ewm.md missing required marker {required_marker!r}."
                    )

        anti_renaming_trigger = re.compile(
            r"\b(?:world[- ]models?|novel|first|superior|outperforms|beats)\b",
            re.IGNORECASE,
        )
        for line in matrix_text.splitlines():
            if not line.strip().startswith("| **CLM-"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 8:
                continue
            (
                claim_id,
                claim_col,
                scope_col,
                _maturity,
                evidence_type,
                _impl,
                _tests,
                benchmark_col,
            ) = cells[:8]
            combined_claim = f"{claim_col} {scope_col}"
            if anti_renaming_trigger.search(combined_claim):
                if "what-is-an-ewm.md" not in line:
                    errors.append(
                        f"Anti-renaming violation in {claim_id}: claim uses demarcation-sensitive terms "
                        "('world model', 'novel', 'first', or superiority) without pointing to "
                        "docs/explanation/what-is-an-ewm.md."
                    )
                ev_clean = evidence_type.strip("` ")
                bench_clean = benchmark_col.strip("` ")
                if ev_clean != "UNVERIFIED_RESEARCH" and (
                    not bench_clean or bench_clean.upper().startswith("N/A")
                ):
                    errors.append(
                        f"Anti-renaming violation in {claim_id}: claim uses demarcation-sensitive terms "
                        "without a comparative experiment/benchmark artifact or UNVERIFIED_RESEARCH status."
                    )

    # 5. Verify inline repository paths in docs/REFERENCES.md exist
    for doc_rel in [
        "docs/REFERENCES.md",
    ]:
        doc_path = REPO_ROOT / doc_rel
        if not doc_path.exists():
            errors.append(f"Missing canonical documentation artifact: {doc_rel}")
            continue
        doc_text = doc_path.read_text(encoding="utf-8")
        for path_match in re.finditer(
            r"`((?:src|tests|scripts|benchmarks|docs|conformance)/[A-Za-z0-9_./\-]+)`",
            doc_text,
        ):
            repo_rel = path_match.group(1).rstrip("/")
            if not (REPO_ROOT / repo_rel).exists():
                errors.append(f"Non-existent repository path in {doc_rel}: {repo_rel!r}")

    # 6. Verify enterprise benchmark suite fingerprint consistency across artifact and docs
    suite_json_path = (
        REPO_ROOT / "docs" / "results" / "artifacts" / "enterprise_benchmark_suite.json"
    )
    if not suite_json_path.exists():
        errors.append("Missing docs/results/artifacts/enterprise_benchmark_suite.json.")
    else:
        suite_data = json.loads(suite_json_path.read_text(encoding="utf-8"))
        suite_fp = str(suite_data.get("suite_fingerprint", ""))
        if len(suite_fp) != 64:
            errors.append(
                f"Invalid suite_fingerprint in enterprise_benchmark_suite.json: {suite_fp!r}"
            )
        else:
            for target_rel in [
                "docs/research/enterprise-benchmark-suite.md",
            ]:
                target_path = REPO_ROOT / target_rel
                if target_path.exists() and suite_fp not in target_path.read_text(encoding="utf-8"):
                    errors.append(
                        f"Stale or missing suite_fingerprint ({suite_fp}) in {target_rel}"
                    )

    # 7. Verify flagship fraud cascade artifact and fingerprint consistency across docs
    flagship_json_path = (
        REPO_ROOT / "docs" / "results" / "artifacts" / "fraud_cascade_flagship.json"
    )
    if not flagship_json_path.exists():
        errors.append("Missing docs/results/artifacts/fraud_cascade_flagship.json.")
    else:
        flagship_data = json.loads(flagship_json_path.read_text(encoding="utf-8"))
        repro_block = flagship_data.get("reproducibility", {})
        out_fp = str(repro_block.get("output_fingerprint", ""))
        scen_fp = str(repro_block.get("scenario_fingerprint", ""))
        if len(out_fp) != 64 or len(scen_fp) != 64:
            errors.append(
                f"Invalid fingerprints in fraud_cascade_flagship.json: output={out_fp!r}, scenario={scen_fp!r}"
            )
        else:
            tut_path = REPO_ROOT / "docs" / "tutorials" / "fraud-call-center-cascade.md"
            if tut_path.exists() and out_fp not in tut_path.read_text(encoding="utf-8"):
                errors.append(
                    f"Stale or missing output_fingerprint ({out_fp}) in docs/tutorials/fraud-call-center-cascade.md"
                )
            ref_path = REPO_ROOT / "docs" / "reference" / "fraud-cascade-scenario.md"
            if ref_path.exists() and scen_fp not in ref_path.read_text(encoding="utf-8"):
                errors.append(
                    f"Stale or missing scenario_fingerprint ({scen_fp}) in docs/reference/fraud-cascade-scenario.md"
                )
        if not flagship_data.get("mechanism_recovery_checks", {}).get("all_passed", False):
            errors.append(
                "fraud_cascade_flagship.json mechanism_recovery_checks.all_passed is not True."
            )
        if (
            not flagship_data.get("four_queries", {})
            .get("leakage_certificate", {})
            .get("passed", False)
        ):
            errors.append(
                "fraud_cascade_flagship.json four_queries.leakage_certificate.passed is not True."
            )

    # 8. Verify decision-first comparative evaluation ladder & pre-registration artifacts
    comp_json_path = (
        REPO_ROOT / "docs" / "results" / "artifacts" / "comparative_evaluation_ladder.json"
    )
    comp_md_path = REPO_ROOT / "docs" / "results" / "decision-first-comparative-evaluation.md"
    prereg_json_path = (
        REPO_ROOT / "docs" / "results" / "artifacts" / "pre_registration_comparative_ladder.json"
    )
    if not comp_json_path.exists():
        errors.append("Missing docs/results/artifacts/comparative_evaluation_ladder.json.")
    elif not comp_md_path.exists():
        errors.append("Missing docs/results/decision-first-comparative-evaluation.md.")
    elif not prereg_json_path.exists():
        errors.append("Missing docs/results/artifacts/pre_registration_comparative_ladder.json.")
    else:
        comp_data = json.loads(comp_json_path.read_text(encoding="utf-8"))
        prereg_data = json.loads(prereg_json_path.read_text(encoding="utf-8"))
        comp_fp = str(comp_data.get("report_fingerprint", ""))
        if len(comp_fp) != 64:
            errors.append(
                f"Invalid report_fingerprint in comparative_evaluation_ladder.json: {comp_fp!r}"
            )
        elif comp_fp not in comp_md_path.read_text(encoding="utf-8"):
            errors.append(
                f"Stale or missing report_fingerprint ({comp_fp}) in docs/results/decision-first-comparative-evaluation.md"
            )
        if comp_data.get("pre_registration_id") != prereg_data.get("pre_registration_id"):
            errors.append(
                "Mismatch between comparative_evaluation_ladder.json and pre_registration_comparative_ladder.json pre_registration_id."
            )
        if not comp_data.get("leakage_certificate", {}).get("passed", False):
            errors.append(
                "comparative_evaluation_ladder.json leakage_certificate.passed is not True."
            )

    return errors


def check_security_automation_controls() -> list[str]:
    """Verify security controls are actually enabled in workflow files, not just documented."""
    errors: list[str] = []
    workflows_dir = REPO_ROOT / ".github" / "workflows"

    # 1. Run check_action_pins.py
    pin_res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_action_pins.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if pin_res.returncode != 0:
        errors.append(f"Unpinned GitHub Actions detected:\n{pin_res.stdout}\n{pin_res.stderr}")

    # 2. Check top-level read-only permissions across all workflows
    for wf_path in sorted(workflows_dir.glob("*.yml")):
        wf_data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        perms = wf_data.get("permissions", {})
        if perms != {"contents": "read"} and perms != "read-all":
            errors.append(
                f"Workflow {wf_path.name} does not set least-privilege top-level `permissions: contents: read` (got {perms!r})"
            )

    # 3. Verify required security workflows and controls exist in code
    required_files_and_markers: list[tuple[Path, list[str]]] = [
        (
            workflows_dir / "ci.yml",
            ["step-security/harden-runner@", "pip-audit --strict", "zizmor"],
        ),
        (
            workflows_dir / "vulnerability-scan.yml",
            ["google/osv-scanner-action@", "anchore/scan-action@"],
        ),
        (
            workflows_dir / "codeql.yml",
            ["github/codeql-action/init@", "github/codeql-action/analyze@"],
        ),
        (
            workflows_dir / "scorecard.yml",
            ["ossf/scorecard-action@"],
        ),
        (
            workflows_dir / "release.yml",
            [
                "id-token: write",
                "attestations: write",
                "actions/attest-build-provenance@",
                "anchore/sbom-action@",
                "cyclonedx-py",
                "pypa/gh-action-pypi-publish@",
                "attestations: true",
            ],
        ),
        (
            REPO_ROOT / ".github" / "dependabot.yml",
            ['package-ecosystem: "pip"', 'package-ecosystem: "github-actions"'],
        ),
        (
            REPO_ROOT / "tests" / "security" / "test_no_credentials.py",
            ["SECRET_PATTERNS", "test_no_hardcoded_secrets_or_credentials"],
        ),
        (
            REPO_ROOT / ".pre-commit-config.yaml",
            ["detect-private-key"],
        ),
    ]

    for file_path, markers in required_files_and_markers:
        if not file_path.exists():
            errors.append(f"Missing required security automation file: {file_path}")
            continue
        text = file_path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in text:
                errors.append(
                    f"Security control marker {marker!r} missing from {file_path.relative_to(REPO_ROOT)}"
                )

    return errors


REQUIRED_EVIDENCE_TIERS: tuple[str, ...] = (
    "THEORETICAL",
    "UNIT_VERIFIED",
    "PROPERTY_VERIFIED",
    "CONFORMANCE_VERIFIED",
    "BENCHMARKED",
    "CALIBRATED",
    "EXTERNALLY_REPRODUCED",
    "UNVERIFIED_RESEARCH",
)

REQUIRED_ADR_053_MARKERS: tuple[str, ...] = (
    "ContextSnapshot",
    "EpistemicState",
    "MechanismKind",
    "MechanismProvenance",
    "ObservationFilter",
    "BeliefState",
    "InfeasibleResidualPolicy",
    "DriftOrigin",
    "AuditMetadataContextDiscovery",
    "DependencyEvidenceNature",
)

REQUIRED_ADR_054_FAMILIES: tuple[str, ...] = (
    "A_DirectDynamics",
    "B_HiddenWorkflowCascades",
    "C_RuntimeConfiguration",
    "D_DeploymentShift",
    "E_PartialObservability",
    "F_LongHorizonBusinessOperation",
    "G_MultiAgentEconomy",
    "H_CausalCounterfactualCorrectness",
    "I_ConservationConstraintStress",
    "J_UncertaintyCalibration",
    "K_OODRegimeShift",
    "L_DecisionQuality",
)

RESULT_CARD_MD_FILES: dict[str, str] = {
    "seir": "docs/results/seir-outbreak.md",
    "beer_game": "docs/results/beer-game-bullwhip.md",
    "world3": "docs/results/world3-resource-depletion.md",
    "calibrated_posterior": "docs/results/calibrated-posterior-s04.md",
    "counterfactual_scm": "docs/results/counterfactual-scm-s01.md",
    "decision_discovery": "docs/results/decision-discovery-s02.md",
    "induction_validation": "docs/results/induction-validation-s09.md",
}


def check_claim_evidence_downgrade_safety(
    repo_root: Path = REPO_ROOT,
    matrix_text_override: str | None = None,
    manifest_override: dict[str, Any] | None = None,
) -> list[str]:
    """Verify Claim-to-Evidence Registry completeness and enforce evidence-tier downgrade safety."""
    errors: list[str] = []
    matrix_path = repo_root / "docs" / "scientific" / "claim-evidence-matrix.md"
    if matrix_text_override is not None:
        matrix_text = matrix_text_override
    elif not matrix_path.exists():
        return ["Missing docs/scientific/claim-evidence-matrix.md."]
    else:
        matrix_text = matrix_path.read_text(encoding="utf-8")

    if manifest_override is not None:
        manifest = manifest_override
    else:
        manifest_path = repo_root / "reproduction_manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )

    # 1. Parse Section 1 (Evidence Taxonomy counts)
    declared_tier_counts: dict[str, int] = {}
    for tier in REQUIRED_EVIDENCE_TIERS:
        m = re.search(rf"\|\s*`{re.escape(tier)}`\s*\|\s*[^|]+\|\s*\**(\d+)\**\s*\|", matrix_text)
        if not m:
            errors.append(
                f"Section 1 of claim-evidence-matrix.md is missing taxonomy row for {tier!r}."
            )
        else:
            declared_tier_counts[tier] = int(m.group(1))

    if declared_tier_counts.get("EXTERNALLY_REPRODUCED", 0) != 0:
        errors.append(
            f"EXTERNALLY_REPRODUCED count in Section 1 must be 0 until independent replication exists "
            f"(got {declared_tier_counts.get('EXTERNALLY_REPRODUCED')})."
        )

    # 2. Parse Section 2 (CLM-* rows)
    clm_rows: list[dict[str, str]] = []
    actual_tier_counts: dict[str, int] = dict.fromkeys(REQUIRED_EVIDENCE_TIERS, 0)

    for line in matrix_text.splitlines():
        if not line.strip().startswith("| **CLM-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 11:
            errors.append(
                f"Malformed CLM row in claim-evidence-matrix.md (expected 11 columns, got {len(cells)}): {cells[0] if cells else line}"
            )
            continue
        row = {
            "claim_id": cells[0].strip("* "),
            "claim": cells[1],
            "scope": cells[2],
            "maturity": cells[3].strip("` "),
            "evidence_type": cells[4].strip("` "),
            "implementation": cells[5],
            "tests": cells[6],
            "benchmark": cells[7],
            "citation": cells[8],
            "limitations": cells[9],
            "verification_command": cells[10].strip("` "),
        }
        clm_rows.append(row)
        ev_t = row["evidence_type"]
        if ev_t not in actual_tier_counts:
            errors.append(f"{row['claim_id']} has unknown evidence_type {ev_t!r}.")
        else:
            actual_tier_counts[ev_t] += 1

    if not clm_rows:
        errors.append("No CLM-* rows found in Section 2 of claim-evidence-matrix.md.")
        return errors

    # 3. Verify Section 1 counts == Section 2 counts and EXTERNALLY_REPRODUCED == 0
    if actual_tier_counts.get("EXTERNALLY_REPRODUCED", 0) != 0:
        errors.append(
            f"Section 2 contains {actual_tier_counts['EXTERNALLY_REPRODUCED']} EXTERNALLY_REPRODUCED rows; must be 0."
        )

    for tier, decl_count in declared_tier_counts.items():
        act_count = actual_tier_counts.get(tier, 0)
        if decl_count != act_count:
            errors.append(
                f"Evidence tier count mismatch for {tier}: Section 1 declares {decl_count}, "
                f"but Section 2 has {act_count} rows."
            )

    # 4. Verify README.md claim count matches Section 2 row count (when checking default matrix)
    if matrix_text_override is None:
        readme_path = repo_root / "README.md"
        if readme_path.exists():
            readme_text = readme_path.read_text(encoding="utf-8")
            m_readme = re.search(r"(\d+)-claim \[Claim-to-Evidence Registry\]", readme_text)
            if m_readme and int(m_readme.group(1)) != len(clm_rows):
                errors.append(
                    f"README.md states {m_readme.group(1)}-claim registry, but claim-evidence-matrix.md has {len(clm_rows)} CLM rows."
                )

    # 5. Verify ADR-053 capabilities and ADR-054 families A-L are covered in Section 2
    section2_blob = "\n".join(
        f"{r['claim']} {r['scope']} {r['implementation']} {r['benchmark']}" for r in clm_rows
    )
    for marker in REQUIRED_ADR_053_MARKERS:
        if marker not in section2_blob:
            errors.append(
                f"ADR-053 capability {marker!r} is missing from Section 2 of claim-evidence-matrix.md."
            )
    for fam in REQUIRED_ADR_054_FAMILIES:
        if fam not in section2_blob:
            errors.append(
                f"ADR-054 benchmark family {fam!r} is missing from Section 2 of claim-evidence-matrix.md."
            )

    # 6. Per-row downgrade guard: verify files exist, -k selectors match real tests, and tier floors hold
    path_pat = re.compile(r"(?:src|tests|scripts|benchmarks|docs|conformance)/[A-Za-z0-9_./\-]+")
    sci_cards = manifest.get("results", {}).get("scientific_cards", {})
    adr_layer = manifest.get("results", {}).get("adr_053_054_evidence_layer", {})
    conf_worlds = manifest.get("results", {}).get("conformance_reference_worlds", {})

    for row in clm_rows:
        cid = row["claim_id"]
        tier = row["evidence_type"]
        for req_field in (
            "claim",
            "scope",
            "implementation",
            "tests",
            "benchmark",
            "limitations",
            "verification_command",
        ):
            if not row[req_field].strip():
                errors.append(f"{cid}: empty required column {req_field!r}.")

        # Check all referenced repo paths in implementation, tests, benchmark, verification_command
        for col_name in ("implementation", "tests", "benchmark", "verification_command"):
            for m_path in path_pat.finditer(row[col_name]):
                rel_p = m_path.group(0).rstrip(".,;:`'\")")
                if not (repo_root / rel_p).exists():
                    errors.append(
                        f"{cid} ({tier}): missing {col_name} path {rel_p!r} — tier must be downgraded or evidence restored."
                    )

        # Verify pytest -k selectors against actual test functions in target test files
        vcmd = row["verification_command"]
        if "pytest" in vcmd and "-k " in vcmd:
            k_match = re.search(r'-k\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))', vcmd)
            test_files = [
                m.group(0).rstrip(".,;:`'\")")
                for m in re.finditer(r"tests/[A-Za-z0-9_./\-]+\.py", vcmd)
            ]
            if k_match and test_files:
                k_expr = k_match.group(1) or k_match.group(2) or k_match.group(3) or ""
                k_tokens = [
                    tok.strip().lower()
                    for tok in re.split(r"\s+(?:or|and)\s+", k_expr)
                    if tok.strip()
                ]
                combined_test_src = ""
                for tf in test_files:
                    tf_path = repo_root / tf
                    if tf_path.exists():
                        combined_test_src += "\n" + tf_path.read_text(encoding="utf-8")
                test_names = [
                    fn.lower()
                    for fn in re.findall(
                        r"def\s+(test_[A-Za-z0-9_]+)|class\s+(Test[A-Za-z0-9_]+)",
                        combined_test_src,
                    )
                    for fn in fn
                    if fn
                ]
                for tok in k_tokens:
                    if not any(tok in tname for tname in test_names):
                        errors.append(
                            f"{cid} ({tier}): verification_command -k token {tok!r} matches no test function in {test_files}."
                        )

        # Tier-specific evidence floor checks
        if tier in (
            "UNIT_VERIFIED",
            "PROPERTY_VERIFIED",
            "CONFORMANCE_VERIFIED",
            "BENCHMARKED",
            "CALIBRATED",
        ):
            if "tests/" not in row["tests"]:
                errors.append(
                    f"{cid}: tier {tier} requires an automated test under tests/ (got {row['tests']!r})."
                )

        if tier == "PROPERTY_VERIFIED":
            test_paths = [
                m.group(0) for m in re.finditer(r"tests/[A-Za-z0-9_./\-]+\.py", row["tests"])
            ]
            has_prop = any("tests/property/" in tp for tp in test_paths)
            if not has_prop:
                for tp in test_paths:
                    tp_file = repo_root / tp
                    if tp_file.exists() and (
                        "hypothesis" in tp_file.read_text(encoding="utf-8")
                        or "@given" in tp_file.read_text(encoding="utf-8")
                        or "metamorphic" in tp_file.read_text(encoding="utf-8").lower()
                    ):
                        has_prop = True
                        break
            if not has_prop:
                errors.append(
                    f"{cid}: tier PROPERTY_VERIFIED requires property/metamorphic tests (got {row['tests']!r})."
                )

        elif tier == "CONFORMANCE_VERIFIED":
            if (
                "conformance/" not in row["implementation"]
                and "conformance/" not in row["benchmark"]
            ):
                errors.append(
                    f"{cid}: tier CONFORMANCE_VERIFIED requires conformance/ reference worlds."
                )
            if conf_worlds:
                for w_name, w_rec in conf_worlds.items():
                    if w_rec.get("status") != "PASSED" or not w_rec.get("passed", False):
                        errors.append(
                            f"{cid}: conformance reference world {w_name!r} did not pass in reproduction_manifest.json."
                        )

        elif tier == "BENCHMARKED":
            b_clean = row["benchmark"].strip("` ")
            if not b_clean or b_clean.upper().startswith("N/A"):
                errors.append(
                    f"{cid}: tier BENCHMARKED cannot have benchmark={row['benchmark']!r}; downgrade to UNIT_VERIFIED or provide a benchmark."
                )
            card_m = re.search(r"--card\s+([a-z0-9_]+)", vcmd)
            if card_m and sci_cards:
                c_key = card_m.group(1)
                c_rec = sci_cards.get(c_key, {})
                if c_rec.get("status") != "PASSED":
                    errors.append(
                        f"{cid}: tier BENCHMARKED references card {c_key!r}, whose status in reproduction_manifest.json is {c_rec.get('status')!r} (not 'PASSED')."
                    )
            if "fraud_cascade_flagship" in row["benchmark"] and adr_layer:
                if adr_layer.get("fraud_cascade_flagship", {}).get("status") != "PASSED":
                    errors.append(
                        f"{cid}: fraud_cascade_flagship status is not 'PASSED' in reproduction_manifest.json."
                    )
            if "comparative_evaluation_ladder" in row["benchmark"] and adr_layer:
                if adr_layer.get("comparative_evaluation_ladder", {}).get("status") != "PASSED":
                    errors.append(
                        f"{cid}: comparative_evaluation_ladder status is not 'PASSED' in reproduction_manifest.json."
                    )
            if "enterprise_benchmark_suite" in row["benchmark"] and adr_layer:
                if adr_layer.get("enterprise_benchmark_suite", {}).get("status") != "PASSED":
                    errors.append(
                        f"{cid}: enterprise_benchmark_suite status is not 'PASSED' in reproduction_manifest.json."
                    )
            if "validity_limits_evaluation" in row["benchmark"] and adr_layer:
                if adr_layer.get("validity_limits_evaluation", {}).get("status") != "PASSED":
                    errors.append(
                        f"{cid}: validity_limits_evaluation status is not 'PASSED' in reproduction_manifest.json."
                    )

        elif tier == "CALIBRATED":
            if (
                "calibration" not in row["implementation"]
                and "conformal" not in row["implementation"]
                and "uncertainty" not in row["implementation"]
            ):
                errors.append(
                    f"{cid}: tier CALIBRATED requires a calibration/conformal implementation module."
                )
            card_m = re.search(r"--card\s+([a-z0-9_]+)", vcmd)
            if card_m and sci_cards:
                c_key = card_m.group(1)
                c_rec = sci_cards.get(c_key, {})
                if c_rec.get("status") != "PASSED" or not c_rec.get("coverage_certified", False):
                    errors.append(
                        f"{cid}: tier CALIBRATED card {c_key!r} must have status='PASSED' and coverage_certified=True."
                    )

    return errors


def check_artifact_to_docs_pipeline(
    repo_root: Path = REPO_ROOT,
    manifest_override: dict[str, Any] | None = None,
) -> list[str]:
    """Verify reproduction manifest coverage, card status coherence, and zero hand-pasted/stale numbers in docs."""
    from benchmarks.enterprise_suite import (
        EnterpriseBenchmarkSuiteReport,
        render_enterprise_benchmark_markdown_doc,
    )
    from ewm_engine.cards.models import validate_result_card_dict
    from ewm_engine.flagship.comparative_evaluation import ComparativeEvaluationReport
    from scripts.run_comparative_evaluation import render_comparative_evaluation_markdown

    errors: list[str] = []

    root_manifest_path = repo_root / "reproduction_manifest.json"
    docs_manifest_path = repo_root / "docs" / "results" / "reproduction_manifest.json"

    if manifest_override is not None:
        manifest = manifest_override
    else:
        if not root_manifest_path.exists():
            return ["Missing root reproduction_manifest.json."]
        if not docs_manifest_path.exists():
            return ["Missing docs/results/reproduction_manifest.json."]
        root_text = root_manifest_path.read_text(encoding="utf-8")
        docs_text = docs_manifest_path.read_text(encoding="utf-8")
        if json.loads(root_text) != json.loads(docs_text):
            errors.append(
                "reproduction_manifest.json and docs/results/reproduction_manifest.json are out of sync."
            )
        manifest = json.loads(root_text)

    # 1. Verify reproduction_manifest.json structure, certification status, and fingerprint
    if not manifest.get("verification_checks_passed", False):
        errors.append("reproduction_manifest.json verification_checks_passed is False.")

    git_dirty = bool(manifest.get("provenance", {}).get("git", {}).get("dirty", False))
    cert_status = manifest.get("certification_status")
    if git_dirty and cert_status == "CERTIFIED_RELEASE":
        errors.append(
            "Dirty working tree cannot claim CERTIFIED_RELEASE in reproduction_manifest.json."
        )
    if git_dirty and cert_status != "REPRODUCIBLE_DIRTY_WORKTREE":
        errors.append(
            f"Expected REPRODUCIBLE_DIRTY_WORKTREE for dirty working tree, got {cert_status!r}."
        )

    from ewm_engine.core._canonical import canonical_sha256

    manifest_copy = {k: v for k, v in manifest.items() if k != "reproduction_fingerprint"}
    expected_fp = canonical_sha256(manifest_copy)
    if manifest.get("reproduction_fingerprint") != expected_fp:
        errors.append(
            f"reproduction_manifest.json reproduction_fingerprint mismatch: "
            f"expected {expected_fp}, got {manifest.get('reproduction_fingerprint')!r}."
        )

    # 2. Verify ADR-053/054 evidence layer coverage in reproduction_manifest.json
    adr_layer = manifest.get("results", {}).get("adr_053_054_evidence_layer", {})
    for req_layer in (
        "enterprise_benchmark_suite",
        "fraud_cascade_flagship",
        "comparative_evaluation_ladder",
        "validity_limits_evaluation",
    ):
        if req_layer not in adr_layer:
            errors.append(
                f"reproduction_manifest.json missing adr_053_054_evidence_layer[{req_layer!r}]."
            )
        elif adr_layer[req_layer].get("status") != "PASSED":
            errors.append(
                f"reproduction_manifest.json adr_053_054_evidence_layer[{req_layer!r}] status is not 'PASSED'."
            )

    # 3. Verify canonical_artifacts and visual_artifacts SHA-256 digests against disk
    for rel_path, meta in manifest.get("canonical_artifacts", {}).items():
        disk_p = repo_root / rel_path
        if not disk_p.exists():
            errors.append(f"Missing canonical artifact on disk: {rel_path}")
            continue
        live_sha = hashlib.sha256(disk_p.read_bytes()).hexdigest()
        if live_sha != meta.get("file_sha256"):
            errors.append(
                f"Stale canonical artifact {rel_path}: manifest file_sha256={meta.get('file_sha256')}, disk={live_sha}."
            )

    for svg_name, recorded_sha in manifest.get("visual_artifacts", {}).items():
        svg_file = svg_name if svg_name.endswith(".svg") else f"{svg_name}.svg"
        svg_p = repo_root / "docs" / "results" / "artifacts" / svg_file
        if not svg_p.exists():
            errors.append(f"Missing visual artifact on disk: docs/results/artifacts/{svg_file}")
            continue
        live_sha = hashlib.sha256(svg_p.read_bytes()).hexdigest()
        if live_sha != recorded_sha:
            errors.append(
                f"Stale visual artifact {svg_file}: manifest={recorded_sha}, disk={live_sha}."
            )

    # 4. Verify all 7 scientific result cards pass validate_result_card_dict() and match embedded JSON in docs/results/*.md
    sci_cards = manifest.get("results", {}).get("scientific_cards", {})
    for card_key, rel_md in RESULT_CARD_MD_FILES.items():
        if card_key not in sci_cards:
            errors.append(f"reproduction_manifest.json missing scientific_cards[{card_key!r}].")
            continue
        card_dict = sci_cards[card_key]
        errors.extend(validate_result_card_dict(card_dict, strict=False))

        md_p = repo_root / rel_md
        if not md_p.exists():
            errors.append(f"Missing result card markdown file: {rel_md}")
            continue
        md_text = md_p.read_text(encoding="utf-8")
        json_m = re.search(r"```json\s*\n(\{.*?\})\s*\n```", md_text, flags=re.DOTALL)
        if not json_m:
            errors.append(f"Missing ```json scorecard block in {rel_md}.")
            continue
        try:
            embedded_dict = json.loads(json_m.group(1))
        except Exception as exc:
            errors.append(f"Invalid JSON block in {rel_md}: {exc}")
            continue
        if embedded_dict != card_dict:
            errors.append(
                f"Embedded JSON scorecard in {rel_md} is out of sync with reproduction_manifest.json[{card_key!r}]."
            )

    # 5. Verify auto-generated markdown reports match their canonical JSON artifacts byte-for-byte
    suite_json_p = repo_root / "docs" / "results" / "artifacts" / "enterprise_benchmark_suite.json"
    suite_md_p = repo_root / "docs" / "research" / "enterprise-benchmark-suite.md"
    if suite_json_p.exists() and suite_md_p.exists():
        suite_report = EnterpriseBenchmarkSuiteReport.model_validate_json(
            suite_json_p.read_text(encoding="utf-8")
        )
        expected_suite_md = render_enterprise_benchmark_markdown_doc(suite_report).strip()
        actual_suite_md = suite_md_p.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if expected_suite_md.replace("\r\n", "\n") != actual_suite_md:
            errors.append(
                "docs/research/enterprise-benchmark-suite.md is out of sync with docs/results/artifacts/enterprise_benchmark_suite.json (hand-edited or stale)."
            )

    comp_json_p = (
        repo_root / "docs" / "results" / "artifacts" / "comparative_evaluation_ladder.json"
    )
    comp_md_p = repo_root / "docs" / "results" / "decision-first-comparative-evaluation.md"
    if comp_json_p.exists() and comp_md_p.exists():
        comp_report = ComparativeEvaluationReport.model_validate_json(
            comp_json_p.read_text(encoding="utf-8")
        )
        expected_comp_md = render_comparative_evaluation_markdown(comp_report).strip()
        actual_comp_md = comp_md_p.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if expected_comp_md.replace("\r\n", "\n") != actual_comp_md:
            errors.append(
                "docs/results/decision-first-comparative-evaluation.md is out of sync with docs/results/artifacts/comparative_evaluation_ladder.json (hand-edited or stale)."
            )

    # 6. Verify prose numbers in docs/results/index.md and docs/results/benchmarks.md match backing artifacts
    index_p = repo_root / "docs" / "results" / "index.md"
    if index_p.exists() and adr_layer:
        index_text = index_p.read_text(encoding="utf-8")
        flag_regret = adr_layer.get("fraud_cascade_flagship", {}).get("myopic_decision_regret_usd")
        if flag_regret is not None and f"{flag_regret:,.2f}" not in index_text:
            errors.append(
                f"docs/results/index.md missing flagship myopic regret {flag_regret:,.2f} from fraud_cascade_flagship.json."
            )
        mae_red = adr_layer.get("comparative_evaluation_ladder", {}).get(
            "rung1_to_rung3_mae_reduction_pct"
        )
        if mae_red is not None and f"{mae_red:.2f}%" not in index_text:
            errors.append(
                f"docs/results/index.md missing comparative ladder MAE reduction {mae_red:.2f}% from comparative_evaluation_ladder.json."
            )
        perf_erosion = adr_layer.get("validity_limits_evaluation", {}).get(
            "performative_utility_erosion_usd"
        )
        if perf_erosion is not None and f"{perf_erosion:,.2f}" not in index_text:
            errors.append(
                f"docs/results/index.md missing performative utility erosion {perf_erosion:,.2f} from reproduction_manifest.json."
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify release version truth, artifact freshness, and security automation."
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional release git tag (e.g. v1.5.0) to verify against package & citation versions",
    )
    args = parser.parse_args()

    all_errors: list[str] = []
    all_errors.extend(check_version_coherence(tag=args.tag))
    all_errors.extend(check_canonical_artifacts())
    all_errors.extend(check_claim_evidence_downgrade_safety())
    all_errors.extend(check_artifact_to_docs_pipeline())
    all_errors.extend(check_security_automation_controls())

    if all_errors:
        print("[FAIL] Release & Security Truth Gate failed:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        "[OK] Release version coherence, canonical artifact freshness, claim-evidence downgrade safety, artifact-to-docs sync, and security automation verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
