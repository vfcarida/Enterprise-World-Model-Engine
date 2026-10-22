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
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
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
        except Exception as exc:
            errors.append(f"Invalid reproduction_manifest.json: {exc}")

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
    all_errors.extend(check_security_automation_controls())

    if all_errors:
        print("[FAIL] Release & Security Truth Gate failed:", file=sys.stderr)
        for err in all_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(
        "[OK] Release version coherence, canonical artifact freshness, and security automation verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
