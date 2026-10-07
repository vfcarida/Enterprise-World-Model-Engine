# Release Checklist

**Release Target:** vX.Y.Z  
**Release Manager:** @vfcarida  
**Standard Authority:** SemVer 2.0.0, PEP 440, NEP-23, SPEC 0, SLSA L2+, PEP 740

---

## 1. Quality & Pre-Release Gates
- [ ] All 20 required CI quality gates green on `main` (Lint, Format, Mypy Strict, Zizmor, Pinning, Towncrier, cffconvert, pip-audit, Packaging, Unit Matrix, Property, Integration, Contract, Security, Examples, Architecture, Package Build, Docs Build, Memray, Numerical).
- [ ] Strict type checking passes with zero errors: `uv run mypy --strict src tests examples benchmarks`.
- [ ] Zizmor GitHub Actions security scan clean: `uvx zizmor --min-severity medium .github/workflows .github/actions`.
- [ ] Action SHA pinning verified: `uv run python scripts/check_action_pins.py`.
- [ ] SPEC 0 version floors verified: `uv run pytest tests/packaging/test_spec0_compliance.py -v`.
- [ ] Packaging tests pass: `uv run pytest tests/packaging/ -v`.
- [ ] Deprecation warning compliance: verify all deprecated APIs emit `DeprecationWarning` with `stacklevel=2` and have test coverage in `tests/unit/test_deprecation.py`.
- [ ] Memory leaks and allocation ceilings verified: `uv run pytest tests/performance/test_memory_gates.py --memray -v`.

---

## 2. Changelog & Documentation Truth
- [ ] Towncrier news fragments compiled into `CHANGELOG.md`:
  ```bash
  uv run towncrier build --version X.Y.Z
  ```
- [ ] Review compiled `CHANGELOG.md` diff; verify consumed fragments in `newsfragments/` are staged for removal.
- [ ] Synchronize version in `CITATION.cff` (`version: X.Y.Z`).
- [ ] Verify `.release-please-manifest.json` reflects target version `X.Y.Z`.
- [ ] Documentation builds with zero warnings in strict mode: `uv run mkdocs build --strict`.
- [ ] Verify `ROADMAP.md` and `README.md` maturity tables match reality.

---

## 3. Dynamic Version Truth & Build Verification
- [ ] Verify clean dynamic versioning (Hatch-VCS):
  - Ensure working tree is clean (`git status`).
  - Verify `no-local-version` scheme produces no `.dev` or `+local` segments on release tags.
- [ ] Build distribution packages locally:
  ```bash
  uv run python -m build
  ```
- [ ] Inspect wheel contents for PEP 561 `py.typed` and generated `_version.py`:
  ```bash
  uv run pytest tests/packaging/test_wheel_contents.py -v
  ```
- [ ] Validate distribution metadata with twine:
  ```bash
  uv run twine check dist/*
  ```

---

## 4. Release Execution & Supply Chain Attestations
- [ ] Commit compiled changelog and metadata:
  ```bash
  git commit -m "chore(release): prepare release vX.Y.Z"
  git push origin main
  ```
- [ ] Create and push signed annotated git tag:
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  git push origin vX.Y.Z
  ```
- [ ] Monitor `.github/workflows/release.yml` execution:
  - [ ] **Version Truth Gate**: asserts `version == tag` and rejects `.dev`/`+local`.
  - [ ] **Dual SBOMs**: CycloneDX JSON (`dist/cyclonedx-bom.json`) and Syft SPDX JSON (`dist/syft-bom.spdx.json`).
  - [ ] **SLSA L2+ Build Provenance**: in-toto provenance attested via `actions/attest-build-provenance`.
  - [ ] **PyPI Trusted Publishing**: published via OIDC with PEP 740 Sigstore digital attestations (`attestations: true`).
  - [ ] **GitHub Release**: published with release assets, SBOMs, and provenance.

---

## 5. Post-Release & Academic Archival
- [ ] **Zenodo DOI Minting**: Verify GitHub release hook triggered Zenodo deposit per `.zenodo.json`; confirm Version DOI and Concept DOI.
- [ ] **Software Heritage Archival**: Trigger archive snapshot on [Software Heritage Save Code Now](https://archive.softwareheritage.org/save/).
- [ ] **Documentation Deployment**: Verify `.github/workflows/docs.yml` deployed documentation to GitHub Pages.
- [ ] Verify PyPI package installs in clean environment:
  ```bash
  uv pip install --no-cache-dir ewm-engine==X.Y.Z
  python -c "import ewm_engine; print('Deployed:', ewm_engine.__version__)"
  ```
