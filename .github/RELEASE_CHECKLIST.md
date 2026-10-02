# Release Checklist

**Release Candidate:** vX.Y.Z  
**Release Manager:** @handle  

---

## 1. Quality & Test Gates
- [ ] Full test suite passes: `uv run pytest -q`
- [ ] Strict type checking passes: `uv run mypy src tests`
- [ ] Code formatting & linting clean: `uv run ruff check src tests && uv run ruff format --check src tests`
- [ ] Import boundaries verified: `uv run pytest tests/architecture/`
- [ ] Contract tests verified: `uv run pytest tests/contract/`
- [ ] Benchmark regression verified: `uv run python benchmarks/run_benchmarks.py`

## 2. Documentation & Compliance
- [ ] Documentation builds with zero warnings: `uv run mkdocs build --strict`
- [ ] `CHANGELOG.md` updated with all changes, deprecations, and fixes
- [ ] `CITATION.cff` and `pyproject.toml` version metadata synchronized
- [ ] No forbidden marketing claims present in docs or release notes

## 3. Packaging & Supply Chain
- [ ] Package builds successfully: `uv build`
- [ ] Wheel and sdist verified: `uv run twine check dist/*`
- [ ] Clean install in fresh virtual environment: `pip install dist/*.whl`
- [ ] CLI smoke test verified: `ewm --version` and `ewm-engine --version`

## 4. Release Execution
- [ ] Git tag created: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
- [ ] Tag pushed to remote: `git push origin vX.Y.Z`
- [ ] GitHub Release drafted with release notes and changelog
- [ ] Packages published to PyPI via trusted publishing / OIDC
