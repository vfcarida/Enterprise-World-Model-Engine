# ACM Artifact Review & Badging Appendix

> **Framework**: ACM Artifact Review and Badging (v1.1 â€” Readiness Appendix Template)
> **Associated Manuscript Draft**: *Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty* (`paper/arxiv/paper.md` â€” *arXiv ID not yet published*)
> **Target Badges (Prepared for Committee Evaluation â€” Not Yet Awarded)**:
> 1. **Artifacts Available** (*pending public Zenodo/Software Heritage release archival*)
> 2. **Artifacts Evaluated - Functional & Reusable**
> 3. **Results Reproduced**

---

## 1. Artifact Identification & Availability (Badge Target: Artifacts Available)

EWM Engine prepares the following archival metadata for **Artifacts Available** evaluation:

- **Archival Repositories**:
  - **Zenodo Metadata**: Configured in [`.zenodo.json`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.zenodo.json) (*Concept and Version DOIs not yet published / to be assigned upon release deposition*).
  - **Software Heritage SWHID**: Archival procedure documented in [`docs/scientific/swhid.md`](swhid.md) (*SWHID not yet published / to be assigned upon public ingestion*).
- **License**: Apache License 2.0 (Open Source Initiative approved).
- **Public Git Forge**: [https://github.com/vfcarida/Enterprise-World-Model-Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine)

---

## 2. Artifact Quality & Reusability (Badge: Evaluated - Functional + Reusable)

EWM Engine satisfies the **Functional & Reusable** criteria:

### A. Completeness & Packaging
- **Standard Packaging**: Fully compliant with PEP 517/518 and PEP 621 via `pyproject.toml`.
- **Pinned Lockfile**: Distributed with `uv.lock` pinning transitive dependencies and cryptographic wheel hashes.
- **Multi-Platform Support**: Tested on Linux (x86_64, aarch64), macOS (Apple Silicon), and Windows 11.
- **Zero Heavy Core Dependencies**: Pure-Python / NumPy core with zero mandatory deep learning frameworks or LLM APIs.

### B. Documentation & Type Safety
- **Strict Typing**: Passes `mypy --strict` with zero typing errors across the entire core API surface.
- **Standardized Architecture Records**: 52 Architecture Decision Records (ADRs) documenting design rationale.
- **Executable Tutorials**: DiÃ¡taxis-structured tutorials and how-to guides validated via automated snippet tests (`scripts/test_doc_snippets.py`).

### C. Testing Rigor
- **Test Suite**: Multi-tier testing suite (`tests/unit/`, `tests/contract/`, `tests/property/`, `tests/integration/`) with gated test coverage $\ge 85\%$.
- **Property-Based Testing**: Validates state immutability and fingerprint uniqueness via Hypothesis.
- **SLSA Provenance**: Conforms to SLSA Build Level 2+ with cryptographic SBOMs.

---

## 3. Independent Reproduction Guide (Badge: Results Reproduced)

The committee can reproduce all experimental claims with a single command.

### System Requirements
- **Hardware**: Standard 2-core x86_64 or ARM64 workstation (no GPU required).
- **RAM**: Minimum 4 GB RAM.
- **Disk**: 1 GB free disk space.
- **Time**: ~15â€“30 seconds for full reproduction.

### Step-by-Step Execution

#### Method 1: Local Shell Execution
```bash
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine
pip install -e ".[all,dsl]"
make reproduce
```

#### Method 2: One-Click Cloud / In-Browser
- **Binder**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) [Launch Interactive Notebook](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb)
- **GitHub Codespaces**: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) [Launch Cloud Dev Container](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine)

#### Method 3: Containerized Docker Run
```bash
docker build -t ewm-engine:pinned -f Dockerfile .
docker run --rm -v $(pwd):/app ewm-engine:pinned python scripts/reproduce.py --manifest /app/reproduction_manifest.json
```

---

## 4. Verification Oracle & Scorecard

During reproduction, `scripts/reproduce.py` evaluates:

1. **Card 1 (SEIR Outbreak)**: Simulates 80 steps of Kermack-McKendrick compartmental ODE; asserts maximum mass-balance drift $\le 10^{-12}$.
2. **Card 2 (Beer Distribution Game)**: Simulates 4-tier supply chain; asserts exact bullwhip ratio of $3.80$.
3. **Card 3 (WORLD3 Global Dynamics)**: Simulates Meadows resource limits; asserts inflection point occurs at step $t=99$.
4. **Card 4 (SBI Calibration)**: Runs TARP coverage test over amortized posterior; asserts TARP error $\le 0.05$.
5. **Card 5 (Twin-Rollout Counterfactual SCM)**: Abduces exogenous noise under baseline and counterfactual intervention; asserts net treatment effect $\Delta \tau = +35.0$.
6. **Card 6 (Decision Discovery)**: Runs PRIM scenario discovery; asserts subspace density $\ge 90\%$ and recall $\ge 80\%$.
7. **Card 7 (Induction Honesty)**: Tests world-model induction against unobserved confounding; asserts fail-closed refusal exception.
8. **WMIF Reference Suite**: Executes 3 reference models (`ref_seed_trivial`, `ref_closed_form_stochastic`, `ref_constraint_stress`); asserts 100% compliance.
9. **SoA State Tensor**: Compares Structure-of-Arrays state serialization against OOP state; asserts bitwise SHA-256 fingerprint identity.

All results are hashed and written to `reproduction_manifest.json`.
