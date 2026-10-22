# ML Reproducibility Challenge (MLRC 2026) Replication Package

> **Venue**: ML Reproducibility Challenge 2026 (Replication Protocol Template)
> **Target Manuscript Draft**: *Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty* (`paper/arxiv/paper.md` â€” *arXiv ID not yet published*)
> **Source Repository**: `https://github.com/vfcarida/Enterprise-World-Model-Engine`
> **Primary Contact**: Vinicius CaridÃ¡ (`vfcarida@gmail.com`)
> **External Archival**: Zenodo & Software Heritage (*not yet published / to be assigned upon release archival*)

---

## 1. Executive Summary for Independent Reviewers

This replication package is designed for independent reviewers participating in the **ML Reproducibility Challenge 2026 (MLRC 2026)**. Unlike empirical machine learning papers that rely on non-deterministic GPU cluster training with wide variance, the **Enterprise World Model Engine (EWM)** is built on deterministic, seed-isolated pseudo-random number generator (PRNG) trees and algebraic conservation layers.

**Headline Reproduction Claim:**
> *Any reviewer, executing our reproduction harness on any modern x86_64 or ARM64 architecture, will reproduce all seven empirical validation cards, all three WMIF reference conformance worlds, and cross-backend SoA state tensor bit-identity, matching our exact published SHA-256 state fingerprints.*

---

## 2. Specific Claims to Verify

Reviewers are asked to evaluate the following empirical and methodological claims:

| Claim ID | System / Benchmark | Hypothesis / Invariant | Expected Published Metric | Allowable Tolerance |
| :--- | :--- | :--- | :--- | :--- |
| **Claim 1** | SEIR Outbreak (Pillar 1) | Mass balance ($S+E+I+R = N$) | Max residual $= 4.54 \times 10^{-13}$ | $\le 10^{-12}$ (IEEE-754 tolerance) |
| **Claim 2** | Beer Game (Pillar 2) | Multi-tier inventory conservation | Bullwhip Ratio $= 3.80$ | Exact $3.80$ |
| **Claim 3** | WORLD3 (Pillar 3) | Non-linear feedback loop stability | Industrial inflection $t=99$ | Step $t=99$ |
| **Claim 4** | SBI Calibration (Pillar 4) | Posterior calibration via TARP | TARP Error $= 0.028 \le 0.05$ | $\le 0.05$ (Calibrated) |
| **Claim 5** | Twin-Rollout SCM (Pillar 5) | Counterfactual noise coupling | Net ATE $= +35.0$ units | Exact $+35.0$ |
| **Claim 6** | PRIM Discovery (Pillar 6) | Vulnerability subspace purity | Density $= 91.2\%$, Recall $= 80\%$ | $\pm 0.5\%$ |
| **Claim 7** | Induction Honesty (Pillar 7) | Fail-closed refusal under confounding | Unqualified claim blocked | Exception raised |
| **Claim 8** | WMIF Conformance | Reference suite execution | 3 / 3 models passed | 100% conformance |
| **Claim 9** | Cross-Backend SoA | Batched state bit-identity | SHA-256 identical to OOP state | Bit-for-bit exact |

---

## 3. Reproduction Protocols (Four Options)

Reviewers may choose any of the following four execution pathways depending on their available tooling:

### Option A: Zero-Install In-Browser via Binder (Recommended for Fast Review)
1. Open the interactive Jupyter notebook directly in your browser:
   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb)
2. In the JupyterLab menu, select **Run $\to$ Run All Cells**.
3. Verify that the final execution cell prints `ALL REPRODUCTION CHECKS PASSED` and renders the validation scorecard table and vector SVG artifacts.

### Option B: Cloud Dev Environment via GitHub Codespaces
1. Open GitHub Codespaces:
   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine)
2. The environment automatically builds from `.devcontainer/devcontainer.json` and runs `python scripts/reproduce.py`.
3. Check the terminal output to inspect the green verification table.

### Option C: Isolated Pinned Docker Container
To reproduce in an air-gapped, isolated OCI container:
```bash
# Build or pull the pinned Docker container
docker build -t ewm-engine:pinned -f Dockerfile .

# Execute reproduction harness and extract manifest
docker run --rm -v $(pwd):/app ewm-engine:pinned python scripts/reproduce.py --manifest /app/reproduction_manifest.json
```

### Option D: Local Shell with `uv` or Standard Virtualenv
```bash
# 1. Clone repository
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine

# 2. Setup environment (Python 3.10, 3.11, or 3.12)
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -e ".[all,dsl]"

# 3. Execute one-command reproduction
make reproduce
# Or equivalently:
python scripts/reproduce.py
```

---

## 4. Output Inspection & Verification Rubric

Upon completion, `scripts/reproduce.py` outputs a cryptographic JSON run-manifest to `reproduction_manifest.json`.

### Reviewer Checklist:
- [ ] **Exit Code**: The reproduction command exits with return code `0`.
- [ ] **Host Environment Provenance**: Verify that `reproduction_manifest.json` logs your local OS, Python version, NumPy version, and BLAS provider (`openblas`, `mkl`, or `accelerate`).
- [ ] **Lockfile Hash**: Verify that `uv_lock_sha256` matches the committed `uv.lock`.
- [ ] **Card Fingerprints**: Verify that the generated SHA-256 hashes in `results` match the committed references.
- [ ] **Vector Visuals**: Inspect `docs/results/artifacts/ground_truth_overlay.svg`, `calibration_pit.svg`, and `multiseed_error_bars.svg` to confirm they reflect dynamic execution data.

---

## 5. Reviewer Report Template (MLRC Submission)

When filing your replication report to the MLRC OpenReview portal, you may use the following structure:

```markdown
# Replication Report: Structural World Models (EWM Engine)

## 1. Summary of Results
- Hardware Platform: [e.g., Apple M3 / AMD Ryzen 9 / GitHub Codespaces 4-core]
- OS & Python: [e.g., Linux 6.5 / Python 3.11.9]
- Reproduction Option Used: [Binder / Codespaces / Docker / Local]
- Overall Status: [SUCCESS / PARTIAL / FAIL]

## 2. Verification of Specific Claims
- Claim 1 (SEIR Mass Balance): [Confirmed residual <= 1e-12 / Failed]
- Claim 2 (Beer Game Bullwhip): [Confirmed 3.80 / Failed]
- Claim 3 (WORLD3 Inflection): [Confirmed t=99 / Failed]
- Claim 4 (SBI TARP Calibration): [Confirmed Error <= 0.05 / Failed]
- Claim 5 (Twin-Rollout SCM): [Confirmed ATE = +35.0 / Failed]
- Claim 6 (PRIM Scenario Discovery): [Confirmed Density 91.2% / Failed]
- Claim 7 (Induction Refusal Gate): [Confirmed Blocked on Confounding / Failed]
- Claim 8 (WMIF Conformance): [Confirmed 3/3 passed / Failed]
- Claim 9 (SoA Bit-Identity): [Confirmed exact SHA-256 match / Failed]

## 3. General Feedback & Discussion
[Reviewer notes on ease of setup, documentation clarity, and robustness to environment changes]
```
