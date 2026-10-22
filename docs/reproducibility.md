# Research Reproducibility & Citation Guide

> **Academic Rigor & Software Heritage Spine**  
> Discipline: Research-software credibility, reproducibility checklists, persistent identifiers, and open science standards.  
> Standards: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), [FORCE11 Software Citation Principles](https://doi.org/10.7717/peerj-cs.86), [CITATION.cff v1.2.0](https://citation-file-format.github.io/), [Zenodo](https://zenodo.org/), [Software Heritage](https://www.softwareheritage.org/).

The **Enterprise World Model Engine (`ewm-engine`)** is engineered from the ground up for scientific rigor and reproducibility. This document outlines the academic citation protocols, persistent identifiers, archival mechanisms, and the reproducibility checklist implemented across the engine.

---

## 1. Academic Citation & Persistent Identifiers

### Native GitHub Citation (`CITATION.cff`)
`ewm-engine` includes a machine-readable [`CITATION.cff`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/CITATION.cff) file at the repository root compliant with the Citation File Format (v1.2.0) schema. GitHub uses this file to provide automated, one-click citations in both **APA** and **BibTeX** formats via the **"Cite this repository"** button on the main repository page.

#### BibTeX Citation
```bibtex
@software{carida2026ewmengine,
  author    = {Caridá, Vinicius},
  title     = {Enterprise World Model Engine: An Open Framework for Modeling, Simulating, and Evaluating Organizational Dynamics},
  year      = {2026},
  url       = {https://github.com/vfcarida/Enterprise-World-Model-Engine},
  version   = {1.5.0},
  license   = {Apache-2.0}
}
```

---

### Zenodo DOI: Concept DOI vs. Version DOI

To ensure permanent archival and academic traceability, `ewm-engine` integrates with **Zenodo** through automated GitHub release webhooks, configured via [`.zenodo.json`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.zenodo.json):

```
       Concept DOI (All Versions)
     ┌────────────────────────────┐
     │ 10.5281/zenodo.<concept>   │ ──► Always resolves to the latest release
     └──────────────┬─────────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌──────────────────┐  ┌──────────────────┐
│  Version v1.0.0  │  │  Version v1.1.0  │
│ 10.5281/zenodo.1 │  │ 10.5281/zenodo.2 │ ──► Fixed, immutable snapshot for citations
└──────────────────┘  └──────────────────┘
```

1. **Concept DOI (`10.5281/zenodo.<concept_id>`)**:
   - Represents the software project conceptually across all releases (*not yet published / to be assigned upon Zenodo release deposition*).

2. **Version DOI (`10.5281/zenodo.<version_id>`)**:
   - Represents a specific, immutable release tarball (*not yet published / to be assigned upon Zenodo release deposition*).

---

### Software Heritage Archival (SWHID - ISO/IEC 18670:2025)

While Zenodo stores release tarballs, [Software Heritage](https://www.softwareheritage.org/) archives the Git development graph, tracking individual commits, trees, and blobs via cryptographic **SoftWare Heritage Identifiers (SWHIDs)** conforming to ISO/IEC 18670:2025 (*public SWHID not yet assigned; see local verification instructions in [Software Heritage Guide](scientific/swhid.md)*).

- **Trigger Archival**: The repository can be submitted via the [Software Heritage Save Code Now](https://archive.softwareheritage.org/save/) service using the repository URL:
  ```text
  https://github.com/vfcarida/Enterprise-World-Model-Engine
  ```
- **Referencing via SWHID**: Specific versions follow the standard intrinsic syntax:
  ```text
  swh:1:rev:<git_commit_sha1>;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine
  ```
  See the dedicated [Software Heritage Guide](scientific/swhid.md) for verification procedures.

---

### Companion Manuscripts (In-Tree Drafts — Not Yet Published)

EWM Engine includes two manuscript drafts in-tree (*neither has been assigned an external DOI or arXiv identifier yet*):

1. **Methods & Benchmark Companion Manuscript**:
   - Title: *Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty*
   - Status: In-tree draft (*arXiv identifier not yet published / to be assigned upon submission*)
   - Draft manuscript: `paper/arxiv/paper.md` and LaTeX source `paper/arxiv/paper.tex`.
   - Checklist: `paper/arxiv/NEURIPS_CHECKLIST.md`.

2. **Journal of Open Source Software (JOSS) Manuscript**:
   - Paper manuscript: `paper/paper.md`
   - Bibliography: `paper/paper.bib`
   - Status: In-tree draft (*JOSS DOI not yet published / to be assigned upon review*).

---

### Replication Protocols & Claim-to-Evidence Registry

- **Claim-to-Evidence Registry**: All public claims, evidence tiers (`THEORETICAL`, `UNIT_VERIFIED`, `PROPERTY_VERIFIED`, `CONFORMANCE_VERIFIED`, `BENCHMARKED`, `CALIBRATED`, `EXTERNALLY_REPRODUCED`, `UNVERIFIED_RESEARCH`), and verification commands are cataloged in [`docs/scientific/claim-evidence-matrix.md`](scientific/claim-evidence-matrix.md).
- **Replication Templates (Prepared for External Review — Not Yet Externally Replicated)**:
  - MLRC protocol template: [`docs/scientific/mlrc_2026_replication_pkg.md`](scientific/mlrc_2026_replication_pkg.md).
  - ReScience C template: [`docs/scientific/rescience_c_submission.md`](scientific/rescience_c_submission.md).
  - ACM Artifact Appendix template: [`docs/scientific/acm_artifact_appendix.md`](scientific/acm_artifact_appendix.md).
  - Pre-registered study protocol: [`docs/scientific/registered_report_protocol.md`](scientific/registered_report_protocol.md).

---

### Run-Manifest Certification Semantics

`scripts/reproduce.py` enforces a strict distinction between clean immutable releases and modified working trees via `certification_status` in `reproduction_manifest.json`:

| Certification State | Criteria | Meaning |
| :--- | :--- | :--- |
| `CERTIFIED_RELEASE` | All 9 checks pass **and** `git status --porcelain` is clean (`dirty == False`) | Reproducible run from an unmodified committed revision |
| `REPRODUCIBLE_DIRTY_WORKTREE` | All 9 checks pass, but `dirty == True` | Numerical checks pass, but working tree has uncommitted modifications (`source_tree_sha256` recorded) |
| `LOCAL_DEVELOPMENT` | Subset of checks run or git metadata unavailable | Partial / local development execution |
| `NON_REPRODUCIBLE` | One or more verification checks failed (`--require-clean` or numerical failure) | Verification failed |

---

### Single-Command Reproduction Layer

Anyone, on any machine, reproduces our published numbers and fingerprints via:

| Method | Environment | Command / Link | Output |
| :--- | :--- | :--- | :--- |
| **Interactive Jupyter** | In-Browser Binder | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) | Interactive verification & SVG figures |
| **Cloud Codespaces** | GitHub Cloud | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) | Complete dev container |
| **Local Terminal** | Python 3.10–3.12 | `make reproduce` | `reproduction_manifest.json` |
| **Pinned Docker** | Docker Engine | `docker run --rm -v $(pwd):/app ewm-engine:pinned` | Verified run-manifest |

---

## 2. Reproducibility Checklist (NeurIPS & Papers-with-Code Standards)

To guarantee that simulations and scientific evaluations conducted with `ewm-engine` are fully verifiable, the framework adheres strictly to the following reproducibility rubric:

| Category | Requirement | EWM Engine Implementation | Verification Gate |
| :--- | :--- | :--- | :--- |
| **Dependencies** | Complete specification of execution environment and dependencies | [`RunManifest.dependency_versions`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/provenance/manifest.py), `uv.lock`, hardware summary | Automated check in `compute_readiness_report` |
| **PRNG Seeding** | Explicit random number generator lifecycle and thread isolation | [`seed_everything()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/seed.py), `np.random.SeedSequence` hierarchical spawning, `OMP_NUM_THREADS=1` | Zero global state; bitwise trajectory equivalence |
| **State Immutability** | Cryptographic state provenance preventing mutations | [`WorldState`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/state.py) deep immutability (`frozen=True`), canonical SHA-256 fingerprinting | Property tests with Hypothesis |
| **Statistical Reporting** | No bare point estimates; distribution reporting | [`evaluate_multiseed()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/multiseed.py) with percentile bootstrap confidence intervals (mean ± 95% CI) | Enforced in `MultiSeedReport` |
| **Rollout Drift** | Measurement of compounding error over simulation horizons | [`RolloutDivergenceResult.compounding_error_ratios`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/experimental/dynamics_eval.py) ($C(h) = \text{err}(h) / (h \cdot \text{err}(1))$, Talvitie 2014) | [`select_reliable_horizon()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/experimental/dynamics_eval.py) safety check |
| **Probabilistic UQ** | Proper scoring rules and distribution-free intervals | CRPS, Gaussian log score/NLL, Brier score, adaptive calibration error (ACE), split conformal prediction | Calibration consistency gate |
| **Causal Honesty** | Gated causal estimation without heuristic over-promotion | Backdoor/frontdoor DAG identifiability check, Doubly Robust OPE with mandatory ESS and clip diagnostics | Block on non-identifiable; no auto-upgrade of `EvidenceLevel` |
| **Sensitivity** | Robustness to unobserved confounding | VanderWeele & Ding E-values attached to estimates and CI bounds; common-support overlap checks ($d \ge 5$ warning) | Refutation battery (placebo, random common cause, subset stability) |
| **ML Production Readiness** | Systematic testing across data, models, infrastructure, monitoring | Google ML Test Score ($\min(\text{Data}, \text{Model}, \text{Infra}, \text{Monitoring})$) | [`compute_readiness_report()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/experimental/readiness.py) |
| **Metamorphic Invariants** | Code change regression auditing | Invariant transformation checks using business rules as the oracle | [`run_metamorphic_regression_gate()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/experimental/readiness.py) |

---

## 3. Practical Reproducibility Example

The following code illustrates how to generate a canonical, self-fingerprinting `RunManifest` and execute multi-seed evaluations with rigorous statistical reporting:

```python
from ewm_engine.core.seed import seed_everything
from ewm_engine.provenance.manifest import create_run_manifest
from ewm_engine.evaluation.multiseed import evaluate_multiseed
from ewm_engine.evaluation.uncertainty import compute_mean_crps, compute_adaptive_calibration_error

# 1. Enforce bitwise determinism across Python, NumPy, and BLAS runtimes
seed_everything(seed=42, deterministic=True)

# 2. Record an auditable run manifest mapping 1:1 to the NeurIPS checklist
manifest = create_run_manifest(
    engine_version="1.5.0",
    seed=42,
    scenario_id="supply_chain_shock",
    initial_state_fingerprint="sha256_canonical_state_fingerprint",
    component_versions={"inventory_dynamics": "1.5.0", "capacity_constraint": "1.5.0"},
    determinism_flags={"deterministic": True, "single_threaded_blas": True},
    execution_duration_sec=1.42,
)

print(f"Canonical Run Manifest Hash: {manifest.manifest_hash}")
print(f"Reproducibility Checklist Items Passed: {len(manifest.reproducibility_checklist)}")


# 3. Multi-seed evaluation reporting mean ± 95% bootstrap confidence intervals
def simulate_loss(rng, seed):
    # Deterministic simulation function per seed
    return float(rng.normal(loc=100.0, scale=12.0))


report = evaluate_multiseed(
    run_fn=simulate_loss,
    metric_name="downstream_loss",
    base_seed=42,
    n_seeds=10,
    confidence_level=0.95,
)

metric = report.metrics["downstream_loss"]
print(
    f"Downstream Loss: {metric.mean:.2f} (95% CI: [{metric.ci_lower:.2f}, {metric.ci_upper:.2f}])"
)
```

---

## 4. Summary of Epistemic Invariants

1. **No Bare Point Estimates**: Evaluations must report variability (standard deviation, interquartile range, or bootstrap confidence intervals).
2. **No Auto-Upgraded Evidence**: Simulations produce dependency traces with `EvidenceLevel.ASSUMED` or `EMPIRICAL_SIMULATED`. Causal claims require formal DAG identifiability and pass-through of sensitivity E-values.
3. **Compounding Horizon Safety**: Multi-step rollouts must measure compounding drift $C(h)$. Receding-horizon planners should never plan past horizons where compounding drift exceeds linear scaling.
