# Research Reproducibility & Citation Guide

> **Academic Rigor & Software Heritage Spine**  
> Discipline: Research-software credibility, reproducibility checklists, persistent identifiers, and open science standards.  
> Standards: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), [FORCE11 Software Citation Principles](https://doi.org/10.7717/peerj-cs.86), [CITATION.cff v1.2.0](https://citation-file-format.github.io/), [Zenodo](https://zenodo.org/), [Software Heritage](https://www.softwareheritage.org/).

The **Enterprise World Model Engine (`ewm-engine`)** is engineered from the ground up for scientific rigor and reproducibility. This document outlines the academic citation protocols, persistent identifiers, archival mechanisms, and the reproducibility checklist implemented across the engine.

---

## 1. Academic Citation & Persistent Identifiers

### Native GitHub Citation (`CITATION.cff`)
`ewm-engine` includes a machine-readable [`CITATION.cff`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/CITATION.cff) file at the repository root compliant with the Citation File Format (v1.2.0) schema. GitHub uses this file to provide automated, one-click citations in both **APA** and **BibTeX** formats via the **"Cite this repository"** button on the main repository page.

#### BibTeX Citation
```bibtex
@software{carida2026ewmengine,
  author    = {Caridá, Vinicius},
  title     = {Enterprise World Model Engine: An Open Framework for Modeling, Simulating, and Evaluating Organizational Dynamics},
  year      = {2026},
  url       = {https://github.com/vfcarida/Enterprise-World-Model-Engine},
  version   = {1.0.0},
  license   = {Apache-2.0}
}
```

---

### Zenodo DOI: Concept DOI vs. Version DOI

To ensure permanent archival and academic traceability, `ewm-engine` integrates with **Zenodo** through automated GitHub release webhooks, configured via [`.zenodo.json`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/.zenodo.json):

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
   - Represents the software project conceptually across all time.
   - Use this when citing the engine generally in conceptual or comparative literature.
   - Resolves dynamically to the latest published release.

2. **Version DOI (`10.5281/zenodo.<version_id>`)**:
   - Represents a specific, immutable release tarball.
   - Use this when reporting experimental benchmarks or empirical findings to guarantee that subsequent readers can inspect the exact code executed.

---

### Software Heritage Archival (SWHID)

While Zenodo stores release tarballs, [Software Heritage](https://www.softwareheritage.org/) archives the entire Git development graph, tracking individual commits, trees, and blobs via cryptographic **SoftWare Heritage Identifiers (SWHIDs)**.

- **Trigger Archival**: The repository is archived via the [Software Heritage Save Code Now](https://archive.softwareheritage.org/save/) service using the repository URL:
  ```text
  https://github.com/vfcarida/Enterprise-World-Model-Engine
  ```
- **Referencing via SWHID**: Specific versions can be referenced with persistent intrinsic identifiers:
  ```text
  swh:1:dir:<directory_hash>;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine
  ```
  This guarantees that code availability does not depend on the continued existence of any single hosting platform.

---

### Journal of Open Source Software (JOSS) Paper

A peer-reviewed software paper describing the mathematical foundations, architecture, and statement of need for `ewm-engine` is maintained under the [`paper/`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/paper/paper.md) directory:

- Paper manuscript: [`paper/paper.md`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/paper/paper.md)
- Bibliography: [`paper/paper.bib`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/paper/paper.bib)

Upon acceptance by JOSS, the `preferred-citation` field in `CITATION.cff` redirects all citation tooling directly to the peer-reviewed journal DOI.

---

## 2. Reproducibility Checklist (NeurIPS & Papers-with-Code Standards)

To guarantee that simulations and scientific evaluations conducted with `ewm-engine` are fully verifiable, the framework adheres strictly to the following reproducibility rubric:

| Category | Requirement | EWM Engine Implementation | Verification Gate |
| :--- | :--- | :--- | :--- |
| **Dependencies** | Complete specification of execution environment and dependencies | [`RunManifest.dependency_versions`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/provenance/manifest.py), `uv.lock`, hardware summary | Automated check in `compute_readiness_report` |
| **PRNG Seeding** | Explicit random number generator lifecycle and thread isolation | [`seed_everything()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/seed.py), `np.random.SeedSequence` hierarchical spawning, `OMP_NUM_THREADS=1` | Zero global state; bitwise trajectory equivalence |
| **State Immutability** | Cryptographic state provenance preventing mutations | [`WorldState`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/state.py) deep immutability (`frozen=True`), canonical SHA-256 fingerprinting | Property tests with Hypothesis |
| **Statistical Reporting** | No bare point estimates; distribution reporting | [`evaluate_multiseed()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/multiseed.py) with percentile bootstrap confidence intervals (mean ± 95% CI) | Enforced in `MultiSeedReport` |
| **Rollout Drift** | Measurement of compounding error over simulation horizons | [`RolloutDivergenceResult.compounding_error_ratios`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/dynamics_eval.py) ($C(h) = \text{err}(h) / (h \cdot \text{err}(1))$, Talvitie 2014) | [`select_reliable_horizon()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/dynamics_eval.py) safety check |
| **Probabilistic UQ** | Proper scoring rules and distribution-free intervals | CRPS, Gaussian log score/NLL, Brier score, adaptive calibration error (ACE), split conformal prediction | Calibration consistency gate |
| **Causal Honesty** | Gated causal estimation without heuristic over-promotion | Backdoor/frontdoor DAG identifiability check, Doubly Robust OPE with mandatory ESS and clip diagnostics | Block on non-identifiable; no auto-upgrade of `EvidenceLevel` |
| **Sensitivity** | Robustness to unobserved confounding | VanderWeele & Ding E-values attached to estimates and CI bounds; common-support overlap checks ($d \ge 5$ warning) | Refutation battery (placebo, random common cause, subset stability) |
| **ML Production Readiness** | Systematic testing across data, models, infrastructure, monitoring | Google ML Test Score ($\min(\text{Data}, \text{Model}, \text{Infra}, \text{Monitoring})$) | [`compute_readiness_report()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/readiness.py) |
| **Metamorphic Invariants** | Code change regression auditing | Invariant transformation checks using business rules as the oracle | [`run_metamorphic_regression_gate()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/readiness.py) |

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
    engine_version="1.0.0",
    seed=42,
    scenario_id="supply_chain_shock",
    initial_state_fingerprint="sha256_canonical_state_fingerprint",
    component_versions={"inventory_dynamics": "1.0.0", "capacity_constraint": "1.0.0"},
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
print(f"Downstream Loss: {metric.mean:.2f} (95% CI: [{metric.ci_lower:.2f}, {metric.ci_upper:.2f}])")
```

---

## 4. Summary of Epistemic Invariants

1. **No Bare Point Estimates**: Evaluations must report variability (standard deviation, interquartile range, or bootstrap confidence intervals).
2. **No Auto-Upgraded Evidence**: Simulations produce dependency traces with `EvidenceLevel.ASSUMED` or `EMPIRICAL_SIMULATED`. Causal claims require formal DAG identifiability and pass-through of sensitivity E-values.
3. **Compounding Horizon Safety**: Multi-step rollouts must measure compounding drift $C(h)$. Receding-horizon planners should never plan past horizons where compounding drift exceeds linear scaling.
