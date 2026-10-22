# Research Reproducibility & Citation Guide

> **Reproducibility & Software Citation**  
> Discipline: Research-software reproducibility, cryptographic run manifests, and structured software citation.  
> Standards: [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), [FORCE11 Software Citation Principles](https://doi.org/10.7717/peerj-cs.86), [CITATION.cff v1.2.0](https://citation-file-format.github.io/).

The **Enterprise World Model Engine (`ewm-engine`)** is engineered for deterministic execution and verifiable numerical reproducibility. This document outlines the software citation metadata, run-manifest certification semantics, and reproducibility checklist implemented in the repository.

---

## 1. Software Citation & Claim Verification

### Native GitHub Citation (`CITATION.cff`)
`ewm-engine` includes a machine-readable [`CITATION.cff`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/CITATION.cff) file at the repository root compliant with the Citation File Format (v1.2.0) schema.

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

### Claim-to-Evidence Registry

All public scientific and engineering claims, evidence tiers (`THEORETICAL`, `UNIT_VERIFIED`, `PROPERTY_VERIFIED`, `CONFORMANCE_VERIFIED`, `BENCHMARKED`, `CALIBRATED`, `EXTERNALLY_REPRODUCED`, `UNVERIFIED_RESEARCH`), explicit boundary conditions, and executable verification commands are cataloged in [`docs/scientific/claim-evidence-matrix.md`](scientific/claim-evidence-matrix.md).

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

Published validation cards, WMIF reference worlds, and the 12-family Enterprise Benchmark Suite can be reproduced via:

| Method | Environment | Command | Output |
| :--- | :--- | :--- | :--- |
| **Local CLI (`uv`)** | Python 3.11–3.13 | `uv run python scripts/reproduce.py` | `reproduction_manifest.json` |
| **Benchmark Suite** | Python 3.11–3.13 | `uv run python -m benchmarks.enterprise_suite --seed 42` | `docs/results/artifacts/enterprise_benchmark_suite.json` |
| **Pinned Docker** | Docker Engine | `docker build -t ewm-engine:pinned -f Dockerfile . && docker run --rm ewm-engine:pinned` | Verified run-manifest |


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
