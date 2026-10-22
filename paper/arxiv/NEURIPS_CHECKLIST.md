# NeurIPS Datasets & Benchmarks Track: Paper & Benchmark Checklist

> **Submission Title**: *Structural World Models: Evaluating Hybrid Conservation, Learned Residual Dynamics, and Runtime Context Discovery Under Organizational Regime Shift*
> **Author**: Vinicius Caridá (Independent Researcher)
> **Artifact URL**: [https://github.com/vfcarida/Enterprise-World-Model-Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine)
> **Permanent DOI**: *Not yet published (to be assigned upon Zenodo release archival)*
> **Software Heritage SWHID**: *Not yet published (to be assigned upon Software Heritage ingestion)*

---

## 1. Claims and Evidence
* **Question 1.1:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
  * **Answer:** **Yes**. The paper formulates and tests a falsifiable hypothesis across 12 synthetic operational families (`A`–`L`) and 7 domain validation cards: combining linear conservation nullspace projection ($\le 10^{-12}$ IEEE-754 tolerance), learned residual dynamics, and runtime context discovery improves transition fidelity, recovers from mid-rollout deployment shifts (`Family D`, recovering in 2 steps after audit lag), and eliminates decision regret (`Family L`) while exposing honest failure modes across all baselines (transient error during audit telemetry lag in `Family D`, partial-observability filtering lag in `Family E`, fail-closed refusal under unobserved confounding in `Family H`, and raw prediction breakdown under unmodeled OOD structural collapse requiring abstention in `Family K`). Numerical reproducibility is scoped via a Three-Tier Reproducibility Contract (`BITWISE_REPRODUCIBLE` for pure-Python/NumPy, `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` for JAX `x64` within `rtol=1e-6, atol=1e-7`, and `NON_CERTIFIED_BACKEND` for CuPy).
* **Question 1.2:** Have you accurately reported the limitations of your work?
  * **Answer:** **Yes**. Section 6 of the paper and Section 9 of `README.md` explicitly document seven operational boundaries: model-conditioned validity ($P_{\text{model}}$ vs. reality), OOD extrapolation degradation, causal identification preconditions, marginal (non-conditional) conformal exchangeability, stylized benchmark scope, IEEE-754 floating-point precision envelopes, and decision-support boundaries.
* **Question 1.3:** Did you discuss whether the data you are using/curating can reflect unfair or undesirable bias?
  * **Answer:** **Yes**. Model Cards (`ModelCard`), Dataset Cards (`DatasetCard`), and `docs/responsible-ai.md` document intended use, out-of-scope use, slice-conditional conformal coverage disparities (`slice_coverages`), and epistemic refusal gates (`ResponsibleAIGate`).

---

## 2. Reproducibility
* **Question 2.1:** Are the execution commands, random seeds, and environment configurations needed to reproduce all reported results clearly described?
  * **Answer:** **Yes**. All seven validation cards (`reproduction_manifest.json`), three WMIF reference worlds, and the 12-family Enterprise Benchmark Suite (`docs/results/artifacts/enterprise_benchmark_suite.json`, `suite_fingerprint = 7dadf02addfc7a434538cc587c8ce8a4dbc2009a4394c323c19384b3a0349c67`) reproduce deterministically on CPU via `python scripts/reproduce.py` and `python -m benchmarks.enterprise_suite --seed 42`. State fingerprints, `SeedSequence` seeds, dependency versions, and worktree cleanliness (`CERTIFIED_RELEASE` vs. `REPRODUCIBLE_DIRTY_WORKTREE`) are audited by `python scripts/check_release_truth.py`.
* **Question 2.2:** Is the software repository publicly accessible under an open-source license?
  * **Answer:** **Yes**. Distributed under the Apache-2.0 license at `https://github.com/vfcarida/Enterprise-World-Model-Engine`.
* **Question 2.3:** Are dependencies, installation steps, and minimal hardware requirements specified?
  * **Answer:** **Yes**. Standard Python 3.11–3.13 with zero mandatory deep learning or GPU dependencies in core (`pydantic>=2.6.0`, `numpy>=1.26.0`, `pyyaml>=6.0`); installable via `pip install -e .`. Pinned Docker, Binder, and GitHub Codespaces devcontainer configurations are provided.

---

## 3. Maintenance, Governance, and Ethics
* **Question 3.1:** Is there a formal maintenance plan, security disclosure policy, and community governance structure?
  * **Answer:** **Yes**. Documented in `GOVERNANCE.md` (ADR-038 RFC process), `SECURITY.md` (RFC 9116), `docs/spec0-policy.md` (Scientific Python SPEC 0), and `docs/stability-policy.md` (SemVer 2.0.0).
* **Question 3.2:** Does the benchmark release personal data or copyrighted proprietary datasets?
  * **Answer:** **No**. All benchmarks are based on synthetic or public mathematical formulations (Kermack–McKendrick SEIR ODE, Sterman Beer Game, Meadows WORLD3, synthetic warehouse logistics, synthetic OCEL 2.0 process logs, and synthetic operational shift families `A`–`L`).
