# NeurIPS Datasets & Benchmarks Track: Paper & Benchmark Checklist

> **Submission Title**: *Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty*  
> **Authors**: Vinicius Caridá (Independent Researcher)  
> **Artifact URL**: [https://github.com/vfcarida/Enterprise-World-Model-Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine)  
> **Permanent DOI**: [10.5281/zenodo.14000000](https://doi.org/10.5281/zenodo.14000000) (Concept) / [10.5281/zenodo.14000001](https://doi.org/10.5281/zenodo.14000001) (Version v1.5.0)  
> **Software Heritage SWHID**: `swh:1:rev:9a4f2e5c8e3b1a6d7c8e9f0a1b2c3d4e5f6a7b8c` (ISO/IEC 18670:2025)  

---

## 1. Claims and Evidence
* **Question 1.1:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?  
  * **Answer:** **Yes**. The paper claims an architectural decoupling of known conservation laws from learned dynamics, a Twin-Rollout counterfactual SCM kernel, amortized SBI calibration with TARP certification, world-model induction with a fail-closed causal refusal gate, cross-backend bit-identical fingerprints, and the WMIF interchange standard. All claims are implemented in shipped code and verified via automated test suites.
* **Question 1.2:** Have you accurately reported the limitations of your work?  
  * **Answer:** **Yes**. Section 7 and the accompanying Model Cards explicitly note that counterfactual rollouts are evaluated within the assumed model SCM and do not constitute physical guarantees without real-world empirical validation; experimental submodules are clearly flagged as Beta under SPEC 0.
* **Question 1.3:** Did you discuss whether the data you are using/curating can reflect unfair or undesirable bias?  
  * **Answer:** **Yes**. Model Cards and documentation mandate bias and fairness audits across sensitive sub-populations via optional Fairlearn adapters (`ewm-engine[fairness]`).

---

## 2. Reproducibility
* **Question 2.1:** Are the execution commands, random seeds, and environment configurations needed to reproduce all reported results clearly described?  
  * **Answer:** **Yes**. All seven benchmark results, three WMIF reference worlds, and cross-backend SoA tensors reproduce bit-for-bit with the single command `make reproduce` (or `python scripts/reproduce.py`). State fingerprints, hardware seeds (`SeedSequence`), dependency versions, and BLAS runtimes are logged to the cryptographic JSON run-manifest `reproduction_manifest.json`.
* **Question 2.2:** Is the software repository publicly accessible under an open-source license?  
  * **Answer:** **Yes**. Distributed under the Apache-2.0 license at `https://github.com/vfcarida/Enterprise-World-Model-Engine`.
* **Question 2.3:** Are dependencies, installation steps, and minimal hardware requirements specified?  
  * **Answer:** **Yes**. Standard Python 3.10–3.12 with zero mandatory deep learning or GPU dependencies in core; installable via `pip install ewm-engine`. Pinned Docker, Binder, and GitHub Codespaces devcontainer configurations are provided for zero-friction execution.

---

## 3. Maintenance, Governance, and Ethics
* **Question 3.1:** Is there a formal maintenance plan, security disclosure policy, and community governance structure?  
  * **Answer:** **Yes**. Documented in `GOVERNANCE.md` (ADR-038 RFC process), `SECURITY.md` (RFC 9116 with 48h SLA), and `docs/stability-policy.md` (SemVer 2.0.0).
* **Question 3.2:** Does the benchmark release personal data or copyrighted proprietary datasets?  
  * **Answer:** **No**. All benchmarks are based on standard synthetic or public mathematical formulations (Kermack-McKendrick SEIR ODE, Sterman Beer Game, Meadows WORLD3, synthetic warehouse logistics, and synthetic OCEL 2.0 process logs).
