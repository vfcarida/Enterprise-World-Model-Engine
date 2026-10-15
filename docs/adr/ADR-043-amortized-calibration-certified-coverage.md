# ADR-043: Amortized Simulation-Based Calibration with Certified Coverage

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-15
* Technical Story: Amortized calibration with certified coverage (`ewm calibrate`)
* Supersedes: None

## Context and Problem Statement

The Enterprise World Model Engine simulates complex operational, organizational, and physical dynamics from declarative specifications (`WorldSpec`). However, realistic enterprise simulators inevitably include tunable continuous parameters—such as depletion rates, market demand elasticity, transfer lead times, or epidemiological transmission coefficients.

Historically, simulators have relied on manual spreadsheet tweaking, unconstrained least-squares curve fitting, or point optimization. This creates three critical failures:
1. **Spurious Point Certainty:** Reporting a single scalar parameter $\hat{\theta}$ conceals parameter confounding, multi-modal likelihoods, and epistemic uncertainty.
2. **Computational Bottleneck:** Standard Markov Chain Monte Carlo (MCMC) must rerun thousands of simulation steps from scratch every time new empirical data arrives.
3. **Catastrophic Miscalibration under Misspecification:** Because all enterprise simulators are simplifications, standard Bayesian inference often becomes dangerously overconfident or collapsed when real data falls outside the simulator's generative manifold. Downstream decision-makers relying on overconfident models make fragile, high-risk commitments.

How should EWM Engine bridge synthetic worlds to real empirical ground truth while maintaining uncompromising epistemic honesty and refusing to report calibrations it cannot certify?

## Decision Drivers

* **Simulator-as-Forward-Model (SBI Motif 3):** Treat the deterministic/seeded EWM engine as the forward generative model $\theta \to x$, defining priors over WorldSpec parameters and simulating $(θ \to x)$ pairs.
* **Amortized Bayesian Inference:** Learn an amortized posterior $p(\theta \mid x_{obs})$ trained once offline and reused in milliseconds across multiple empirical observations (Lavin et al., 2021; Gloeckler et al., Simformer, ICML 2024; Boelts et al., 2024).
* **Non-Point Invariant:** The calibration engine must **never** return a bare scalar or single point estimate; it must always emit full posterior distributions, standard deviations, covariances, and 90% credible intervals.
* **Mandatory Coverage Certification:** Posteriors must undergo rigorous coverage testing—sampling-based **TARP** (Lemos et al., ICML 2023) and **Posterior-SBC** (arXiv:2502.03279)—as a mandatory, non-optional gate.
* **Misspecification Robustness:** Compute standardized statistical distance between observed data and the simulator predictive support (Huang et al., NeurIPS 2023). Flag models that cannot express empirical observations.
* **Refusal Doctrine:** An uncertified, under-covered, or misspecified posterior must be **flagged and refused for downstream decision use** (`decision_usable=False`, `ewm decide` refusal), mirroring the causal "refuse when non-identifiable" doctrine.
* **Epistemic Honesty:** Calibrated $\neq$ causally identified. Certified calibration promotes models to `EvidenceLevel.PREDICTIVE`, but never auto-promotes to `CAUSAL` or `STRUCTURAL`.
* **Zero Mandatory Heavy Dependencies:** Core must provide a lightweight, pure NumPy kernel-conditioned locally linear Bayesian amortizer, with optional delegation to PyTorch/`sbi` under the `[sbi]`/`[calibrate]` extra.

## Considered Options

* **Option 1: Point Optimization via Least Squares (SciPy / Nelder-Mead)** — Fit single parameters minimizing squared trajectory error. Rejected because point estimates mask parameter uncertainty and provide zero coverage guarantees.
* **Option 2: Classical Per-Observation MCMC (PyMC / Stan)** — Run MCMC chains for each new empirical observation. Rejected because MCMC requires thousands of sequential simulator invocations per dataset, is computationally prohibitive for interactive enterprise analysis, and provides no coverage certification.
* **Option 3: Amortized SBI with Mandatory Coverage Certification & Refusal Gate (Chosen)** — Learn an amortized posterior over simulator parameters, evaluate TARP joint coverage and Posterior-SBC rank uniformity, detect model misspecification, and refuse uncertified models for decision making.

## Decision Outcome

Chosen option: **Option 3: Amortized SBI with Mandatory Coverage Certification & Refusal Gate**.

### Consequences

#### Positive:
* **Principled Real-World Grounding:** Seamlessly connects synthetic EWM worlds to empirical data series without manual guesswork.
* **Amortized Speed:** Evaluating the posterior on new empirical telemetry takes milliseconds at test time.
* **Honest Uncertainty Quantification:** Downstream decision-making operates on full parameter distributions with 90% credible intervals and full covariance matrices.
* **Strict Safety Gate:** Uncertified or misspecified models are explicitly blocked from driving automated decision optimizations (`ewm decide`).
* **Zero Heavy Dependencies in Core:** Pure NumPy amortizer runs in any environment without PyTorch or CUDA.
* **Reproducibility:** Seeded runs produce bit-for-bit identical calibration artifacts (`target_data_hash`, SHA-256 manifest).

#### Negative:
* Training amortized estimators requires an upfront batch of forward simulations (e.g. 50–200 simulations).
* Highly nonlinear or discontinuous dynamics may require larger simulation batches to satisfy stringent TARP coverage thresholds.

## Pros and Cons of the Options

### Option 1: Point Optimization via Least Squares
* Good: Fast to compute on single trajectories.
* Bad: Conceals parameter confounding; treats epistemic uncertainty as zero; provides no coverage certification.

### Option 2: Classical Per-Observation MCMC
* Good: Exact asymptotic Bayesian posterior under well-specified priors.
* Bad: Extremely slow; cannot be amortized across scenarios; does not check coverage or misspecification automatically.

### Option 3: Amortized SBI with Mandatory Certification (Chosen)
* Good: Fast amortized test-time evaluation; full distribution output; automated TARP and SBC coverage tests; active refusal gate on misspecification.
* Bad: Requires offline simulation batch generation.

## Links

* Lavin et al., "Simulation Intelligence: Towards a New Generation of Scientific Methods", arXiv:2112.03235.
* Gloeckler et al., "All-in-one Simulation-Based Inference (Simformer)", ICML 2024, arXiv:2404.09636.
* Boelts et al., "sbi reloaded", arXiv:2411.17337.
* Lemos et al., "Sampling-based Accuracy and Coverage for Simulation-Based Inference (TARP)", ICML 2023, arXiv:2302.03026.
* Huang et al., "Robust Simulation-Based Inference under Model Misspecification", NeurIPS 2023, arXiv:2305.15871.
* Kennedy & O'Hagan, "Bayesian calibration of computer models", JRSS-B 2001.
