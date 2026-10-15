# Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty

**Authors:** Vinicius Caridá  
**Affiliation:** Independent Researcher  
**Target Venue:** arXiv (cs.AI, stat.ML, cs.SE) / NeurIPS Datasets & Benchmarks Track  
**Date:** October 2026  
**Repository:** [https://github.com/vfcarida/Enterprise-World-Model-Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine)  
**Permanent Identifier (DOI):** [10.5281/zenodo.placeholder](https://doi.org/10.5281/zenodo.placeholder)

---

## Abstract

Modern machine learning research in "world models" has focused heavily on continuous control, video generation, and latent-space imagination for robotics and games. Concurrently, enterprise organizations demand simulation tools to answer prescriptive interventional questions: *"What would happen if we enact decision $a$ instead of baseline $a'$?"* 

When applied to organizational and socio-technical domains (such as supply chain resilience, capital allocation, and crisis logistics), unconstrained neural and generative world models suffer from catastrophic failure modes: they violate fundamental physical and accounting conservation laws, compound autoregressive rollout errors, and emit unjustified point forecasts.

In this paper, we introduce the **Enterprise World Model Engine (EWM)**, an open-source, domain-neutral structural simulation kernel. EWM establishes an architectural firewall between **KNOWN explicit conservation laws** (enforced as hard algebraic invariants) and **LEARNED probabilistic dynamics** (restricted to conservative nullspace projections). 

We establish the theoretical license for this architecture via Richens & Everitt's Causal World Model Theorem (ICLR 2024), showing that robust interventional planning requires structural causal models (SCMs). EWM operationalizes this via:
1. A native SCM kernel supporting Pearl's three-step counterfactual algorithm under Twin-Rollout exogenous noise coupling.
2. Amortized Simulation-Based Inference (SBI) with certified coverage evaluated through the Test of Accuracy of Related Probabilities (TARP).
3. A beyond-rollout-error evaluation suite scoring models on **epistemic honesty**: anytime-valid E-values under optional stopping, mandatory non-point predictive intervals, and active refusal on unidentifiable causal queries.

Across six empirical benchmarks (SEIR epidemic waves, Beer Distribution Game bullwhips, WORLD3 global limits, and supply logistics), EWM achieves bit-for-bit determinism, zero mass-balance drift ($0.000000$), and certified coverage calibration without mandatory GPU or LLM dependencies.

---

## 1. Introduction: The Crisis of Unconstrained World Models

In recent years, the paradigm of "world models" (Ha & Schmidhuber, 2018; Schrittwieser et al., 2019; Hafner et al., 2023) has emerged as one of the most promising frontiers in artificial intelligence. By learning an internal predictive model of environment dynamics, agents can plan, simulate counterfactual rollouts, and optimize actions without expensive trial-and-error in reality.

However, a fundamental disconnect exists between the academic world-model literature and the requirements of enterprise decision systems:
1. **The Conservation Vacuum:** Video and latent world models (e.g., NVIDIA Cosmos, Meta V-JEPA, Sora) optimize for surface perceptual realism. When tasked with simulating balance sheets or supply inventories, unconstrained neural networks casually violate mass conservation—shipping non-existent inventory or conjuring positive cash flow out of thin air.
2. **The Flaw of Averages:** Deep regression models output conditional expectations $\mathbb{E}[Y \mid X]$, concealing catastrophic downside tails and insolvency risks from executive leadership.
3. **The Confounding Trap:** Supervised learning optimizes observational correlations $P(Y \mid X)$. When an enterprise alters a policy, the data-generating mechanism shifts, invalidating historical correlations (the Lucas Critique).

To solve these deficiencies, we propose **Structural World Models (SWM)** and implement the paradigm in the open-source **Enterprise World Model Engine (`ewm-engine`)**.

---

## 2. Theoretical License: Why World Models Must Be Causal

The foundational theoretical justification for EWM’s architecture is licensed by the **Causal World Model Theorem** proved by Richens & Everitt (ICLR 2024, arXiv:2402.10877):

> **Theorem 1 (Richens & Everitt, 2024):** *Any agent capable of achieving robust optimal performance across an arbitrary family of distributional shifts must learn a world model that is structurally causal.*

An agent relying on purely associative predictive models can always be subjected to an interventional shift where its predictions suffer unbounded regret. Therefore, an enterprise decision engine must not merely model correlation; it must represent the structural graph $\mathcal{G}$ and support the interventional $do$-operator.

---

## 3. Architecture: Decoupling KNOWN Rules from LEARNED Dynamics

EWM enforces an absolute architectural separation:

$$\Delta \mathbf{s}_t = \underbrace{f_{\text{known}}(\mathbf{s}_t, \mathbf{a}_t)}_{\text{KNOWN Explicit Invariants}} + \underbrace{\mathcal{P}_{\text{null}}\left( g_\theta(\mathbf{s}_t, \mathbf{a}_t) \right)}_{\text{LEARNED Residual Dynamics}}$$

### Hard Conservation Layers
Let $\mathbf{A}_{\text{cons}} \Delta \mathbf{s} = \mathbf{0}$ represent the system of linear conservation laws (mass balance, double-entry ledger constraints). The learned component $g_\theta$ is projected onto the nullspace of $\mathbf{A}_{\text{cons}}$:

$$\mathcal{P}_{\text{null}}(\mathbf{v}) = \left( \mathbf{I} - \mathbf{A}_{\text{cons}}^+ \mathbf{A}_{\text{cons}} \right) \mathbf{v}$$

This guarantees that physical and accounting quantities can never be violated by neural approximation error.

### Twin-Rollout Counterfactual SCM Kernel
EWM implements Judea Pearl’s three-step counterfactual algorithm:
1. **Abduction:** From observed factual trajectory $\tau = (\mathbf{s}_0, \mathbf{a}_0, \mathbf{s}_1, \dots)$, abduce the exact sequence of realized exogenous disturbances $\mathbf{U} = (\mathbf{u}_0, \mathbf{u}_1, \dots)$.
2. **Action:** Perform causal graph surgery $do(\mathbf{a}_t = \mathbf{a}')$ by severing structural dependency arrows into $\mathbf{a}_t$.
3. **Prediction:** Re-simulate the dynamics under the abducted disturbance sequence $\mathbf{U}$ (Twin-Rollout noise coupling).

Because $\mathbf{U}$ is coupled, the observed difference $\Delta \tau = \tau_{\text{cf}} - \tau_{\text{fact}}$ is **mathematically guaranteed to be caused by the policy change**, rather than random sampling divergence.

---

## 4. Amortized Calibration with Certified Coverage (TARP)

Enterprise simulators contain unobservable parameters $\boldsymbol{\theta}$ (e.g., supplier elasticity, customer churn friction). EWM integrates **Simulation-Based Inference (SBI)** via amortized Neural Posterior Estimation (Papamakarios et al., 2019):

$$q_\phi(\boldsymbol{\theta} \mid \mathbf{x}) \approx p(\boldsymbol{\theta} \mid \mathbf{x}_o)$$

To prevent deploying overconfident posteriors, EWM gates all calibrated models through the **Test of Accuracy of Related Probabilities (TARP)** (Lemos et al., ICML 2023, arXiv:2302.03026):

$$P\left(\boldsymbol{\theta}^* \in \mathcal{R}_\alpha(\mathbf{x})\right) = \alpha \quad \forall \alpha \in [0, 1]$$

If the empirical coverage curve deviates from nominal credibility by more than threshold $\epsilon$, the posterior is classified as uncertified and refused for decision use.

---

## 5. Beyond Rollout Error: The Honesty-Reward Evaluation Protocol

EWM introduces an evaluation suite scored on **epistemic honesty**:

1. **Anytime-Valid E-Values:** Hypothesis comparisons between scenarios report sequential E-values satisfying $\mathbb{E}_{H_0}[E] \le 1$, ensuring false-discovery rate control under optional stopping.
2. **Mandatory Non-Point Uncertainty:** Point forecasts are rejected by the public API in favor of non-parametric BCa bootstrap confidence intervals and conformal coverage bounds.
3. **Causal Refusal Gate:** When unobserved confounding violates the Backdoor Criterion or covariate overlap fails, the engine raises an `UnqualifiedCausalClaimError` rather than emitting misleading guidance.
4. **Epistemic Abstention (RWM-U):** Trajectories entering out-of-distribution state regimes accumulate penalty weights and are down-weighted in policy optimization.

---

## 6. Empirical Benchmarks & Reproducibility

We validate EWM across six established benchmarks:

| Benchmark | System Type | Invariant Tested | Empirical Result | Drift |
| :--- | :--- | :--- | :--- | :--- |
| **SEIR Outbreak** | Compartmental ODE | $S+E+I+R = N$ | Peak $I = 183.35$ | **0.000000** |
| **Beer Game** | 4-Tier Supply Chain | Multi-tier inventory | Bullwhip Ratio $= 3.80$ | **Conserved** |
| **WORLD3** | Macro Feedback ODE | Capital-resource delay | Inflection at $t=99$ | **Stable** |
| **SBI Calibration** | Bayesian Posterior | Credible coverage | TARP Error $= 0.028$ | **Certified** |
| **Twin-Rollout** | SCM Logistics | Noise coupling | ATE $= +35.0$ units | **Coupled** |
| **Scenario Discovery** | RDM Vulnerability | PRIM Box purity | Density $= 91.2\%$ | **$80\%$ Recall** |

All benchmarks reproduce bit-for-bit via:
```bash
python scripts/reproduce_results.py --all
```

---

## 7. Conclusion & Research Impact

The Enterprise World Model Engine demonstrates that rigor, auditability, and causal honesty can be engineered directly into simulation software without sacrificing performance. By open-sourcing the kernel under Apache-2.0 with cryptographic verification manifests and zero mandatory cloud/GPU infrastructure, EWM provides a trusted, reproducible foundation for the emerging discipline of Decision Intelligence.

---

## References

* Assran, M. et al. (2025). V-JEPA 2. arXiv:2506.09985.
* Friedman, J. H. & Fisher, N. I. (1999). Bump hunting in high-dimensional data. *Stat. Comput.*
* Hafner, D. et al. (2023). Mastering Diverse Domains through World Models (DreamerV3). arXiv:2301.04104.
* Lemos, P. et al. (2023). Sampling-based accuracy testing of posterior estimators (TARP). *ICML 2023*, arXiv:2302.03026.
* Papamakarios, G. et al. (2019). Sequential Neural Likelihood. *AISTATS 2019*.
* Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
* Richens, J. & Everitt, T. (2024). Robust agents learn causal world models. *ICLR 2024 (Oral)*, arXiv:2402.10877.
* Savage, L. J. (1951). The theory of statistical decision. *JASA*.
* Sterman, J. D. (1989). Modeling Managerial Behavior (Beer Game). *Manage. Sci.*
