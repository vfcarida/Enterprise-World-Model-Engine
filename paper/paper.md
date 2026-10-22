---
title: 'Enterprise World Model Engine: An Open Framework for Modeling, Simulating, and Evaluating Organizational Dynamics'
tags:
  - Python
  - world models
  - enterprise simulation
  - counterfactual reasoning
  - causal inference
  - operations research
  - uncertainty quantification
  - reproducible research
  - interchange standards
authors:
  - name: Vinicius Caridá
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 16 October 2026
bibliography: paper.bib
---

# Summary

Modern decision intelligence systems in operational and socio-technical domains frequently struggle with counterfactual evaluation: answering *"what happens if we intervene?"* rather than merely forecasting *"what is likely to happen under status-quo trends?"* The **Enterprise World Model Engine (`ewm-engine`)** is an open-source, domain-neutral Python framework designed to model, simulate, and evaluate complex organizational systems under interventional policy changes.

In contrast to purely observational predictive pipelines or task-centric workflow engines, `ewm-engine` executes action-conditioned forward rollouts ($\mathbb{E}_{P_{\text{model}}}[S_{t+H} \mid \text{do}(A_t)]$) that decouple **known explicit invariants and conservation laws** from **learned probabilistic residual dynamics** [@pearl2009causality; @scholkopf2021causal]. It provides deterministic state fingerprinting and hierarchical pseudo-random number generator (PRNG) seed trees on a pure-Python/NumPy core with zero mandatory deep learning or large language model (LLM) dependencies.

# Statement of Need

Operational organizations—such as supply chains, disaster response networks, healthcare delivery systems, and financial logistics—are governed by a combination of hard operational boundaries (physical capacity limits, legal regulations, conserved inventory) and stochastic human or environmental interactions. Existing open-source software addresses distinct subsets of this problem:

1. **Discrete-Event & Agent-Based Simulators**: Frameworks such as SimPy [@teamsimpy2020] and Mesa [@kazil2020mesa] provide expressive procedural simulation loops, but leave counterfactual scenario branching, post-transition conservation projection, and causal identifiability gates to application-level code.
2. **System Dynamics**: Equation-based tools such as PySD [@houghton2015pysd] model continuous stock-and-flow feedback loops, but do not natively combine typed relational entity graphs, pre-action rejection gates, and split-conformal uncertainty calibration in a single schema.
3. **Deep World Models**: Neural world model architectures such as DreamerV3 [@hafner2023mastering] and V-JEPA 2 [@assran2025vjepa2] learn latent transition dynamics from high-dimensional observations, but when applied directly to conserved enterprise ledgers without structural projection, they accumulate rollout error [@talvitie2014model] and violate physical/accounting conservation invariants.

`ewm-engine` bridges these paradigms through a hybrid structural-probabilistic kernel. Researchers and engineers define typed world states, phase-aware pre/post-transition constraints, and modular dynamics, evaluating interventions with explicit uncertainty quantification, reproducible run manifests, and fail-closed causal identifiability gates.

# State of the Field

In enterprise decision-making, observational predictive models are frequently misapplied as prescriptive simulators. A core principle of causal inference is that observational associations $P(Y \mid X)$ do not equal interventional distributions $P(Y \mid \text{do}(X))$ under confounding [@pearl2009causality]. When an organization alters an operating policy, historical covariate correlations shift (the Lucas Critique).

Recent theoretical work motivates causal structure in world models: Richens & Everitt [@richens2024robust] proved that, under local mechanism interventions on finite discrete variables, an agent achieving vanishing regret across arbitrary distributional shifts must encode an internal model isomorphic to the environment's structural causal model (SCM). `ewm-engine` operationalizes this motivation by combining explicit structural equations and linear conservation nullspace projection with learned residual dynamics and runtime context discovery.

# Software Architecture and Empirical Validation

`ewm-engine` (`v1.5.0`) is organized around six core architectural pillars:

1. **Grey-Box Conservation Projection (KNOWN vs. LEARNED)**: State transitions decompose into known explicit rules $f_{\text{known}}$ and learned residual dynamics $g_\theta$ projected onto the conservation nullspace: $\Delta \mathbf{s}_t = f_{\text{known}}(\mathbf{s}_t, \mathbf{a}_t) + (\mathbf{I} - \mathbf{A}_{\text{cons}}^+ \mathbf{A}_{\text{cons}}) g_\theta(\mathbf{s}_t, \mathbf{a}_t)$, preserving linear invariants within IEEE-754 double-precision tolerance ($\le 10^{-12}$; observed max residual $4.55 \times 10^{-13}$ over 80 steps on SEIR).
2. **Twin-Rollout Counterfactual SCM Kernel**: Implements Pearl's three-step counterfactual algorithm (abduction of exogenous disturbances $\mathbf{U}$, action surgery $\text{do}(\mathbf{a}_t = \mathbf{a}')$, and prediction under coupled $\mathbf{U}$), isolating simulated intervention effects from Monte Carlo noise divergence.
3. **World-Model Induction with Fail-Closed Refusal Gate**: Automates candidate structure discovery from event logs (OCEL 2.0) and time series while gating causal assertions behind Backdoor, Frontdoor, and Shpitser–Pearl ID checks, explicit `identification_provenance`, VanderWeele & Ding confounding **Sensitivity E-values** [@vanderweele2017evalues] (strictly separated from sequential **safe-testing e-processes** [@grunwald2024evalues] in `SafeTestingEProcess`), and DoWhy-style refuters. Unidentifiable queries fail closed (`UnqualifiedCausalClaimError`).
4. **Empirical Calibration & Uncertainty Diagnostics**: Evaluates rollouts using continuous ranked probability scores (CRPS) [@gneiting2007strictly], split-conformal intervals with finite-sample marginal coverage bounds under exchangeability, and **Tests of Accuracy with Random Points (TARP)** [@lemos2023tarp] with per-instance random reference points. Scenario comparisons report non-parametric bootstrap confidence intervals (`percentile`, `basic`, `normal`, `studentized`, `bca`) with a 6-part uncertainty taxonomy while labeling zero-variance deterministic deltas as `(det)`.
5. **Three-Tier Numerical Reproducibility Contract**: Enforces `BITWISE_REPRODUCIBLE` SHA-256 state fingerprints across pure-Python and vectorized NumPy Structure-of-Arrays (SoA) execution under pinned runtimes (`OMP_NUM_THREADS=1`), `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` (`rtol=1e-6, atol=1e-7`) for JAX `x64`, and `NON_CERTIFIED_BACKEND` for uncertified GPU atomic reductions.
6. **World Model Interchange Format (WMIF v1.0)**: Defines a versioned interchange format verified by three standalone reference oracle worlds (`ref_seed_trivial`, `ref_closed_form_stochastic`, `ref_constraint_stress`).

# Limitations and Scope Boundaries

To ensure responsible scientific use, `ewm-engine` documents explicit boundaries: (1) counterfactual and bootstrap estimates are conditioned on the declared model $P_{\text{model}}$ and do not automatically capture unmodeled organizational mechanisms; (2) learned residual dynamics degrade under severe out-of-distribution shifts outside training support (flagged by `SupportBoundaryOODDetector`); (3) observational structure learning identifies only Markov equivalence classes (CPDAG/PAG) and cannot detect omitted latent confounders; (4) split-conformal coverage guarantees are marginal under exchangeability, not conditional; and (5) all outputs are decision-support artifacts rather than autonomous execution commands.

# Reproducibility

`ewm-engine` provides single-command CPU reproduction (`python scripts/reproduce.py` and `python -m benchmarks.enterprise_suite --seed 42`) backed by committed JSON manifests (`reproduction_manifest.json` for the 7 domain validation cards and `docs/results/artifacts/enterprise_benchmark_suite.json` for the 12-family enterprise benchmark suite) and audited by `python scripts/check_release_truth.py`.

# Acknowledgements

The author thanks the open-source scientific Python community for foundational tools including NumPy, Pydantic, NetworkX, and PyYAML, as well as the researchers whose work in causality, world models, and reproducible benchmarking shaped the design of `ewm-engine`.

# References
