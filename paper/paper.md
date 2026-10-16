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
    orcid: 0009-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 16 October 2026
bibliography: paper.bib
---

# Summary

Modern decision intelligence systems in operational and socio-technical domains frequently struggle with counterfactual evaluation: answering *"what happens if we intervene?"* rather than merely forecasting *"what is likely to happen under status-quo trends?"* The **Enterprise World Model Engine (`ewm-engine`)** is an open-source, domain-neutral Python framework designed to model, simulate, and evaluate complex organizational systems under interventional policy changes. 

In contrast to purely observational predictive pipelines or task-centric workflow engines, `ewm-engine` executes action-conditioned forward rollouts ($\mathbb{E}[S_{t+H} \mid \text{do}(A_t)]$) that strictly decouple **known explicit invariants and conservation laws** from **learned probabilistic dynamics** [@pearl2009causality; @scholkopf2021causal]. It provides bitwise reproducibility via cryptographic state fingerprinting and hierarchical pseudo-random number generator (PRNG) seed trees, while keeping core simulation execution completely free of mandatory deep learning or large language model (LLM) dependencies.

# Statement of Need

Operational organizations—such as supply chains, disaster response networks, healthcare delivery systems, and financial logistics—are governed by a combination of hard operational boundaries (physical capacity limits, legal regulations, conserved inventory) and stochastic human or environmental interactions. Existing research software addresses portions of this challenge, but reveals acute gaps:

1. **Simulation Tools**: Discrete-event simulators such as SimPy [@teamsimpy2020] and agent-based frameworks like Mesa [@kazil2020mesa] provide procedural state mutation, but lack native counterfactual scenario branching, formal post-transition invariant enforcement, and standardized epistemic provenance.
2. **System Dynamics**: Equation-based tools like PySD [@houghton2015pysd] model aggregate continuous feedback loops, but do not natively accommodate discrete relational graph states, normative action rejections, or off-policy evaluation diagnostics.
3. **Deep World Models**: Neural world model architectures such as DreamerV3 [@hafner2023mastering] and V-JEPA 2 [@assran2025vjepa2] learn latent transition dynamics directly from high-dimensional observations. However, when applied to enterprise domains, unconstrained neural models suffer from compounding rollout error [@talvitie2014model] and frequently violate fundamental mass and financial conservation laws.

`ewm-engine` addresses this gap by providing a symbolic-probabilistic hybrid kernel. Researchers and operations engineers define formal world states, phase-aware pre/post-transition constraints, and modular dynamics. The engine guarantees that interventions are evaluated with explicit uncertainty quantification, reproducible run manifests, and rigorous causal gates.

# State of the Field

In enterprise decision-making, predictive models are frequently misapplied as prescriptive simulators. A core tenet of causal reasoning is that observational associations do not imply interventional consequences [@pearl2009causality]. When an organization alters a policy, historical covariates shift, violating observational assumptions (the Lucas Critique).

Recent theoretical work establishes why world models must be causal: Richens & Everitt [@richens2024robust] proved that any agent achieving robust optimal performance across arbitrary distributional shifts must learn a structurally causal world model. EWM operationalizes this theoretical license by coupling symbolic conservation laws with learned residual dynamics, guaranteeing that learned approximations cannot breach systemic invariants.

# Software Architecture and Scientific Rigor

`ewm-engine` is built around six core architectural pillars:

1. **Architectural Firewall (KNOWN vs. LEARNED)**: State transitions decompose into known explicit invariants $f_{\text{known}}$ and learned dynamics $g_\theta$ projected onto the conservation nullspace: $\Delta \mathbf{s}_t = f_{\text{known}}(\mathbf{s}_t, \mathbf{a}_t) + (\mathbf{I} - \mathbf{A}_{\text{cons}}^+ \mathbf{A}_{\text{cons}}) g_\theta(\mathbf{s}_t, \mathbf{a}_t)$.
2. **Twin-Rollout Counterfactual SCM Kernel**: Implements Pearl's three-step counterfactual algorithm (abduction of exogenous disturbances $\mathbf{U}$, action surgery $\text{do}(\mathbf{a}_t = \mathbf{a}')$, and prediction under coupled $\mathbf{U}$), mathematically isolating policy effects from sampling divergence.
3. **World-Model Induction with Honest Refusal Gate**: Automates transition discovery from process logs (OCEL 2.0) while gating causal assertions behind backdoor/frontdoor identifiability checks, sensitivity E-values [@vanderweele2017evalues], and refuters. Unidentifiable queries fail closed with informative explanations.
4. **Certified Calibration & Epistemic Honesty**: Evaluates rollouts using continuous ranked probability scores (CRPS), adaptive calibration error (ACE) [@gneiting2007strictly], and TARP coverage certification [@lemos2023tarp]. Hypothesis comparisons use anytime-valid E-values [@grunwald2024evalues] under optional stopping.
5. **Cross-Backend Bit-Identical Fingerprints**: Ensures bitwise state hash identity across pure-Python, vectorized NumPy, and compiled JAX backends via Structure-of-Arrays (SoA) state layouts and deterministic seed trees.
6. **World Model Interchange Format (WMIF)**: Defines a standardized, versioned interchange format (Level 1, Packages: `graph_state`, `conformal_intervals`, `event_sourced_replay`) paired with an executable, standalone conformance suite.

# Reproducibility & Research Impact

`ewm-engine` provides a single-command reproducibility layer (`make reproduce`) backed by a cryptographic JSON run-manifest (`reproduction_manifest.json`) verifying all benchmark cards, figures, and SHA-256 state fingerprints. Strangers can reproduce all published results in-browser via Binder, in the cloud via GitHub Codespaces, or locally via pinned Docker containers without local environment friction.

# Generative AI Usage Disclosure

In compliance with the Journal of Open Source Software guidelines, generative artificial intelligence tools (Google Gemini / Antigravity IDE) were utilized during development as an agentic pair-programming assistant to assist with code scaffolding, test case synthesis, and documentation drafting. All code logic, mathematical formulations, test suites, and written analyses were verified, refined, and validated by the author.

# Acknowledgements

The author thanks the open-source scientific Python community for foundational tools including NumPy, Pydantic, NetworkX, and PyYAML, as well as the researchers whose foundational work in causality, world models, and reproducible benchmarking shaped the architectural principles of `ewm-engine`.

# References
