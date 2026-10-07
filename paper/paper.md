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
authors:
  - name: Vinicius Caridá
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 06 October 2026
bibliography: paper.bib
---

# Summary

Modern decision intelligence systems in operational and socio-technical domains frequently struggle with counterfactual evaluation: answering *"what happens if we intervene?"* rather than merely forecasting *"what is likely to happen under status-quo trends?"* The **Enterprise World Model Engine (`ewm-engine`)** is an open-source, domain-neutral Python framework designed to model, simulate, and evaluate complex organizational systems under interventional policy changes. 

In contrast to purely observational predictive pipelines or task-centric workflow engines, `ewm-engine` executes action-conditioned forward rollouts ($\mathbb{E}[S_{t+H} \mid \text{do}(A_t)]$) that strictly decouple **known explicit invariants and conservation laws** from **learned probabilistic dynamics** [@pearl2009causality; @scholkopf2021causal]. It provides bitwise reproducibility via cryptographic state fingerprinting and hierarchical pseudo-random number generator (PRNG) seed trees, while keeping core simulation execution completely free of mandatory deep learning or large language model (LLM) dependencies.

# Statement of Need

Operational organizations—such as supply chains, disaster response networks, healthcare delivery systems, and financial logistics—are governed by a combination of hard operational boundaries (physical capacity limits, legal regulations, conserved inventory) and stochastic human or environmental interactions. Existing research software addresses portions of this challenge, but reveals acute gaps:

1. **Simulation Tools**: Discrete-event simulators such as SimPy [@teamsimpy2020] and agent-based frameworks like Mesa [@kazil2020mesa] provide procedural state mutation, but lack native counterfactual scenario branching, formal post-transition invariant enforcement, and standardized epistemic provenance.
2. **System Dynamics**: Equation-based tools like PySD [@houghton2015pysd] model aggregate continuous feedback loops, but do not natively accommodate discrete relational graph states, normative action rejections, or off-policy evaluation diagnostics.
3. **Deep World Models**: Neural world model architectures such as DreamerV3 [@hafner2023mastering] and V-JEPA 2 [@assran2025vjepa2] learn latent transition dynamics directly from high-dimensional observations. However, when applied to enterprise domains, unconstrained neural models suffer from compounding rollout error [@talvitie2014model] and frequently hallucinate violations of fundamental conservation laws.

`ewm-engine` fills this gap by acting as a symbolic-probabilistic hybrid kernel. Researchers and operations engineers define formal world states, phase-aware pre/post-transition constraints, and modular dynamics. The engine guarantees that interventions are evaluated with explicit uncertainty quantification, reproducible run manifests, and rigorous causal gates.

# State of the Field

In enterprise decision-making, predictive models are frequently misapplied as prescriptive simulators. A core tenet of causal reasoning is that observational associations do not imply interventional consequences [@pearl2009causality]. When an organization changes a policy, historical covariates shift, violating observational assumptions and degrading unverified models. 

Recent benchmarks demonstrate that surface realism in model rollouts does not imply structural validity. While value-aware model learning [@farahmand2017vaml] optimizes dynamics for downstream planning, systemic invariants are required as oracle verification gates. `ewm-engine` operationalizes this balance by providing an open evaluation harness where learned neural or graph neural network (GNN) dynamics can be audited against explicit symbolic rules, compounding drift metrics, and off-policy estimators.

# Software Architecture and Scientific Rigor

`ewm-engine` is built around four core architectural pillars designed for high scientific rigor:

1. **Immutable State & Bitwise Reproducibility**: World states are deeply immutable snapshot models with canonical SHA-256 state fingerprints. Simulation rollouts derive deterministic random streams using NumPy `SeedSequence` trees. A canonical `RunManifest` captures hardware metadata, resolved configuration hashes, dependency versions, and determinism flags matching the NeurIPS Reproducibility Checklist.
2. **Phase-Aware Normative Constraints**: Invariant checks operate across distinct execution phases: `PRE_ACTION` (normative action rejection) and `POST_TRANSITION` (rollout invalidation on conservation law breach).
3. **Probabilistic Evaluation & Conformal Bounds**: Evaluates rollouts using strictly proper scoring rules, including continuous ranked probability scores (CRPS) and adaptive calibration error (ACE) [@gneiting2007strictly]. Finite-sample distribution-free coverage is guaranteed via split conformal prediction and adaptive conformal inference (ACI) [@angelopoulos2021conformal].
4. **Gated Causal Inference & Off-Policy Diagnostics**: Enforces strict epistemic honesty. DAG identifiability (backdoor/frontdoor) gates estimation: unidentified queries return explicit confounding explanations rather than ungrounded numbers. Off-policy evaluation (OPE) via Doubly Robust estimation mandates diagnostics (Kish's effective sample size, propensity clip rates) and computes VanderWeele & Ding sensitivity E-values [@vanderweele2017evalues] alongside common support overlap gates [@damour2021overlap]. Aleatoric and epistemic uncertainties are treated under formal unidentifiability limits [@mucsanyi2024trustworthy], preventing unjustified claims of certainty.
5. **Production Readiness & Metamorphic Testing**: Evaluates systemic readiness via the Google ML Test Score rubric [@breck2017mltest] and regression testing against domain metamorphic relations.

# Research Impact Statement

`ewm-engine` enables empirical research across computational social science, operations research, and safe reinforcement learning. By offering a standardized benchmark harness with reproducible run manifests, researchers can objectively compare structural world models, measure super-linear compounding error in multi-step rollouts, and evaluate counterfactual interventions in civic logistics and organizational systems without risking real-world systemic failures.

# Generative AI Usage Disclosure

In compliance with the Journal of Open Source Software guidelines, generative artificial intelligence tools (Google Gemini / Antigravity IDE) were utilized during development as an agentic coding assistant to assist with code scaffolding, test case synthesis, and documentation drafting. All code logic, mathematical formulations, test suites, and written analyses were verified, refined, and validated by the author.

# Acknowledgements

The author thanks the open-source scientific Python community for foundational tools including NumPy, Pydantic, NetworkX, and PyYAML, as well as the researchers whose foundational work in causality, world models, and reproducible benchmarking shaped the architectural principles of `ewm-engine`.

# References
