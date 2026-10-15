# Community Outreach & Ecosystem Plan

> Objective: Systematic, credibility-first dissemination of EWM Engine across academic, open-source, and applied engineering communities.

---

## 1. Curated Awesome Lists

We propose targeted Pull Requests to curated GitHub lists that index software in simulation, causal inference, and decision intelligence:

| Repository / List | Category Target | Submission Description | Prerequisite Asset |
| :--- | :--- | :--- | :--- |
| **`awesome-causal-inference`** | Causal World Models & Simulators | *EWM Engine: Structural causal simulation engine with Abduction-Action-Prediction twin rollouts and identifiability gating.* | S01 SCM Kernel + arXiv Paper |
| **`awesome-decision-intelligence`** | Simulation & Scenario Discovery | *EWM Engine: Scenario discovery (PRIM / CART) and minimax regret optimization under deep uncertainty.* | S02 Decision Module |
| **`awesome-python`** | Science and Data Analysis / Simulation | *EWM Engine: Deterministic, domain-neutral system dynamics and causal world model engine.* | Pure Python + JOSS paper |
| **`awesome-system-dynamics`** | Python Frameworks | *EWM Engine: Conservation-preserving system dynamics engine with stock-flow consistency and feedback stability.* | WORLD3 & SEIR Validation Packs |
| **`awesome-simulation`** | Discrete & Hybrid Simulation | *EWM Engine: Grey-box hybrid simulation separating explicit conservation laws from learned dynamics.* | S03 Grey-Box Engine |

---

## 2. Academic Venues & Workshop Targets

| Target Venue | Track / Workshop | Primary Asset Presented | Relevance & Alignment |
| :--- | :--- | :--- | :--- |
| **NeurIPS** | *Datasets and Benchmarks Track* | Companion Paper + Benchmark Suite (`paper/arxiv/`) | Fully audited with NeurIPS D&B checklist (`NEURIPS_CHECKLIST.md`). |
| **ICML** | *Workshop on Simulation-Based Inference (SBI)* | Amortized Calibration with Certified Coverage (`S04`) | TARP empirical coverage validation on non-trivial simulator parameters. |
| **CLeaR** | *Conference on Causal Learning and Reasoning* | Counterfactual Twin-Rollout SCM Engine (`S01`) | Direct implementation of Pearl's three-step counterfactual logic in dynamic graphs. |
| **ISDC** | *International System Dynamics Conference* | Stock-Flow Grey-Box Dynamics (`S03`) | Bridging traditional system dynamics with machine learning and automated calibration. |
| **Winter Simulation Conference (WSC)** | *Hybrid Simulation & Agent-Based Modeling* | Deterministic Branch Isolation & Provenance | Bit-exact cryptographic auditability in enterprise simulation. |

---

## 3. Open-Source Ecosystem Alignments

EWM Engine actively interfaces with and complements existing tools in the scientific Python ecosystem:

- **DoWhy / PyWhy**: EWM Engine handles time-series rollout simulation and twin-rollout counterfactuals; integration harnesses allow validating graphs with DoWhy's refutation tests.
- **EMA-Workbench**: EWM's `ewm decide` exports run ensembles directly compatible with exploratory modeling and analysis workflows.
- **Scikit-learn**: Follows scikit-learn API design principles (`fit`, `predict`, `score`), example gallery structure, and documentation clarity.
- **NumFOCUS**: Long-term governance roadmap aims for NumFOCUS Affiliated Project status (see [NumFOCUS Sponsorship Roadmap](docs/numfocus-sponsorship.md)).

---

## 4. Outreach Principles & Code of Ethics

1. **Lead with Proof, Never Spam**: All community engagement must link to reproducible code, open benchmarks, or verified mathematical results.
2. **Transparent Validation**: Never claim simulated results represent real-world enterprise deployments unless verified and cited. Reference results must always be explicitly labeled as synthetic or historical validation benchmarks.
3. **Respect Maintainers**: Submissions to awesome lists or partner projects must strictly follow the target repository's contribution guidelines.
4. **Foster Academic Citations**: Make citing EWM Engine effortless by keeping `CITATION.cff`, Zenodo DOIs, and BibTeX snippets synchronized across all releases.
