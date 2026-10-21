# Verified Scientific, Executive & Engineering References

This document consolidates the primary-source academic literature, frontier AI research lab publications, enterprise governance frameworks, and open-source architectural standards implemented across the **Enterprise World Model Engine (`ewm-engine`)**.

---

## Pillar 1: Scientific, Academic & Frontier AI Lab Foundations

### 1. World Models & Action-Conditioned Latent Dynamics
- **Ha, D., & Schmidhuber, J. (2018).** *World Models.* Advances in Neural Information Processing Systems (NeurIPS 2018). DOI: `10.5281/zenodo.1207631` / `arXiv:1803.10122`.
  - **Application in EWM:** Foundational separation between world state representation, transition dynamics model, and policy/controller evaluation.
- **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023).** *Mastering Diverse Domains through World Models (DreamerV3).* `arXiv:2301.04104` (Nature, 2025).
  - **Application in EWM:** Symlog input/target transformation ($\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$) in `TorchNeuralResidualDynamics` to stabilize multi-scale enterprise resource learning without domain-specific normalization tuning.
- **Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023).** *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023). `arXiv:2301.08243`.
- **Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., et al. (2025).** *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* Meta FAIR. `arXiv:2506.09985`.
  - **Application in EWM:** Action-conditioned predictive world modeling and receding-horizon Model Predictive Control (`RecedingHorizonController` in `ewm_engine.simulation.mpc`).
- **Vezhnevets, A. S., Agapiou, J. P., Aharon, A., Ziv, R., Matyas, J., Duéñez-Guzmán, E. A., et al. (2023).** *Generative Agent-Based Modeling with Actions Grounded in Physical, Social, or Digital Space using Concordia.* Google DeepMind. `arXiv:2312.03664`.
  - **Application in EWM:** Game-Master arbitration pattern in `EWMWorldArbiter` (`ewm_engine.integrations.concordia`), where agent intents are translated into candidate `Action` objects, checked against hard world constraints, and converted into grounded observations.

### 2. Structural Causal Models, Identifiability & World-Model Necessity
- **Pearl, J. (2009).** *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. DOI: `10.1017/CBO9780511803160`.
  - **Application in EWM:** Three-step structural counterfactual kernel (Abduction–Action–Prediction) in `ewm_engine.simulation.counterfactual`, Backdoor Criterion verification, and probabilities of causation ($PN$, $PS$, $PNS$).
- **Richens, J., & Everitt, T. (2024).** *Robust Agents Learn Causal World Models.* International Conference on Learning Representations (ICLR 2024 Oral). Google DeepMind. `arXiv:2402.10877`.
  - **Application in EWM:** Theoretical grounding for why agents and policies capable of surviving distributional and interventional shifts must encode the causal structure of their environment rather than purely observational correlations.
- **Schölkopf, B., Locatello, F., Bauer, S., Ke, N. R., Kalchbrenner, N., Goyal, A., & Bengio, Y. (2021).** *Toward Causal Representation Learning.* Proceedings of the IEEE, 109(5), 612–634. DOI: `10.1109/JPROC.2021.3058954`.
  - **Application in EWM:** Epistemic hierarchy (`EvidenceLevel`) separating observational/predictive associations from interventional and counterfactual mechanisms.
- **Spirtes, P., Glymour, C., & Scheines, R. (2000).** *Causation, Prediction, and Search* (2nd ed.). MIT Press.
  - **Application in EWM:** PC and FCI constraint-based causal discovery in `ewm_engine.induction.causal`, explicitly preserving Markov Equivalence Classes (CPDAG/PAG) and marking unoriented edges as `DIRECTIONAL_UNCERTAIN` and bidirected edges as `CONFOUNDED`.
- **Bello, K., Aragam, B., & Ravikumar, P. (2022).** *DAGMA: Learning DAGs via M-Matrices and a Log-Determinant Acyclicity Characterization.* Advances in Neural Information Processing Systems (NeurIPS 2022). `arXiv:2209.08037`.
  - **Application in EWM:** Continuous score-based causal structure learning member in the `ewm induce causal` ensemble.
- **Reisach, A., Seiler, C., & Weichwald, S. (2021).** *Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy to Game.* Advances in Neural Information Processing Systems (NeurIPS 2021). `arXiv:2102.13647`.
- **Kaiser, M., & Sipos, M. (2022).** *Unsuitability of NOTEARS for Causal Graph Discovery when Dealing with Dimensional Quantities.* Neural Processing Letters, 54, 1587–1595. `arXiv:2104.05441`.
  - **Application in EWM:** Variance-standardization invariance checks and refusal to assert single DAGs from scale-sensitive continuous scores without interventional or refutation backing.
- **Sharma, A., & Kiciman, E. (2020).** *DoWhy: An End-to-End Library for Causal Inference.* Microsoft Research. `arXiv:2011.04216`.
  - **Application in EWM:** Automated causal refutation gates (`refute_placebo_treatment`, `refute_random_common_cause`, `refute_data_subset`) in `ewm_engine.induction.validation`.
- **VanderWeele, T. J., & Ding, P. (2017).** *Sensitivity Analysis in Observational Research: Introducing the E-Value.* Annals of Internal Medicine, 167(4), 268–274. DOI: `10.7326/M16-2607`.
- **Grünwald, P., de Heide, R., & Koolen, W. (2024).** *Safe Testing.* Journal of the Royal Statistical Society Series B (Statistical Methodology), 86(5), 1091–1128. DOI: `10.1111/stan.12328`.
- **D'Amour, A., Ding, P., Feller, A., Lei, L., & Sekhon, J. (2021).** *Overlap in Observational Studies with High-Dimensional Covariates.* Journal of Econometrics, 221(2), 644–654. DOI: `10.1016/j.jeconom.2020.04.045`.

### 3. Grey-Box Conservation Laws, Symbolic Regression & Process Induction
- **Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).** *Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems (SINDy).* Proceedings of the National Academy of Sciences (PNAS), 113(15), 3932–3937. DOI: `10.1073/pnas.1517384113`.
  - **Application in EWM:** Sequentially Thresholded Least Squares (STLSQ) symbolic rate-law discovery in `ewm_engine.experimental.flywheel` and `ewm_engine.induction.dynamics`.
- **Wu, Z., & Tegmark, M. (2020).** *AI Poincaré: Machine Learning Conservation Laws from Trajectories.* Physical Review E / `arXiv:2011.04698`.
  - **Application in EWM:** SVD nullspace estimation of state-transition differences and seeded Monte Carlo stress testing for conservation invariant discovery in `ewm_engine.induction.invariants`.
- **Berti, A., Koren, I., Adams, J. N., Park, G., Knopp, B., Graves, N., et al. (2024).** *OCEL (Object-Centric Event Log) 2.0 Specification.* RWTH Aachen / PADS. `arXiv:2403.01975`.
  - **Application in EWM:** Object-centric process induction and token-based replay fitness/precision conformance scoring in `ewm_engine.induction.process`.
- **Fey, M., Hu, W., Huang, K., Lenssen, J. E., Ranjan, R., Robinson, J., et al. (2024).** *Position: Relational Deep Learning — Graph Representation Learning on Relational Databases (RelBench).* International Conference on Machine Learning (ICML 2024). `arXiv:2407.20060`.
  - **Application in EWM:** Strict temporal point-in-time (`as_of`) anti-leakage cutoff enforcement during relational and DataFrame world-model induction (`ewm_engine.induction.connectors`).

### 4. Decision Science under Deep Uncertainty & Simulation-Based Inference
- **Lempert, R. J., Groves, D. G., Popper, S. W., & Bankes, S. C. (2006).** *A General, Analytic Method for Generating Robust Strategies and Narrative Scenarios.* Management Science, 52(4), 514–528. DOI: `10.1287/mnsc.1050.0472`.
  - **Application in EWM:** Robust Decision Making (RDM) workflow in `DecisionEngine` (`ewm_engine.decision`).
- **Friedman, J. H., & Fisher, N. I. (1999).** *Bump Hunting in High-Dimensional Data.* Statistics and Computing, 9(2), 123–143. DOI: `10.1023/A:1008979822606`.
  - **Application in EWM:** Patient Rule Induction Method (PRIM) peeling algorithm for interpretable scenario vulnerability discovery (`ewm_engine.decision.discovery`).
- **Elmachtoub, A. N., & Grigas, P. (2022).** *Smart "Predict, then Optimize".* Management Science, 68(1), 9–26. DOI: `10.1287/mnsc.2020.3922`.
  - **Application in EWM:** Decision-focused regret evaluation (`DecisionRegretResult`) prioritizing downstream policy optimality over raw point-prediction MSE.
- **McPhail, C., Maier, H. R., Kwakkel, J. H., Giuliani, M., Castelletti, A., & Westra, S. (2018).** *Robustness Metrics: How Are They Calculated, When Should They Be Used and Why Do They Give Different Results?* Earth's Future, 6(2), 169–191. DOI: `10.1002/2017EF000649`.
  - **Application in EWM:** Multi-metric robustness comparison across Starr's Domain Criterion, Hurwicz Criterion, Savage Minimax Regret, and Laplace Insufficient Reason (`ewm_engine.decision.robustness`).
- **Friston, K. (2010).** *The Free-Energy Principle: A Unified Brain Theory?* Nature Reviews Neuroscience, 11(2), 127–138. DOI: `10.1038/nrn2787`.
  - **Application in EWM:** Active Inference Expected Free Energy ($G(\pi) = \text{Pragmatic Risk} - w \cdot \text{Epistemic Information Gain}$) scorer in `ewm_engine.decision.efe`.
- **Cranmer, K., Brehmer, J., & Louppe, G. (2020).** *The Frontier of Simulation-Based Inference.* Proceedings of the National Academy of Sciences (PNAS), 117(48), 30055–30062. DOI: `10.1073/pnas.1912789117`.
- **Lemos, P., Coogan, A., Hezaveh, Y., & Perreault-Levasseur, L. (2023).** *Sampling-Based Accuracy Testing of Posterior Estimators for General Inference (TARP).* International Conference on Machine Learning (ICML 2023). `arXiv:2302.03026`.
  - **Application in EWM:** Test of Accuracy with Random Points (TARP) Empirical Coverage Probability (ECP) auditing and split-conformal recalibration in `ewm_engine.calibration.coverage`.
- **Gneiting, T., & Raftery, A. E. (2007).** *Strictly Proper Scoring Rules, Prediction, and Estimation.* Journal of the American Statistical Association, 102(477), 359–378. DOI: `10.1198/016214506000001437`.
  - **Application in EWM:** Continuous Ranked Probability Score (CRPS) in `ewm_engine.evaluation.uncertainty`.

---

## Pillar 2: Executive, Strategic & Regulatory Governance Frameworks

- **NIST (2023).** *Artificial Intelligence Risk Management Framework (AI RMF 1.0).* National Institute of Standards and Technology, U.S. Department of Commerce. NIST AI 100-1. DOI: `10.6028/NIST.AI.100-1`.
  - **Application in EWM:** Direct architectural mapping of `GOVERN`, `MAP`, `MEASURE`, and `MANAGE` functions into `ResponsibleAIGate` (`ewm_engine.core.trust`), `EvidenceLevel`, and `InductionValidationGate` (`docs/governance/compliance-mapping.md`).
- **European Parliament & Council of the European Union (2024).** *Regulation (EU) 2024/1689 Laying Down Harmonised Rules on Artificial Intelligence (EU AI Act).* Official Journal of the European Union.
  - **Application in EWM:** Article 12 (Record-keeping & automatic logging via `Provenance` and `SystemicTrace`), Article 13 (Transparency & epistemic standing via `EvidenceLevel` and `ModelCard`), Article 14 (Human oversight via `OpenQuestion` surfacing and `ResponsibleAIGate`), and Article 15 (Accuracy, robustness & cybersecurity via `evaluate_model_reliability` and `StrictSafeLoader`).
- **Gartner Research (2024–2026).** *Decision Intelligence Platforms, Digital Twin of an Organization (DTO), and AI Trust, Risk, and Security Management (AI TRiSM).*
  - **Application in EWM:** Executive positioning of EWM Engine as an auditable Decision Intelligence and DTO simulation substrate with built-in AI TRiSM runtime enforcement (`docs/executive/`).
- **MIT Sloan Management Review & Boston Consulting Group (BCG Henderson Institute).** *Generating Value with Decision-Focused AI and Simulation-Grounded Strategy.*
  - **Application in EWM:** Executive use-case framing for COO (supply-chain resilience), CFO (capital allocation & CVaR tail risk), CRO (correlated stress testing), and CSO (multi-agent war-gaming) in `docs/executive/use-cases/`.
- **Forrester Research & McKinsey & Company.** *Digital Twins and Generative AI in Enterprise Operations: Separating Predictive Correlation from Operational Causality.*
  - **Application in EWM:** Enterprise maturity model (`docs/executive/maturity-model.md`) progressing from descriptive analytics (L1) and predictive forecasting (L2) to constraint-aware simulation (L3), counterfactual twin rollouts (L4), and autonomous competence-gated actuation (L5).

---

## Pillar 3: Open-Source Competitive Ecosystem & Interoperability Standards

- **Model Context Protocol (MCP)** — Anthropic / Open-Source Specification (`modelcontextprotocol.io`).
  - **Application in EWM:** Native JSON-RPC 2.0 stdio server (`ewm_engine.mcp.server`) exposing 8 deterministic world-model tools and 4 URI resources so LLM agents can query state, validate constraints, run counterfactuals, and audit causal induction without hallucinating physics.
- **MLCommons Croissant 1.0 (`mlcommons.org/croissant`)** — Standardized JSON-LD metadata format for machine-learning datasets and simulation benchmarks.
  - **Application in EWM:** Exported by `ReproducibilityCard.to_croissant_jsonld()` (`ewm_engine.cards.models`).
- **Software Heritage Persistent Identifiers (SWHID) & Citation File Format (`CITATION.cff` 1.2.0)** — Archival source-code identification and scholarly citation standards.
  - **Application in EWM:** `CITATION.cff`, `.zenodo.json`, and `docs/scientific/swhid.md`.
- **Functional Mock-up Interface (FMI 2.0 / 3.0) & Discrete-Event / Agent-Based Ecosystem (`SimPy`, `Mesa`, `PySD`, `EMA Workbench`, `DoWhy`, `SALib`, `Gymnasium`)**.
  - **Application in EWM:** Clean hexagonal interoperability adapters in `ewm_engine.cosim`, `ewm_engine.integrations`, and `ewm_engine.decision`, plus the **World Model Interchange Format (WMIF v1.0)** (`ewm_engine.serialization.wmif`).

---

## Pillar 4: Gold-Standard SWE, ML Engineering & Reproducibility Practices

- **Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011).** *Parallel Random Numbers: As Easy as 1, 2, 3.* Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC11). DOI: `10.1145/2063384.2063405`.
  - **Application in EWM:** Counter-based and `SeedSequence.spawn` hierarchical PRNG isolation guaranteeing bit-identical rollouts regardless of worker count or execution order.
- **Chen, T. Y., Cheung, S. C., & Yiu, S. M. (2018).** *Metamorphic Testing: A New Approach for Generating Next Test Cases.* ACM Computing Surveys, 51(1), 1–27. DOI: `10.1145/3143561`.
  - **Application in EWM:** Five structural metamorphic relations in `ewm_engine.evaluation.reliability` verifying dynamics models without requiring ground-truth oracle labels on every unseen state.
- **Scientific Python SPEC 0 & NEP 23** — Minimum supported dependencies and programmatic deprecation lifecycle policies (`docs/spec0-policy.md`, `docs/deprecation-policy.md`).
- **Diátaxis Documentation Architecture** — Systematic separation of Tutorials, How-To Guides, Reference, and Explanation (`mkdocs.yml`).
