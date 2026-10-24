# What Is an Enterprise World Model? Scientific Demarcation & Scope

Enterprise World Model Engine (`ewm-engine`) is an open-source **research program** and computational testbed. Its purpose is not to assert that "world models" are universally superior to existing analytical disciplines, nor to rebrand established modeling paradigms under a fashionable label. Instead, it exists to investigate, honestly and comparatively, when—and whether—explicitly modeling the coupled dynamics, constraints, and epistemic uncertainty of an organizational system yields measurably better decisions than simpler alternatives.

---

## 1. Working Definition & The Two Guiding Questions

### Working Definition

An **Enterprise World Model (EWM)** is a **computational, domain-scoped representation** that models:

1. The **relevant states** of an organizational environment (typed entities, conserved and bounded resources, relational topology, and institutional/policy context $K_t$, distinguished from partial observations $O_t$ and belief states $B_t$);
2. Their **dependencies**, explicitly graded by the epistemic nature of their supporting evidence (`DependencyEvidenceNature`);
3. Their **temporal evolution** across hybrid deterministic, stochastic, rule-governed, or learned transition mechanisms; and
4. **When—and only when—there is sufficient identification evidence**, the **effects of actions and interventions** ($do(\cdot)$),

in order to support **prediction, simulation, planning, and decision evaluation under uncertainty**.

Five boundary conditions are part of this definition:

- **Domain-scoped, not omniscient:** An EWM does not need to represent an entire corporation. It models a bounded operational domain (e.g., an inventory allocation network, an order-to-cash workflow, a clinical capacity pool) and makes its scope, boundary assumptions, and omitted variables explicit.
- **Explicit uncertainty and epistemic state:** It separates ontic state $S_t$ ("what is true in the modeled system") from epistemic state $P_t$ ("what the engine knows, infers, assumes, or holds identification evidence for").
- **Hybrid representation:** Its mechanisms may combine accounting identities, business rules, statistical models, structural equations, discrete-event queues, optimization solvers, and neural or foundation-model components.
- **Predicting evolution $\neq$ knowing interventional effects:** Forecasting future trajectories under an observational distribution $P(S_{t+1} \mid S_t)$ is fundamentally distinct from identifying the causal effect of a policy intervention $P(S_{t+1} \mid S_t, do(A_t = a))$.
- **Conditional utility:** An EWM is only *useful* if it demonstrates measurable benefit on clearly defined decision tasks and operating conditions relative to strong baselines.

### The Two Guiding Research Questions

Every module, benchmark, and experiment in `ewm-engine` serves two linked, falsifiable questions:

1. **The Empirical Decision Question (Conditional & Falsifiable):**
   > *Under which concrete operational conditions does explicitly modeling the dynamics, constraints, and feedback of a corporate system let you decide better than simply predicting its outcomes?*
2. **The Demarcation Question:**
   > *What, if anything, distinguishes an Enterprise World Model from a conventional combination of predictors, simulators, causal graphs, and optimizers?*

!!! important "A Negative Result Is a First-Class Contribution"
    This research program does **not** assume that an EWM is superior, novel, or cost-justified. Those are hypotheses to be implemented, stress-tested against mature baselines, and potentially **refuted**. Demonstrating that a calibrated predictor + linear program matches or beats an EWM under stationary, unconstrained conditions—or that an EWM improves state forecasting while failing to reduce decision regret—is a primary scientific result of this repository.

---

## 2. Positioning Against Mature Neighbors

Organizational modeling is a mature field with decades of rigorous theory across econometrics, operations research, simulation, and machine learning. An EWM shares foundational primitives with nine neighboring disciplines. The table below positions EWM neutrally against each neighbor without straw-manning.

| Discipline / Paradigm | Primary Sources | Core Scope & Strengths | What EWM Shares | How EWM Differs in Scope or Contract (Candidate Distinction) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Structural Causal Models (SCM) & $do$-Calculus** | Pearl (2009); Spirtes, Glymour & Scheines (2000); Peters, Janzing & Schölkopf (2017); Richens & Everitt (2024) | Formal identification of interventional ($P(Y \mid do(X))$) and counterfactual ($P(Y_{x'} \mid X=x, Y=y)$) distributions via DAGs, backdoor/frontdoor criteria, and structural equations $V_i := f_i(\text{PA}_i, U_i)$. | Explicit graph surgery (`StructuralCausalModel.intervene`), backdoor identifiability checks, DoWhy-style refuters, and strict refusal to promote observational associations to causal standing. | Classical SCMs typically analyze static or short-horizon equilibrium causal effects over a fixed variable set. EWM embeds SCM identification gates inside multi-step, resource-constrained, stateful rollouts where institutional policy context $K_t$ and capacity bounds co-evolve with actions. |
| **2. System Dynamics (SD)** | Forrester (1961); Sterman (2000) | Aggregate stocks, flows, delay structures, and nonlinear feedback loops modeled via coupled ordinary differential or difference equations. | Conservation of stocks/resources, rate-governed inflows/outflows, and feedback-driven regime shifts over extended horizons. | System Dynamics operates primarily on continuous macroscopic aggregates with fixed structural equations. EWM tracks typed discrete entities and relational graphs alongside aggregate resources, and attaches explicit epistemic standing and uncertainty calibration to each mechanism. |
| **3. Discrete-Event Simulation (DES)** | Law (2014); Banks et al. (2010); Zeigler, Praehofer & Kim (2000) | Queueing networks, resource contention, stochastic service/arrival times, and event-driven state transitions in operational processes. | Stateful step/event progression, capacity constraints, seeded CRN (Common Random Numbers) variance reduction, and trajectory-level KPI evaluation. | A conventional DES assumes its programmed transition logic and routing probabilities are causally valid for any what-if parameter change. EWM explicitly tracks whether a transition is a known mechanism vs. an observational fit, refusing or flagging unidentifiable counterfactual policy shifts. |
| **4. Agent-Based Models (ABM)** | Bonabeau (2002); Macal & North (2010); Kazil, Masad & Crooks (2020) | Bottom-up emergence from heterogeneous, autonomous actors interacting locally under behavioral rules. | Typed `Entity` instances, local interaction topologies (`Relationship`), and multi-actor action/arbitration protocols. | ABMs focus on emergent phenomena from postulated micro-behaviors, often facing severe calibration and identifiability challenges. EWM focuses on governed operational decision evaluation with explicit feasibility projection ($\Pi_{\mathcal{C}}$), epistemic provenance, and conformal/Bayesian calibration. |
| **5. Digital Twins** | Grieves & Vickers (2017); Tao et al. (2019); National Academies (2024) | Bidirectional telemetry synchronization between a physical asset/factory and its high-fidelity virtual replica for monitoring and control. | State estimation/filtering from observations ($O_t \to B_t$), drift detection (`detect_world_drift`), and shadow-mode counterfactual replay. | Engineering digital twins rely on first-principles physics (CAD, Navier-Stokes, kinematics) where governing equations are invariant. Enterprise systems are governed by mutable contracts, human incentives, and partial observability where causal mechanisms are partially unknown and drift with policy changes. |
| **6. Process Mining (Object-Centric Event Logs)** | van der Aalst (2016); Carmona et al. (2018); Berti, Koren & van der Aalst (2023, OCEL 2.0) | Discovery, conformance checking, and performance enhancement of business workflows from historical event logs (XES, OCEL 2.0). | Object-centric lifecycle extraction (`ewm induce`), replay fitness/precision conformance validation, and bottleneck identification. | Process mining excels at descriptive discovery and replay diagnostics of historical control-flow. Moving from descriptive process logs to counterfactual policy evaluation requires causal identification and resource-conservation constraints beyond control-flow miners alone. |
| **7. Probabilistic Graphical & State-Space Models** | Kalman (1960); Koller & Friedman (2009); Durbin & Koopman (2012); Murphy (2023) | Principled Bayesian filtering, smoothing, and uncertainty propagation over latent states $S_t$ given noisy observations $O_t$. | Explicit `ObservationSpec`, `BeliefState`, epistemic/aleatoric decomposition, and calibrated predictive intervals. | State-space models capture $P(S_{t+1} \mid S_t)$ and $P(O_t \mid S_t)$ under a fixed probability measure, but do not by themselves distinguish observational state conditioning from structural causal surgery $do(A_t)$ or enforce hard organizational feasibility projections. |
| **8. Operations Research (OR) & Mathematical Optimization** | Bertsimas & Tsitsiklis (1997); Birge & Louveaux (2011); Powell (2022) | Prescriptive optimization (LP, MILP, stochastic programming, MDP/MPC) over well-defined objective functions and algebraic constraints. | Hard/soft constraint checking, feasible-action projection residuals, and Receding-Horizon Model Predictive Control (MPC). | Optimization solvers prescribe optimal actions *given* a fixed mathematical model of dynamics and constraints, but do not manage the epistemic provenance of how those constraints/dynamics were induced, whether causal effects are identified, or when structural drift invalidates the formulation. |
| **9. Neural & Foundation World Models** | Ha & Schmidhuber (2018); Schrittwieser et al. (2020, MuZero); Hafner et al. (2023, DreamerV3); LeCun (2022, JEPA); Assran et al. (2025, V-JEPA 2) | Learning compact latent representations and latent transition dynamics $z_{t+1} = f_\theta(z_t, a_t)$ from high-dimensional sensory streams (pixels, video, continuous control) for planning or reinforcement learning. | Rollout-based planning (MPC/tree search), latent or residual dynamics learning (`LearnedDynamics`, `LinearResidualDynamics`), and evaluating models via downstream control performance. | See §2.1 below on why results from games, robotics, and video prediction do not transfer automatically to enterprise decision systems. |

### 2.1 Why Game, Control, and Vision World-Model Results Do Not Transfer Automatically to Enterprises

Recent breakthroughs in neural world models—such as **World Models** (Ha & Schmidhuber, 2018), **MuZero** (Schrittwieser et al., 2020), **DreamerV3** (Hafner et al., 2023), and **JEPA / V-JEPA 2** (LeCun, 2022; Assran et al., 2025)—demonstrate remarkable success in Atari, board games, continuous robotics, and self-supervised video representation learning. However, six structural differences prevent those results from transferring directly to enterprise decision-making:

1. **No simulator-as-ground-truth or cheap online exploration:** In MuZero or Dreamer, the agent can execute millions of trial-and-error episodes against an exact game engine or physics simulator, quickly correcting action-conditioned bias. An enterprise cannot run $10^6$ randomized pricing, layoff, or supply-chain re-routing experiments in production to learn latent action effects.
2. **Confounded observational logs vs. autonomous exploration:** Historical enterprise data (ERP tables, CRM logs, event streams) is generated by human managers and legacy policies acting on unobserved signals. Fitting a latent transition model $z_{t+1} = f_\theta(z_t, a_t)$ directly to observational logs confounds the state-transition function with the historical logging policy $\pi_{\text{beh}}(a_t \mid S_t, U_t)$.
3. **Invariant physics vs. mutable institutional context ($K_t$):** Pixel dynamics in video (V-JEPA 2) or rigid-body mechanics in robotics obey stationary physical laws. Enterprise dynamics are heavily mediated by contracts, SLAs, tax rules, credit limits, and organizational hierarchies ($K_t$) that change discontinuously by executive decree.
4. **Hard accounting conservation and auditability:** In latent-space world models (Dreamer, MuZero, JEPA), predictions occur in an abstract vector space $\mathbb{R}^d$ where exact double-entry accounting, non-negativity of physical inventory, and contractual SLA clauses are not guaranteed to hold—and where a regulator or CFO cannot audit *why* a specific trajectory was rejected.
5. **Relational and object-centric topology:** Enterprise state is not a fixed-resolution pixel grid or dense proprioceptive vector; it is a time-varying, typed relational graph of orders, suppliers, shipments, accounts, and queues.
6. **Asymmetric, high-stakes decision regret:** In enterprise operations, a single uncalibrated out-of-distribution action can trigger cascading insolvency or SLA defaults. Point-estimate rollouts in an uncalibrated latent space provide no formal guarantee against tail-risk violations.

---

## 3. Candidate Differentiators: Hypotheses to Be Tested

!!! warning "Anti-Renaming Principle: Integration Alone Is Not a Contribution"
    Bundling a causal graph, a discrete-event simulator, a neural residual model, and an MPC optimizer behind a unified Python API is **software engineering**, not a scientific contribution by itself. Never call a dependency graph, a conventional simulator, or an ensemble of predictors a "world model" merely because they are wired together.

Accordingly, `ewm-engine` frames its candidate architectural differentiators strictly as **falsifiable hypotheses to be tested against simpler baselines**:

- **Hypothesis H1 (Decision-First Regime Advantage):** Explicitly coupling typed state, conservation constraints, and multi-step rollout dynamics reduces realized decision regret relative to direct supervised predictors ($a^* = \arg\max_a \hat{f}(x, a)$) **specifically when** actions induce delayed feedback, multi-resource contention, or shifting constraint boundaries—and offers **zero or negative net value** (due to specification and variance overhead) in single-step, unconstrained, stationary regimes.
- **Hypothesis H2 (Epistemic Gating & Refusal Value):** Grading every dependency by the nature of its evidence (`DependencyEvidenceNature`) and failing closed when an interventional query traverses an unidentifiable `STATISTICAL_ASSOCIATION` or unverified `CAUSAL_HYPOTHESIS` prevents costly false-intervention decisions under confounding and regime shift.
- **Hypothesis H3 (Ontic/Epistemic Separation & Calibrated Uncertainty):** Separating ontic state $S_t$, observation/belief state $B_t$, and epistemic state $P_t$—while propagating calibrated uncertainty intervals into a risk-adjusted decision contract—reduces tail-risk violations (CVaR, constraint breaches) compared to point-estimate simulation or uncalibrated latent rollouts.
- **Hypothesis H4 (Hybrid Pluralism under Governance):** Combining known functional accounting rules (`KNOWN_FUNCTIONAL`) and business-rule mechanisms (`BUSINESS_RULE_MECHANISM`) with learned statistical or residual components (`STATISTICAL_ASSOCIATION`, `LearnedDynamics`) inside a hard feasibility projection ($\Pi_{\mathcal{C}}$) outperforms both pure hand-coded rules (which miss complex nonlinear patterns) and pure unconstrained black-box models (which violate physical/accounting invariants under distribution shift).

---

## 4. Dependency Evidence-Nature Taxonomy

To prevent the silent conflation of correlation and causation (Vision §7.2), `ewm-engine` classifies every dependency and mechanism along two reconciled axes in [`ewm_engine.provenance.evidence`](../reference/api/provenance.md):

1. **`DependencyEvidenceNature` (`EvidenceNature`)** — *why* we believe the dependency exists (the nature of its source evidence); and
2. **`EvidenceLevel`** — the *effective query-capability rung* propagated across trajectories and enforced by `ResponsibleAIGate`.

| `DependencyEvidenceNature` | Canonical `EvidenceLevel` | Meaning & Source of Evidence | Permitted Query Classes (`DependencyQueryKind`) | Causal Refusal Gate Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `KNOWN_FUNCTIONAL` | `STRUCTURAL` | Deterministic accounting identities, conservation laws, physical capacities, or engineered formulas (e.g., $\text{Stock}_{t+1} = \text{Stock}_t + \text{In}_t - \text{Out}_t$). | `OBSERVATIONAL_PREDICTION`, `STATE_EVOLUTION`, `INTERVENTIONAL_DO`, `COUNTERFACTUAL_TWIN` | Permitted across all query classes provided domain support/overlap holds. |
| `BUSINESS_RULE_MECHANISM` | `ASSERTED` | Enforced institutional, contractual, workflow, or software-encoded policy mechanisms active under context $K_t$ (e.g., SLA penalty schedules, automated reorder rules). | `OBSERVATIONAL_PREDICTION`, `STATE_EVOLUTION`, `INTERVENTIONAL_DO`, `COUNTERFACTUAL_TWIN` | Permitted while the governing policy context $K_t$ remains active; flagged if policy drift (`DriftKind.CONTEXT_DRIFT`) occurs. |
| `EXPERIMENTALLY_SUPPORTED` | `INTERVENTIONAL` | Directional causal effect empirically identified via randomized experiments ($do$-interventions, A/B tests) or validated quasi-experiments passing refutation and overlap checks. | `OBSERVATIONAL_PREDICTION`, `STATE_EVOLUTION`, `INTERVENTIONAL_DO` (plus `COUNTERFACTUAL_TWIN` only when structural functional invertibility is also proven) | Permitted for interventional $do(\cdot)$ queries within common support; refuses out-of-support extrapolation. |
| `CAUSAL_HYPOTHESIS` | `ASSUMED` (or `QUASI_CAUSAL` with identification certificate) | Domain-posited or structure-learned directed hypothesis not yet backed by experimental verification. | `OBSERVATIONAL_PREDICTION`, `STATE_EVOLUTION` (upgrades to `INTERVENTIONAL_DO` **only** with explicit `is_identifiable=True`, `overlap_satisfied=True`, and `identification_provenance`) | **Refuses** `INTERVENTIONAL_DO` and `COUNTERFACTUAL_TWIN` (`UnqualifiedCausalClaimError`) unless explicit identification evidence is supplied. |
| `STATISTICAL_ASSOCIATION` | `PREDICTIVE` | Observational correlation, predictive regression, or learned conditional distribution $P(Y \mid X)$ without causal identification. | `OBSERVATIONAL_PREDICTION`, `STATE_EVOLUTION` **only** | **Strictly refused** for `INTERVENTIONAL_DO` and `COUNTERFACTUAL_TWIN` under all circumstances—never silently promoted to causal. |

---

## 5. Structural Core + Plural Components, Judged on Decisions

A common misconception in applied AI is to posit a universal hierarchy of modeling paradigms—either claiming that "symbolic rules are deterministic and trustworthy while ML/LLMs are probabilistic and untrustworthy," or conversely that "end-to-end neural models obsolete explicit structure." Both claims are scientifically false:

1. **Determinism of execution $\neq$ epistemic certainty:** A hand-written business rule executes deterministically in code, yet if it encodes an outdated heuristic or an unverified manager hunch, its epistemic uncertainty regarding real-world outcomes is high. Conversely, a neural network or LLM with greedy decoding (`temperature=0`) and fixed weights is also a deterministic mathematical function at inference time, while a stochastic discrete-event simulator with known queueing physics is explicitly probabilistic.
2. **No component class is second-class:** In `ewm-engine`, rules, conservation equations, classical statistical estimators, specialized ML models (`LearnedDynamics`, `LinearResidualDynamics`), foundation/LLM components, and mathematical optimizers **coexist as plural pluggable mechanisms** inside a shared structural core.
3. **Selection by operational regime and decision impact:** The right mechanism for a dependency depends on data density, out-of-distribution behavior, latency, cost, auditability requirements, governance, and downstream decision sensitivity—not ideological preference. Every component is judged by the same contract: **does it improve calibrated decision quality under explicit epistemic guarantees?**

---

## 6. Blunt Guide: When You Do NOT Need an EWM

Building and maintaining an explicit world model carries real specification, calibration, and governance costs. **Do not use an Enterprise World Model** in the following regimes:

1. **Information extraction, summarization, and document search:** If your task is parsing invoices, summarizing contracts, routing support tickets, or querying enterprise knowledge bases (RAG), use information retrieval and foundation models directly. There is no coupled dynamical state or multi-step intervention to simulate.
2. **Single-step classification or stationary tabular prediction:** If you are predicting credit default, customer churn, or fraud where the action applies to a single independent unit ($i.i.d.$) without capacity contention, inventory coupling, or delayed feedback loops, a well-calibrated supervised classifier or uplift model is simpler, faster, and typically more accurate.
3. **Bounded deterministic workflow automation (RPA):** If a business process is fully specified by deterministic software rules with no resource bottlenecks, uncertainty, or trade-off decisions, standard workflow orchestration or rule engines suffice.
4. **Purely physical/engineering asset mirroring:** If the system is a physical turbine, chemical reactor, or CNC machine governed by known partial differential equations, use a domain-specific physics simulator or engineering Digital Twin (e.g., Modelica, ANSYS).
5. **Well-specified static resource allocation:** If the objective, capacities, and linear technological coefficients are known and stationary over a single period, formulate and solve a standard Linear or Mixed-Integer Program (LP/MILP) directly.
6. **Single-process queueing without cross-domain coupling or causal ambiguity:** If your goal is sizing staffing levels for a single call center with known arrival and service distributions, a conventional Discrete-Event Simulation (DES) or Erlang-C model is sufficient.

Use an EWM **only** when decisions must navigate **coupled multi-step dynamics, shared resource constraints, partial observability, regime/policy shifts, and mixed epistemic evidence** where simpler predictors or single-paradigm simulators demonstrably fail.

---

## 6b. Validity Limits: Performativity, Endogeneity, Adversarial Adaptation & Verifiable Explainability

Even where an EWM is appropriate, **its own deployment erodes its validity** through four structural mechanisms enforced in `ewm_engine`:

1. **Performativity & the Lucas Critique (`DriftKind.POLICY_INDUCED_SHIFT`):** Deploying an optimized policy $\pi$ alters the behavioral elasticities of the human and institutional environment it modeled ($P(S_{t+1} \mid S_t, A_t; \pi) \neq P(S_{t+1} \mid S_t, A_t; \pi_0)$). `check_plan_performative_validity` flags when a candidate plan invalidates the static behavioral parameters assumed by its own rollout simulator.
2. **Closed-Loop Endogeneity (`DriftKind.ENDOGENOUS_FEEDBACK_SHIFT`):** Operational actions $A_t = \pi(S_t, O_t, U_t)$ are taken in response to state and unrecorded operator urgency $U_t$, inducing simultaneity bias that invalidates static observational correlations (`STATISTICAL_ASSOCIATION`).
3. **Strategic Adversarial Adaptation (`DriftKind.ADVERSARIAL_ADAPTATION_SHIFT`):** In security, fraud, and competitive pricing regimes, strategic adversaries adapt to deployed thresholds, eroding static policy utility over time ($T_{\text{degrade}}$).
4. **Verifiable Mechanistic Explainability vs. Narrative Plausibility (`VerifiableMechanisticExplanation` & `verify_narrative_explanation_fidelity`):** Free-text or LLM-generated prose is never accepted as evidence that a valid mechanism was used (`is_standalone_narrative_evidence=False`). Explanations are deterministic projections of the recorded `MechanismProvenance`, `ContextSnapshot` version/fingerprint, and `SystemicTrace`, and `verify_narrative_explanation_fidelity` rejects any narrative asserting mechanisms, context keys, or causal standing not present in the trace.
5. **Uncertainty-Typed Abstention (`ValidityLimitAbstentionGate`):** When observed states leave the declared `ValidityDomain`, when performative/adversarial drift fires, or when any component of the 6-part `UncertaintySourceType` taxonomy (`ALEATORIC`, `PARAMETER`, `STRUCTURAL_MODEL`, `RUNTIME_CONTEXT`, `MONTE_CARLO_ERROR`, `MODEL_DISCREPANCY_VS_REALITY`) exceeds threshold, the engine emits `DO_NOT_USE_ME_HERE`, reports which uncertainty source triggered abstention, requests human review, and falls back to a pre-defined safe operational policy.

---

## 7. Primary References

1. **Assran, M., et al.** (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. arXiv:2506.09985.
2. **Banks, J., Carson, J. S., Nelson, B. L., & Nicol, D. M.** (2010). *Discrete-Event System Simulation* (5th ed.). Pearson.
3. **Berti, A., Koren, I., & van der Aalst, W. M. P.** (2023). *OCEL (Object-Centric Event Log) 2.0 Specification*. RWTH Aachen University.
4. **Bertsimas, D., & Tsitsiklis, J. N.** (1997). *Introduction to Linear Optimization*. Athena Scientific.
5. **Birge, J. R., & Louveaux, F.** (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.
6. **Bonabeau, E.** (2002). Agent-based modeling: Methods and techniques for simulating human systems. *Proceedings of the National Academy of Sciences*, 99(suppl_3), 7280–7287.
7. **Carmona, J., van Dongen, B., Solti, A., & Weidlich, M.** (2018). *Conformance Checking: Relating Processes and Models*. Springer.
8. **Durbin, J., & Koopman, S. J.** (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
9. **Forrester, J. W.** (1961). *Industrial Dynamics*. MIT Press.
10. **Grieves, M., & Vickers, J.** (2017). Digital Twin: Mitigating unpredictable, undesirable emergent behavior in complex systems. In *Transdisciplinary Perspectives on Complex Systems* (pp. 85–113). Springer.
11. **Ha, D., & Schmidhuber, J.** (2018). World Models. *Advances in Neural Information Processing Systems (NeurIPS)*, arXiv:1803.10122.
12. **Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T.** (2023). Mastering Diverse Domains through World Models (DreamerV3). *Nature* (2025) / arXiv:2301.04104.
13. **Kalman, R. E.** (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
14. **Kazil, J., Masad, D., & Crooks, A.** (2020). Utilizing Python for Agent-Based Modeling: The Mesa Framework. In *Social, Cultural, and Behavioral Modeling (SBP-BRiMS)*, Springer.
15. **Koller, D., & Friedman, N.** (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT Press.
16. **Law, A. M.** (2014). *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill.
17. **LeCun, Y.** (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview Technical Report*.
18. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
19. **Peters, J., Janzing, D., & Schölkopf, B.** (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT Press.
20. **Powell, W. B.** (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley.
21. **Richens, J., & Everitt, T.** (2024). Robust agents learn causal world models. *International Conference on Learning Representations (ICLR)*, arXiv:2402.10877.
22. **Schrittwieser, J., et al.** (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero). *Nature*, 588, 604–609.
23. **Spirtes, P., Glymour, C., & Scheines, R.** (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
24. **Sterman, J. D.** (2000). *Business Dynamics: Systems Thinking and Modeling for a Complex World*. McGraw-Hill.
25. **Tao, F., Zhang, H., Liu, A., & Nee, A. Y. C.** (2019). Digital Twin in Industry: State-of-the-Art. *IEEE Transactions on Industrial Informatics*, 15(4), 2405–2415.
26. **van der Aalst, W. M. P.** (2016). *Process Mining: Data Science in Action* (2nd ed.). Springer.
27. **Zeigler, B. P., Praehofer, H., & Kim, T. G.** (2000). *Theory of Modeling and Simulation* (2nd ed.). Academic Press.
