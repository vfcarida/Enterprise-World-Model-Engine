# Architecture Reconciliation Report

**Date**: 2026-10-30  
**Version**: v1.5.0  
**Auditor**: Vinicius Caridá  

---

## 1. Summary

Reconciled [`ARCHITECTURE.md`](../../ARCHITECTURE.md) claims against the actual `v1.5.0` repository source tree (`src/ewm_engine/`), root public API (`ewm_engine.__all__`), [`README.md`](../../README.md), and [`ROADMAP.md`](../../ROADMAP.md).

### Classification Totals (20 Architectural Capabilities Audited)
- **`IMPLEMENTED_STABLE` (`v1.5.0` Core & Public API)**: **8** capabilities
- **`IMPLEMENTED_BETA` (`v1.5.0` In-Tree Beta)**: **4** capabilities
- **`IMPLEMENTED_EXPERIMENTAL` (`v1.5.0` `experimental/` Namespace)**: **3** capabilities
- **`PLANNED` (`v2.0+` Vertical Target / Research Prototypes Outside `v1.5.0` Core Public API)**: **5** vertical/router capabilities (labeled `[PLANNED]` for `v2.0+` production graduation, with in-tree experimental research scaffolding isolated under `src/ewm_engine/routing/`, `src/ewm_engine/graph/`, `src/ewm_engine/financial/`, and `src/ewm_engine/aiops/` and excluded from `ewm_engine.__all__` and `v1.5.0` core certification claims)

---

## 2. Feature Status Table

| Feature | ARCHITECTURE.md Claim | Actual Status (`v1.5.0` Core Scope) | Source Evidence |
| :--- | :--- | :--- | :--- |
| **Core Simulation Kernel** | Deterministic Monte Carlo rollout, `SeedSequence` branching, bitwise reproducibility | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/simulation/engine.py`, `src/ewm_engine/core/world.py` (exported in `ewm_engine.__all__`) |
| **Immutable State & Context ($S_t, K_t$)** | Typed `WorldState`, canonical SHA-256 fingerprints, `ContextSnapshot`, `BitemporalLedger` | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/core/state.py`, `src/ewm_engine/core/context.py`, `src/ewm_engine/core/bitemporal.py` |
| **Phase-Aware Constraint Engine** | `PRE_ACTION` and `POST_TRANSITION` validation gates with `HARD`/`SOFT` severities | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/constraints/base.py`, `src/ewm_engine/constraints/registry.py` |
| **Pluggable & Projected Residual Dynamics** | `DynamicsModel` protocol, `DeterministicTransferDynamics`, KKT `ConservationProjectionLayer` | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/dynamics/`, `src/ewm_engine/experimental/conservation.py` |
| **Scenario Evaluation & Uncertainty Suite** | Paired/independent bootstrap CIs, split-conformal intervals, e-processes (`compare_scenarios`) | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/evaluation/comparison.py`, `src/ewm_engine/evaluation/conformal.py` |
| **Systemic Provenance & Model/Scenario Cards** | `SystemicTrace` DAGs with `EvidenceLevel` tiers and SHA-256 signed cards | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/provenance/trace.py`, `src/ewm_engine/cards/models.py` |
| **World Specification Language (`.wsl`) & WMIF** | Declarative `.wsl` compiler and WMIF v1.0 conformance reference worlds | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/serialization/wsl.py`, `src/ewm_engine/serialization/wmif.py` |
| **Co-Simulation, Multi-Agent & FastAPI Service** | Master co-simulation loop (`FMI`, `SimPy`, `Mesa`, `PySD`), multi-agent mediator, REST API | `IMPLEMENTED_STABLE` (`[IMPLEMENTED]`) | `src/ewm_engine/cosim/`, `src/ewm_engine/multiagent/`, `src/ewm_engine/service/` |
| **Flagship Fraud Cascade & 5-Rung Ladder** | Fraud $\to$ Declines $\to$ Call-Center cascade and `EWME-PREREG-20261023-LADDER` comparative evaluation | `IMPLEMENTED_BETA` (`[IMPLEMENTED]`) | `src/ewm_engine/flagship/fraud_cascade.py`, `src/ewm_engine/flagship/comparative_evaluation.py` |
| **Validity Limits & Abstention Gate** | Performativity, adversarial adaptation, and `"DO_NOT_USE_ME_HERE"` abstention gate | `IMPLEMENTED_BETA` (`[IMPLEMENTED]`) | `src/ewm_engine/flagship/validity_limits.py` |
| **Counterfactual SCM Kernel & Twin Rollout** | Pearlian abduction-action-prediction twin rollout with backdoor identifiability checks | `IMPLEMENTED_BETA` (`[IMPLEMENTED]`) | `src/ewm_engine/simulation/counterfactual.py`, `src/ewm_engine/experimental/causal.py` |
| **Decision Discovery, SBI Calibration, Induction & MCP** | `ewm decide` (PRIM), `ewm calibrate` (TARP), `ewm induce` (refusal gate), `ewm mcp` server | `IMPLEMENTED_BETA` (`[IMPLEMENTED]`) | `src/ewm_engine/decision/`, `src/ewm_engine/calibration/`, `src/ewm_engine/induction/`, `src/ewm_engine/mcp/` |
| **Torch Neural Residual & Relational GNN Dynamics** | PyTorch MLP residual with symlog scaling and heterogeneous relational GNN | `IMPLEMENTED_EXPERIMENTAL` (`[IMPLEMENTED]`) | `src/ewm_engine/experimental/dynamics_torch.py`, `src/ewm_engine/experimental/graph_dynamics.py` |
| **3-Tier Array Backends (`NumPy`, `JAX`, `CuPy`)** | `BITWISE_REPRODUCIBLE`, `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`, `NON_CERTIFIED_BACKEND` | `IMPLEMENTED_EXPERIMENTAL` (`[IMPLEMENTED]`) | `src/ewm_engine/simulation/backends.py`, `src/ewm_engine/simulation/batched.py` |
| **SWE Ecosystem & Domain Composition** | Software-engineering change-impact/deployment dynamics and multi-domain composition | `IMPLEMENTED_EXPERIMENTAL` (`[IMPLEMENTED]`) | `src/ewm_engine/experimental/swe_ecosystem.py`, `src/ewm_engine/experimental/domain_composition.py` |
| **Mechanistic Router** | "Three-tier processing hierarchy (`<100ms` deterministic, BGE semantic, LLM arbitrator)" | `PLANNED` (`[PLANNED]` for `v2.0+`; unexported research prototype in `src/ewm_engine/routing/`) | Not exported in `ewm_engine.__all__` or `v1.5.0` `reproduction_manifest.json`; prototype modules in `src/ewm_engine/routing/router.py` |
| **CT-TGNN (`ContinuousGraphRegistry` & `EvolveGCN`)** | "In-memory temporal graph registry and recurrent GCN weight evolution" | `PLANNED` (`[PLANNED]` for `v2.0+`; unexported research prototype in `src/ewm_engine/graph/`) | Not exported in `ewm_engine.__all__` or `v1.5.0` `reproduction_manifest.json`; prototype modules in `src/ewm_engine/graph/evolvegcn.py` |
| **Financial Vertical — Market-1T & EDGAR** | "1Hz limit order books (`Market1TLoader`) and SEC EDGAR (`EDGARLoader`) bitemporal pipelines" | `PLANNED` (`[PLANNED]` for `v2.0+`; unexported research prototype in `src/ewm_engine/financial/data/`) | Not in `ewm_engine.__all__` or `v1.5.0` core release scope; top-level `from ewm_engine.financial import Market1TLoader` raises `ImportError` (resides in submodule `financial.data.market1t_loader`) |
| **Financial Vertical — JEPA & Contagion Predictor** | "Context/EMA target encoders (`FinancialJEPAEncoder`), `VICRegLoss`, `FinancialTransitionPredictor`" | `PLANNED` (`[PLANNED]` for `v2.0+`; unexported research prototype in `src/ewm_engine/financial/models/`) | Not in `ewm_engine.__all__` or `v1.5.0` core release scope; requires optional `[ml]` PyTorch runtime |
| **AIOps Vertical — Prometheus, Jaeger, STGCN, CausalRCA & Chaos Mesh** | "Telemetry ingestion, `STGCNEncoder`, `CausalRCA`, `RemediationGenerator`, `ChaosMeshHarness`" | `PLANNED` (`[PLANNED]` for `v2.0+`; unexported research prototype in `src/ewm_engine/aiops/`) | Not in `ewm_engine.__all__` or `v1.5.0` core release scope; top-level `from ewm_engine.aiops import PrometheusCollector` raises `ImportError` (resides in submodule `aiops.data.prometheus_collector`) |

---

## 3. Contradictions Identified and Resolved

1. **Contradiction 1 — `ARCHITECTURE.md` Omitted the `v1.5.0` Core Architecture and Presented `v2.0+` Verticals Without Status Labels**:
   - *Before*: `ARCHITECTURE.md` described only the Mechanistic Router, CT-TGNN, Financial World Model, and AIOps World Model without status tags, while omitting the `v1.5.0` Stable/Beta/Experimental core subsystems documented in `README.md` and `ROADMAP.md`.
   - *Resolution*: Restructured [`ARCHITECTURE.md`](../../ARCHITECTURE.md) into two clearly labeled parts:
     - **Part I — Implemented `v1.5.0` Core, Beta & Experimental Architecture (`[IMPLEMENTED]`)**: Covers the Core Simulation Engine, Immutable State & Bitemporal Context, Phase-Aware Constraint Engine, Pluggable & Grey-Box Projected Residual Dynamics, Scenario Evaluation & Decision Intelligence, Flagship Fraud Cascade & 5-Rung Comparative Ladder, and Experimental Subsystems (`swe_ecosystem`, `domain_composition`, `dynamics_torch`).
     - **Part II — Planned `v2.0+` Vertical Extensions & Research Prototypes (`[PLANNED]`)**: Preserves the full technical specifications for the Mechanistic Router, CT-TGNN (`EvolveGCN`), Financial World Model Vertical, and AIOps World Model Vertical with explicit `[PLANNED]` (`v2.0+`) status banners clarifying that these vertical tracks are outside the `v1.5.0` core public API.

2. **Contradiction 2 — `README.md` Navigation Table Linked Unlabeled Vertical Model/Dataset Cards**:
   - *Before*: `README.md` line 277 linked vertical model/dataset cards alongside `v1.5.0` core reference docs without indicating they belong to `v2.0+` planned verticals, and line 264 used the phrase "financial ledgers".
   - *Resolution*: Updated [`README.md`](../../README.md) so `v1.5.0` documentation links in `README.md` strictly reflect `v1.5.0` core/flagship artifacts (and replaced "financial ledgers" with "monetary ledgers"), eliminating any ambiguity in `README.md`.

3. **Contradiction 3 — `ROADMAP.md` Did Not Explicitly List the `v2.0+` Vertical Tracks**:
   - *Before*: `ROADMAP.md` covered `v1.0.0` through `v2.0.0` core tracks (`T1`–`T9`, SCM, Decision, Calibration, Induction, MCP, WSL) but did not explicitly list the Mechanistic Router, Financial World Model, or AIOps World Model under future planned milestones.
   - *Resolution*: Added a dedicated **`v2.1+ Future Vertical Extensions (Planned — Future Release)`** section in [`ROADMAP.md`](../../ROADMAP.md) covering the Mechanistic Router, Financial World Model Vertical, and AIOps World Model Vertical as planned future milestones (`v2.0+`).

---

## 4. Recommendations

1. **Enforce Explicit Status Badges in [`ARCHITECTURE.md`](../../ARCHITECTURE.md)**:
   - Every section heading in `ARCHITECTURE.md` must carry either `[IMPLEMENTED]` (with tier `v1.5.0 STABLE`, `v1.5.0 BETA`, or `v1.5.0 EXPERIMENTAL`) or `[PLANNED]` (`v2.0+ Future Milestone / Research Prototype`).
2. **Maintain Strict Separation Between `v1.5.0` Certified Core and `v2.0+` Vertical Prototypes**:
   - Keep `ewm_engine.__all__` and `README.md` scoped strictly to `v1.5.0` certified core capabilities until the `v2.0+` verticals complete external benchmark certification.
