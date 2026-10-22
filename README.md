<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Enterprise World Model Engine Logo" src="assets/logo-light.svg" width="680" height="130">
  </picture>
</p>

<p align="center">
  <em>Deterministic, auditable what-if simulation kernel for socio-technical systems — no GPU/LLM required.</em><br>
  <strong>An open-source structural world-model engine evaluated on calibrated uncertainty, hard conservation invariants, and explicit causal guardrails.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ewm-engine/"><img src="https://img.shields.io/pypi/v/ewm-engine?color=blue&label=PyPI" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/ewm-engine/"><img src="https://img.shields.io/pypi/pyversions/ewm-engine?style=flat" alt="Python Versions"></a>
  <a href="https://pepy.tech/project/ewm-engine"><img src="https://static.pepy.tech/badge/ewm-engine" alt="PyPI Downloads"></a>
  <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vfcarida/Enterprise-World-Model-Engine/ci.yml?branch=main&label=CI&style=flat" alt="CI Status"></a>
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat" alt="Documentation"></a>
  <a href="scripts/check_coverage.py"><img src="https://img.shields.io/badge/coverage-%E2%89%A585%25%20gated-success?style=flat" alt="Coverage Gate"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/types-mypy%20strict-1f2937?style=flat" alt="Mypy Strict"></a>
  <a href="docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md"><img src="https://img.shields.io/badge/SLSA-Build%20L2%2B-blue?style=flat" alt="SLSA L2+"></a>
  <a href="https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
  <a href="https://codespaces.new/vfcarida/Enterprise-World-Model-Engine"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat" alt="License: Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/"><b>Docs</b></a> ·
  <a href="#60-second-quickstart"><b>Quickstart</b></a> ·
  <a href="#verified-results--evidence"><b>Evidence</b></a> ·
  <a href="#project-roadmap--maturity"><b>Maturity</b></a> ·
  <a href="#limitations--non-goals"><b>Limitations</b></a> ·
  <a href="#reproduce-in-one-command"><b>Reproduce</b></a> ·
  <a href="docs/scientific/claim-evidence-matrix.md"><b>Claim Registry</b></a> ·
  <a href="docs/research/enterprise-benchmark-suite.md"><b>Benchmarks (A–L)</b></a> ·
  <a href="docs/positioning/landscape.md"><b>Landscape</b></a> ·
  <a href="#citation"><b>Cite</b></a>
</p>

---

## 1. What EWM Engine Is

**Enterprise World Model Engine (`ewm-engine`)** is an open-source, action-conditioned Python simulation kernel that models, simulates, and audits the operational consequences of candidate interventions across typed socio-technical state, explicit resource constraints, and calibrated uncertainty—without requiring a GPU or large language model (LLM) in the core loop.

> [!NOTE]
> **Structural vs. Visual World Models:** Unlike pixel or video world models (such as NVIDIA Cosmos, Google Genie, or Meta V-JEPA 2) that predict sensory frames or latent embeddings, `ewm-engine` simulates explicit organizational state (inventories, financial ledgers, supply topologies, service queues, and regulatory rules). See the [Competitive Landscape](docs/positioning/landscape.md).

---

## 2. The Problem It Solves

Enterprise analytics and supervised machine learning are primarily built to estimate observational associations $P(Y \mid X)$ (*"What is likely to happen if current correlations continue?"*). Operational decision-makers and autonomous planning agents face a structurally different question:

$$\text{"What could happen if we enact intervention } \text{do}(A_t) \text{ instead of the baseline policy?"}$$

When an organization changes dispatch rules, reallocates working capital, or reroutes supply lines, it alters the underlying data-generating process:
- **Purely observational regressors** misestimate action effects under confounding and degrade when runtime configurations shift (the Lucas Critique).
- **Unconstrained neural or LLM predictors** do not enforce physical or accounting balance equations, permitting negative inventories or unbooked cash creation over multi-step rollouts.
- **Static rule engines** enforce known bounds but miss empirical friction, perishability, and stochastic demand surges.

`ewm-engine` provides a domain-independent simulation substrate where candidate policies are evaluated against explicit constraints, coupled counterfactual noise streams, and multi-metric robustness criteria before actions are taken in reality.

---

## 3. Maturity & Status

- **Current Release:** `v1.5.0` (`Development Status :: 4 - Beta`, Apache-2.0 License)
- **Supported Runtime:** Python `>=3.11` (`3.11`, `3.12`, `3.13`) and NumPy `>=1.26.0` under [Scientific Python SPEC 0](docs/spec0-policy.md)
- **Core Dependencies:** `pydantic>=2.6.0`, `numpy>=1.26.0`, `pyyaml>=6.0` (zero mandatory ML, solver, or cloud dependencies in core)
- **Verification Gates:** Strict `mypy` typing, `ruff` lint/format, `>=85%` branch coverage gate (`>=90%` on core modules), Hypothesis property/metamorphic tests, selective mutation ratchet (`scripts/run_mutation_gate.py`), untrusted-input fuzz smoke (`scripts/run_fuzz_smoke.py`), and a 19-claim [Claim-to-Evidence Registry](docs/scientific/claim-evidence-matrix.md).

---

## 4. Distinguishing Architecture

`ewm-engine` separates state evolution into three explicitly governed layers (detailed in [State Management & Formalism](docs/architecture/state-management.md) and [ADR-053](docs/adr/ADR-053-formal-enterprise-world-model-semantics-and-context.md)):

1. **Known Structural Rules & Phase-Aware Constraints ($f_{\text{known}}, \mathcal{V}_{\text{pre}}, \mathcal{V}_{\text{post}}$):** Typed entities, directed relationships, and bounded resources ($R_t \in [R_{\min}, R_{\max}]$). Pre-action constraints reject invalid actions prior to transition; post-transition hard constraints mark violating trajectories `INVALID` without silently fabricating or clamping state.
2. **Feasible-Set Projected Residual Dynamics ($\mathcal{P}_{\mathcal{C}}(g_\theta)$):** Optional empirical or neural residual models capture unmodeled friction and stochastic demand while KKT nullspace projections (`ConservationProjectionLayer`) and explicit `InfeasibleResidualPolicy` rules (`PROJECT`, `REJECT`, `ABSTAIN`, `INVALIDATE`) preserve linear conservation laws within IEEE-754 double-precision tolerance ($\le 10^{-12}$).
3. **Runtime Context ($K_t$) & Epistemic Guardrails:** Separates ontic state ($S_t$), partial observations ($O_t$), belief state ($B_t$), and runtime deployment context ($K_t$). Causal effect queries check backdoor/frontdoor identifiability on the declared `CausalGraph` and fail closed (`UnqualifiedCausalClaimError`) when unobserved confounding leaves backdoor paths open.

---

<a id="60-second-quickstart"></a>

## 5. Minimal Seeded Quickstart

Install the core package:

```bash
pip install ewm-engine
```

Simulate a deterministic baseline against a counterfactual resource transfer intervention in 12 lines of seeded Python:

```python
import ewm_engine as ewm
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

# 1. Define enterprise state with bounded resources
world = ewm.World(
    state=ewm.WorldState(
        resources=[
            ewm.Resource(id="stock_a", current=100.0, max_value=200.0),
            ewm.Resource(id="stock_b", current=20.0, max_value=200.0),
        ]
    ),
    dynamics=DeterministicTransferDynamics(),
)
# 2. Simulate baseline vs. isolated counterfactual branch with fixed seed
baseline = world.simulate(horizon=5, seed=42)
counterfactual = world.branch().simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 30.0,
                "source_resource": "stock_a",
                "target_resource": "stock_b",
            },
        )
    ],
    horizon=5,
    seed=42,
)
# 3. Compare scenarios and inspect the summary table
diff = ewm.compare_scenarios(
    baseline, [counterfactual], metrics=["resource_stock_a", "resource_stock_b"]
)
print(diff.summary_table())
```

```text
=== Scenario Comparison (Baseline: Baseline) ===
Estimand: E_{P_model}[Y_candidate] - E_{P_model}[Y_baseline] (in-simulator P_model; not P_real unless externally validated)
---------------------------------------------------------------------------------------------------------------------------
Metric                    | Scenario             | Mean (Std)         | p50 [p05, p95]       | Delta vs Base [CI] (*sig)   
---------------------------------------------------------------------------------------------------------------------------
resource_stock_a          | Baseline             | 100.00 (+/-0.00)   | 100.00 [100.00, 100.00] | -                           
                          | Counterfactual       | 70.00 (+/-0.00)    | 70.00 [70.00, 70.00] | -30.00 (-30.0%) [-30.00, -30.00] (det)
---------------------------------------------------------------------------------------------------------------------------
resource_stock_b          | Baseline             | 20.00 (+/-0.00)    | 20.00 [20.00, 20.00] | -                           
                          | Counterfactual       | 50.00 (+/-0.00)    | 50.00 [50.00, 50.00] | +30.00 (+150.0%) [+30.00, +30.00] (det)
---------------------------------------------------------------------------------------------------------------------------
```

> [!NOTE]
> Zero-variance deterministic comparisons are explicitly labeled `(det)` (`status_label="deterministic_difference"`, `is_significant=False`) rather than claiming statistical significance (`*`). Stochastic Monte Carlo comparisons (`samples >= 2`) report non-parametric bootstrap confidence intervals under $P_{\text{model}}$.

---

<a id="verified-results--evidence"></a>

## 6. Verified Results & Evidence

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero-proof-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/hero-proof-light.svg">
    <img alt="EWM Engine Proof Hero: Empirical Ground Truth vs Structural Simulation" src="assets/hero-proof-dark.svg" width="940">
  </picture>
</p>
<p align="center">
  <em>SEIR validation card: Analytical ground-truth curve overlaid on EWM forward simulation. Closed-system conservation (S + E + I + R = N) holds within IEEE-754 double-precision tolerance (&lt;= 1e-12; observed max residual 4.55e-13 across 80 steps). Shaded band indicates 90% split-conformal marginal coverage. Regenerate via <code>python scripts/generate_hero_proof.py</code>.</em>
</p>

All empirical claims are verified by committed, machine-readable benchmark artifacts ([`reproduction_manifest.json`](reproduction_manifest.json) and [`docs/results/artifacts/enterprise_benchmark_suite.json`](docs/results/artifacts/enterprise_benchmark_suite.json)):

| Evaluation Suite / Card | Scope & Invariant Tested | Verified Empirical Result | Evidence Tier & Command |
| :--- | :--- | :--- | :--- |
| **Enterprise Benchmark Suite (Families A–L)** | 12 synthetic enterprise failure-mode families comparing `rules-only`, `learned-only`, `static hybrid`, and `hybrid + runtime context discovery` across 13 orthogonal metrics | `CONSTRAINT_PRESERVATION = 1.0000` (vs. `0.7823` for `learned-only`); post-shift recovery in `2` steps on Family D (`MAE = 0.158` vs. `>34.6` unrecovered for static baselines); `DECISION_REGRET = 0.1557` (vs. `46.25` for `learned-only`) | `BENCHMARKED` — `python -m benchmarks.enterprise_suite --seed 42` |
| **SEIR Epidemic Outbreak** | Closed-population conservation ($S+E+I+R=1000$) over 80 steps | Observed max residual $4.55 \times 10^{-13} \le 10^{-12}$; 90% split-conformal coverage $= 1.00$ | `BENCHMARKED` — `python scripts/reproduce.py` |
| **Beer Game & WORLD3 Packs** | Multi-echelon supply bullwhip (Sterman, 1989) & non-linear resource depletion (Meadows et al., 1972) | Bullwhip amplification ratio $= 3.80$; WORLD3 industrial peak at step $t=99$ | `BENCHMARKED` — `python scripts/reproduce.py` |
| **Amortized SBI & TARP** | Empirical posterior coverage across 200 synthetic holdout simulations | Mean TARP calibration error $= 0.0283 \le 0.05$ threshold | `CALIBRATED` — `python scripts/reproduce.py` |
| **Counterfactual SCM & Induction Gate** | Abducted-noise twin rollout & fail-closed refusal under unobserved confounding | Coupled net treatment effect $= +35.0$ units; `0` unidentifiable causal strings emitted | `BENCHMARKED` — `python scripts/reproduce.py` |

See the full **[Claim-to-Evidence Registry](docs/scientific/claim-evidence-matrix.md)** and **[Enterprise Benchmark Suite (A–L) Report](docs/research/enterprise-benchmark-suite.md)**.

---

## 7. Architecture Overview

```mermaid
flowchart LR
    accTitle: Enterprise World Model Engine Architecture Dataflow
    accDescr: End-to-end dataflow illustrating how state observations, runtime context, and constraints initialize the world state, feed into the dynamics engine with actions and exogenous shocks, and pass through constraint verification and scenario evaluation to produce systemic traces.

    OBS[/"Observations & Context K_t"\] --> STATE["World State S_t"]
    RULES{{"Rules & Constraints"}} --> STATE
    EVENTS>Exogenous Events E_t] --> ENGINE[["Dynamics Engine (Known + Projected Residual)"]]
    STATE --> ENGINE
    ACTION(["Action / Intervention A_t"]) --> ENGINE
    ENGINE --> FUTURE[("Future State Distribution")]
    FUTURE --> VERIFY{"Pre/Post Constraint Gate"}
    VERIFY --> EVAL[["Scenario & Robustness Evaluation"]]
    EVAL --> TRACE[/"Systemic Trace & Provenance"/]
    AGENTS{{"Actors / Planners"}} --> ACTION
    EVAL --> AGENTS
```

- **State & Context (`ewm_engine.core`):** Immutable `WorldState` snapshots with canonical SHA-256 fingerprints, `ContextSnapshot` ($K_t$), `ObservationFilter`, and $O(1)$ isolated `world.branch()` snapshots (`AC-005`).
- **Constraints (`ewm_engine.constraints`):** `PRE_ACTION` and `POST_TRANSITION` validation gates with `HARD` and `SOFT` severities (`AC-006`–`AC-008`).
- **Dynamics (`ewm_engine.dynamics`):** Pluggable `DynamicsModel` protocol supporting deterministic rules, stochastic processes, and `LearnedDynamics` residuals with `InfeasibleResidualPolicy` enforcement.
- **Simulation & Counterfactuals (`ewm_engine.simulation`):** Hierarchical `SeedSequence` Monte Carlo rollouts, Pearlian three-step `StructuralCausalModel` twin rollouts, receding-horizon control (`MPC`), and three-tier array backends.
- **Evaluation & Decision (`ewm_engine.evaluation`, `ewm_engine.decision`):** Paired/independent bootstrap confidence intervals, split-conformal intervals, safe-testing e-processes, PRIM scenario vulnerability discovery, and multi-metric robustness analysis.
- **Provenance & Cards (`ewm_engine.provenance`, `ewm_engine.cards`):** Inspectable `SystemicTrace` DAGs tagged with `EvidenceLevel` tiers and Mitchell/Gebru/Croissant-compliant `ModelCard`, `DatasetCard`, and `ScenarioCard` artifacts.

---

<a id="project-roadmap--maturity"></a>

## 8. Capabilities by Maturity Tier

Capabilities are governed across three explicit stability tiers ([Stability Policy](docs/stability-policy.md)):

| Tier | Guarantee & Namespace | Representative Subsystems |
| :--- | :--- | :--- |
| **Stable (`1.x`)** | Backward-compatible root public API (`ewm_engine.__all__`), deterministic CPU execution, zero mandatory ML/solver extras | Core simulation kernel, `World` & immutable `WorldState`, phase-aware `ConstraintEngine`, `DeterministicTransferDynamics`, `compare_scenarios` (5 bootstrap methods + 6-part uncertainty taxonomy), `SplitConformalCalibrator`, `SystemicTrace`, JSON Schema Draft 2020-12 & safe YAML loader, event-sourced `TraceLog`/`ResultStore`, `ModelCard`/`DatasetCard`/`ScenarioCard`, OR-Tools/SciPy/Z3/Gymnasium adapters, FMI/SimPy/Mesa/PySD co-simulation, reporting & FastAPI service |
| **In-Tree Beta (`1.5.0`)** | Verified by unit, contract, and benchmark suites; domain-specific optional extras (`[calibrate]`, `[causal]`, `[induction]`, `[dsl]`, `[mcp]`) | Twin-Rollout Counterfactual SCM (`ewm_engine.simulation.counterfactual`), Grey-Box `ConservationProjectionLayer` & SINDy flywheel, PRIM scenario discovery & multi-metric robustness (`ewm decide`), Amortized SBI calibration & TARP diagnostics (`ewm calibrate`), OCEL 2.0 / causal / invariant induction with refusal gate (`ewm induce`), World Specification Language (`.wsl`) & WMIF v1.0 conformance, MCP stdio server (`ewm mcp`), 12-family Enterprise Benchmark Suite (A–L) |
| **Experimental (`ewm_engine.experimental.*`)** | Research prototypes isolated behind optional extras (`[ml]`, `[jax]`, `[cuda]`); interfaces may evolve across minor releases | `TorchNeuralResidualDynamics` (PyTorch MLP residual with symlog scaling), `GraphNeuralDynamics` (heterogeneous relational GNN), `JaxBackend` (`NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`), `CuPyBackend` (`NON_CERTIFIED_BACKEND`) |

See [ROADMAP.md](ROADMAP.md) and [Platform Capability Tracks](docs/architecture/platform-tracks.md) for the complete subsystem table.

---

<a id="limitations--non-goals"></a>

## 9. Limitations & Non-Goals

### Operational & Scientific Limitations

1. **Simulation Validity Depends on Model Assumptions:** All rollout distributions, bootstrap intervals, and counterfactual deltas are computed *under the declared simulation model* ($P_{\text{model}}$). If a domain model omits a critical real-world feedback loop, in-model metrics cannot automatically measure `model_discrepancy_vs_reality` without external holdout telemetry.
2. **Learned Dynamics & Unmodeled Regime Shifts:** While runtime context discovery ($K_t$) recovers from parameter shifts represented in `ContextSnapshot` (after any audit telemetry delay, as shown in Benchmark Family D), unmodeled structural phase changes outside the context schema (Benchmark Family K) cause raw state-prediction breakdown and require `CompetenceEstimator` abstention.
3. **Causal Conclusions Require Identification:** `world.counterfactual()` and `check_backdoor_identifiability()` evaluate causal validity relative to the declared `CausalGraph`. If latent confounders exist in reality but are omitted from the graph specification, graph-based checks cannot detect them. Observational sensitivity analyses (`compute_sensitivity_e_value`) bound, but do not eliminate, unmeasured confounding.
4. **Calibration & Conformal Assumptions:** Split-conformal prediction intervals guarantee finite-sample **marginal** coverage ($\mathbb{P}(Y_{\text{test}} \in \hat{C}(X_{\text{test}})) \ge 1-\alpha$) under **exchangeability**, not distribution-free conditional coverage ($\mathbb{P}(Y \in \hat{C}(X) \mid X=x)$). Under covariate or regime shift, coverage degrades until recalibrated. TARP (`compute_tarp_coverage`) is an empirical finite-sample diagnostic, not a formal proof of real-world posterior truth.
5. **Three-Tier Numerical Reproducibility Envelope:**
   - `BITWISE_REPRODUCIBLE` (`NumPyBackend` / pure-Python): Bit-identical SHA-256 state fingerprints require an identical pinned runtime (same Python, NumPy, IEEE-754 `float64` dtype, single-threaded BLAS `OMP_NUM_THREADS=1`, and `SeedSequence` tree). Universal bit-identity across arbitrary CPU ISAs, compilers, or multi-threaded BLAS reductions is not claimed.
   - `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` (`JaxBackend` in `float64`): Certified within `rtol=1e-6, atol=1e-7` due to XLA fusion and FMA instruction selection.
   - `NON_CERTIFIED_BACKEND` (`CuPyBackend`): Non-associative GPU floating-point atomic reductions are non-deterministic; refused when `certify_equivalence=True`.
6. **Decision Support, Not Autonomous Truth:** Model outputs, scenario rankings, and PRIM vulnerability boxes are decision-support artifacts for human operators and governed pipelines, not autonomous ground truth.

### Explicit Non-Goals

- **Not a Video, Pixel, or 3D World Model:** Does not render camera frames, point clouds, or 3D physics meshes.
- **Not an LLM Agent Orchestrator:** External agents may call `ewm-engine` via Python or MCP (`ewm mcp`), but the simulation kernel itself is a deterministic mathematical engine, not a prompt workflow runner.
- **Not an Assumption-Free Causal Discovery Oracle:** `ewm induce causal` outputs Markov Equivalence Classes (CPDAG/PAG) and refuses to assert unique causal directions from observational data alone without interventional or refutation backing.

---

## 10. Documentation Map (Diátaxis)

Full documentation is published at **[https://vfcarida.github.io/Enterprise-World-Model-Engine/](https://vfcarida.github.io/Enterprise-World-Model-Engine/)**:

| Diátaxis Mode | Purpose | Key Entry Points |
| :--- | :--- | :--- |
| **Tutorials** *(Learning-oriented)* | Guided, end-to-end lessons for new users | [Getting Started](docs/getting-started.md) · [The Whole Game in 10 Minutes](docs/tutorials/whole-game-in-ten-minutes.md) · [Building Your First World](docs/tutorials/first-world.md) · [Simulating Interventions](docs/tutorials/counterfactual-interventions.md) · [CivicFlow Walkthrough](docs/examples/civicflow.md) |
| **How-To Guides** *(Goal-oriented)* | Step-by-step operational recipes | [Counterfactual Twin-Rollouts](docs/guides/counterfactual-twin-rollouts.md) · [Scenario Discovery with PRIM](docs/guides/scenario-discovery-prim.md) · [Enforcing Conservation Layers](docs/guides/conservation-layers.md) · [Amortized Calibration](docs/guides/calibration.md) · [Cookbook Index](docs/cookbook/index.md) |
| **Reference** *(Information-oriented)* | Exact APIs, schemas, CLI, cards, and evidence | [API Reference](docs/reference/api/index.md) · [CLI Reference](docs/reference/cli.md) · [JSON Schemas](docs/reference/schemas.md) · [Claim-to-Evidence Registry](docs/scientific/claim-evidence-matrix.md) · [Verified Bibliography](docs/REFERENCES.md) · [Model & Scenario Cards](docs/cards/TEMPLATE.md) |
| **Explanation** *(Understanding-oriented)* | Mathematical foundations, trade-offs, and benchmarks | [What is a World Model?](docs/concepts/world-models.md) · [Causal Epistemology](docs/concepts/causality.md) · [Uncertainty & Rollouts](docs/concepts/uncertainty.md) · [Enterprise Benchmark Suite (A–L)](docs/research/enterprise-benchmark-suite.md) · [Competitive Landscape](docs/positioning/landscape.md) · [Handbook](docs/handbook/index.md) |

---

<a id="reproduce-in-one-command"></a>

## 11. Reproducibility

Under a pinned Python/NumPy environment (`uv sync --extra dev`), all benchmark cards, WMIF reference worlds, and the 12-family Enterprise Benchmark Suite can be reproduced with single commands:

```bash
# 1. Verify the 7 scientific validation cards, 3 WMIF oracle worlds, and SoA fingerprints
python scripts/reproduce.py

# 2. Run the 12-family Enterprise Benchmark Suite (A-L) across all 4 baselines
python -m benchmarks.enterprise_suite --seed 42

# 3. Verify release version coherence, schema freshness, and security controls
python scripts/check_release_truth.py
```

| Environment | Launch / Command | Scope |
| :--- | :--- | :--- |
| **Local CLI (`uv`)** | `uv run python scripts/reproduce.py` | Verifies [`reproduction_manifest.json`](reproduction_manifest.json) (`CERTIFIED_RELEASE` on clean git worktrees; `REPRODUCIBLE_DIRTY_WORKTREE` when uncommitted edits exist) |
| **Interactive Binder** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) | Executes reproduction notebook in browser |
| **GitHub Codespaces** | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) | Pre-configured container dev environment |
| **Pinned Docker** | `docker run --rm -v $(pwd):/app ewm-engine:pinned` | Isolated Linux `float64` container with `OMP_NUM_THREADS=1` |

See the [Reproducibility & Citation Guide](docs/reproducibility.md) and [Performance & Reproducibility Contract](docs/performance.md).

---

## 12. Security, Governance & Contributing

- **Supply-Chain & Runtime Security ([SECURITY.md](SECURITY.md), [ADR-034](docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md)):** Safe YAML deserialization (`StrictSafeLoader`), closed `WorldFactory` registries (`AC-003`), 100% 40-char commit-SHA pinned GitHub Actions (`scripts/check_action_pins.py`), least-privilege workflow permissions, CodeQL, OSSF Scorecard, `pip-audit`, OSV-Scanner/Grype, CycloneDX + Syft SBOMs, and PyPI OIDC Trusted Publishing with PEP 740 / SLSA build provenance attestations.
- **Contributing & Governance:** Contributions are welcome! Review [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SPEC 0 Support Policy](docs/spec0-policy.md), and [Deprecation Policy](docs/deprecation-policy.md).

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vfcarida"><img src="https://avatars.githubusercontent.com/u/1043329?v=4" width="100px;" alt="Vinicius Caridá"/><br /><sub><b>Vinicius Caridá</b></sub></a><br /><a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Code">💻</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Documentation">📖</a> <a href="#design-vfcarida" title="Design">🎨</a> <a href="#ideas-vfcarida" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-vfcarida" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-vfcarida" title="Maintenance">🚧</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/pulls?q=is%3Apr+reviewed-by%3Avfcarida" title="Reviewed Pull Requests">👀</a> <a href="#security-vfcarida" title="Security">🛡️</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

---

<a id="citation"></a>

## 13. Citation & Publication Status

If you use `ewm-engine` in research or technical evaluations, please cite the software via [`CITATION.cff`](CITATION.cff) or the BibTeX entry below:

```bibtex
@software{carida2026ewmengine,
  author    = {Carid{\'a}, Vinicius},
  title     = {Enterprise World Model Engine: An Open Framework for Modeling, Simulating, and Evaluating Organizational Dynamics},
  year      = {2026},
  url       = {https://github.com/vfcarida/Enterprise-World-Model-Engine},
  version   = {1.5.0},
  license   = {Apache-2.0}
}
```

- **In-Tree Companion Manuscripts (Drafts — Not Yet Externally Published):**
  - *Methods & Benchmark Manuscript*: [`paper/arxiv/paper.md`](paper/arxiv/paper.md) and [`paper/arxiv/paper.tex`](paper/arxiv/paper.tex) (*arXiv identifier not yet assigned*).
  - *JOSS Software Manuscript*: [`paper/paper.md`](paper/paper.md) and [`paper/paper.bib`](paper/paper.bib) (*JOSS DOI not yet assigned*).
- **External Archival Metadata:** Configured in [`.zenodo.json`](.zenodo.json) and [`docs/scientific/swhid.md`](docs/scientific/swhid.md) (*Zenodo DOI and public SWHID not yet minted*).
