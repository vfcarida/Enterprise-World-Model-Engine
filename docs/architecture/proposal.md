# Architecture Proposal: Enterprise World Model Engine (EWM Engine)

**Status:** Approved & Implemented (v0.1.0)  
**Target:** Open-Source Reference Implementation for Action-Conditioned Organizational Simulation  
**License:** Apache License 2.0  

---

## A. Architecture Summary

### 1. Intellectual Context & Core Thesis
Traditional machine learning systems optimize predictive mappings:
$$\hat{Y} = f_\theta(X)$$
Examples include intent classification, demand forecasting, customer churn prediction, and predictive maintenance. While foundation models have transformed this landscape by learning reusable representations across modalities, **observational prediction and representation are not equivalent to maintaining an executable model of how a system evolves under intervention**.

An **Enterprise World Model** formalizes an executable transition abstraction:
$$P(S_{t+1} \mid S_t, A_t, E_t, \theta)$$
where the system state $S_t$ encapsulates:
- Structural topology (entities, hierarchical relations, dependencies);
- Finite physical and organizational resources (inventory, budgets, capacity, personnel);
- Cumulative organizational memory and historical trajectories;
- Active regulatory constraints and operational policies;
- Exogenous environmental context.

**Core Definition:**  
> An *Enterprise World Model* is an action-conditioned, uncertainty-aware representation of an organization's or ecosystem's evolving state, integrating learned dynamics, agent behavior, exogenous events, memory, and explicit operational constraints to evaluate possible consequences of interventions before deployment.

### 2. The Four Epistemic Distinctions
EWM Engine strictly delineates four fundamentally different analytical questions:

| Epistemic Question | Formal Question | Scope in EWM Engine |
| :--- | :--- | :--- |
| **Prediction** | What is likely to happen next? ($P(Y \mid X)$) | Auxiliary input / contextual forecast |
| **Simulation** | How could this system evolve over time? ($P(S_{1:T} \mid S_0, \theta)$) | Core rollout capability |
| **Intervention** | What could happen if we perform action $X$? ($P(S_{1:T} \mid \text{do}(X))$) | Primary design objective |
| **Decision Evaluation** | How do consequences of $X$ compare with $Y$ and $Z$? | Multi-scenario counterfactual comparison |

EWM Engine explicitly rejects the conflation of observational correlation with interventional causality: observational prediction alone does not identify causal effects without structural assumptions.

### 3. What EWM Engine Is vs. What It Is Not

**What EWM Engine Is:**
- An integration architecture for executable socio-technical world dynamics.
- A hybrid simulation substrate combining explicit domain rules with probabilistic dynamics.
- A counterfactual branching engine that allows organizations to simulate *"What if we do this instead?"* before real-world deployment.
- The environment model that autonomous agents or decision-makers consult before acting.

**What EWM Engine Is NOT:**
- Another chatbot, RAG, or prompt-orchestration framework.
- An AutoGen, LangGraph, or CrewAI replacement (agents interface as external actors).
- Purely an RL environment wrapper or discrete-event simulator replacement.
- A financial market simulator or 3D digital-twin visualization tool.
- A proprietary black-box neural predictor or JEPA-only implementation.

### 4. Core Architectural Principle: Structure vs. Probabilistic Dynamics
The single most critical architectural tenet of EWM Engine is:

$$\text{KNOWN STRUCTURE} \quad \oplus \quad \text{LEARNED / STOCHASTIC DYNAMICS}$$

Deterministic organizational knowledge (conservation of physical goods, legal capacity limits, connectivity topology, precedence constraints) must **never** be forced into the stochastic weights of a neural network when it can be represented explicitly as verifiable constraints.

```
+--------------------------------------------------------------------------+
| KNOWN STRUCTURE (Explicit Rules & Constraints)                           |
| - Resource conservation laws         - Physical capacity bounds          |
| - Topological route availability     - Organizational policies           |
| - Logical precedence & invariants    - Safety & regulatory boundaries    |
+--------------------------------------------------------------------------+
                                    +
+--------------------------------------------------------------------------+
| LEARNED / STOCHASTIC DYNAMICS (Probabilistic Transitions)                |
| - Consumer / regional demand shocks  - Behavioral adaptation             |
| - Weather & environmental cascades   - Network congestion latency        |
| - Unmodeled residual friction        - Agent interaction patterns        |
+--------------------------------------------------------------------------+
```

### 5. Formal Canonical World State Representation
The state of the enterprise at discrete time step $t$ is formalized as:
$$S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$$

where:
1. $G_t = (V_t, E_t)$ represents the topological entity graph (nodes $V$ = warehouses, shelters, machines; edges $E$ = supplies, connected_to, owns).
2. $R_t$ represents bounded, measurable resources (inventory, vehicle fleet capacity, power, budget).
3. $M_t$ represents cumulative organizational memory, event logs, and operational counters.
4. $\Gamma_t$ represents active operational rules and constraints currently enforced.
5. $C_t$ represents exogenous environmental context (rainfall level, market indices, ambient disruption).

All `WorldState` instances are immutable snapshots equipped with deterministic SHA-256 fingerprinting for auditable provenance.

---

## B. Complete Repository Structure

```
ewm-engine/
|-- .github/
|   |-- workflows/
|   |   |-- ci.yml
|   |   `-- docs.yml
|   |-- ISSUE_TEMPLATE/
|   |   |-- bug_report.yml
|   |   |-- feature_request.yml
|   |   `-- research_proposal.yml
|   `-- PULL_REQUEST_TEMPLATE.md
|-- benchmarks/
|   `-- run_benchmarks.py
|-- docs/
|   |-- adr/
|   |   |-- ADR-001-python-first.md
|   |   |-- ADR-002-pluggable-dynamics.md
|   |   |-- ADR-003-separation-constraints-dynamics.md
|   |   |-- ADR-004-no-llm-dependency.md
|   |   |-- ADR-005-deterministic-reproducibility.md
|   |   `-- ADR-006-python-first-world-specification.md
|   |-- architecture/
|   |   |-- overview.md
|   |   |-- proposal.md
|   |   |-- simulation-lifecycle.md
|   |   `-- state-management.md
|   |-- concepts/
|   |   |-- causality.md
|   |   |-- constraints.md
|   |   |-- systemic-traces.md
|   |   |-- uncertainty.md
|   |   `-- world-models.md
|   |-- examples/
|   |   |-- civicflow.md
|   |   `-- minimal-world.md
|   |-- api/
|   |   `-- reference.md
|   |-- getting-started.md
|   `-- index.md
|-- examples/
|   |-- minimal_world/
|   |   `-- run.py
|   `-- civicflow/
|       |-- __init__.py
|       |-- actors.py
|       |-- constraints.py
|       |-- dynamics.py
|       |-- run.py
|       `-- world.py
|-- src/
|   `-- ewm_engine/
|       |-- __init__.py
|       |-- __main__.py
|       |-- exceptions.py
|       |-- actors/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- rule_based.py
|       |   `-- stochastic.py
|       |-- cli/
|       |   |-- __init__.py
|       |   `-- main.py
|       |-- constraints/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- registry.py
|       |   |-- results.py
|       |   `-- standard.py
|       |-- core/
|       |   |-- __init__.py
|       |   |-- actions.py
|       |   |-- entities.py
|       |   |-- events.py
|       |   |-- resources.py
|       |   |-- state.py
|       |   |-- types.py
|       |   `-- world.py
|       |-- dynamics/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- composite.py
|       |   |-- deterministic.py
|       |   |-- learned.py
|       |   `-- stochastic.py
|       |-- evaluation/
|       |   |-- __init__.py
|       |   |-- comparison.py
|       |   |-- metrics.py
|       |   `-- uncertainty.py
|       |-- integrations/
|       |   `-- README.md
|       |-- provenance/
|       |   |-- __init__.py
|       |   |-- evidence.py
|       |   |-- metadata.py
|       |   `-- trace.py
|       `-- simulation/
|           |-- __init__.py
|           |-- branching.py
|           |-- engine.py
|           |-- scenario.py
|           `-- trajectory.py
|-- tests/
|   |-- conftest.py
|   |-- integration/
|   |   |-- test_civicflow.py
|   |   `-- test_minimal_world.py
|   |-- property/
|   |   `-- test_properties.py
|   |-- regression/
|   |   `-- test_regression.py
|   `-- unit/
|       |-- test_actors.py
|       |-- test_branching.py
|       |-- test_constraints.py
|       |-- test_core.py
|       |-- test_dynamics.py
|       |-- test_evaluation.py
|       |-- test_provenance.py
|       `-- test_simulation.py
|-- CHANGELOG.md
|-- CITATION.cff
|-- CODE_OF_CONDUCT.md
|-- CONTRIBUTING.md
|-- GOVERNANCE.md
|-- LICENSE
|-- mkdocs.yml
|-- pyproject.toml
|-- README.md
|-- ROADMAP.md
`-- SECURITY.md
```

---

## C. Core Public API Proposal

### 1. Domain Primitives

```python
from ewm_engine.core import (
    Entity,
    Relationship,
    Resource,
    WorldState,
    Action,
    Intervention,
    ExogenousEvent,
    World,
)

# Immutable Domain Entities
warehouse = Entity(
    id="warehouse_north",
    entity_type="warehouse",
    attributes={"location": "Sector A", "hub": True},
)

# Bounded Resources with Capacity Metrics
inventory = Resource(
    id="stock_north",
    name="Clean Water Pallets",
    current_value=150.0,
    min_value=0.0,
    max_value=300.0,
    unit="pallets",
)

# Immutable State Representation S_t
state = WorldState(
    step=0,
    entities={"warehouse_north": warehouse},
    resources={"stock_north": inventory},
    memory={"cumulative_shortage": 0.0},
    context={"inundation_risk": 0.15},
)
```

### 2. Pluggable Dynamics Engine

```python
from ewm_engine.dynamics import (
    DynamicsModel,
    TransitionResult,
    DeterministicTransferDynamics,
    StochasticDemandDynamics,
    CompositeDynamics,
)

# Composition of structural transfer with stochastic consumer demand
dynamics = CompositeDynamics(
    models=[
        DeterministicTransferDynamics(),
        StochasticDemandDynamics(
            demands={"stock_south": (15.0, 3.0)},  # mean, std
            unmet_demand_memory_key="unmet_south",
        ),
    ]
)
```

### 3. Constraints with Failure Provenance

```python
from ewm_engine.constraints import (
    ConstraintRegistry,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
    ActionTransferAvailabilityConstraint,
)

constraints = ConstraintRegistry()
constraints.register(ResourceNonNegativeConstraint("stock_north"))
constraints.register(ResourceCapacityConstraint("stock_north", max_capacity=300.0))
constraints.register(ActionTransferAvailabilityConstraint())
```

### 4. Actors & Policies

```python
from ewm_engine.actors import Actor, ActorContext
from ewm_engine.core import Action


class ThresholdReplenishmentActor(Actor):
    def __init__(self, target_resource: str, source_resource: str, threshold: float, amount: float):
        self.actor_id = "replenisher"
        self.target = target_resource
        self.source = source_resource
        self.threshold = threshold
        self.amount = amount

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        res = state.get_resource(self.target)
        if res and res.current_value <= self.threshold:
            return [
                Action(
                    action_type="transfer_resource",
                    actor_id=self.actor_id,
                    parameters={
                        "source": self.source,
                        "target": self.target,
                        "amount": self.amount,
                    },
                )
            ]
        return []
```

### 5. Counterfactual Branching & Comparative Evaluation

```python
from ewm_engine.simulation import SimulationEngine, Scenario, branch_world
from ewm_engine.evaluation import compare_scenarios

# 1. Initialize World
world = World(initial_state=state, dynamics=dynamics, constraints=constraints)

# 2. Branch Counterfactual Scenarios from Identical Initial State
world_a = branch_world(world)
world_b = branch_world(world)

world_b.add_actor(
    ThresholdReplenishmentActor("stock_south", "stock_north", threshold=40.0, amount=40.0)
)

# 3. Simulate Monte Carlo Rollouts
engine = SimulationEngine()
result_a = engine.simulate(
    Scenario(name="Status Quo", world=world_a, horizon=24, samples=100, seed=42)
)
result_b = engine.simulate(
    Scenario(name="Proactive Policy", world=world_b, horizon=24, samples=100, seed=42)
)

# 4. Compare Trajectory Distributions
comparison = compare_scenarios(
    baseline=result_a, candidates=[result_b], metrics=["resource_stock_south", "violations_count"]
)
print(comparison.summary_table())
```

---

## D. Dependency Strategy: Minimal, Pure Core

To ensure maximum scientific longevity and enterprise portability, the core engine adheres to strict zero-forced-dependency rules:

| Dependency | Purpose | Type |
| :--- | :--- | :--- |
| `python >= 3.11` | Modern runtime with pattern matching, `StrEnum`, strict typing | Core Runtime |
| `pydantic >= 2.0` | High-performance schema validation, serialization, immutability | Core Required |
| `numpy >= 1.26` | High-speed vectorized quantile computation and PRNG streams | Core Required |
| `networkx >= 3.0` | Graph representation for systemic dependency DAGs and traces | Core Required |
| `torch` / `scikit-learn` | Optional learned dynamics regression models | Optional `[ml]` |
| `z3-solver` / `ortools` | Optional SMT constraint solving and MILP optimization | Optional `[solvers]` |
| `pytest`, `hypothesis` | Test runner and property-based verification | Dev `[dev]` |
| `ruff`, `mypy` | High-speed linting and strict static type verification | Dev `[dev]` |
| `mkdocs-material` | Scientific documentation site generator | Dev `[dev]` |

**Zero Forced Requirements:**
The core engine runs completely offline without GPUs, LLM APIs, cloud services, Docker, or external databases.

---

## E. Simulation Lifecycle & Execution Flow

```
                      +-----------------------------+
                      |   OBSERVATIONS / CONTEXT   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |      INITIAL WORLD STATE    |
                      |  S_0 = (G_0, R_0, M_0, ...) |
                      +-----------------------------+
                                     |
                +--------------------+--------------------+
                |                                         |
     [Baseline World]                           [Counterfactual Branch]
     Intervention: None                         Intervention: Policy B
                |                                         |
                +--------------------+--------------------+
                                     |
                                     v
                 +---------------------------------------+
                 |    MONTE CARLO SEED SEQUENCE SPAWN    |
                 |      SeedSequence(seed).spawn(K)      |
                 +---------------------------------------+
                                     |
                   Trajectory Rollout k = 1 ... K:
                                     |
    +---> Step t = 0 ... T-1:        |
    |                                v
    |            +---------------------------------------+
    |            |   1. SAMPLE EXOGENOUS EVENTS (E_t)    |
    |            |      Rainfall, shocks, breakdowns     |
    |            +---------------------------------------+
    |                                |
    |                                v
    |            +---------------------------------------+
    |            |   2. PROPOSE ACTIONS (Actors / Policy)|
    |            +---------------------------------------+
    |                                |
    |                                v
    |            +---------------------------------------+
    |            |   3. PRE-VALIDATE PROPOSED ACTIONS    |
    |            |   (Reject invalid action transfers)   |
    |            +---------------------------------------+
    |                                |
    |                                v
    |            +---------------------------------------+
    |            |   4. DYNAMICS MODEL TRANSITION        |
    |            |   next_state, trace = transition(...) |
    |            +---------------------------------------+
    |                                |
    |                                v
    |            +---------------------------------------+
    |            |   5. POST-VALIDATE NEW STATE S_{t+1}  |
    |            |   (Record hard/soft violations)       |
    |            +---------------------------------------+
    |                                |
    |                                v
    |            +---------------------------------------+
    |            |   6. RECORD STEP & SYSTEMIC TRACE     |
    |            +---------------------------------------+
    |                                |
    +--- (t < horizon - 1) ----------+
                                     |
                                     v
                 +---------------------------------------+
                 |   TRAJECTORY DISTRIBUTION REDUCTION   |
                 |   Mean, Std, Median, p05, p95, CVaR05 |
                 +---------------------------------------+
                                     |
                                     v
                 +---------------------------------------+
                 |      MULTI-SCENARIO COMPARISON        |
                 |  Delta metrics vs baseline, p-values  |
                 +---------------------------------------+
```

---

## F. Testing Plan & Invariants

| Test Suite | File / Scope | Verified Behavior & Invariant |
| :--- | :--- | :--- |
| **Unit: Core** | `test_core.py` | State immutability, entity graph indexing, resource bounding, deterministic SHA-256 fingerprinting. |
| **Unit: Dynamics** | `test_dynamics.py` | Deterministic inventory conservation, stochastic demand unmet logging, composite chaining. |
| **Unit: Constraints**| `test_constraints.py`| Hard rejection vs soft penalty provenance, offending value and entity capture. |
| **Unit: Simulation** | `test_simulation.py`| Reproducibility: `SeedSequence(42)` produces bit-identical trajectories across runs. |
| **Unit: Provenance** | `test_provenance.py`| Mermaid export, DAG edge generation, evidence level propagation. |
| **Property (Hypothesis)** | `test_properties.py`| **Invariant 1:** Resource non-negativity under clamped operations.<br>**Invariant 2:** State round-trip serialization preservation.<br>**Invariant 3:** Branch independence (modifying a branch cannot alter original state). |
| **Integration** | `test_minimal_world.py`<br>`test_civicflow.py` | End-to-end multi-step Monte Carlo rollouts; policy comparison delta calculation. |
| **Regression** | `test_regression.py` | Trajectory step-count integrity, delta calculation under invariant metrics. |

---

## G. Documentation Plan

1. **Getting Started**: Installation, Quickstart, Core Mental Model (`getting-started.md`).
2. **Foundational Concepts**:
   - `world-models.md`: Formal mathematical definition, distinction from agent-only and forecasting models.
   - `causality.md`: The four questions, Pearl's ladder, $P(Y \mid X)$ vs $P(Y \mid \text{do}(X))$, epistemic tiers.
   - `uncertainty.md`: Monte Carlo rollouts, quantiles ($p_{05}, p_{50}, p_{95}$), and Tail Risk ($\text{CVaR}_{05}$).
   - `constraints.md`: Separation of structure from learned dynamics; failure provenance.
   - `systemic-traces.md`: Dependency DAGs vs causal graphs; Mermaid and NetworkX export.
3. **Architecture Specifications**:
   - `overview.md`, `state-management.md`, `simulation-lifecycle.md`, `proposal.md`.
4. **Architecture Decision Records (ADRs)**:
   - `ADR-001` through `ADR-006` documenting rationale for pure-Python core, pluggability, and immutability.
5. **Executable Case Studies**:
   - `examples/minimal-world.md`: 50-line supply chain inventory balancer.
   - `examples/civicflow.md`: Municipal flood disaster logistics research benchmark.

---

## H. v0.1 Scope: Boundary Matrix

```
+--------------------------------------------------------------------------+
| MUST HAVE (v0.1.0 MVP - Implemented & Verified)                          |
| - Immutable WorldState S_t = (G_t, R_t, M_t, \Gamma_t, C_t)             |
| - Deterministic SHA-256 fingerprinting for states and scenarios          |
| - Bounded Resource primitives with utilization metrics                   |
| - Pluggable Dynamics Protocol (Deterministic, Stochastic, Composite)     |
| - First-class Constraints (Hard vs Soft) with failure provenance         |
| - Seeded Monte Carlo Simulation Engine with branching                    |
| - Multi-quantile statistical evaluation & ScenarioComparison table      |
| - SystemicTrace dependency DAG with Mermaid flowchart export             |
| - Minimal World & CivicFlow disaster relief reference examples           |
| - 85%+ test coverage, strict Ruff, strict MyPy, MkDocs documentation     |
+--------------------------------------------------------------------------+
| SHOULD HAVE (Immediate v0.2 Enhancements)                                |
| - Declarative YAML/JSON World Specification schema parser                |
| - Receding-horizon online simulation (Model Predictive Control loop)     |
| - TransitionDataset batch export for training offline world models       |
| - Lightweight terminal visualization utilities                           |
+--------------------------------------------------------------------------+
| FUTURE (Long-Term Research Roadmap)                                      |
| - Recurrent State Space Models (RSSM) & JEPA latent representations      |
| - Relational Graph Neural Network (GNN) dynamics                         |
| - Formal SMT / Z3 symbolic verification adapters                         |
| - Google OR-Tools MILP logistics solvers                                 |
| - Causal discovery and instrumental variable evaluation                  |
| - Multi-agent cascade stress benchmarks                                  |
+--------------------------------------------------------------------------+
```

---

## I. Architectural Risks & Mitigations

1. **State Snapshot Memory Overhead in Long Horizons**:
   *Risk:* Generating thousands of Monte Carlo samples across long horizons could create memory pressure if complete state graphs are deeply cloned.
   *Mitigation:* `WorldState` attributes are immutable dataclasses/Pydantic models with structural sharing; trajectories record compact step records, instantiating full snapshots only on demand.

2. **Randomness & Concurrency Leakage**:
   *Risk:* Python's global `random` or unseeded `numpy.random` calls could compromise run reproducibility.
   *Mitigation:* Global RNG is prohibited. The simulation engine uses `np.random.SeedSequence(seed).spawn(samples)` to assign an isolated `np.random.Generator` to every rollout trajectory.

3. **Silent State Corruption vs Constraint Verification**:
   *Risk:* Clamping or raising exceptions immediately on resource overflow prevents the constraint engine from observing and penalizing violations.
   *Mitigation:* Resources allow unconstrained values when `enforce_bounds=False`, permitting `ConstraintRegistry` to record failures with full provenance.

4. **Pseudo-Causal Overclaiming in Systemic Traces**:
   *Risk:* Users might interpret simulation dependency traces as empirically proven physical causality.
   *Mitigation:* Explicit `EvidenceLevel` tagging on every node and transition. Traces are explicitly documented as *simulated dependency graphs*, not econometric causal discoveries.

---

## J. Implementation Milestones

- **Milestone 1 (Foundation):** Environment setup, `pyproject.toml`, core domain primitives (`WorldState`, `Entity`, `Resource`, `Action`, `Event`).
- **Milestone 2 (Dynamics):** `DynamicsModel` protocol, deterministic transfer, stochastic demand, and sequential composite dynamics.
- **Milestone 3 (Constraints):** `Constraint` base, `ConstraintRegistry`, hard/soft failure provenance reporting.
- **Milestone 4 (Simulation):** `SimulationEngine`, `Scenario`, `Trajectory`, deterministic PRNG seeding.
- **Milestone 5 (Branching & Comparison):** `branch_world`, multi-quantile statistical reduction, `compare_scenarios`.
- **Milestone 6 (Provenance):** `EvidenceLevel`, SHA-256 fingerprints, `SystemicTrace` with Mermaid DAG export.
- **Milestone 7 (Reference Examples):** Minimal Supply Chain and CivicFlow Disaster Response.
- **Milestone 8 (Documentation & Governance):** MkDocs Material site, ADRs, `README.md`, `ROADMAP.md`, Apache 2.0 license.
- **Milestone 9 (Quality Gates & Verification):** Strict Ruff, MyPy, 33/33 tests passing, 88.25% coverage, wheel build verification.
