# Specification-Driven Development (EWM Engine v1 Contract)

> **Status:** Canonical v1 Contract  
> **Source of Truth:** This document defines the authoritative architecture, operating invariants, and acceptance criteria for the Enterprise World Model Engine (`ewm-engine`). It supersedes historical implementation prompts.

---

## 1. Product Mission & Boundaries

> **EWM Engine is not another agent framework. It is the environment model agents can use to ask "what happens if I do this?" before they act.**

### Epistemic Stance
EWM Engine explicitly separates what is structurally known from what is learned or uncertain:
$$\text{KNOWN STRUCTURE} \quad \oplus \quad \text{LEARNED / STOCHASTIC DYNAMICS}$$

- **Prediction** ($P(Y \mid X)$): Passive observational forecast.
- **Simulation** ($P(S_{1:T} \mid S_0, \theta)$): Stochastic state progression.
- **Intervention** ($P(S_{1:T} \mid \text{do}(X))$): Action-conditioned counterfactual rollout.
- **Evaluation**: Comparative multi-scenario delta analysis with explicit uncertainty quantiles.

Simulation runs do **not** constitute causal evidence on their own. Systemic traces record dependency lineage with default evidence level `EvidenceLevel.ASSUMED`.

---

## 2. Hard Invariants & Operating Rules

1. **Semantic Immutability**: All `WorldState`, `Entity`, `Resource`, `Relationship`, and core records are deeply immutable. State transitions emit fresh snapshots with deterministic SHA-256 fingerprinting.
2. **Explicit Randomness**: Random streams are derived strictly from `SeedSequence` per rollout. No component may touch global `random` or global `numpy.random`.
3. **Explicit Constraints**: Invariants and physical conservation bounds are evaluated first-class (`ConstraintPhase.PRE_ACTION` and `ConstraintPhase.POST_TRANSITION`). Constraint failures never silently disappear.
4. **Safe Serialization**: Configuration loading (`WorldSpecification`) relies exclusively on safe parsing (`yaml.safe_load`, JSON). No `eval`, no `exec`, no dynamic arbitrary module imports, and no network requests in core simulation loops.
5. **No Forced Dependencies**: Core requires only `pydantic>=2.6` and `numpy>=1.26`. All ML frameworks (PyTorch), solvers (Z3), graph tools (NetworkX), and LLMs remain optional integrations.

---

## 3. Module Boundaries & Architecture

- `core`: Foundations (`WorldState`, `Entity`, `Resource`, `Relationship`, `Action`, `World`). Must NOT depend on `simulation`, `evaluation`, or `integrations`.
- `constraints`: Invariants and registry. Evaluates actions and states. Must NOT import from `dynamics` or `simulation`.
- `dynamics`: Transition operators. Must NOT import from `simulation`.
- `provenance`: Audit metadata and systemic trace DAG. Must NOT import from `simulation`.
- `simulation`: Rollout orchestrator, branching, and MPC. Imports `core`, `constraints`, `dynamics`, `provenance`.
- `evaluation`: Multi-scenario comparisons and uncertainty distributions.
- `integrations` / `experimental`: Optional bridges (agent adapters, solvers, learned models). Must NOT be imported by core or simulation.

---

## 4. Governance & Change Process

1. Any modification to the Stable public API (`ewm_engine.__all__`) requires an approved **API Change Proposal** (`.github/API_CHANGE_PROPOSAL.md`).
2. Architectural decisions are recorded as ADRs in `docs/adr/` using `.github/ADR_TEMPLATE.md`.
3. Release candidates must pass all criteria in `.github/RELEASE_CHECKLIST.md`.
