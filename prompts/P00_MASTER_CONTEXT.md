# P00 — Master Context (read before any other prompt)

> This file is shared context for the EWM Engine evolution prompts (`P01`–`P12`).
> Paste it (or reference it) at the top of any session that runs one of the feature prompts.
> It does not itself ask you to change code.

## What the project is

**Enterprise World Model Engine (EWM Engine)** — an open-source, domain-neutral Python framework for defining, simulating, evaluating, and (eventually) learning the dynamics of complex organizational and socio-technical systems. Its purpose is to answer **"what could happen if we do this?"** (intervention / scenario comparison), not merely **"what is likely next?"** (prediction).

The engine separates **what is KNOWN** (explicit rules/constraints) from **what is LEARNED** (probabilistic dynamics). Agents/policies *orbit* the engine; they are not the engine.

Repository: `vfcarida/Enterprise-World-Model-Engine`. Primary language: Python 3.11+. Everything in the repo (code, comments, docstrings, docs, tests, templates, errors, CLI) is in **English**. The authoritative spec `docs/specs/spec-driven-development.md` is written in pt-BR by owner decision — it is the one English exception.

## Current state (the starting line for this evolution)

The repo has **already shipped `v1.0.0` as a stable contract**. All 24 acceptance criteria (`AC-001`…`AC-024`) are satisfied and mapped to tests in `docs/specs/v1-convergence.md`. A post-release audit (`AUDIT-001`…`AUDIT-006`) is closed. Do **not** re-implement v1.0.0; **build forward** (v1.1 → v2.0+).

Stable public surface (`ewm_engine.__all__`): `Action, Constraint, ConstraintPhase, ConstraintResult, ConstraintSeverity, DynamicsModel, Entity, EvidenceLevel, ExogenousEvent, Provenance, Relationship, Resource, Scenario, SimulationEngine, SimulationResult, TraceEdge, Trajectory, TrajectoryStatus, TransitionResult, World, WorldState, compare_scenarios, __version__`.

Package tree (abridged):
```
src/ewm_engine/
  core/        state, world, actions, entities, events, resources, spec, types, _canonical
  dynamics/    base, deterministic, stochastic, composite, learned (experimental)
  constraints/ base, registry, results, standard
  actors/      base, rule_based, stochastic
  simulation/  engine, scenario, trajectory, branching, metrics, mpc (experimental)
  evaluation/  metrics, comparison, uncertainty, pareto
  provenance/  evidence, trace, metadata, html_visualizer
  serialization/ json, yaml
  hooks/       protocol
  integrations/ agents, solvers, ortools, gym   (ALPHA adapters)
  experimental/ (re-exports LearnedDynamics, LinearResidualDynamics, TransitionDataset, MPC)
  cli/ main
tests/ unit integration property contract security architecture regression examples benchmark
docs/ adr(16) concepts architecture api specs examples integrations audit
schemas/ *.schema.json (JSON Schema Draft 2020-12)
```

Core runtime deps are only `pydantic>=2.6`, `numpy>=1.26`, `pyyaml>=6`. Optional extras: `graphs` (networkx), `solvers` (z3), `ml` (torch, gymnasium), `rl` (gymnasium), `docs`, `dev`, `all`.

## Non-negotiable invariants (apply to EVERY prompt)

1. **Authority chain.** `docs/specs/spec-driven-development.md` > accepted ADRs > feature specs > contract/acceptance tests > source > README/guides. A change that contradicts the spec must change the spec (with justification), not route around it via an ADR.
2. **FUTURE items stay out of core.** Learned/neural dynamics, JEPA, RSSM, GNN, LLM actors, solvers, distributed/GPU, DSL are **adapters or `experimental`** until an ADR promotes them. Never add Torch/solver/LLM imports to `core`, `dynamics` (base), `constraints`, `simulation`, `evaluation`, or `provenance`.
3. **State is semantically immutable.** `WorldState` and all core models are `frozen=True` with defensive copying; branches must not share mutable state. Any new state-touching code preserves this and is covered by a property test.
4. **RNG is explicit.** Use `np.random.Generator` derived from `SeedSequence`; never `random` or global `numpy.random`. Determinism: *same logical state + config + component versions + seed + supported env → same logical trajectory.*
5. **Constraints keep hard/soft × pre/post semantics.** Hard pre-action rejects the action; hard post-transition invalidates the rollout; the core NEVER silently projects an invalid state to a valid one.
6. **Safe serialization only.** No `pickle`/`eval`/`exec`, no import-by-string, no YAML custom tags, no network calls in import/parse/sim core. Untrusted YAML/JSON → safe load → plain data → Pydantic → trusted registry.
7. **Causal honesty.** Simulation ≠ causal inference. `P(Y|X) ≠ P(Y|do(X))`. Traces are **dependency/systemic traces**; the relation `"causes"` is forbidden on `TraceEdge`. `EvidenceLevel` is declared by components, defaults to `ASSUMED`, and is never auto-upgraded.
8. **No overclaiming.** Forbidden in docs/README without matching evidence: *state-of-the-art, first-ever, revolutionary, production-ready, production-safe, enterprise-ready, causally correct, guarantees optimal decisions, hyper-realistic*. Use maturity labels (Stable / Experimental / Planned Adapter / Research).
9. **No fake features.** Do not create empty classes to claim support. If not implemented, it goes in the roadmap, not the code.
10. **SemVer + API change control.** `1.x` must not break the Stable surface. Any change to `__all__`, public signatures, serialized fields, schema semantics, enum values, exception contract, simulation ordering, or fingerprint algorithm needs an **API Change Proposal** (template in `.github/API_CHANGE_PROPOSAL.md`). Boundary/dependency/determinism/security/causal changes need an **ADR**.

## Required workflow for a non-trivial change

Follow the project's own Spec-Driven flow:

```
Feature spec (docs/specs/features/NNN-slug/spec.md)
  → Architecture/API impact + ADR if a boundary/dependency/semantics changes
  → plan.md → tasks.md
  → code + tests + docs + schema updates
  → convergence check (implementation matches spec/plan/tasks)
  → PR using .github/PULL_REQUEST_TEMPLATE.md
```

Branch naming: `feat/… fix/… docs/… refactor/… research/…`. Commit style: `type(scope): summary` with a `Refs: AC-xxx` footer. Default squash merge.

## CI quality gates every PR must pass

`ruff check` (0), `ruff format --check` (0), `mypy --strict` (0) over `src tests examples benchmarks`; unit/property/integration/contract/security/architecture/example tests 100% pass; `mkdocs build --strict` clean; wheel+sdist build and clean-install Quickstart; coverage ≥90% on `core/simulation/constraints/provenance` and ≥85% overall. Keep all GitHub Actions pinned to commit SHAs; releases use OIDC Trusted Publishing.

## Key external references (2024–2026) that justify the direction

- **V-JEPA 2 / V-JEPA 2-AC** — Assran, LeCun, Ballas et al., Meta FAIR, arXiv:2506.09985 (2025). Action-conditioned *latent* world model; template for a future `LearnedDynamics` adapter.
- **Navigation World Models** — Bar, Zhou, Darrell, LeCun, CVPR 2025, arXiv:2412.03572. World model that *plans by rolling out under constraints* — precedent for a planning/controller layer.
- **DreamerV3** — Hafner et al. RSSM latent model-based RL; shape of a learned `DynamicsModel`.
- **Toward Causal Representation Learning** — Schölkopf et al., 2021 (arXiv:2102.11107); plus the causal-world-model line. Grounds `EvidenceLevel` and the v2.0 causal extension.
- **Meta Agents Research Environments (ARE) / Gaia2** — Meta, 2025, arXiv:2509.17158. Environments = rules + tools + content + verifiers, separated from agents; motivates an agent-evaluation harness on EWM worlds.
- **NVIDIA Cosmos; DeepMind Genie 2/3; World Labs** (2024–2025) — the "world foundation model" wave. EWM is the *symbolic/hybrid organizational* counterpart, not a pixel/video model — use this contrast for positioning.
- Engineering: SemVer 2.0.0; Python Packaging User Guide (extras); Pydantic v2 JSON Schema; JSON Schema 2020-12; Ruff; pytest + Hypothesis; GitHub Trusted Publishing + CodeQL; OpenTelemetry Python (API-only in libraries).

A fuller, primary-source-verified reference list (with arXiv IDs across world models, causality, agent-eval environments, org simulation, uncertainty quantification, ML engineering, OR, and the platform-completeness tracks behind `P13`–`P16`) is in **`REFERENCES.md`** beside this file. The expanded roadmap those platform tracks implement is in **`../01_EXPANDED_ROADMAP.md`**. Three convergent 2024–2026 findings should shape the work:
1. observational/predictive world models carry **no** interventional or counterfactual guarantee (now empirically proven);
2. surface realism ≠ systemic understanding → a learned backend must pass a **systemic-invariant consistency gate**;
3. **verifiable, state-based scoring** is the gold standard; LLM-as-judge is scoped only to free-form content.

> Treat these as *motivation and shape*, never as mandatory architecture. The engine must not cristallize on any single model family.
