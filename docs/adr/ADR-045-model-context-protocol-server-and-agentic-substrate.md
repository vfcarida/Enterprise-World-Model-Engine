# ADR-045: Model Context Protocol (MCP) Server and Deterministic Agentic Substrate

- **Status:** Accepted
- **Date:** 2026-10-16
- **Deciders:** Core Engineering, Agentic Systems & Interoperability Teams
- **Consulted:** Causal SCM Team, Evaluation Team, Security Team
- **Informed:** Community Maintainers, Release Engineers

---

## Context

The artificial intelligence and autonomous agent ecosystem has converged on an essential architectural principle: **do not let the LLM be the world model; let the LLM propose, orchestrate, and interpret, while a deterministic, causal engine executes, validates, and evaluates state transitions.**

Published empirical evidence decisively underscores this necessity:
1. **Implicit World Model Incoherence:** Vafa et al. (NeurIPS 2024 / ICML 2025, arXiv:2406.03689) demonstrate that generative models fail to maintain coherent internal state representations, frequently hallucinating physical states, violating basic conservation laws, and predicting non-descendant changes under interventions.
2. **Generative Physics-IQ Deficit:** Kang et al. (ICML 2025, arXiv:2411.02385) prove that perceptual plausibility is decoupled from physical truth; autoregressive and diffusion decoders invent numbers rather than respecting invariant laws.
3. **Causal Fallibility of Language Agents:** Gkountouras et al. (arXiv:2410.19923) demonstrate that language agents without an external causal oracle produce spurious correlations and confabulate counterfactual outcomes.

The Model Context Protocol (MCP, specification revision 2025-06-18) has emerged as the universal protocol for connecting agentic hosts (Claude Desktop, Cursor, OpenAI Agents SDK, LangGraph, Google ADK, and pydantic-ai) to external computational tools and verifiable resources.

To solve the distribution barrier and make the Enterprise World Model Engine universally accessible to the entire agentic install base without framework-specific glue code, EWM Engine requires a native MCP server and first-class agentic toolsets.

---

## Decision

We introduce an official MCP server, first-class `pydantic-ai` integration, and a deterministic Game-Master arbiter under optional extras (`[mcp]` and `[agents]`):

1. **Isolation in Optional Extras (`[mcp]`, `[agents]`):**
   - The runtime core of `ewm-engine` remains strictly minimal (`pydantic>=2.6`, `numpy>=1.26`, `pyyaml>=6.0`).
   - The official `mcp` SDK (v2.3.0+) and `pydantic-ai` live exclusively in optional extras `[mcp]` and `[agents]`.
   - Core unit and regression tests run hermetically without requiring `mcp` or `pydantic-ai`.

2. **Safe Boundary and Untrusted Caller Model:**
   - Every tool call receives inputs from untrusted LLMs or autonomous agents.
   - All tool arguments are strictly validated through Pydantic schemas configured with `extra="forbid"`.
   - Hard execution bounds are enforced: `MAX_HORIZON = 100`, `MAX_SAMPLES = 1000`, and `MAX_ENTITIES = 500`.
   - Specifications are parsed exclusively through `safe_load_yaml`; zero `eval()`, `exec()`, or custom deserialization tags are allowed.

3. **Exposed MCP Tools, Resources, and Prompts:**
   - **Tools:**
     - `ewm.simulate(world_yaml, horizon, samples, seed)`: Forward stochastic simulation based strictly on declared rules and seed; returns state fingerprint, evidence level, and trace URIs.
     - `ewm.counterfactual(world_yaml, intervention, seed, horizon)`: Evaluates Pearl three-step counterfactual trajectories under Twin-Rollout exogenous noise coupling.
     - `ewm.check_constraints(world_yaml, action)`: Validates actions against physical, operational, and financial invariants prior to execution.
     - `ewm.decide(world_yaml, candidates, satisficing_threshold, robustness_metric, horizon, samples, seed)`: Ranks decision candidates under parametric uncertainty.
     - `ewm.explain(identifier)`: Retrieves the cryptographic provenance trace DAG for a simulation or decision.
   - **Resources:**
     - Dynamic queryable resources: `ewm://worlds/{world_id}`, `ewm://runs/{run_id}`, and `ewm://traces/{trace_id}`.
   - **Prompts:**
     - Contextual prompt templates: `author_what_if` and `explain_result`.

4. **Deterministic Game-Master Bridge (`EWMWorldArbiter`):**
   - Implements the Concordia/Smallville Game-Master pattern where LLMs propose natural language or structured intent, and EWM acts as the deterministic `WorldArbiter`:
     $$\text{LLM Intent} \xrightarrow{\text{Parse}} \text{Action} \xrightarrow{\text{Validate}} \text{Transition} \xrightarrow{\text{Audit}} \text{Observation} + \text{SHA-256 Fingerprint}$$
   - Pre-action constraint checks reject non-compliant intents, returning grounded feedback while leaving the world state untouched.

5. **Provenance-Strict Mode (`--provenance-strict`):**
   - **Low-Evidence Default:** Any WorldSpec received from an agentic host defaults to `EvidenceLevel.ASSUMED`, preventing LLM auto-promotion of evidence tiers.
   - **Refusal to Assert Non-Identifiable Queries:** Interventions targeting unobserved confounders or missing structural equations explicitly return `status="NOT_IDENTIFIED"` and `refusal=True` with an E-value sensitivity score, rather than hallucinating numbers.
   - **Cryptographic Audit Trail:** Every interaction logs an immutable `AuditEvent` with SHA-256 fingerprints of inputs, outputs, and timestamps, directly supporting NIST AI RMF 1.0 (Measure/Manage) and EU AI Act (Article 14 Human Oversight) compliance.

---

## Consequences

### Positive
- Instantly connects EWM Engine to the entire modern agentic ecosystem (Claude Desktop, Cursor, OpenAI Agents SDK, LangGraph, pydantic-ai) with zero bespoke glue code.
- Establishes mathematical and causal honesty as an API contract: the server refuses non-identifiable queries instead of inventing plausible figures.
- Provides regulatory compliance artifacts (NIST AI RMF, EU AI Act) out-of-the-box for enterprise agent deployments.
- Ensures core runtime remains lightweight, fast, and dependency-free.

### Negative / Trade-offs
- Maintaining parity between MCP tool signatures and evolving core APIs requires disciplined contract testing.
- Supporting both stdio and SSE transports introduces dependency on the official `mcp` SDK networking layer.
