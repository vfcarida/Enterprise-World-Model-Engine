# ADR-023: Safe Declarative World Specification Language (WSL)

**Status:** Accepted  
**Date:** 2026-10-05  
**Context:** EWM Engine v2.0 Candidate (Prompt P10 / Horizon C)  
**Related Specs:** `FEAT-003`, `ADR-004`, `ADR-006`, `ADR-011`, `ADR-022`

---

## Context

In `ADR-006`, we deferred the introduction of a general-purpose Domain-Specific Language (DSL) until the programmatic Python API stabilized. Over minor releases `v1.0.0` through `v1.1.0`, interface patterns for entities, relationships, bounded resources, transition models, constraints, and scenario evaluations have proven robust.

Enterprise users now require a unified, declarative serialization format—the **World Specification Language (WSL)**—to share world definitions across teams, archive model baselines, and author simulations in language-agnostic YAML or JSON.

Crucially, because specification files are frequently exchanged across organizational boundaries or submitted by untrusted agents, the specification engine must adhere to strict security invariants: zero remote code execution, zero arbitrary imports, and zero custom deserialization hooks.

---

## Decision

### 1. Minimal, Declarative Superset (`wsl/2.0.0`)
We introduce `WSLDocument` (`ewm_engine.serialization.wsl`), an expressive declarative superset of the legacy `WorldSpec`:
- Versioned schema identifier: `schema_version = "wsl/2.0.0"` or `"2.0.0"`.
- Declarative top-level blocks: `metadata`, `temporal`, `entities`, `relationships`, `resources`, `dynamics`, `constraints`, `event_sources`, and `scenarios`.
- Full round-trip compilation: `compile_wsl(doc) -> World` and `export_wsl(world) -> WSLDocument`.

### 2. Strict Rejection of Scope Creep & Expression Languages
To maintain complete safety, WSL is strictly **declarative data**, not a Turing-complete programming language:
- **NO Embedded Code / Eval**: No Python expressions, math formula parsers, or string interpolation hooks (`${...}`).
- **NO Custom YAML Tags**: Deserialization uses `StrictSafeLoader` which raises `SerializationSecurityError` on any custom tag (e.g. `!python/object`, `!cmd`).
- **NO Import-by-String**: Models and constraints cannot reference arbitrary Python module strings; all dynamic components must reference registered `type_id` strings resolved through a programmatic `ComponentRegistry`.

### 3. Pipeline Invariant
The security pipeline remains immutable:
$$\text{Untrusted YAML} \longrightarrow \text{StrictSafeLoader} \longrightarrow \text{Plain Primitives} \longrightarrow \text{Pydantic Validation} \longrightarrow \text{Trusted Registry} \longrightarrow \text{World}$$

### 4. Machine-Readable Schema Artifact
We generate and commit a canonical JSON Schema:
- Path: `docs/schemas/wsl-v2.schema.json`.
- Enables client-side IDE auto-completion, linting, and CI validation against enterprise model specifications.

---

## Consequences

### Positive
- **Safe Interoperability**: Non-programmers and external tools can author and validate world definitions safely.
- **Hermetic Security**: Untrusted YAML files cannot execute arbitrary commands or trigger deserialization vulnerabilities.
- **Formal Grammar**: Versioned schema with explicit migration paths for future enhancements.

### Trade-offs & Mitigations
- Lack of embedded expressions requires complex business logic to be encapsulated in registered Python components; this is an intentional design choice preserving auditability and security.
