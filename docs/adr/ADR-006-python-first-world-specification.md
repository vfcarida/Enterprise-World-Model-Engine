# ADR-006: Python-First World Specification Before Freezing a DSL

## Status
Accepted

## Context
Designing a declarative Domain-Specific Language (DSL) in YAML or JSON prematurely risks freezing awkward grammar before core abstractions have matured under real-world usage.

## Decision
Prioritize Python-first programmatic world definition for v0.1.0. WorldState, entities, relationships, dynamics, and constraints are defined via standard Python objects and Pydantic models. A declarative DSL will be considered in future releases once interface patterns stabilize.

## Consequences
- **Positive**: Full IDE autocomplete, compile-time type checking, refactoring support, and maximum expressive freedom.
- **Negative**: Non-programmers cannot directly author world models without Python knowledge in v0.1.0.
