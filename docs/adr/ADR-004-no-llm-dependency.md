# ADR-004: Zero Dependency on LLMs in Core Engine

## Status
Accepted

## Context
Recent AI frameworks frequently couple execution loops to proprietary LLM APIs, resulting in network fragility, nondeterministic behavior, latency, and vendor lock-in.

## Decision
The core engine has zero mandatory dependencies on LLMs, cloud APIs, or GPU hardware. All core abstractions run entirely offline. External LLM agents interface as optional actors via the `Actor` protocol in `ewm_engine.integrations`.

## Consequences
- **Positive**: Complete offline execution, fast unit testing, strict determinism, and zero cloud API costs.
- **Negative**: Users wishing to simulate natural language agents must write or import integration adapters.
