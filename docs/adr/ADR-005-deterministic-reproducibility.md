# ADR-005: Deterministic Reproducibility and Scenario Provenance

## Status
Accepted

## Context
Scientific research and high-stakes enterprise decision-making require that experiments can be independently audited and bit-identically reproduced across different compute environments.

## Decision
Mandate explicit pseudo-random generator passing (`rng: numpy.random.Generator`) across all stochastic dynamics and actors. Generate independent child seeds via `SeedSequence.spawn()`. Calculate cryptographic SHA-256 fingerprints across initial world state, configuration, and random seeds.

## Consequences
- **Positive**: Exact bit-level reproducibility of all simulation trajectories; guaranteed repeatability for regression testing and scientific publication.
- **Negative**: Developers must never call global `random.random()` or `np.random.rand()`.
