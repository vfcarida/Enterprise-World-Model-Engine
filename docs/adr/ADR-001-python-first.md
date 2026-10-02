# ADR-001: Python-First Architecture

## Status
Accepted

## Context
Enterprise world modeling requires interoperability across data science, scientific computing, operations research, and modern AI engineering ecosystems.

## Decision
Implement EWM Engine in modern Python (3.11+) with strict typing (`mypy --strict`), Pydantic models for validated immutability, and NumPy for vectorization.

## Consequences
- **Positive**: Low developer onboarding friction, rapid iteration, and direct interoperability with NumPy, PyTorch, SciPy, and NetworkX.
- **Negative**: Single-threaded Python execution speed for massive-scale rollouts (mitigated in future releases via C-extensions or Ray distribution).
