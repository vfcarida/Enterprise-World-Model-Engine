# ADR-003: Explicit Constraints vs. Neural Weights

## Status
Accepted

## Context
Many modern world-model proposals attempt to learn physical laws and operational constraints directly from observational data using neural network penalty terms. In production enterprise systems, this causes safety hazards and physical hallucinations.

## Decision
Separate what the system *knows* as an explicit structural rule from what the system *learns* probabilistically. Implement constraints as first-class evaluatable objects (`Constraint` protocol and `ConstraintRegistry`) supporting hard and soft enforcement with complete provenance.

## Consequences
- **Positive**: Strict guarantees that physical capacities and logistical rules are never breached silently.
- **Negative**: Requires domain experts to explicitly register known operational rules rather than relying solely on raw data training.
