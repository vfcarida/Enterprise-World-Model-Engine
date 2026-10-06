# Constraints Engine API Reference

This module defines first-class constraint verification protocols, phase semantics (`PRE_ACTION` vs `POST_TRANSITION`), and standard resource constraints.

---

## Protocols & Registry

::: ewm_engine.constraints.base
    options:
      show_root_heading: true
      show_source: false
      members:
        - Constraint

::: ewm_engine.constraints.registry
    options:
      show_root_heading: true
      show_source: false
      members:
        - ConstraintRegistry

---

## Results & Phasing

::: ewm_engine.constraints.results
    options:
      show_root_heading: true
      show_source: false
      members:
        - ConstraintResult
        - ConstraintPhase
        - ConstraintSeverity

---

## Standard Constraints

::: ewm_engine.constraints.standard
    options:
      show_root_heading: true
      show_source: false
      members:
        - ResourceCapacityConstraint
        - ResourceNonNegativeConstraint
        - ActionTransferAvailabilityConstraint
