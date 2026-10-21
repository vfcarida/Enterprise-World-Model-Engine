# World-Model Induction & Honesty Gate API Reference

This module provides automated structural, process, causal, dynamical, and conservation-invariant discovery from enterprise telemetry, guarded by the mandatory fail-closed `InductionValidationGate`.

---

## Induction Honesty & Validation Gate

::: ewm_engine.induction.validation
    options:
      show_root_heading: true
      show_source: false
      members:
        - InductionValidationGate
        - InductionValidationReport
        - RefusedElement
        - OpenQuestion
        - RefuterResult
        - validate_induction
        - run_causal_refutations

---

## Induction Audit Report

::: ewm_engine.induction.report
    options:
      show_root_heading: true
      show_source: false
      members:
        - InductionReport
        - CandidateInvariant
        - ProvisionalLink
        - ConformanceScore
        - CausalEquivalenceClassSummary
        - InducedRateLawSummary

---

## Causal Equivalence Class Discovery (PC, FCI, DAGMA, LiNGAM)

::: ewm_engine.induction.causal
    options:
      show_root_heading: true
      show_source: false
      members:
        - CausalEdge
        - CausalEdgeType
        - CausalEquivalenceClass
        - induce_causal

---

## Governing Dynamics & Conservation Invariant Discovery

::: ewm_engine.induction.dynamics
    options:
      show_root_heading: true
      show_source: false
      members:
        - InducedRateLaw
        - induce_dynamics

::: ewm_engine.induction.invariants
    options:
      show_root_heading: true
      show_source: false
      members:
        - InvariantHypothesis
        - induce_invariants

---

## Relational Schema, DataFrame & OCEL 2.0 Process Induction

::: ewm_engine.induction.schema
    options:
      show_root_heading: true
      show_source: false
      members:
        - induce_schema

::: ewm_engine.induction.connectors
    options:
      show_root_heading: true
      show_source: false
      members:
        - DataFrameSchemaMapping
        - induce_from_dataframe

::: ewm_engine.induction.process
    options:
      show_root_heading: true
      show_source: false
      members:
        - induce_process
