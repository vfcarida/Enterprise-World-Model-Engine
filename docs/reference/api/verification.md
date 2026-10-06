# Trajectory Verification API Reference

This module implements multi-step trajectory verification: oracle-graph temporal DAG verification and Signal Temporal Logic (STL) robustness monitoring (Track T3).

---

## Oracle Graph Verifier

::: ewm_engine.verification.oracle
    options:
      show_root_heading: true
      show_source: false
      members:
        - OracleGraph
        - OracleNode
        - OracleEdge
        - OracleVerificationResult
        - evaluate_oracle_graph

---

## Property Specifications

::: ewm_engine.verification.spec
    options:
      show_root_heading: true
      show_source: false
      members:
        - PropertySpec
        - compute_property_hash
        - fold_properties_into_fingerprint

---

## Signal Temporal Logic (STL)

::: ewm_engine.verification.temporal
    options:
      show_root_heading: true
      show_source: false
      members:
        - STLFormula
        - TemporalMonitor
