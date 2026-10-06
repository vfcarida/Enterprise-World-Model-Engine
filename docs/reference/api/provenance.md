# Systemic Traces & Provenance API Reference

This module defines epistemic trace graphs, evidence tiers (`EvidenceLevel`), and simulation metadata provenance.

---

## Systemic Traces

::: ewm_engine.provenance.trace
    options:
      show_root_heading: true
      show_source: false
      members:
        - SystemicTrace
        - TraceNode
        - TraceEdge

---

## Epistemic Evidence Levels

::: ewm_engine.provenance.evidence
    options:
      show_root_heading: true
      show_source: false
      members:
        - EvidenceLevel

---

## Metadata & Provenance

::: ewm_engine.provenance.metadata
    options:
      show_root_heading: true
      show_source: false
      members:
        - SimulationMetadata
        - Provenance
        - ComponentVersion
