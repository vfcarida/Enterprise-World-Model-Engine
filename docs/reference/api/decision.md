# Decision-Focused Scenario Discovery & Robustness API Reference

This module implements Robust Decision Making (RDM) under deep uncertainty: Patient Rule Induction Method (PRIM) and CART scenario vulnerability discovery, multi-metric decision robustness (Starr, Hurwicz, Savage Minimax Regret, Laplace), Smart "Predict-then-Optimize" decision regret, and Active Inference Expected Free Energy (EFE) scoring.

---

## Decision Engine & Reports

::: ewm_engine.decision.engine
    options:
      show_root_heading: true
      show_source: false
      members:
        - DecisionEngine

::: ewm_engine.decision.report
    options:
      show_root_heading: true
      show_source: false
      members:
        - DecisionReport

---

## Scenario Vulnerability Discovery (PRIM & CART)

::: ewm_engine.decision.discovery
    options:
      show_root_heading: true
      show_source: false
      members:
        - PrimBox
        - ScenarioDiscoveryResult
        - run_prim
        - run_cart

---

## Multi-Metric Robustness & Decision Regret

::: ewm_engine.decision.robustness
    options:
      show_root_heading: true
      show_source: false
      members:
        - RobustnessScores
        - DecisionRegretResult
        - DecisionComparisonResult
        - evaluate_decision_robustness

---

## Active Inference Expected Free Energy (EFE)

::: ewm_engine.decision.efe
    options:
      show_root_heading: true
      show_source: false
      members:
        - ExpectedFreeEnergyScore
        - ActiveInferenceScorer
