# EWM Engine Cookbook: Task-Oriented Recipes

> *"Copy-pasteable, verified engineering recipes for common simulation-based decision workflows in Python."*

---

## 1. Overview

The **EWM Engine Cookbook** provides focused, task-oriented code recipes to solve specific operational problems. Unlike conceptual guides or tutorials, every recipe provides a concise, production-ready pattern you can adapt to your own domain.

---

## 2. Available Recipes

| Recipe | Primary Capability | When to Use | Link |
| :--- | :--- | :--- | :--- |
| **Constrained Multi-Hub Inventory Reroute** | Conservation Invariants & Interventions | Balancing logistics networks under sudden regional stockout risks. | [View Recipe](recipe-supply-reroute.md) |
| **Macroeconomic Stress Testing with PRIM** | Scenario Discovery (`ewm decide`) | Automatically finding the failure subspace across multi-variate parameter shocks. | [View Recipe](recipe-stress-test.md) |
| **Auditing Counterfactuals with Mermaid Diffs** | Twin-Rollout SCM & Provenance | Producing inspectable visual causal graphs for board and audit committees. | [View Recipe](recipe-counterfactual-audit.md) |

---

## 3. How to Use These Recipes

Every recipe follows a 3-step pattern:
1. **The Scenario:** The concrete problem and assumptions.
2. **The Recipe Code:** A modular, self-contained Python function or script.
3. **Verification & Audit:** How to verify that physical conservation and determinism hold.
