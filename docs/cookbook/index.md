# EWM Engine Cookbook: By-Question Recipes

> *"Copy-pasteable, verified engineering recipes for common simulation-based decision workflows in Python."*

---

## 1. Overview

The **EWM Engine Cookbook** is organized around concrete operational **questions**. Every headline capability has an isolated, runnable recipe with explicit verification and audit guarantees.

---

## 2. Recipe Gallery (Organized by Question)

| Question | Capability | Link |
| :--- | :--- | :--- |
| **"How do I test a policy change before deploying it?"** | Counterfactual Branching & Bootstrap CIs | [View Recipe](policy-change.md) |
| **"How do I calibrate my world model to historical event logs?"** | SBI Amortized Posterior & TARP Coverage | [View Recipe](historical-calibration.md) |
| **"How do I enforce strict conservation and accounting laws?"** | Bounded Resources & Hard Invariant Gates | [View Recipe](conservation-laws.md) |
| **"How do I connect autonomous LLM agents via MCP?"** | Model Context Protocol Tool Interface | [View Recipe](agentic-mcp.md) |
| **"How do I discover failure scenarios under deep uncertainty?"** | Scenario Discovery & PRIM Patient Rule Induction | [View Recipe](recipe-stress-test.md) |
| **"How do I audit counterfactuals with visual diffs?"** | Twin-Rollouts & Mermaid Causal Graphs | [View Recipe](recipe-counterfactual-audit.md) |
| **"How do I rebalance multi-hub inventories under supply shocks?"** | Constrained Inventory Reroute | [View Recipe](recipe-supply-reroute.md) |

---

## 3. How to Use These Recipes

Every recipe follows a 3-step pattern:
1. **The Scenario:** The concrete problem and assumptions.
2. **The Recipe Code:** A modular, self-contained Python script or function.
3. **Verification & Audit:** How to verify that physical conservation, determinism, and epistemic honesty hold.
