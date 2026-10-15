# ADR-041: Decision-Focused Scenario Discovery and Multi-Robustness Adjudication

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-15
* Technical Story: S02 — Decision-Focused Scenario Discovery (`ewm decide`)
* Supersedes: None

## Context and Problem Statement

Enterprise world models simulate complex organizational, financial, and socio-technical systems to support high-stakes executive decisions under deep uncertainty. Historically, world models evaluated simulations primarily through **rollout accuracy, trajectory calibration, and Pareto scorecards**.

However, decision science, Robust Decision Making (RDM; Lempert et al., 2003, 2006), and Decision-Focused Learning (DFL; Elmachtoub & Grigas, 2022; Mandi et al., 2023) show that:
1. **Low prediction error does not guarantee good decisions:** Forecast errors near decision boundaries produce massive downstream losses, while errors in unconstrained regimes may be completely benign.
2. **Generating scenarios is not discovering them:** Monte Carlo clouds show *what could happen*, but do not isolate the concise parameter boundaries *under which a policy fails*.
3. **The single-metric illusion:** Under deep uncertainty, no single optimization criterion exists. In practice, Min-Max Regret, Tail Regret (CVaR), and Satisficing often select completely different candidate decisions (McPhail et al., 2018). Collapsing deep uncertainty into a single winner misleads leadership.

How should EWM Engine implement a decision layer that discovers policy vulnerability subspaces and adjudicates multi-robustness trade-offs with scientific humility?

## Decision Drivers

* **Decision Quality Over Prediction Error:** Evaluate policies directly by downstream regret across plausible futures (DFL / RDM).
* **Interpretable Vulnerability Discovery:** Provide Patient Rule Induction Method (PRIM) and CART to mine the uncertainty space for orthogonal bounding boxes ("strategy X breaks when demand > 45 and delay > 3.2").
* **Multi-Robustness Reporting and Refusal to Rank:** Compare decisions side-by-side across Min-Max Regret, 90th percentile/CVaR Regret, and Starr's Satisficing. Prohibit artificial consensus: **the engine must explicitly refuse to declare a single winner when metrics disagree**.
* **Active Information Value:** Support Active Inference Expected Free Energy (EFE) scoring, mathematically valuing actions that reduce the engine's own uncertainty.
* **Lightweight Core & Zero Bloat:** Implement pure NumPy PRIM and CART in core; keep EMA Workbench and heavy solvers as optional extras (`[decision]`).
* **Deterministic Reproducibility (Invariant 5):** All discoveries and evaluations must be reproducible and fingerprinted.

## Considered Options

* **Option 1: Rely solely on Monte Carlo Accuracy and Pareto Evaluation** — Existing status quo. Relegates the engine to a passive simulator; fails to identify failure boundaries; leaves regret unmodeled.
* **Option 2: Mandatory External Dependency on EMA Workbench or Optimization Solvers** — Outsource PRIM, CART, and regret to `ema-workbench` or Pyomo. Introduces heavy dependencies (matplotlib, pandas, scipy, statsmodels) into core, breaking the lightweight engine promise.
* **Option 3: Pure NumPy Decision Layer with Optional EMA Workbench Adapter (Chosen)** — Implement native, deterministic PRIM, CART, multi-robustness metrics, and Active Inference EFE scoring in `ewm_engine.decision`; expose CLI `ewm decide`; provide optional adapter for EMA Workbench (`[decision]`); enforce programmatic refusal to rank when criteria disagree.

## Decision Outcome

Chosen option: **Option 3: Pure NumPy Decision Layer with Optional EMA Workbench Adapter**.

### Consequences

* **Good, because**:
  * **Interpretable Vulnerability Regions:** PRIM and CART identify actionable, bounded subspaces where candidate policies break, surfaced with quasi-p-values.
  * **Multi-Robustness Side-by-Side:** Reports Min-Max Regret, CVaR(90) Regret, Starr's Satisficing, and Expected Regret without collapsing them into an arbitrary scalar.
  * **Epistemic Refusal to Rank:** Protects decision-makers from overconfident algorithmic rankings by refusing to pick a single winner when metrics disagree (McPhail et al., 2018).
  * **Active Exploration Incentive:** Expected Free Energy (EFE) scoring natively rewards exploratory, uncertainty-reducing actions.
  * **Zero Mandatory Dependencies:** Runs entirely on Python standard library and NumPy.
  * **CLI & API Ergonomics:** `ewm decide` produces executive tables, structured JSON, and Markdown reports.
* **Bad, because**:
  * Executive users seeking a simple "best policy" answer must confront metric trade-offs and epistemic disagreements.
  * Non-convex, highly irregular failure regions may require multiple overlapping PRIM boxes.

### Confirmation

* Unit tests in `tests/unit/test_decision_discovery.py` verify PRIM and CART discovery, coverage/density calculation, and quasi-p-values.
* Unit tests in `tests/unit/test_decision_robustness.py` verify multi-robustness calculations and refusal to rank under metric disagreement.
* Unit tests in `tests/unit/test_decision_efe.py` verify pragmatic vs. epistemic decomposition under Active Inference.
* Contract tests in `tests/contract/test_decision_contract.py` verify Invariant 5 (fingerprint determinism), prohibition of "guarantees optimal", and `ewm decide` CLI execution.

## Pros and Cons of the Options

### Option 1: Existing Simulator Status Quo
* Good: Zero engineering work.
* Bad: Fails to answer "what should we do and where does it break?"; vulnerable to the curse of point prediction.

### Option 2: Mandatory Heavy External Toolkits
* Good: Avoids reimplementing PRIM.
* Bad: Adds heavy dependency burden, non-deterministic RNGs, and complex cross-platform installation issues.

### Option 3: Pure NumPy Core Decision Layer & Pluggable Adapter
* Good: Lightweight, bit-reproducible, mathematically rigorous, and strictly enforces epistemic humility.
* Bad: Requires maintaining native PRIM and CART implementations.

## More Information

* Elmachtoub, A. N., & Grigas, P. (2022). "Smart 'Predict-then-Optimize'." *Management Science*, 68(1), 9-26. [arXiv:1710.08005](https://arxiv.org/abs/1710.08005).
* Lempert, R. J., Popper, S. W., & Bankes, S. C. (2003). *Shaping the Next One Hundred Years: New Methods for Quantitative, Long-Term Policy Analysis*. RAND MR-1626.
* McPhail, C., et al. (2018). "Robustness metrics: How choice of metric changes which policy wins." *Environmental Modelling & Software*, 107, 142-155.
* Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.
* Feature Specification: [FEAT-009](file:///docs/specs/features/FEAT-009-decision-focused-scenario-discovery.md)
* Related ADRs: [ADR-003](ADR-003-explicit-constraints.md), [ADR-005](ADR-005-deterministic-reproducibility.md), [ADR-016](ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md), [ADR-028](ADR-028-experimentation-suite-doe-backtesting-calibration-optimization.md), [ADR-039](ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md), [ADR-040](ADR-040-twin-rollout-counterfactual-scm-kernel.md).
