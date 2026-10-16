# ADR-046: Developer Experience, Teaching Errors, and Zero-Install WebAssembly Playground

## Context and Problem Statement

For complex technical infrastructure (scientific simulation, causal inference, and decision twins), adoption frequently tops out low unless developer experience (DX) and zero-install verification are prioritized from Day 1. Real-world analysis of comparable toolkits demonstrates that breakout adoption (Ruff 50k★, uv 90k★, Polars 40k★, DuckDB 42k★, Marimo 23k★) is consistently driven by:
1. A **60-second first win**: A canonical quickstart going from `pip install` to an impressive result against user data in $\le 15$ lines.
2. **Errors that teach**: Refusals that act as rigorous onboarding rather than frustrating dead-ends.
3. **Zero-install live verification**: A client-side WebAssembly playground shareable by URL.
4. **Rich notebook representations**: Visual HTML cards (`_repr_html_`) for core models so data scientists experience high-leverage interactive feedback.

How should EWM Engine structure its first-win ergonomics, error feedback, client-side browser playground, and coordinated launch strategy?

## Considered Options

1. **Status Quo (Multi-step boilerplate & raw exception messages):** Require 60+ lines of code to instantiate scenarios, schedulers, and engines. Return generic `ValueError` or raw stack traces when constraints or causal identifiability checks fail.
2. **Server-Side Hosted Playground & Complex Cloud Sandbox:** Host a JupyterHub or modal container backend to execute simulations in the cloud.
3. **Ergonomic First-Class Quickstart, Teaching Errors, Pyodide WASM, and Coordinated Launch:**
   - Streamline `World.simulate(actions=...)` to achieve a calibrated counterfactual with bootstrap error bars in 12 lines.
   - Implement `TeachingError` hierarchy returning structured two-line explanations ("Why I refused" + "How to resolve").
   - Ship a 100% client-side Pyodide WebAssembly playground with Plotly fan charts and URL hash sharing.
   - Implement `_repr_html_` across 7 core models for rich notebook DX.
   - Author a question-oriented cookbook and stage a synchronized single-day launch pack.

## Decision Outcome

Chosen option: **Option 3**, because:
- **Zero Runtime Dependencies Added:** EWM Engine's mandatory core (`pydantic`, `numpy`, `pyyaml`) already runs natively client-side inside Pyodide WebAssembly without native C extension recompilation or cloud backends.
- **Honesty-as-DX:** When EWM Engine refuses an operation (unobserved confounding, uncertified calibration, capacity limit breach), returning structured remediation guidance turns epistemic rigor into a feature that educates the user on causal inference and physical conservation.
- **Privacy & Air-Gap Compliance:** The client-side WASM playground runs completely inside the user's browser sandbox, ensuring zero data leakage for enterprise evaluators.

## Consequences

### Positive
- A new user evaluates a calibrated counterfactual in $\le 15$ lines / $<60$ seconds.
- Anyone can run, manipulate, and share live simulations via URL without installing Python.
- Notebook environments automatically render interactive progress gauges, status badges, and delta tables.
- Refusals guide users toward mathematically sound causal formulations.

### Negative / Trade-offs
- The WebAssembly playground requires a one-time ~15 MB browser download (Pyodide Python 3.12 runtime + standard library).
- Browser sandbox memory is capped at ~2 GB and single-threaded. These boundaries are explicitly documented in the playground banner.

## Pros and Cons of the Options

### Option 1: Status Quo
- Good: Zero extra code or documentation required.
- Bad: High barrier to entry; users abandon setup before seeing a calibrated counterfactual.

### Option 2: Server-Side Hosted Sandbox
- Good: Full multi-core server performance.
- Bad: Recurring cloud hosting bills, security vulnerabilities from user code execution, violates air-gap privacy principles.

### Option 3: Ergonomic Quickstart, Teaching Errors & Pyodide WASM (Selected)
- Good: 100% client-side, zero cloud hosting costs, zero privacy risk, reproducible URL sharing, exceptional first-minute conversion.
- Bad: Requires maintaining Pyodide compatibility and keeping doc snippets verified in CI.

## Links
- [ADR-037: Release Engineering & Version Truth](ADR-037-release-engineering-version-truth-and-deprecation-policy.md)
- [ADR-045: Model Context Protocol Server & Agentic Substrate](ADR-045-model-context-protocol-server-and-agentic-substrate.md)
- [Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877)](https://arxiv.org/abs/2402.10877)
