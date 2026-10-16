# Client-Side WASM Playground (Zero-Install Simulation)

> *Run full Monte Carlo enterprise simulations and Pearl counterfactuals 100% in your browser using Pyodide WebAssembly. No Python install, zero mandatory server, zero data upload.*

---

## 1. What Is the EWM WebAssembly Playground?

Because the **Enterprise World Model Engine (EWM)** core runtime is strictly decoupled from heavy GPU dependencies and proprietary neural APIs (operating purely on `pydantic`, `numpy`, and `pyyaml`), its mandatory execution stack compiles and runs natively inside client WebAssembly engines via [Pyodide](https://pyodide.org/).

This delivers a unique advantage in decision tooling: **instant, verifiable evaluation without installing a single package**.

<div style="margin: 24px 0; border: 1px solid #334155; border-radius: 8px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.3);">
  <iframe src="../playground/index.html" width="100%" height="820" style="border: none; display: block; background: #0b0f19;"></iframe>
</div>

> [!TIP]
> **Open Standalone Playground:**  
> You can also run the playground in full-screen mode at [docs/playground/index.html](playground/index.html) or share any live experiment using the **"Share URL"** button (which encodes seeds, horizon, and intervention parameters directly into the URL hash).

---

## 2. Architectural Honesty & Browser Limits

In adherence to EWM Engine's **Honesty-as-DX** principle, the in-browser sandbox operates with explicit operational boundaries:

| Dimension | WebAssembly Sandbox (Browser) | Native Python Runtime (`pip install ewm-engine`) |
| :--- | :--- | :--- |
| **Download Size** | ~15 MB one-time bundle (Pyodide Python 3.12 runtime + standard library + NumPy) | ~25 MB standard PyPI wheel installation |
| **Memory Ceiling** | ~2 GB WebAssembly heap limit (enforced by browser) | Limited only by host system physical memory (RAM) |
| **Concurrency** | Single-threaded JavaScript event loop execution | Full local multi-processing (`ProcessPoolExecutor`) and distributed cluster backends (`Ray`) |
| **Data Privacy** | 100% Client-Side. No telemetry, zero packets emitted | Fully local execution, air-gapped certified |
| **Determinism** | Bitwise deterministic PRNG matching native Python seeds | Bitwise deterministic across OS platforms (Linux, macOS, Windows) |

---

## 3. How URL Sharing Works (Marimo Pattern)

The playground uses deterministic URL parameter hashing inspired by the [Marimo](https://marimo.io) reactive notebook pattern:

- Every slider modification instantly synchronizes with `window.location.hash`:
  ```text
  https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/#preset=warehouse&h=25&delta=35&seed=101&samples=10
  ```
- Clicking **"Share URL"** writes the exact reproducible configuration to your clipboard.
- Recipients loading the link reconstruct the identical pseudo-random generator state and trajectory envelope immediately upon opening the page.
