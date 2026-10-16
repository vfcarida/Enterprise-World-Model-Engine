# EWM Engine Coordinated Single-Day Launch Pack

> **Role:** Open-Source GTM & Developer Experience Lead.  
> **Core Rule:** *The launch fires only once — fire it when install (`T01`), docs, MCP (`T02`), and the demo playground (`T03`) are all live.*  
> **Citable Thesis:** *Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877): Any agent that robustly generalizes across distribution shifts must learn an approximate causal world model separating structural rules from exogenous variations. EWM Engine builds that model explicitly and auditably.*

---

## 1. Coordinated Launch Sequencing Gate

The single-day launch coordinates Show HN, Reddit (`r/Python`, `r/MachineLearning`), an authored X/Bluesky thread, and curated awesome-list PRs in a synchronized push.

| Gate Condition | Verification Check | Status |
| :--- | :--- | :--- |
| **1. PyPI Install Live** | `pip install ewm-engine` installs cleanly on clean Python 3.11/3.12 | Verified |
| **2. Docs Site Deployed** | `https://vfcarida.github.io/Enterprise-World-Model-Engine/` resolves 200 OK | Verified |
| **3. MCP Server Live** | `python -m ewm_engine.mcp.server` handles JSON-RPC tool calls | Verified |
| **4. WASM Playground Live** | `https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/` runs client-side | Verified |
| **5. Proof Hero in CI** | `python scripts/generate_hero_proof.py` matches committed SVG | Verified |

---

## 2. Show HN Submission

### Title
```text
Show HN: EWM Engine – A causal, calibrated simulation kernel for enterprise decisions (zero GPU/LLM)
```

### URL
```text
https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/
```

### Author First Comment (Post immediately upon submission)
```markdown
Hi HN,

Most recent "world models" fall into two camps: generative video models (Cosmos, Sora) that simulate visual pixels, or autoregressive LLMs (LangGraph, CrewAI) that simulate text token sequences.

While impressive, both fail when leadership teams or autonomous agents ask an operational question: *"What could happen if we enact policy A instead of B?"* Generative models accumulate errors over multi-step horizons, hallucinate resources out of thin air, and cannot state when a counterfactual query is mathematically non-identifiable.

We built Enterprise World Model Engine (EWM) to solve this:
1. **Explicit Separation of Known vs. Learned:** Physical, logistical, and accounting conservation laws (mass balance, cash solvency, capacity limits) hold strictly by construction. Learned stochastic behaviors (demand, transit delays) run within these explicit bounds.
2. **Native Pearlian Counterfactuals (Rung 3):** Evaluates three-step counterfactual rollouts (Abduction -> Action -> Prediction) with noise-coupled Twin Rollouts (Richens & Everitt, ICLR 2024 oral, arXiv:2402.10877).
3. **Scored on Honesty, Not Just Optimism:** Mandates conformal prediction bands (90% coverage) and bootstrap confidence intervals. Refuses to rank decisions when unobserved confounding violates structural identifiability.
4. **Teaching Errors as DX:** Every refusal returns a 2-line explanation: why it refused, and the identifiable alternative.
5. **Zero Mandatory GPU / 100% Client-Side WASM:** The core engine is pure Python (Pydantic + NumPy + PyYAML). You can run the entire simulation kernel client-side in your browser via our WebAssembly playground.

### 60-Second Win (≤15 lines):
```python
import ewm_engine as ewm
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

world = ewm.World(
    state=ewm.WorldState(resources=[
        ewm.Resource(id="stock_a", current=100.0, max_value=200.0),
        ewm.Resource(id="stock_b", current=20.0, max_value=200.0),
    ]),
    dynamics=DeterministicTransferDynamics(),
)
baseline = world.simulate(horizon=5, seed=42)
counterfactual = world.branch().simulate(
    actions=[ewm.Action(type="transfer_resource", parameters={"quantity": 30.0, "source_resource": "stock_a", "target_resource": "stock_b"})],
    horizon=5, seed=42,
)
diff = ewm.compare_scenarios(baseline, [counterfactual], metrics=["resource_stock_a", "resource_stock_b"])
print(diff.summary_table())
```

- **WASM In-Browser Playground:** https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/
- **Documentation & Handbook:** https://vfcarida.github.io/Enterprise-World-Model-Engine/
- **GitHub Repository:** https://github.com/vfcarida/Enterprise-World-Model-Engine
- **MCP Server for Claude / Cursor:** `pip install "ewm-engine[mcp]"` -> `python -m ewm_engine.mcp.server`

We would love your thoughts on the architecture, the browser playground, and how you approach counterfactual decision testing in your own organizations.
```

---

## 3. Reddit Posts

### A. r/Python

**Title:**
```text
EWM Engine: In-browser WebAssembly simulation playground + 12-line calibrated counterfactuals
```

**Post Body:**
```markdown
Hey r/Python,

We just released **Enterprise World Model Engine (EWM Engine)** v1.5.0 (`pip install ewm-engine`).

A big challenge with enterprise simulators (SimPy, Arena) is that they either require writing monolithic event loops by hand or lack native counterfactual evaluation and causal identifiability checks. On the other end, LLM agents hallucinate quantities and breach balance equations.

EWM Engine is an open-source structural simulation kernel designed for Python 3.11+:
- **12-Line First Win:** Go from import to an audited counterfactual comparison with bootstrap confidence intervals in <60 seconds.
- **Client-Side WASM Playground:** Because the core is pure Pydantic + NumPy + PyYAML (no GPU/PyTorch required), we compiled the engine to run 100% in-browser via Pyodide. You can test scenarios, adjust sliders, and share the exact state via URL: https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/
- **Errors That Teach:** Refusals (capacity violations, non-identifiable causal queries) return structured Pydantic-style teaching errors with the reason, remediation, and docs link.
- **Model Context Protocol (MCP):** Connect external LLMs (Claude, Cursor, LangGraph) via `ewm-engine[mcp]` so agents simulate consequences before taking real actions.

GitHub: https://github.com/vfcarida/Enterprise-World-Model-Engine
Docs: https://vfcarida.github.io/Enterprise-World-Model-Engine/
```

### B. r/MachineLearning

**Title:**
```text
[Project] EWM Engine – Action-Conditioned Structural World Models & Noise-Coupled Counterfactuals (ICLR 2024 Oral Grounding)
```

**Post Body:**
```markdown
Hello r/MachineLearning,

We are sharing **Enterprise World Model Engine (EWM)**, an open-source framework for building, simulating, and evaluating structural world models of complex organizational systems.

### Motivation & Theory:
Recent theoretical work (Richens & Everitt, ICLR 2024 oral, arXiv:2402.10877) proved that any agent capable of robust out-of-distribution generalization must learn an approximate causal world model. Meanwhile, empirical analyses of autoregressive LLMs (Vafa et al., arXiv:2406.03689) show that transformer next-token predictors struggle to maintain physical conservation laws or track multi-step state transitions over long horizons.

### Framework Architecture:
1. **Strict Invariants by Construction:** Hard conservation laws (mass, energy, cash flow) are enforced by explicit pre-action and post-transition validation gates rather than soft loss penalties.
2. **Pearl Rung-3 Counterfactuals:** Evaluates structural causal counterfactuals via three-step Abduction-Action-Prediction with exogenous noise coupling (Twin-Rollout).
3. **Mandatory Uncertainty Intervals:** Rejects bare point predictions; produces conformal prediction bands and bootstrap confidence intervals.
4. **Epistemic Honesty Refusal:** Formally evaluates backdoor/frontdoor identifiability; refuses to emit causal claims when unobserved confounding cannot be blocked.

- Repository: https://github.com/vfcarida/Enterprise-World-Model-Engine
- Reproducible SEIR Benchmark (exact analytical ODE overlay): `make reproduce`
- Interactive In-Browser Demo: https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/
```

---

## 4. Authored 10-Post Thread (X / Bluesky)

```text
1/10
Why do AI agents fail at long-horizon planning?

Richens & Everitt (ICLR 2024 oral) proved the reason mathematically: any agent that robustly generalizes under distribution shift MUST learn a causal world model.

Today, we're releasing the Enterprise World Model Engine (EWM):
https://github.com/vfcarida/Enterprise-World-Model-Engine

2/10
Most recent "world models" generate video pixels (Sora, Cosmos) or autoregress text (LLMs).

In operational systems, that leads to catastrophic drift: hallucinations, money created out of thin air, and zero guarantees that physical conservation laws hold.

3/10
EWM Engine separates KNOWN structural physics from LEARNED stochastic dynamics:
- Hard conservation laws hold by construction.
- Pluggable dynamics evaluate candidate policies.
- Epistemic traces track causal dependencies.

Zero mandatory GPU or cloud API keys.

4/10
The 60-Second First Win:
Go from `pip install ewm-engine` to an audited counterfactual with bootstrap error bars in 12 lines of code:

[Image: Code snippet from quickstart]

5/10
Because EWM core is pure Python (Pydantic + NumPy), it runs 100% client-side in WebAssembly!

Test scenarios, slide intervention parameters, and share reproducible rollouts via link—zero server required:
https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/

6/10
Honesty-as-DX: "Errors that Teach".
When an intervention violates a physical invariant or attempts a non-identifiable do-query, EWM refuses with a structured explanation:
- Why I refused
- How to resolve it
- Link to documentation

7/10
For agent builders: EWM Engine exposes native Model Context Protocol (MCP) endpoints.
Connect Claude, Cursor, or OpenAI Agents to simulate consequences of actions before taking them:
`pip install "ewm-engine[mcp]"`

8/10
Reproducibility Hero:
Overlaid on our SEIR benchmark, EWM's discrete forward rollout matches analytical ground-truth ODEs bit-for-bit with exact mass balance conservation across all 80 steps.

Reproduce with: `make reproduce`

9/10
We also shipped a comprehensive handbook & question-oriented cookbook:
- Testing policy changes
- Calibrating to telemetry (TARP certified coverage)
- Stress-testing under deep uncertainty (PRIM)
https://vfcarida.github.io/Enterprise-World-Model-Engine/cookbook/

10/10
Try EWM Engine today:
📦 `pip install ewm-engine`
💻 Repo: https://github.com/vfcarida/Enterprise-World-Model-Engine
🌐 Playground: https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/

Give it a star ⭐ and let us know what you simulate!
```

---

## 5. Awesome-List PR Submissions

### 1. `awesome-python`
- **Category:** `Simulation` or `Data Analysis`
- **Entry:**
  ```markdown
  * [EWM Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine) - Deterministic, auditable structural simulation kernel for socio-technical systems and counterfactual decision evaluation.
  ```

### 2. `awesome-mcp-servers`
- **Category:** `Simulation & Evaluation`
- **Entry:**
  ```markdown
  * [Enterprise World Model](https://github.com/vfcarida/Enterprise-World-Model-Engine) - Decision twin and counterfactual consequence simulator for AI agents via Model Context Protocol.
  ```

### 3. `awesome-causality`
- **Category:** `Libraries / Software`
- **Entry:**
  ```markdown
  * [EWM Engine](https://github.com/vfcarida/Enterprise-World-Model-Engine) - Structural causal simulation kernel implementing Pearl's three-step counterfactual algorithm with noise-coupled Twin Rollouts and identifiability gates.
  ```
