# Model Context Protocol (MCP) & Agentic Substrate

> *"Don't let the LLM be the world model. Let it propose; let EWM execute, validate, and prove — same seed, same fingerprint, every time; it refuses non-identifiable claims instead of inventing a number."*

---

## 1. Why an Engine for Autonomous Agents?

The generative AI and autonomous agent community has converged on an invariant empirical lesson: **large language models cannot serve as reliable internal world models.**

Independent published evaluations confirm the failure modes of generative models acting as simulators:
1. **Implicit World Models Lack Coherence:** [Vafa et al. (NeurIPS 2024 / ICML 2025, arXiv:2406.03689)](https://arxiv.org/abs/2406.03689) demonstrate that generative sequence models fail to build coherent world models. Even when predictive likelihood is high, interventions $do(X = x)$ cause ungrounded perturbations on non-descendants and invert physical dynamics.
2. **The Generative Physics-IQ Deficit:** [Kang et al. (ICML 2025, arXiv:2411.02385)](https://arxiv.org/abs/2411.02385) prove that perceptual plausibility is decoupled from physical truth; autoregressive transformers confabulate numbers and breach conservation laws while producing "realistic-looking" text.
3. **Causal Failures in Language Agents:** [Gkountouras et al. (arXiv:2410.19923)](https://arxiv.org/abs/2410.19923) demonstrate that language agents without an external causal oracle hallucinate counterfactual outcomes and fall prey to spurious correlations.

**EWM Engine is the deterministic simulation kernel that solves this.** 

Through the **Model Context Protocol (MCP, rev 2025-06-18)**, EWM becomes callable by Claude Desktop, Cursor, OpenAI Agents SDK, LangGraph, Google ADK, and pydantic-ai without bespoke glue code. Every tool returns a canonical SHA-256 state fingerprint, an explicit `EvidenceLevel`, and an inspectable trace URI.

---

## 2. Installation

The MCP server and agent integrations live strictly in optional extras, preserving EWM's minimal core runtime:

```bash
# Official MCP server (Claude Desktop, Cursor, MCP clients)
pip install "ewm-engine[mcp]"

# Native Pydantic-AI agent toolset
pip install "ewm-engine[agents]"

# Or install both
pip install "ewm-engine[mcp,agents]"
```

---

## 3. Host Configuration Snippets

### Claude Desktop

Add EWM Engine to your Claude Desktop configuration file:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "ewm": {
      "command": "ewm",
      "args": ["mcp", "--provenance-strict"]
    }
  }
}
```

### Cursor IDE

Add EWM to your Cursor project MCP configuration (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ewm-engine": {
      "command": "ewm",
      "args": ["mcp"]
    }
  }
}
```

### Pydantic-AI Agent

Because EWM Engine core models are built natively on Pydantic v2, integration with `pydantic-ai` offers zero serialization overhead:

```python
from ewm_engine.integrations.pydantic_ai import create_ewm_agent, get_ewm_toolset

# Option A: Instantiate ready-to-run agent
agent = create_ewm_agent(model="openai:gpt-4o", provenance_strict=True)

# Option B: Add EWM tools to an existing agent
from pydantic_ai import Agent

custom_agent = Agent(
    model="anthropic:claude-3-5-sonnet-20241022",
    tools=get_ewm_toolset(provenance_strict=True),
    system_prompt="Validate all operational interventions using ewm_check_constraints before simulating.",
)
```

### LangGraph Workflow Node

Integrate deterministic EWM simulation directly as a LangGraph graph node:

```python
from ewm_engine.integrations.pydantic_ai import EwmAgentToolset

toolset = EwmAgentToolset(provenance_strict=True)


def simulate_policy_node(state: dict) -> dict:
    result = toolset.ewm_simulate(
        world_yaml=state["world_spec"],
        horizon=state.get("horizon", 5),
        samples=state.get("samples", 10),
        seed=state.get("seed", 42),
    )
    return {
        "simulation_result": result,
        "state_fingerprint": result["state_fingerprint"],
        "evidence_level": result["evidence_level"],
    }
```

### OpenAI Agents SDK

Connect via standard input/output transport:

```python
from agents import Agent
from agents.mcp import StdioServerTransport

strategy_agent = Agent(
    name="Enterprise Supply Chain Arbiter",
    instructions="Use EWM MCP tools to simulate consequences before recommending actions.",
    mcp_transports=[StdioServerTransport(command="ewm", args=["mcp", "--provenance-strict"])],
)
```

---

## 4. Exposed MCP Tools

| Tool Name | Parameters | Returns | Epistemic Description |
|:---|:---|:---|:---|
| `ewm.simulate` | `world_yaml`, `horizon`, `samples`, `seed` | State fingerprint, run manifest, trace URI | Forward stochastic rollout under declared dynamics laws and fixed seed. |
| `ewm.counterfactual` | `world_yaml`, `intervention`, `seed`, `horizon` | Factual & counterfactual fingerprints, E-value sensitivity | Pearl three-step counterfactual evaluation coupled via Twin-Rollout exogenous noise abduction. |
| `ewm.check_constraints` | `world_yaml`, `action` | `is_valid`, hard/soft violations, state fingerprint | Evaluates physical, financial, and operational invariants prior to stepping state. |
| `ewm.decide` | `world_yaml`, `candidates`, `satisficing_threshold`, `robustness_metric` | Ranked alternatives, robustness scores, best candidate | Multi-robustness scenario optimization (regret/satisficing) under deep uncertainty. |
| `ewm.explain` | `identifier` | Cryptographic Systemic Trace DAG | Inspectable provenance DAG with causal lineage and audit hashes. |

---

## 5. Dynamic Resources & Prompts

### Dynamic Resources

Host agents can inspect simulation artifacts directly using standard MCP resource URIs:

- `ewm://worlds/{world_id}`: Declarative specification for a world identified by its canonical SHA-256 fingerprint.
- `ewm://runs/{run_id}`: Execution run manifest containing sample metadata, seed, horizon, and metrics.
- `ewm://traces/{trace_id}`: Cryptographic Systemic Trace DAG showing step-by-step state mutations.

### Prompts

- `author_what_if`: Formulates a valid operational intervention respecting conservation laws and pre-action checks.
- `explain_result`: Guides the host LLM to interpret simulation outcomes citing SHA-256 fingerprints and disclaimers without confabulating real-world causality.

---

## 6. The Deterministic Game-Master Bridge (Concordia Pattern)

In multi-agent generative environments ([Google DeepMind Concordia](https://github.com/google-deepmind/concordia), Smallville), LLMs frequently hallucinate physics and violate game-world rules. 

`ewm_engine.integrations.concordia.EWMWorldArbiter` provides a deterministic arbiter:

```python
from ewm_engine.core.world import World
from ewm_engine.core.spec import WorldSpec, WorldFactory
from ewm_engine.integrations.concordia import EWMWorldArbiter

# 1. Initialize world
spec = WorldSpec.from_yaml(world_yaml)
world = WorldFactory().create_world(spec)
arbiter = EWMWorldArbiter(world)

# 2. Agent proposes natural language or structured intent
verdict = arbiter.arbitrate_intent(
    agent_id="supplier_alpha",
    intent="transfer 50.0 from inventory_north to inventory_south",
)

if verdict.status == "APPROVED":
    print(f"Observation for agent: {verdict.grounded_observation}")
    print(f"Cryptographic trace: {verdict.trace_id}")
else:
    print(f"Action rejected: {verdict.violations}")
```

---

## 7. Provenance-Strict Mode (`--provenance-strict`)

When running with `--provenance-strict`, EWM activates compliance audit safeguards:

1. **Lowest Evidence Default:** LLM-authored WorldSpecs default to `EvidenceLevel.ASSUMED`. No LLM can self-certify empirical or interventional standing.
2. **Refusal to Assert Non-Identifiable Queries:** If an intervention targets an unobserved confounder or an entity lacking structural equations, `ewm.counterfactual` **refuses** to emit a point estimate, returning `status="NOT_IDENTIFIED"` and sensitivity E-values.
3. **Regulatory Audit Trail:** Every interaction logs an immutable `AuditEvent` with SHA-256 fingerprints of inputs and outputs:
   - **NIST AI RMF 1.0:** Maps to Measure 2.3 (Causal validity tracking) and Manage 1.2 (Systemic risk auditing).
   - **EU AI Act:** Maps to Article 14 (Human Oversight) and Article 15 (Accuracy and Robustness).

---

## 8. Registry Listings

EWM Engine is cataloged in the open-source Model Context Protocol indexes:
- `modelcontextprotocol/servers`: Universal verified enterprise world model server.
- `awesome-mcp-servers`: Listed under Simulation, Decision Support, and Scientific Computing.
