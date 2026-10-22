# Recipe: How Do I Connect Autonomous LLM Agents via MCP?

> **Question:** *How can external AI agents (Claude Desktop, Cursor, OpenAI Agents, LangGraph) use EWM Engine as a trusted decision twin via the Model Context Protocol (MCP)?*

---

## The Scenario

Autonomous agent architectures (ReAct, plan-and-solve) fail when asked to predict the consequence of irreversible enterprise decisions. Pure LLMs hallucinate numbers and breach operational constraints.

By exposing EWM Engine via **Model Context Protocol (MCP)**, agents can propose candidate interventions, evaluate counterfactual consequences under uncertainty, and query epistemic evidence levels before executing real-world actions.

---

## The Recipe Code

### Step 1: Install the MCP Extension

```bash
pip install "ewm-engine[mcp]"
```

### Step 2: Configure MCP Server (e.g., Claude Desktop or Cursor)

Add the following block to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ewm-engine": {
      "command": "python",
      "args": ["-m", "ewm_engine.mcp.server"]
    }
  }
}
```

### Step 3: Programmatic Interaction via Python Client

```python
from ewm_engine.mcp.server import EwmMcpServer, ServerConfig, SimulateInput

server = EwmMcpServer(
    config=ServerConfig(default_spec_path="examples/worlds/supply_chain_world.yaml")
)

# Tool: Simulate consequence of an intervention before taking it
result = server.simulate(
    SimulateInput(
        spec_path="examples/worlds/supply_chain_world.yaml",
        horizon=5,
        samples=10,
        seed=42,
    )
)
print("Simulation Status:", result["status"])
print("Evidence Level:", result["evidence_level"])
```

---

## Verification & Audit Guarantees

1. **Epistemic Standing:**  
   Every simulation consequence returned over MCP carries an auditable `EvidenceLevel` (`PREDICTIVE`, `INTERVENTIONAL`, or `COUNTERFACTUAL`).
2. **Refusal Honesty:**  
   If an agent proposes an action that breaches physical conservation laws or non-identifiable causal structures, the MCP endpoint returns a teaching refusal error rather than hallucinated optimistic metrics.
