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
import json
from ewm_engine.mcp.server import inspect_state, simulate_intervention

# Tool 1: Inspect active enterprise state and resource balances
state_summary = inspect_state(world_spec="tests/fixtures/civicflow/tiny_flood_v1.yaml")
print("State Summary:", json.loads(state_summary)["resources"])

# Tool 2: Simulate consequence of an intervention before taking it
simulation_output = simulate_intervention(
    action_type="transfer_resource",
    parameters={
        "quantity": 25.0,
        "source_resource": "clean_water_depot",
        "target_resource": "zone_a_water",
    },
    horizon=5,
    samples=10,
    seed=42,
)

result = json.loads(simulation_output)
print("Simulation Status:", result["status"])
print("Expected Delta:", result.get("deltas"))
print("Evidence Level:", result.get("evidence_level"))
```

---

## Verification & Audit Guarantees

1. **Epistemic Standing:**  
   Every simulation consequence returned over MCP carries an auditable `EvidenceLevel` (`PREDICTIVE`, `INTERVENTIONAL`, or `COUNTERFACTUAL`).
2. **Refusal Honesty:**  
   If an agent proposes an action that breaches physical conservation laws or non-identifiable causal structures, the MCP endpoint returns a teaching refusal error rather than hallucinated optimistic metrics.
