# Agent Evaluation & OR Auditability

The **Agent Evaluation** example demonstrates the "engine != agent" architectural paradigm, inspired by ARE and Gaia2 ([arXiv:2509.17158](https://arxiv.org/abs/2509.17158)).

---

## Scenario Description

A central medical supply hub services three regional hospitals:
- **Central Depot**: 200 units initial stock (capacity 500 units).
- **Corridor Links**: Maximum 60 units transportable along any single corridor in one simulation step.
- **Hospitals**:
  - `hospital_north`: 15 units initial stock (consumes 30 units/step)
  - `hospital_central`: 20 units initial stock (consumes 45 units/step)
  - `hospital_south`: 10 units initial stock (consumes 25 units/step)

### Evaluated Agent Policies

1. **Status Quo (Baseline)**: No actions taken. Rapidly runs out of hospital stock, leaving 255 unserved patient requests.
2. **Greedy Myopic Agent**: Dispatches 100 units to the hospital with the lowest perceived stock. Rebuffed by the engine's hard corridor capacity constraint (max 60 units), resulting in rejected actions and 3 violations.
3. **Proportional Heuristic**: Splits available hub stock equally (20 units to each hospital). Stays within corridor limits, but creates an unserved deficit at `hospital_central` due to asymmetric demand.
4. **CP-SAT Optimization Agent**: Formulates a joint integer allocation with Google OR-Tools CP-SAT, respecting corridor limits and minimizing unmet demand. Achieves lowest cumulative unserved demand with zero violations.

---

## Executing the Example

Run directly from the repository root:

```bash
uv run python examples/agent_evaluation/run.py
```

Or via Python:

```python
from examples.agent_evaluation.run import run_agent_evaluation

results = run_agent_evaluation()
print(results["comparison"].summary_table())
print("Oracle Scores:", results["oracle_scores"])
print("Unsat Core Certificate:", results["unsat_core"])
```

---

## Key Takeaways

- **Verifiable State Invariants**: Agent performance is scored strictly on the deterministic world-state trajectory, avoiding subjective LLM-as-a-judge hallucinations.
- **Unsat Core Proofs**: When crisis demand exceeds physical limits, OR-Tools CP-SAT extracts the exact minimal set of conflicting supply and demand assumptions as an auditable certificate.
