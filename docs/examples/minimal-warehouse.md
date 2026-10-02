# Minimal Warehouse Acceptance Scenario

The **Minimal Warehouse** acceptance fixture (AC-012) represents the normative reference specification for EWM Engine's core inventory balancing, constraint verification, and scenario branching mechanics.

---

## Normative Specification

```yaml
scenario:
  id: "warehouse-transfer-v1"
  horizon: 1
  seed: 42

warehouses:
  warehouse_a:
    capacity: 150.0
    inventory: 100.0
  warehouse_b:
    capacity: 150.0
    inventory: 20.0

demand:
  warehouse_b: 50.0
```

### Evaluated Scenarios and Normative Outcomes

1. **Baseline (No Transfer)**:
   - `served_demand`: $20.0$
   - `unserved_demand`: $30.0$
   - `final_inventory`: `{"warehouse_a": 100.0, "warehouse_b": 0.0}`
   - `hard_constraint_violations`: $0$

2. **Intervention (Transfer 30 A $\to$ B before demand)**:
   - `served_demand`: $50.0$
   - `unserved_demand`: $0.0$
   - `final_inventory`: `{"warehouse_a": 70.0, "warehouse_b": 0.0}`
   - `hard_constraint_violations`: $0$

3. **Rejection (Transfer 120 units)**:
   - Requested transfer exceeds source inventory ($120.0 > 100.0$).
   - `HARD + PRE_ACTION` constraint violation on `available_inventory`.
   - Action is rejected before state transition; inventory remains intact.

---

## Executing the Normative Fixture

You can run the normative acceptance simulation directly via the CLI:

```bash
# Run the Minimal Warehouse acceptance scenario
ewm example minimal
```

Or run via Python:

```python
from examples.minimal_warehouse.run import run_minimal_warehouse

comparison = run_minimal_warehouse()
print(comparison.summary_table())
```
