# Minimal Warehouse Acceptance Fixture (AC-012)

This example is the normative minimal acceptance fixture for the Enterprise World Model Engine.

## Topology & Specification
- Two warehouses: `warehouse_a` (capacity: 150, initial stock: 100), `warehouse_b` (capacity: 150, initial stock: 20).
- Fixed customer demand at `warehouse_b`: 50 units.
- Horizon: 1 step.
- Random Seed: 42.

## Normative Numerical Outcomes

### 1. Baseline (Status Quo: No Transfer)
- `served_demand`: `20.0`
- `unserved_demand`: `30.0`
- `final_inventory`: `{"warehouse_a": 100.0, "warehouse_b": 0.0}`
- `hard_constraint_violations`: `0`

### 2. Intervention (Proactive Transfer: 30 units A -> B before demand)
- `served_demand`: `50.0`
- `unserved_demand`: `0.0`
- `final_inventory`: `{"warehouse_a": 70.0, "warehouse_b": 0.0}`
- `hard_constraint_violations`: `0`

### 3. Action Rejection (Over-Transfer: 120 units A -> B)
- Requested transfer of 120 exceeds available source inventory of 100 in `warehouse_a`.
- Pre-action constraint `available_inventory` rejects the action before state transition.
- 0 units transferred; hard violation recorded.

## Running the Example
```bash
python examples/minimal_warehouse/run.py
```
Or via CLI:
```bash
ewm example minimal_warehouse
```
