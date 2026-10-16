# Worked Example: OCEL 2.0 Process Induction & Calibrated Counterfactual

This example demonstrates the enterprise-native data-to-world-model workflow:
**Point EWM Engine at an Object-Centric Event Log (OCEL 2.0) and propose an auditable, replay-validated simulation twin.**

## What This Demonstrates

1. **Ingestion of OCEL 2.0 (`log.json`)**:
   - Parses declared object types (`order`, `package`) and their instance attributes.
   - Extracts object-to-object relationships (O2O, such as package `fulfills` order).
2. **Native Inductive Miner (Soundness-Guaranteed)**:
   - Discovers Directly-Follows Graphs per object lifecycle with mathematical soundness (no deadlocks, no livelocks).
3. **Candidate Conservation Laws**:
   - Mines token and flow quantities (`inventory_stock`) to propose non-negativity and balance constraints.
4. **Conformance Checking as Validation Gate**:
   - Replays observed event traces against the discovered process model.
   - Maps trace fitness ($f$) and precision ($p$) directly to epistemic evidence tiers:
     - $f \ge 0.95 \land p \ge 0.90 \implies$ `STRUCTURAL` evidence level.
5. **Calibrated Policy Counterfactual**:
   - Compiles the induced `WorldSpec` into an executable `World`.
   - Simulates baseline trajectories vs an intervention scenario ("Add Expedited Approval Queue").
   - Evaluates differences with non-parametric bootstrap confidence intervals and significance testing.

## Running the Example

```bash
python examples/ocel_procurement/run.py
```

Or via CLI:

```bash
ewm induce process --ocel examples/ocel_procurement/log.json --out induced_procurement.yaml --report procurement_report.json
```
