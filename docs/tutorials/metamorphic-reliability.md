# Tutorial: Causal Correctness & Metamorphic Testing

> **Learning Goal**: Learn how to evaluate an enterprise world model beyond rollout error — using metamorphic relations (MRs), hard conservation checks, and out-of-support abstention.

---

## 1. Why Rollout Accuracy is an Insufficient Metric
In machine learning, models are evaluated on test-set Mean Squared Error (MSE). However, a model can achieve $99\%$ accuracy on past peacetime data while failing catastrophic physical sanity tests:
* It generates goods out of thin air when demand spikes.
* It reverses the direction of a known causal mechanism.
* It predicts smooth operations when extrapolated far beyond historical experience.

**Metamorphic Testing ([ADR-044](../adr/ADR-044-causal-correctness-metamorphic-evaluation.md))** tests models without needing an explicit ground-truth oracle: it verifies whether **necessary relationships between inputs and outputs hold invariant under transformations**.

---

## 2. Implementing a Metamorphic Test

```python
from ewm_engine.evaluation.metamorphic import MetamorphicTestEngine, MetamorphicRelation

# Define Metamorphic Relation: Monotonicity under Resource Injection
# If initial inventory at Node A is strictly increased by Delta > 0,
# final unserved customer demand must be strictly less than or equal to baseline.
mr_monotonicity = MetamorphicRelation(
    name="inventory_monotonicity_under_injection",
    transform_input=lambda state: state.with_resource_delta("stock_a", +25.0),
    verify_output=lambda factual_traj, transformed_traj: (
        transformed_traj.final_metric("unserved_demand")
        <= factual_traj.final_metric("unserved_demand") + 1e-6
    ),
)

engine = MetamorphicTestEngine(world=world)
report = engine.run_suite(relations=[mr_monotonicity], test_scenarios=50)

print(f"Metamorphic Pass Rate: {report.pass_rate:.1%}")
print(f"Violations Detected:   {report.violations_count}")
```

---

## 3. The Metamorphic Reliability Scorecard
* **Invariant Conservation (Physics-IQ):** Hard checks that mass, energy, and accounting rules hold bit-for-bit across 100% of steps.
* **Causal Directionality:** Intervening positively on a bottleneck must not worsen the throughput of downstream dependencies.
* **Epistemic Abstention (RWM-U):** Automatically flagging when a test scenario ventures outside the historical support envelope.
