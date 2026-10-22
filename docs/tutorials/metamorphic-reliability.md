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
from ewm_engine.evaluation.metamorphic import (
    STANDARD_MR_CARDS,
    MetamorphicRelationCard,
    MetamorphicSuite,
)

# Instantiate the standard 6-card metamorphic test suite
suite = MetamorphicSuite(cards=STANDARD_MR_CARDS)
report = suite.evaluate(spec=world_spec, scenario=scenario)

print(f"Metamorphic Pass Rate: {report.pass_rate:.1%}")
print(f"All Relations Passed:  {report.all_passed}")
```

---

## 3. The Metamorphic Reliability Scorecard
* **Invariant Conservation (Physics-IQ):** Hard checks that mass, energy, and accounting rules hold within declared IEEE-754 numerical tolerance ($\le 10^{-12}$) across 100% of steps.
* **Causal Directionality:** Intervening positively on a bottleneck must not worsen the throughput of downstream dependencies.
* **Epistemic Abstention (RWM-U):** Automatically flagging when a test scenario ventures outside the historical support envelope.
