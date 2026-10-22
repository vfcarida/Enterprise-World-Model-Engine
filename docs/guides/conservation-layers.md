# How-To Guide: Enforcing Architectural Conservation Laws

> **Goal**: Attach a linear or algebraic conservation constraint to custom dynamic models to eliminate floating-point drift and guarantee exact physical balance.

---

## 1. Overview
In `ewm-engine`, conservation laws are enforced at two levels:
1. **Pre/Post-Transition Invariant Gates:** Reject transitions that violate physical rules ([`ResourceCapacityConstraint`](../reference/api/constraints.md)).
2. **Architectural Nullspace Projection:** Proactively projects continuous state transitions into the exact subspace of conserving vectors ([`ConservationLayer`](../reference/api/dynamics.md)).

---

## 2. Step-by-Step Procedure

### Step 1: Define Conservation Matrix $A_{\text{cons}}$
For $D$ state variables where certain groups must sum to zero (e.g. transfers, double-entry credits and debits), write the linear system $A_{\text{cons}} \Delta s = 0$:

```python
import numpy as np
from ewm_engine.experimental.conservation import (
    ConservationProjectionLayer,
    LinearConservationLaw,
)

# Example: 3 accounts (Cash, Accounts Receivable, Inventory).
# Total enterprise liquid asset value must be invariant under internal reallocation:
# delta_cash + delta_ar + delta_inv = 0
law = LinearConservationLaw(
    matrix=np.array([[1.0, 1.0, 1.0]]),
    resource_ids=["cash", "ar", "inv"],
)
layer = ConservationProjectionLayer(laws=[law])
```

### Step 2: Project Raw Dynamics Deltas
Inside your custom dynamics transition function:

```python
# Raw continuous change proposed by regression or actor policy
raw_delta = np.array([25.0, -10.0, -12.0])  # Drift = +3.0

# Exact orthogonal projection into nullspace:
valid_delta = layer.project(raw_delta)

print("Projected Delta:", valid_delta)
print("Residual Drift:", np.dot(A, valid_delta)[0])
```

---

## 3. Best Practices
* **Zero Overhead in Pure Discrete Worlds:** For discrete token transfers, use [`ActionTransferAvailabilityConstraint`](../reference/api/constraints.md) rather than continuous projection.
* **Physics-IQ Gate Tracking:** All projections are logged and evaluated by the Physics-IQ reliability monitor ([ADR-044](../adr/ADR-044-causal-correctness-metamorphic-evaluation.md)).
