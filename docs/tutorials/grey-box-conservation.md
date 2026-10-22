# Tutorial: Grey-Box Dynamics & Hard Conservation Layers

> **Learning Goal**: Learn how to train learned neural dynamics while architecturally guaranteeing that physical conservation laws (mass, energy, double-entry balances) are never violated.

---

## 1. Why Pure Neural Networks Fail in World Modeling

Unconstrained neural networks (e.g. MLPs, LSTMs, Transformers) learn smooth statistical approximations. When forecasting multi-step inventory or cash flow dynamics, small floating-point errors accumulate:
* Step 1: Mass balance drift = $+0.05$ units.
* Step 10: Mass balance drift = $+4.20$ units.
* Step 50: The network generates $50$ units of phantom goods out of thin air.

**The EWM Grey-Box Paradigm ([ADR-042](../adr/ADR-042-grey-box-conservation-flywheel.md)):**
$$\Delta s_t = \underbrace{f_{\text{known}}(s_t, a_t)}_{\text{Known Conservation Law}} + \underbrace{\mathcal{P}_{\text{null}}\left( g_\theta(s_t, a_t) \right)}_{\text{Learned Residual Projected into Nullspace}}$$

The neural network is architecturally constrained to predict strictly within the nullspace of the conservation matrix $A_{\text{cons}} \Delta s = 0$.

---

## 2. Implementing a Conservation Layer

```python
import numpy as np
from ewm_engine.experimental.conservation import (
    ConservationProjectionLayer,
    LinearConservationLaw,
)

# 1. Define Linear Conservation Invariant: Delta Stock A + Delta Stock B = 0
law = LinearConservationLaw(
    matrix=np.array([[1.0, 1.0]]),
    resource_ids=["stock_a", "stock_b"],
)
layer = ConservationProjectionLayer(laws=[law])
```

---

## 3. The Neuro-Symbolic Flywheel

In EWM’s neuro-symbolic flywheel ([ADR-042](../adr/ADR-042-grey-box-conservation-flywheel.md)), residual neural dynamics are continuously audited:
1. **Residual Auditing:** If the neural residual $g_\theta$ learns a systematic correction that persists across scenarios, the symbolic equation discovery engine (SINDy / PySR) proposes an algebraic equation.
2. **Promotion to Known:** The discovered rule is reviewed, approved, and added to the declarative YAML specification (`WSL`), moving dynamics from "learned black-box" to "known auditable code."
