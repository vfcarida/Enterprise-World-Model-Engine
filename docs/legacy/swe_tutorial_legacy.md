# Software Engineering Progressive-Delivery & Shared-Pool Contention Tutorial

**Module**: `ewm_engine.experimental.swe_ecosystem` (`EXPERIMENTAL`)  
**Executable Script**: [`examples/swe/ci_deployment_cascade.py`](../../examples/swe/ci_deployment_cascade.py)  
**Test Suite**: [`tests/unit/test_swe_application.py`](../../tests/unit/test_swe_application.py)  

---

## 1. Problem Formulation: Static Call Graphs vs. Stateful Resource Contention

In a microservice architecture (`auth_lib`, `inventory_cache`, `orders_db`, `checkout_service`, `api_gateway`, `payment_worker`), enabling `strict_jwt_flag=True` in `auth_lib` with `3x` retries on `checkout_service` exhausts the shared `orders_db` connection pool and starves `payment_worker`—even though `payment_worker` has **zero static dependency** on `auth_lib` or `checkout_service`.

At the same time, this tutorial highlights three explicit domain transfer boundaries (`SWE-N1`–`SWE-N3`):
1. **`SWE-N1` (Static Graph Wins Compile Breaks)**: For deterministic compile/schema breaks, static dependency analysis (`Rung 1`) achieves `F1 = 1.000` with `1` specification component vs. `10` for `Rung 5`.
2. **`SWE-N2` (Historical ML Wins Stationary Test Selection)**: Under stationary configuration, GBT on co-change history (`Rung 3`) beats `Rung 5` (`0.950` vs. `0.920` recall).
3. **`SWE-N3` (Linear Conservation Does Not Transfer to Logic Bugs)**: Stoichiometric conservation (`A * dx = 0`) applies to bounded operational queues and connection pools, **not** discrete code logic bugs.

---

## 2. Running the Example

```bash
python examples/swe/ci_deployment_cascade.py
```

Expected output:

```text
Static dependents of auth_lib:    ('api_gateway', 'auth_lib', 'checkout_service')
Dynamic impacted under burst:     ('api_gateway', 'auth_lib', 'checkout_service', 'orders_db', 'payment_worker')
payment_worker missed statically: True
Static graph deployment regret:   $39,814.77
Dynamic EWM deployment regret:    $0.00
Negative findings confirmed:      3/3
```
