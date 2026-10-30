# Corporate Treasury & Liquidity Cascade Tutorial

**Executable Module**: [`examples/finance/treasury_liquidity_cascade.py`](../../examples/finance/treasury_liquidity_cascade.py)  
**Test Suite**: [`examples/finance/test_finance_application.py`](../../examples/finance/test_finance_application.py), [`tests/unit/test_finance_application.py`](../../tests/unit/test_finance_application.py)  

---

## 1. Overview

This domain tutorial demonstrates how to model a **corporate treasury and intraday liquidity buffer** using the `v1.5.0` stable core (`World`, `WorldState`, `Resource`, `Action`, `DeterministicTransferDynamics`, and `compare_scenarios`):

- **Conservation Invariant**: Total liquidity across `operating_cash` and `reserve_buffer` is strictly conserved (`max_ledger_conservation_drift <= 1e-12`).
- **Counterfactual Decision**: Evaluates whether proactively transferring `$400,000` from `reserve_buffer` to `operating_cash` prevents a regulatory intraday liquidity floor breach (`$350,000` minimum) and eliminates the `$85,000` overdraft penalty.

---

## 2. Running the Example

```bash
python examples/finance/treasury_liquidity_cascade.py
```

Expected output:

```text
Initial ledger total:         $1,200,000.00
Baseline operating cash:      $250,000.00
Counterfactual operating cash:$650,000.00
Max ledger conservation drift:0.000e+00
Baseline regret:              $483,000.00
Counterfactual regret:        $0.00
```
