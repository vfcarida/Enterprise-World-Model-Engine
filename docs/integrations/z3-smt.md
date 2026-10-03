# Formal Verification with Z3 SMT

The `Z3ConstraintAdapter` connects the formal [Z3 Theorem Prover (SMT)](https://github.com/Z3Prover/z3) with EWM Engine's constraint protocol.

---

## Symbolic Invariant Verification

Unlike empirical unit tests that check a single numerical value, SMT solvers symbolically prove that **no possible state transition** can violate an invariant:

```python
from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.integrations.solvers import Z3ConstraintAdapter

def verify_budget_bounds(state, action, z3):
    # Create symbolic integer/real variables
    solver = z3.Solver()
    cash = z3.Real("cash")
    spend = z3.Real("spend")
    
    # Assert constraints
    solver.add(cash >= 0)
    current_cash = state.get_resource("cash").current
    solver.add(cash == current_cash)
    
    # Check if a proposed spend can cause negative cash
    spend_qty = action.parameters.get("amount", 0.0) if action else 0.0
    solver.add(spend == spend_qty)
    solver.add(cash - spend < 0)
    
    # If SAT, a violating condition is possible
    if solver.check() == z3.sat:
        return False, "Symbolic invariant violated: spend exceeds cash reserves", {"deficit": str(solver.model())}
    return True, "Symbolic invariant holds", {}

constraint = Z3ConstraintAdapter(
    constraint_id="symbolic_cash_reserve",
    solver_fn=verify_budget_bounds,
    severity=ConstraintSeverity.HARD,
)
```
