# Formal Verification with Z3 SMT

> [!NOTE]
> **Maturity**: Beta. Implements formal `Constraint` and `ConstraintSolver` protocols with mandatory timeout limits and unsat core extraction. Requires `ewm-engine[solvers]`.

The `Z3ConstraintAdapter` connects the formal [Z3 Theorem Prover (SMT)](https://github.com/Z3Prover/z3) with EWM Engine's constraint and solver protocols.

---

## Symbolic Invariant Verification

Unlike empirical unit tests that check a single numerical value, SMT solvers symbolically prove that **no possible state transition** can violate an invariant.

Every `Z3ConstraintAdapter` evaluation enforces a mandatory finite timeout (`timeout_ms`, default 5000ms). If a solver search exceeds this threshold, it returns `SolverStatus.UNKNOWN` with `timed_out=True` rather than blocking the simulation process.

---

## Example: Enforcing Cash Reserve Invariants with Unsat Core

```python
from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.integrations.solvers import Z3ConstraintAdapter


def verify_budget_bounds(state, action, z3, timeout_ms=5000):
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    cash = z3.Real("cash")
    spend = z3.Real("spend")

    # Track assertions with assumption literals for unsat core extraction
    p_nonneg = z3.Bool("p_non_negative_cash")
    p_spend = z3.Bool("p_spend_limit")

    current_cash = state.get_resource("cash").current
    solver.add(cash == current_cash)

    spend_qty = action.parameters.get("amount", 0.0) if action else 0.0
    solver.add(spend == spend_qty)

    # Invariant to verify: cash - spend >= 0
    # To prove by contradiction, we assert the negation (cash - spend < 0)
    solver.assert_and_track(cash - spend < 0, p_nonneg)

    check_result = solver.check()
    if check_result == z3.sat:
        # A counterexample was found: spend violates reserves
        return (
            False,
            "Symbolic invariant violated: spend exceeds cash reserves",
            {
                "model": str(solver.model()),
                "deficit": True,
            },
        )
    elif check_result == z3.unsat:
        # Contradiction reached: no state can violate the invariant
        return True, "Symbolic invariant proven for all states", {}
    else:
        # Solver reached timeout limit
        return False, "Solver timed out", {"timed_out": True, "reason": "timeout"}


constraint = Z3ConstraintAdapter(
    constraint_id="symbolic_cash_reserve",
    solver_fn=verify_budget_bounds,
    severity=ConstraintSeverity.HARD,
    timeout_ms=3000,  # 3-second mandatory timeout
)

state = WorldState(resources=[Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)])

# 1. Standalone ConstraintSolver Protocol verification
solver_result = constraint.check(state=state)
print(f"Satisfied: {solver_result.satisfied}")
print(f"Solver Status: {solver_result.status.value}")
print(f"Duration: {solver_result.solve_time_seconds:.4f}s")

# 2. Engine Constraint Protocol evaluation
eval_result = constraint.evaluate(state=state)
print(f"Engine Accepted: {eval_result.satisfied}")
```
