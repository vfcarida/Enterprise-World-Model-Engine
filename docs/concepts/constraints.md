# First-Class Constraint Engine

In traditional machine learning, operational limits are often shoehorned into neural loss functions as soft penalty weights ($\mathcal{L} + \lambda \mathcal{L}_{\text{constraint}}$). In real-world enterprise operations, this approach fails: a warehouse cannot physically store 250 pallets if its physical limit is 200, and a vehicle cannot navigate a submerged roadway regardless of penalty coefficients.

In EWM Engine, **constraints are first-class computational citizens**.

---

## Hard vs. Soft Constraints

EWM Engine classifies rules into two distinct enforcement categories:

### 1. Hard Constraints (`ConstraintSeverity.HARD`)
- **Semantics**: Inviolable physical boundaries, legal statutes, or strict logistical invariants.
- **Action Validation**: If a proposed action breaches a hard constraint (e.g. attempting to dispatch more stock than currently on hand), the action is **rejected pre-transition**.
- **State Validation**: If an evolved state violates a hard constraint, it is flagged as a fatal operational failure and recorded in the audit log.

### 2. Soft Constraints (`ConstraintSeverity.SOFT`)
- **Semantics**: Operational guidelines, quality-of-service targets, or non-fatal policy preferences.
- **Evaluation**: The action or state transition is permitted, but numerical penalties are recorded and aggregated to evaluate policy degradation.

---

## Complete Constraint Provenance

When a constraint evaluates to unsatisfied, EWM Engine captures full structural provenance:

```python
class ConstraintResult(BaseModel):
    satisfied: bool
    constraint_id: ConstraintId
    constraint_version: str
    severity: ConstraintSeverity
    message: str
    violating_entities: tuple[EntityId, ...]
    violating_resources: tuple[ResourceId, ...]
    violating_values: dict[str, Any]
    step: int | None
    preceding_action_id: ActionId | None
    penalty: float
```

This ensures that stakeholders do not receive opaque error codes or unexplainable failure rates. The engine answers:
1. **Which specific rule was breached?**
2. **What semantic version of the rule was in effect?**
3. **What simulation step and continuous timestamp did it occur at?**
4. **What exact numerical values breached the threshold?**
5. **Which agent action preceded the violation?**
