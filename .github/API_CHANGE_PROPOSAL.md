# API Change Proposal (ACP-001): Constraint Protocol Phase Signature & Invalidation Semantics

- **Target Release:** v0.2.0 (pre-1.0)
- **Author:** Maintainer Team (@maintainer)
- **Status:** Approved
- **Impact Level:** Breaking (Pre-1.0 Refactoring)
- **Related Spec:** `docs/specs/spec-driven-development.md` (Constraints section)
- **Related ADR:** `docs/adr/ADR-008-constraint-phase-semantics-and-rollout-invalidation.md`
- **Acceptance Criteria:** AC-006, AC-007, AC-008

---

## 1. Summary
Update the public `Constraint` protocol method `evaluate` to accept `actions: Sequence[Action] = ()` and keyword-only `phase: ConstraintPhase`. Add `phase: ConstraintPhase` and `entity_ids: tuple[str, ...]` to `ConstraintResult`. Implement normative rollout termination on post-transition hard constraint violations, marking rollouts as `TrajectoryStatus.INVALID`.

## 2. Motivation
The authoritative v1 specification (`docs/specs/spec-driven-development.md`) requires:
1. Operational distinction between `PRE_ACTION` (action feasibility validation) and `POST_TRANSITION` (state invariant evaluation).
2. Four normative outcomes:
   - `HARD + PRE_ACTION + violated` → reject action before transition.
   - `SOFT + PRE_ACTION + violated` → record result, continue.
   - `HARD + POST_TRANSITION + violated` → mark rollout `TrajectoryStatus.INVALID`, abort rollout, do not repair or fabricate state.
   - `SOFT + POST_TRANSITION + violated` → record result, continue.
3. Strict prohibition of silent projection of invalid states to valid manifolds within the core.

## 3. Proposed API Diff
```python
# Before
class Constraint(Protocol):
    def evaluate(
        self,
        state: WorldState,
        action: Action | None = None,
    ) -> ConstraintResult: ...

# After
class Constraint(Protocol):
    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult: ...
```

In `ConstraintResult`:
```python
# Added fields
phase: ConstraintPhase = Field(default=ConstraintPhase.POST_TRANSITION)
entity_ids: tuple[EntityId, ...] = Field(default_factory=tuple)
```

In `Trajectory`:
```python
# Trajectory status transition and locking
trajectory.status: TrajectoryStatus  # Mutable during rollout, read-only after finalize()
trajectory.finalize() -> None
```

## 4. Backwards Compatibility & Migration Strategy
- **Is this a breaking change?** Yes, for custom implementations of `Constraint`. Because the project is currently pre-1.0 (v0.1.0 -> v0.2.0), breaking changes are permitted under SemVer 2.0.0 section 4, with explicit documentation and migration guidance.
- **Migration Strategy:**
  - Implementers of `Constraint` should update their `evaluate` method to accept `actions: Sequence[Action] = (), *, phase: ConstraintPhase`.
  - Built-in constraints (`ResourceCapacityConstraint`, `ResourceNonNegativeConstraint`, `ActionTransferAvailabilityConstraint`) provide default values for `phase` so direct single-argument evaluation continues to function for ad-hoc checks.
  - Consumers checking `res.violating_entities` continue to work via bidirectional attribute synchronization with `res.entity_ids`.

## 5. Affected Files & Tests
- Protocol: `src/ewm_engine/constraints/base.py`
- Models: `src/ewm_engine/constraints/results.py`
- Built-in constraints: `src/ewm_engine/constraints/standard.py`, `src/ewm_engine/integrations/solvers.py`, `examples/civicflow/constraints.py`
- Registry: `src/ewm_engine/constraints/registry.py`
- Engine: `src/ewm_engine/simulation/engine.py`
- Trajectory: `src/ewm_engine/simulation/trajectory.py`
- Tests:
  - `tests/unit/test_constraints.py`
  - `tests/integration/test_hard_constraints.py` (AC-006)
  - `tests/integration/test_soft_constraints.py` (AC-007)
  - `tests/integration/test_post_transition_violation.py` (AC-008)

## 6. Checklist
- [x] Added/updated contract and unit tests in `tests/unit/test_constraints.py`
- [x] Added integration tests for AC-006, AC-007, AC-008
- [x] Confirmed `ewm_engine.__all__` includes `ConstraintPhase`, `ConstraintSeverity`, `ConstraintResult`, `TrajectoryStatus`
- [x] Added `docs/adr/ADR-008-constraint-phase-semantics-and-rollout-invalidation.md`
- [x] Type checking passes with 0 errors in strict mode (`uv run mypy src tests`)
- [x] Linter and formatter pass with 0 errors (`uv run ruff check src tests`)
