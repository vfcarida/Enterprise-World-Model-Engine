# ADR-008: Constraint Phase Semantics and Rollout Invalidation

## Status
Accepted

## Related Spec & Issues
- Spec: `docs/specs/spec-driven-development.md` (Constraints & Simulation Lifecycle sections)
- Acceptance Criteria: AC-006, AC-007, AC-008
- Milestone: M2

## Context
In previous iterations, the `Constraint` protocol's `evaluate` method received only `(state, action: Action | None)`. The engine did not distinguish between pre-action feasibility validation and post-transition state verification, nor did it enforce the normative failure modes demanded by the v1 specification. In particular, a post-transition hard constraint breach was recorded in the systemic trace but allowed the rollout to continue simulating future steps, silently masking fatal invariant breaches.

The authoritative specification (`docs/specs/spec-driven-development.md`) defines four non-negotiable normative outcomes:
1. `HARD + PRE_ACTION + violated` → reject action before transition; do not pass to dynamics.
2. `SOFT + PRE_ACTION + violated` → record result, continue execution.
3. `HARD + POST_TRANSITION + violated` → mark rollout as `TrajectoryStatus.INVALID`, terminate rollout immediately, do not silently project or repair state.
4. `SOFT + POST_TRANSITION + violated` → record result, continue execution.

## Decision
1. Update `Constraint.evaluate` in `ewm_engine.constraints.base` to:
   ```python
   def evaluate(
       self,
       state: WorldState,
       actions: Sequence[Action] = (),
       *,
       phase: ConstraintPhase,
   ) -> ConstraintResult: ...
   ```
2. Update `ConstraintResult` to include `phase: ConstraintPhase` and `entity_ids: tuple[str, ...]` with bidirectional synchronization with `violating_entities`.
3. In `ConstraintRegistry`:
   - `validate_actions` evaluates constraints with `phase=ConstraintPhase.PRE_ACTION`. Actions violating `HARD` constraints are excluded from accepted actions.
   - `validate_state` evaluates constraints with `phase=ConstraintPhase.POST_TRANSITION`.
4. In `SimulationEngine`:
   - After dynamics transition and post-transition validation, if any `HARD` constraint is violated:
     - Mark the trajectory as `TrajectoryStatus.INVALID`.
     - Record the step record without advancing simulation time (`final_next_state = raw_next_state`), strictly prohibiting silent state repairs.
     - Abort the horizon loop for that rollout immediately.
   - Finalize the trajectory (`trajectory.finalize()`) to guarantee immutable status upon completion.
5. In `SimulationResult`:
   - Expose `completed_count`, `invalid_count`, `failed_count`.
   - Update `violation_rate()` to account for invalid rollouts and violations.

## Consequences
- **Positive**: Strict adherence to the v1 specification; exact operational semantics for hard vs. soft constraints; zero silent state projection or fabrication; complete auditability of failure modes.
- **Negative / Breaking Change**: Custom implementations of `Constraint` must update their `evaluate` method signature to accept `actions: Sequence[Action] = ()` and keyword-only `phase: ConstraintPhase`.
