# API Change Proposal (ACP-002): Scenario Branching & Scheduled Actions

- **Target Release:** v0.2.0 (pre-1.0)
- **Author:** Maintainer Team (@maintainer)
- **Status:** Approved
- **Impact Level:** Non-breaking enhancement
- **Related Spec:** `docs/specs/spec-driven-development.md` (Scenario & Counterfactual Branching)
- **Related ADR:** `docs/adr/ADR-012-determinism-reproducibility-and-branch-isolation.md`
- **Acceptance Criteria:** AC-004, AC-005, AC-009

---

## 1. Summary
Enhance `Scenario` with support for `initial_state` binding, explicit `scheduled_actions: tuple[ScheduledAction, ...]`, and a non-mutating `branch(...)` method that yields an independent counterfactual branch while preserving `initial_state` identity.

## 2. Motivation
The authoritative v1 specification requires:
1. Deterministic simulation execution with scheduled interventions/actions across time steps.
2. Counterfactual branching (`Scenario.branch`) that creates a new scenario referencing the exact same initial state (fingerprint identity) without mutating the parent scenario or sharing mutable state.
3. Clean per-rollout RNG stream spawning (`np.random.SeedSequence(seed).spawn(samples)`).

## 3. Proposed API Diff
```python
class ScheduledAction(BaseModel):
    step: int = Field(ge=0, description="0-indexed simulation time step at which action is triggered")
    action: Action = Field(description="Action to execute")

class Scenario(BaseModel):
    # Added fields
    initial_state: WorldState | None = Field(default=None, description="Optional pinned initial world state")
    scheduled_actions: tuple[ScheduledAction, ...] = Field(default_factory=tuple)

    def branch(
        self,
        scenario_id: str,
        *,
        name: str | None = None,
        scheduled_actions: Sequence[ScheduledAction] | None = None,
        seed: int | None = None,
        intervention: Intervention | None = None,
        metadata: Mapping[str, Any] | None = None,
        horizon: int | None = None,
        samples: int | None = None,
    ) -> Scenario: ...
```

In `SimulationEngine`:
```python
# Self-check determinism helper
def verify_determinism(self, world: World, scenario: Scenario) -> bool: ...
```

## 4. Backwards Compatibility & Migration Strategy
- **Is this a breaking change?** No. All new fields have defaults (`initial_state=None`, `scheduled_actions=()`). Existing `Scenario` instantiation without these fields continues to work without modification.
- **Migration Strategy:** Existing code continues to function seamlessly. Callers desiring counterfactual exploration can invoke `scenario.branch(...)` or supply `scheduled_actions=[ScheduledAction(step=t, action=act)]`.

## 5. Affected Files & Tests
- Models: `src/ewm_engine/simulation/scenario.py`
- Branching: `src/ewm_engine/simulation/branching.py`
- Engine: `src/ewm_engine/simulation/engine.py`
- Schema: `schemas/scenario.schema.json`
- Tests:
  - `tests/property/test_determinism_properties.py` (AC-004)
  - `tests/property/test_branch_isolation.py` (AC-005)
  - `tests/property/test_monte_carlo.py` (AC-009)
  - `tests/property/test_no_global_rng.py`
