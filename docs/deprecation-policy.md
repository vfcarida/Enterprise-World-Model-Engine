# Deprecation & Stability Policy

- **Status**: Active
- **Standard**: NEP-23 Compatible / SemVer 2.0.0 Compliant
- **Authority**: Enterprise World Model Engine Maintainers

---

## 1. Overview & Principles

Enterprise World Model Engine (`ewm-engine`) provides strict backwards-compatibility guarantees for enterprise simulation pipelines and mission-critical decision workflows.

To balance long-term stability with architectural innovation, EWM Engine adopts a formal deprecation lifecycle modeled directly on **NEP-23 (NumPy Enhancement Proposal 23: Backwards Compatibility and Deprecation Policy)** and **Semantic Versioning 2.0.0**.

---

## 2. Core Deprecation Invariants

1. **Minimum Deprecation Window**: Any public API scheduled for removal must remain deprecated for **at least two minor releases** or **one full calendar year**, whichever is longer.
   - *Example*: A feature marked deprecated in `v1.5.0` cannot be removed until `v1.7.0` at the earliest.
2. **No Deprecations or Removals in Patch Releases**:
   - Patch releases (`1.5.x`) are strictly reserved for non-breaking bug fixes, security patches, and performance optimizations.
   - Deprecations may only be introduced in minor releases (`1.x.0`).
   - Removals may only occur in scheduled minor or major releases adhering to the two-minor window.
3. **Explicit Runtime Warnings**:
   - Deprecated code must emit standard Python `DeprecationWarning` with `stacklevel=2` so tracebacks point to the calling user code rather than internal wrappers.
   - Deprecation warnings must never use standard `logging.warning()` (which cannot be caught by static analyzers or pytest warning filters).
4. **Mandatory Alternatives**:
   - Every deprecation notice must specify a recommended migration path or alternative replacement API.

---

## 3. Developer Implementation

### 3.1 Function / Class Deprecation (`@deprecated`)

Decorate deprecated functions, methods, or classes with `@deprecated`:

```python
from ewm_engine.core.deprecation import deprecated

@deprecated(
    since="1.5.0",
    removed_in="1.7.0",
    alternative="Use World.branch() instead.",
)
def clone_world(world: World) -> World:
    """Legacy helper for branching worlds."""
    return world.branch()
```

The decorator automatically:
- Emits `DeprecationWarning` with `stacklevel=2` upon invocation.
- Appends a `.. deprecated:: 1.5.0` marker to the docstring for MkDocs/Sphinx documentation rendering.
- Attaches `__deprecated__`, `__deprecated_since__`, and `__deprecated_removed_in__` metadata attributes for automated introspection.

### 3.2 Granular Parameter / Branch Deprecation (`deprecate`)

For deprecating specific arguments or conditional branches inside active functions:

```python
from ewm_engine.core.deprecation import deprecate

def run_simulation(world: World, scenario: Scenario, legacy_mode: bool = False) -> SimulationResult:
    if legacy_mode:
        deprecate(
            feature="run_simulation(legacy_mode=True)",
            since="1.5.0",
            removed_in="1.7.0",
            alternative="Omit legacy_mode and use standard scenario parameters.",
        )
    ...
```

---

## 4. Quality Gates & Testing Discipline

1. **Pytest Warning Capture**: Every deprecated API must have an explicit automated test asserting that invoking it triggers the expected `DeprecationWarning`:
   ```python
   with pytest.warns(DeprecationWarning, match=r"'clone_world' was deprecated in v1\.5\.0"):
       clone_world(world)
   ```
2. **Zero Unexpected Deprecations (`-W error`)**: CI runs test suites with strict warning treatment, ensuring that any undeclared or accidental deprecation immediately fails the build.
3. **Newsfragment Requirement**: Any pull request introducing a deprecation must include a Towncrier newsfragment of type `.removal` (e.g., `newsfragments/412.removal.md`).

---

## 5. Scope & Maturity Exceptions

| Scope / Namespace | Deprecation Required? | Policy |
| :--- | :---: | :--- |
| **Root Public API (`ewm_engine.*`)** | **Yes** | Governed strictly by NEP-23 (2 minor releases / 1 year). |
| **Core Primitives (`ewm_engine.core.*`)** | **Yes** | Governed strictly by NEP-23. |
| **Beta Adapters (`ewm_engine.adapters.*`)** | **Yes** | At least 1 minor release deprecation cycle. |
| **Experimental (`ewm_engine.experimental.*`)** | **No** | Subject to API evolution across minor releases without deprecation cycles. |
| **Private Internals (`ewm_engine._*`)** | **No** | Private implementation details may change without notice. |
