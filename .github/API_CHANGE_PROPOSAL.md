# API Change Proposal (ACP-XXX): [Descriptive Title]

- **Target Release:** [e.g. v1.1.0 / v2.0.0]
- **Author:** [Name / GitHub Handle]
- **Status:** [Draft | Under Review | Approved | Rejected | Implemented]
- **Impact Level:** [Non-breaking enhancement | Deprecation | Breaking change (2.0.0+ only)]
- **Related Spec:** [Path to specification section in docs/specs/ or ewm.md]
- **Related ADR:** [docs/adr/ADR-XXX-... if applicable]
- **Acceptance Criteria:** [e.g. AC-024, etc.]

---

## 1. Summary
[A concise 1-2 paragraph description of the proposed API change, deprecation, or extension.]

## 2. Motivation & Use Case
[Why is this change necessary? What problem does it solve for users or the engine? Note any spec requirements or invariants addressed.]

## 3. Proposed API Surface Diff
```python
# Before
class Example:
    def existing_method(self, arg1: str) -> None:
        ...

# After
class Example:
    def existing_method(
        self,
        arg1: str,
        new_optional_param: int = 0,  # Rule: new params in 1.x minor MUST be optional with default
    ) -> None:
        ...
```

### Affected Symbols in `ewm_engine.__all__`
- [ ] No change to `__all__`
- [ ] New public symbol proposed: `[SymbolName]` (Classified as: Stable | Experimental)
- [ ] Deprecated symbol: `[SymbolName]` (Requires ≥1 minor and ≥90 days notice before removal)

### Schema Changes (if applicable)
- Schema file(s): `schemas/[name].schema.json`
- Invariant check: Any new fields must be optional with default values; existing fields cannot be removed in `1.x`.

## 4. Backwards Compatibility & Migration Strategy
- **Is this a breaking change?** (Note: Breaking changes to Stable symbols are prohibited in 1.x per [Stability Policy](docs/stability-policy.md)).
- **Deprecation lifecycle:** (If deprecating, specify deprecation warning, documentation in CHANGELOG, and expected removal target).
- **Migration instructions:** [Step-by-step guidance for callers migrating to the updated API].

## 5. Affected Files & Tests
- Public API / Interface:
- Implementation:
- Schemas:
- Enforcing tests:
  - Contract test: `tests/contract/test_api_compatibility.py`
  - Integration / Unit tests:

## 6. Review & Approval Checklist
- [ ] Reviewed against `docs/stability-policy.md` invariants.
- [ ] Contract snapshot updated in `tests/contract/test_api_compatibility.py` (if approved).
- [ ] Deprecation warning emitted at runtime using `warnings.warn(..., DeprecationWarning, stacklevel=2)` (if deprecating).
- [ ] CHANGELOG entry drafted under `## [Unreleased]`.
- [ ] Documentation updated in `docs/`.
- [ ] CI quality gates pass (`mypy`, `ruff`, `pytest`, `coverage`, `packaging`).
- [ ] Approved by core maintainers.
