# API Change Proposal (ACP-XXXX)

- **Target Release:** vX.Y.Z
- **Author:** Name (@handle)
- **Status:** Draft | Under Review | Approved | Rejected
- **Impact Level:** Breaking | Non-Breaking (Additive) | Deprecation

---

## 1. Summary
Brief explanation of the proposed changes to the public API surface.

## 2. Motivation
Why is this change necessary? Which use cases or architectural requirements motivate it?

## 3. Proposed API Diff
```python
# Before
from ewm_engine import OldSymbol

# After
from ewm_engine import NewSymbol
```

## 4. Backwards Compatibility & Migration Strategy
- Is this a breaking change?
- What is the deprecation schedule? (e.g. at least 1 minor version / 90 days)
- How do existing users migrate?

## 5. Affected Files & Tests
- Files in `src/ewm_engine/`
- Contract tests in `tests/contract/`
- Documentation in `docs/`

## 6. Checklist
- [ ] Added/updated contract tests in `tests/contract/test_public_api.py`
- [ ] Updated `__all__` in `src/ewm_engine/__init__.py`
- [ ] Updated API documentation in `docs/api/reference.md`
- [ ] Added note to `CHANGELOG.md`
