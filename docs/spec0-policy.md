# SPEC 0 Support Policy

- **Status**: Active
- **Standard**: Scientific Python Ecosystem Coordination (SPEC 0)
- **Authority**: Enterprise World Model Engine Maintainers

---

## 1. Overview

The Enterprise World Model Engine (`ewm-engine`) participates in the scientific Python ecosystem and adheres to **[SPEC 0 — Minimum Supported Versions](https://scientific-python.org/specs/spec-0000/)**.

SPEC 0 defines a predictable, time-based cadence for dropping support for aging Python versions and upstream scientific dependencies, allowing the engine to leverage modern language features, security improvements, and performance optimizations without premature breaking changes.

---

## 2. Policy Windows

1. **Python Language Support (36-Month Window)**:
   - EWM Engine drops support for a Python minor version **36 months (3 years)** after its initial stable release.
   - *Example*: Python 3.10 was released in October 2021 and was retired from the support matrix in late 2024. Python 3.11 (released October 2022) is supported through late 2025/2026.
2. **Core Dependencies Support (24-Month Window)**:
   - EWM Engine drops support for core dependency releases (such as `numpy`) **24 months (2 years)** after their initial release date.
   - *Example*: NumPy 1.25 was retired; NumPy `>=1.26.0` is the active floor.

---

## 3. Current Version Support Matrix

| Technology | Minimum Floor | Supported Range | Status |
| :--- | :---: | :---: | :--- |
| **Python** | `>=3.11` | 3.11, 3.12, 3.13 | 3.10 and earlier dropped per SPEC 0 |
| **NumPy** | `>=1.26.0` | 1.26.x, 2.x | 1.25 and earlier dropped per SPEC 0 |
| **Pydantic**| `>=2.6.0` | 2.6+ | Pydantic v1 dropped |
| **PyYAML** | `>=6.0.1` | 6.0+ | Modern safe loader |

---

## 4. Automated CI Enforcement

Compliance with SPEC 0 is enforced on every commit and pull request via the automated packaging test suite:

- [`tests/packaging/test_spec0_compliance.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/packaging/test_spec0_compliance.py): Inspects `pyproject.toml` to assert that `requires-python` and `numpy` floors do not regress below the mandated SPEC 0 calendar thresholds.
- Multi-OS CI matrix tests against all active supported Python versions, with advisory canary builds for free-threaded Python (`3.13t`) and prerelease interpreters (`3.14-dev`).
