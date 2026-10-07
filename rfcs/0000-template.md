# RFC-XXXX: [Short Title of Proposal]

- **RFC Number:** XXXX
- **Status:** Draft | Under Review | Last Call | Accepted | Implemented | Rejected | Superseded
- **Author(s):** Name (@github_handle)
- **Created Date:** YYYY-MM-DD
- **Target Release:** vX.Y.Z
- **Steering Council Sponsor:** Vinicius Caridá (@vfcarida)
- **Related ADRs:** [ADR-XXX](../docs/adr/ADR-XXX.md)
- **Related Issues / PRs:** #123

---

## 1. Executive Summary
Provide a concise 1-paragraph summary of the proposed feature, architectural paradigm, or policy change. What problem does it solve and who is impacted?

---

## 2. Motivation & Problem Statement
- What is the current limitation or shortcoming in EWM Engine?
- Why can this not be solved with existing abstractions (`DynamicsModel`, `Constraint`, `WorldState`, etc.)?
- What are the concrete user or researcher personas benefiting from this?

---

## 3. Detailed Specification & Design
- Describe the complete architectural and mathematical design.
- Include class diagrams or Mermaid workflows where applicable.
- Specify exact data models, invariants, and serialization behaviors.
- Detail edge-case handling, error definitions, and algorithmic complexities ($O(N)$).

```python
# Example interface sketch
class ProposedProtocol(Protocol):
    def evaluate(self, state: WorldState) -> EvaluationResult:
        ...
```

---

## 4. Backwards Compatibility & SemVer Impact
- Does this change break any existing public APIs exported in `ewm_engine.__all__`?
- Does it require a deprecation cycle following NEP-23 / [ADR-037](../docs/adr/ADR-037-release-engineering-version-truth-and-deprecation-policy.md)?
- What is the migration path for existing users and downstream adapters?

---

## 5. Security, Invariants & Supply Chain
- Does this proposal introduce any new external third-party dependencies?
- How are physical conservation laws and state immutability preserved?
- Are there arbitrary execution risks (rejection of `eval`/`exec`/`pickle`)?

---

## 6. Alternatives Considered
- What alternative designs or implementations were investigated?
- Why were they rejected in favor of this proposal?

---

## 7. Implementation Plan & Test Strategy
- Which unit, property, contract, and architecture tests will validate this feature?
- What milestone will this ship in on the [Public Roadmap](../ROADMAP.md)?
- Who will lead the reference implementation?

---

## 8. Unresolved Questions
- What open trade-offs remain to be decided during the review period?
