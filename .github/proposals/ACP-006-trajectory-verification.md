# API Change Proposal (ACP-006): Trajectory Verification — Oracle-Graph and Temporal Logic over Traces

- **Target Release:** v1.3.0 "Trajectory Verification"
- **Author:** Verification Engineering Team
- **Status:** Implemented
- **Impact Level:** Additive subpackage (`ewm_engine.verification`)
- **Related Spec:** `docs/specs/features/FEAT-005-trajectory-verification.md`, `01_EXPANDED_ROADMAP.md` (Track T3)
- **Related ADR:** `docs/adr/ADR-027-trajectory-verification-oracle-graph-and-temporal-logic.md`
- **Acceptance Criteria:** AC-041 through AC-045, AC-024 (Zero Breaking Changes)

---

## 1. Summary

This proposal introduces the **Trajectory Verification Subsystem** (`ewm_engine.verification`) for the Enterprise World Model Engine. It provides formal, audit-ready verification of whole simulation trajectories against structured temporal and causal requirements:
1. **Oracle-Graph Verifier (Core, stdlib/Pydantic)**: Ports the Meta ARE pattern (arXiv:2509.17158) evaluating an expected event DAG against `SystemicTrace` across Consistency, Causality, and Timing axes.
2. **Property-Spec DSL & Cryptographic Provenance Folding (Core, Pydantic)**: Versioned declarative property schema with canonical SHA-256 hashing folded directly into simulation and result fingerprints.
3. **Bounded-Future Discrete STL Monitor (Core, pure-NumPy)**: Zero-dependency discrete temporal logic monitor evaluating bounded always, eventually, and until formulas over trajectory timeseries.
4. **Full STL/MTL via RTAMT (`[stl]` Optional Extra)**: Quantitative robustness evaluation and empirical robustness distributions across Monte Carlo rollouts.
5. **Documented External Stubs**: MoonLight STREL (spatio-temporal reach-and-escape logic) and LLM soft-check verifier using host-provided callables without LLM SDK dependencies.

---

## 2. Motivation & Use Cases

- **Beyond Per-Step Constraints:** Real enterprise requirements span multiple steps (e.g. SLA recovery windows, sustained capacity limits).
- **Agreement vs. Prose Evaluation:** Meta ARE empirical findings demonstrated 0.98 inter-annotator agreement for structured oracle graphs versus 0.72 for unstructured LLM judging.
- **Robustness Margins:** Decision makers need to quantify *how close* a policy was to violation ($\rho > 0$), not just binary compliance.

---

## 3. Proposed Public API Surface Diff

The proposal maintains strict SemVer 1.x backwards compatibility by preserving `LOCKED_V1_STABLE_SURFACE` in root `ewm_engine.__all__`. All new symbols reside in `ewm_engine.verification`:

```python
# ==============================================================================
# ewm_engine.verification
# ==============================================================================

# Property Specification DSL & Provenance
class PropertySpec(BaseModel): ...


def compute_property_hash(prop: PropertySpec) -> str: ...
def fold_properties_into_fingerprint(
    base_fingerprint: str, properties: Sequence[PropertySpec]
) -> str: ...


# Oracle Graph Verifier
class OracleNode(BaseModel): ...


class OracleEdge(BaseModel): ...


class OracleGraph(BaseModel):
    @classmethod
    def create(
        cls,
        nodes: Sequence[OracleNode] | Mapping[str, OracleNode] = (),
        edges: Sequence[OracleEdge] = (),
    ) -> OracleGraph: ...


class VerificationViolation(BaseModel): ...


class OracleVerificationResult(BaseModel): ...


def evaluate_oracle_graph(
    oracle: OracleGraph, trace: SystemicTrace
) -> OracleVerificationResult: ...


# Core Bounded Discrete STL Monitor
class STLFormula(ABC): ...


class PredicateFormula(STLFormula): ...


class NotFormula(STLFormula): ...


class AndFormula(STLFormula): ...


class OrFormula(STLFormula): ...


class ImpliesFormula(STLFormula): ...


class AlwaysFormula(STLFormula): ...


class EventuallyFormula(STLFormula): ...


class UntilFormula(STLFormula): ...


class STLVerdict(NamedTuple / BaseModel): ...


def evaluate_stl_bounded(
    formula: STLFormula, trajectory_or_signals: Trajectory | Mapping[str, Sequence[float]]
) -> STLVerdict: ...
def extract_trajectory_signals(trajectory: Trajectory) -> dict[str, np.ndarray]: ...
def predicate(variable: str, operator: str, threshold: float) -> PredicateFormula: ...
def always(child: STLFormula, k1: int = 0, k2: int = 0) -> AlwaysFormula: ...
def eventually(child: STLFormula, k1: int = 0, k2: int = 0) -> EventuallyFormula: ...
def until(phi: STLFormula, psi: STLFormula, k1: int = 0, k2: int = 0) -> UntilFormula: ...
def implies(antecedent: STLFormula, consequent: STLFormula) -> ImpliesFormula: ...


# Optional Extra [stl] Backend (RTAMT)
def is_rtamt_available() -> bool: ...


class RobustnessDistribution(BaseModel): ...


class RTAMTEvaluationBackend:
    def __init__(
        self, formula: str, variables: Mapping[str, str], spec_name: str = "TrajectorySTLSpec"
    ) -> None: ...
    def evaluate_signals(self, signals: Mapping[str, Sequence[float]]) -> float: ...
    def evaluate_trajectory(self, trajectory: Trajectory) -> float: ...
    def evaluate_monte_carlo(
        self, trajectories: Sequence[Trajectory]
    ) -> RobustnessDistribution: ...


# Stubs
class MoonLightSTRELAdapter: ...


class LLMSoftCheckAdapter: ...
```

---

## 4. Backwards Compatibility & Verification

1. **Zero Breaking Changes:** `tests/contract/test_api_compatibility.py` passes with zero modifications to the v1.0.0 locked surface.
2. **Zero Core Dependencies:** `tests/contract/test_core_dependencies.py` includes `"verification"` in core packages and confirms zero heavy dependencies are imported.
3. **Cross-Validation:** `tests/integration/test_rtamt_stl_verification.py` proves equivalence between the core NumPy monitor and RTAMT on overlapping temporal formulas.
