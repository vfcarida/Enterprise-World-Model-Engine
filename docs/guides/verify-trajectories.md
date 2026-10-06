# Verifying Trajectories: Oracle Graphs and Temporal Logic

This guide demonstrates how to express, monitor, and formally verify temporal, causal, and quantitative requirements over whole simulation trajectories using `ewm_engine.verification`.

---

## 1. Oracle-Graph Verification over Systemic Traces

The Oracle-Graph verifier (conforming to the Meta ARE 3-axis methodology, [arXiv:2509.17158](https://arxiv.org/abs/2509.17158)) checks whether an expected sequence of events occurred with exact parameters, proper causality, and within valid timing windows.

### Example: Verifying Flood Response Pipeline

Suppose our disaster response scenario requires:
1. An exogenous surge event must be detected.
2. An evacuation alert action must be dispatched within 1 to 2 steps after the surge.
3. Supply delivery must complete within 2 to 4 steps after the alert.

```python
from ewm_engine.verification import (
    OracleEdge,
    OracleGraph,
    OracleNode,
    evaluate_oracle_graph,
)

# 1. Define the Oracle Graph DAG
oracle = OracleGraph.create(
    nodes=[
        OracleNode(
            node_id="expected_surge",
            category="event",
            label_pattern="surge",
            expected_details={"severity": 2.5},
        ),
        OracleNode(
            node_id="expected_alert",
            category="action",
            label_pattern="alert",
            expected_details={"region": "sector_7"},
        ),
        OracleNode(
            node_id="expected_delivery",
            category="state_change",
            label_pattern="delivered",
            expected_details={"shelter_id": "S1"},
        ),
    ],
    edges=[
        OracleEdge(
            source="expected_surge",
            target="expected_alert",
            relation="triggers",
            min_step_delay=1,
            max_step_delay=2,
        ),
        OracleEdge(
            source="expected_alert",
            target="expected_delivery",
            relation="enables",
            min_step_delay=2,
            max_step_delay=4,
        ),
    ],
)

# 2. Evaluate against the trajectory's systemic trace
trace = trajectory.systemic_trace
result = evaluate_oracle_graph(oracle, trace)

# 3. Inspect multi-axis outcome
print(f"Satisfied: {result.satisfied}")
print(f"Consistency Score: {result.consistency_score:.2f}")
print(f"Causality Score: {result.causality_score:.2f}")
print(f"Timing Score: {result.timing_score:.2f}")

if not result.satisfied:
    for violation in result.violations:
        print(f"[{violation.axis.upper()}] {violation.message}")
        print(f"  Details: {violation.details}")
```

---

## 2. Discrete Bounded STL Monitoring (Core, Pure-NumPy)

For quantitative constraints over timeseries signals (resources, state variables, step metrics), the core engine provides a bounded-future discrete STL monitor with zero heavy dependencies.

### Example: Bounded SLA Recovery and Buffer Bounds

```python
from ewm_engine.verification import (
    always,
    eventually,
    predicate,
    until,
    evaluate_stl_bounded,
)

# Requirement 1: Warehouse buffer never drops below 10.0 for steps [0, 10]
phi_safety = always(predicate("buffer_stock", ">=", 10.0), k1=0, k2=10)

# Requirement 2: If stock drops, recovery occurs within 3 steps
phi_recovery = eventually(predicate("buffer_stock", ">=", 50.0), k1=0, k2=3)

# Evaluate over simulation trajectory
verdict = evaluate_stl_bounded(phi_safety, trajectory)

print(f"Safety Held: {verdict.satisfied}")
print(f"Robustness Margin: {verdict.robustness:.2f}")  # min (buffer_stock - 10.0)
print(f"Violating Steps: {verdict.violation_steps}")
```

---

## 3. Full STL & Robustness Distributions via RTAMT (`[stl]` Extra)

When evaluating complex continuous Signal Temporal Logic formulas or measuring policy safety margins across Monte Carlo rollouts, install the optional extra:

```bash
pip install 'ewm-engine[stl]'
```

### Evaluating Monte Carlo Robustness Distributions

```python
from ewm_engine.verification import RTAMTEvaluationBackend

# Initialize RTAMT backend with STL formula and typed variables
backend = RTAMTEvaluationBackend(
    formula="always(buffer_stock >= 15.0)",
    variables={"buffer_stock": "float"},
    spec_name="BufferSafetySpec",
)

# Evaluate across 50 Monte Carlo rollouts
rollouts = simulation_result.trajectories
dist = backend.evaluate_monte_carlo(rollouts)

print(f"Formula: {dist.formula}")
print(f"Rollout Count: {dist.rollout_count}")
print(f"Satisfaction Probability: {dist.satisfaction_probability:.2%}")
print(f"Mean Robustness: {dist.mean_robustness:.2f}")
print(f"5th Percentile Robustness (p05): {dist.quantiles['p05']:.2f}")
print(f"Conditional Value-at-Risk (CVaR 5%): {dist.cvar_05:.2f}")
```

> [!TIP]
> **Why Robustness Matters:** A policy with 100% satisfaction probability but a CVaR 5% of $+0.1$ is operating on a knife-edge. A policy with CVaR 5% of $+12.4$ has substantial buffer capacity against unmodeled real-world disruptions.

---

## 4. Declarative Property Specifications & Provenance

To ensure auditability, define requirements using `PropertySpec` and fold their canonical hashes into simulation fingerprints:

```python
from ewm_engine.verification import PropertySpec, fold_properties_into_fingerprint

spec_1 = PropertySpec(
    property_id="req_buffer_positive",
    name="Buffer Non-Negative Invariant",
    property_type="stl_bounded",
    definition={"formula": "always(buffer_stock >= 0.0)"},
    parameters={"k1": 0, "k2": 10},
)

spec_2 = PropertySpec(
    property_id="req_alert_response",
    name="Alert Response Within 2 Steps",
    property_type="oracle_graph",
    definition={"nodes": ["surge", "alert"], "edges": [("surge", "alert")]},
    parameters={"min_step_delay": 1, "max_step_delay": 2},
)

# Every spec has a deterministic canonical SHA-256 hash
print(f"Spec 1 Hash: {spec_1.property_hash}")

# Fold into provenance fingerprint (permutation invariant)
base_fp = simulation_result.provenance.fingerprint
auditable_fp = fold_properties_into_fingerprint(base_fp, [spec_1, spec_2])
print(f"Auditable Provenance Fingerprint: {auditable_fp}")
```

---

## 5. External Verification Adapters

The engine includes documented stubs for external verification frameworks:

### MoonLight STREL (Spatio-Temporal Logic over Topological Worlds)
`MoonLightSTRELAdapter` provides a bridge to the MoonLight JVM engine for graph-world verification:
```python
from ewm_engine.verification import MoonLightSTRELAdapter

# Requires JVM and [strel] extra
try:
    strel = MoonLightSTRELAdapter(script_path="models/network_topology.strel")
except Exception as e:
    print(f"Setup needed: {e}")
```

### LLM Soft-Check Adapter (Zero SDK Dependency)
For inspecting natural-language trace attributes or unstructured qualitative conditions, `LLMSoftCheckAdapter` accepts an external callable without coupling the engine to any vendor SDK:

```python
from ewm_engine.verification import LLMSoftCheckAdapter

# User-provided async or sync callable
def my_llm_checker(prompt: str) -> str:
    # Call OpenAI, Anthropic, or local model here
    return "The evacuation log meets the municipal safety protocol. YES."

adapter = LLMSoftCheckAdapter(llm_callable=my_llm_checker, rubric="Municipal Flood Protocol")
is_valid = adapter.verify_node(node_label="evacuation_notice", details={"tone": "urgent"})
print(f"Soft-check Passed: {is_valid}")
```
