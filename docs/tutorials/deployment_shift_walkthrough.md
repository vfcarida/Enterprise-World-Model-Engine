# ADR-053 Deployment-Shift Walkthrough (`K_t`, `O_t`, `B_t`, and Drift Recovery)

**Architecture Reference**: [ADR-053](../adr/ADR-053-formal-enterprise-world-model-semantics-and-context.md), [Benchmark Family D](../research/enterprise-benchmark-suite.md)  
**Executable Script**: [`examples/deployment_shift_walkthrough.py`](../../examples/deployment_shift_walkthrough.py)  

---

## 1. Learning Objective

This tutorial walks through a complete, executable **ADR-053** simulation demonstrating why separating:
- **Ontic State ($S_t$)**: What is physically/operationally true in the world,
- **Partial Observation ($O_t$)**: What an actor can observe via `ObservationFilter`,
- **Belief State ($B_t$)**: Estimated hidden state and growing epistemic uncertainty via `BeliefState`,
- **Epistemic State ($P_t$)**: Provenance, `EvidenceLevel`, and `DriftSignal` annotations via `EpistemicState`, and
- **Runtime Context ($K_t$)**: Versioned configuration and feature flags discovered via `AuditMetadataContextDiscovery`

enables `ewm-engine` to recover from silent deployment configuration shifts within $\Delta t_{\text{audit}} = 2$ steps, whereas static baselines suffer permanent post-shift error (`MAE = 35.0`).

---

## 2. Running the Walkthrough

Execute the standalone walkthrough script from the repository root:

```bash
python examples/deployment_shift_walkthrough.py
```

Expected output:

```text
Pre-shift MAE:              0.0000
Peak post-shift MAE:        35.0000
Recovered post-shift MAE:   0.0000
Static unrecovered MAE:     35.0000
Steps until recovery:       2
Hidden resource masked:     True
Ontic fingerprint invariant:True
```

---

## 3. Step-by-Step Code Breakdown

### 3.1 Discovering Runtime Context ($K_t$) with Fail-Closed Evidence Semantics

`AuditMetadataContextDiscovery` parses structured deployment audit logs into immutable, SHA-256 fingerprinted `ContextSnapshot` instances and caps runtime-discovered evidence at `EvidenceLevel.PREDICTIVE` unless an explicit `identification_certificate_id` is provided:

```python
from ewm_engine.core import AuditMetadataContextDiscovery

discovery = AuditMetadataContextDiscovery()
k_shifted = discovery.discover_context(
    {
        "context_id": "ctx_routing_v2",
        "version": "2.0.0",
        "parent_version": "1.0.0",
        "environment_regime": "post_deployment_shift",
        "parameters": {"routing_multiplier": 1.7, "base_inflow": 50.0},
        "feature_flags": {"aggressive_retry": True},
    }
)
```

### 3.2 Enforcing Partial Observability ($O_t$) and Updating Belief ($B_t$)

`ObservationFilter` projects $S_t \mapsto O_t$ so that `ops_controller` can observe `visible_queue` while `hidden_retry_buffer` remains strictly masked:

```python
observed_o_t = obs_filter.observe(state, actor_id="ops_controller")
assert "hidden_retry_buffer" not in observed_o_t.resources
belief = belief_updater.update_belief(belief, observed_o_t, actor_id="ops_controller")
```

### 3.3 Ontic vs. Epistemic Fingerprint Separation ($S_t$ vs. $P_t$)

`separate_ontic_epistemic()` splits the state into a pure ontic `WorldState` $S_t$ and an `EpistemicState` $P_t$ carrying `DriftSignal(drift_kind=DriftKind.DEPLOYMENT_CONFIGURATION_DRIFT, drift_origin=DriftOrigin.POLICY_INDUCED_SHIFT)`. Updating epistemic audit trails or drift signals never alters `ontic_s_t.fingerprint`.
