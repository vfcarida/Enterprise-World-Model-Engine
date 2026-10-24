# Tutorial: Build and Run the Fraud → Call-Center Reference Scenario

!!! important "Epistemic Scope: Controlled Synthetic Environment (`CONTROLLED_SYNTHETIC_KNOWN_MECHANISMS_NOT_REAL_WORLD_EFFICACY`)"
    **Capability demonstrated in a controlled, synthetic environment with explicitly KNOWN mechanisms and ground-truth parameters — NOT proven efficacy in a real operation.**
    All numbers on this page are deterministically generated from [`docs/results/artifacts/fraud_cascade_flagship.json`](../results/artifacts/fraud_cascade_flagship.json) (scenario fingerprint `abd41502e780d6479f759d408450c95daae0ee999a9b351c9d96cdb7b9c6bf7e`, output fingerprint `946c8c864449ebbfc2ebfe6a9c8c9f656414aa607d81417ee0e46e47703fb60f`).

---

## 1. Why This Scenario Exists

This tutorial walks through `ewm-engine`'s flagship reference domain: **Fraud Authorization Rule Change → Transaction Declines → Customer Reactions → Call-Center Demand & Wait-Time Cascade**.

It makes the project's central research question concrete and falsifiable:

> *When a fraud team tightens an authorization threshold $\tau$ during a card-testing attack, validating only the immediate handler (Layer 1) makes the change look optimal—fraud chargebacks drop. Yet in a coupled enterprise system, false-positive declines trigger automated account locks (Layer 2), customer retries and contact-center call surges (Layer 3), and queue congestion, SLA breach penalties, and customer churn (Layer 4).*

For the complete state, parameter, and equation specification, see the companion [Fraud → Call-Center Cascade Reference](../reference/fraud-cascade-scenario.md) and [What Is an Enterprise World Model?](../explanation/what-is-an-ewm.md).

---

## 2. Step 1 — Run the Reproducible Flagship Script

You can execute the full seeded scenario and regenerate the canonical JSON artifact with a single command:

```bash
python scripts/run_fraud_cascade_flagship.py
```

Or run the fast CI smoke check (which verifies mechanism recovery, bitemporal leakage guards, `DataLeakageCertificate`, and fingerprint identity):

```bash
python scripts/run_fraud_cascade_flagship.py --smoke
```

Expected output:

```text
[OK] Flagship scenario 'fraud_declines_call_center_cascade_v1' verified.
     Epistemic Label      : CONTROLLED_SYNTHETIC_KNOWN_MECHANISMS_NOT_REAL_WORLD_EFFICACY
     Runtime Env          : Python 3.12.10 | NumPy 2.5.3 | Commit 0a40d050e4bb
     Scenario Fingerprint : abd41502e780d6479f759d408450c95daae0ee999a9b351c9d96cdb7b9c6bf7e
     Output Fingerprint   : 946c8c864449ebbfc2ebfe6a9c8c9f656414aa607d81417ee0e46e47703fb60f
     Mechanism Recovery   : PASS (6/6 checks within tolerance)
     Bitemporal Guard     : PASS (withheld=11 future-knowledge records at t=5.0)
     Leakage Certificate  : PASS (overlap=0)
     Planning Regret      : Myopic=124774.05 (myopic_aggressive_tau_0.40_agents_32) vs. Cascade-EWM=0.0 (coordinated_ewm_tau_0.50_agents_42_sms)
```

---

## 3. Step 2 — Instantiate the Environment, Context $K_t$, and Partial Observability $O_t$

```python
from ewm_engine.flagship import (
    FraudCascadeEnvironment,
    FraudCascadeParameters,
    build_default_fraud_context,
)

# 1. Ground-truth parameters (conserved pool N_0 = 10,000 customers)
params = FraudCascadeParameters()
ctx_v1 = build_default_fraud_context(params, auth_threshold=0.65, policy_version="v1_baseline")
env = FraudCascadeEnvironment(params, initial_context=ctx_v1)

# 2. Seeded rollout (20 steps)
records, ledger, ontic_states, observed_states = env.rollout(horizon=20, seed=42)

# 3. Verify partial observability O_t = h(S_t, K_t, xi_t):
#    The decision-maker sees queue wait times and declines, NOT latent churn propensity
assert "avg_wait_seconds" in observed_states[-1].resources
assert "latent_churn_propensity" not in observed_states[-1].resources
```

---

## 4. Step 3 — Inspect Bitemporal Data Discipline (Event-Time vs. Knowledge-Time)

Every event recorded in `BitemporalLedger` carries both `event_time` ($t_{\text{event}}$) and `knowledge_time` ($t_{\text{know}} = t_{\text{event}} + \Delta t_{\text{lag}}$):

- `auth_telemetry` arrives immediately ($\Delta t_{\text{lag}} = 0$ steps), with a metric definition change at step $t = 12$ from `v1_gross_declines` to `v2_deduplicated_declines`.
- `contact_center_telemetry` arrives with $\Delta t_{\text{lag}} = 1$ step lag.
- `fraud_settlement` emits a preliminary estimate at $t + 1$ and a superseding chargeback settlement revision at $t + 4$.
- `churn_audit` reconciles confirmed customer churn at $t + 6$.

```python
from ewm_engine.core import BitemporalLeakageError

# Point-in-time view as of decision_time = 5.0
pit_t5 = ledger.as_of(5.0)
print("Visible records at t=5.0:", len(pit_t5.visible_records))  # 14
print(
    "Withheld future-knowledge records (event_time <= 5 < knowledge_time):",
    pit_t5.withheld_future_knowledge_count,
)  # 11

# Attempting to read step-5 settled chargebacks (knowledge_time = 9.0) at t = 5.0 fails closed:
try:
    ledger.read_record_at("fraud_final_step_5", decision_time=5.0)
except BitemporalLeakageError as err:
    print("Blocked leakage:", err)
```

---

## 5. Step 4 — Execute the Four Capability Queries

Calling `run_four_capability_queries(params, train_seed=101, eval_seed=202, horizon=20)` evaluates all four capability tiers on the same controlled environment:

```python
from ewm_engine.flagship import run_four_capability_queries

results = run_four_capability_queries(params, train_seed=101, eval_seed=202, horizon=20)
```

### Summary of Artifact-Generated Results (`eval_seed=202`, Fraud-Wave $4.5\times$ on Steps $6\dots 15$)

| Policy / Query | Auth Threshold $\tau$ | Staffed Agents | Step-Up SMS | Myopic Layer-1 Score ($\$$/step) | Total Fraud Loss ($\$$) | Peak Wait Time (s) | SLA Breach Steps | Churned Customers | Realized Decision Utility ($\$$) | Decision Regret ($\$$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Status Quo (No Intervention)** | `0.65` | `32` | `False` | `18,776.89` | `118,360.80` | `90.05` | `0` | `648.99` | `296,721.22` | `43,560.27` |
| **Myopic Layer-1 Tightening (`do(tau=0.40)`)** | `0.40` | `32` | `False` | **`22,692.60`** | `58,054.84` | **`204.05`** | **`10`** | **`1,517.58`** | `215,507.44` | **`124,774.05`** |
| **Moderate Rule-Only (`do(tau=0.52)`)** | `0.52` | `32` | `False` | `21,389.40` | — | — | `0` | `891.08` | `295,918.70` | `44,362.79` |
| **Coordinated Cascade EWM (`do(tau=0.50, agents=42, sms=True)`)** | **`0.50`** | **`42`** | **`True`** | `21,685.34` | `78,731.17` | **`90.05`** | **`0`** | **`515.25`** | **`340,281.49`** | **`0.00`** |
| **Overstaffed Conservative (`do(tau=0.60, agents=70, sms=True)`)** | `0.60` | `70` | `True` | `19,917.58` | — | — | `0` | `460.89` | `277,600.14` | `62,681.35` |

### Key Takeaway

1. **Myopic Layer-1 Optimization (`myopic_aggressive_tau_0.40_agents_32`)** achieves the highest Layer-1 score (`$22,692.60`/step) and cuts direct fraud loss from `$118,360.80` to `$58,054.84`, **but** triggers a secondary call-center cascade (peak wait `204.05s`, `10` SLA breach steps, `1,517.58` churned customers), yielding `$124,774.05` in decision regret.
2. **Coordinated Cascade-Aware Planning (`coordinated_ewm_tau_0.50_agents_42_sms`)** balances threshold tightening ($\tau = 0.50$) with Layer-2 SMS step-up rescue and Layer-3/4 contact-center staffing ($42$ agents), keeping peak wait time at `90.05s` (`0` SLA breaches) and reducing decision regret to `$0.00`.
