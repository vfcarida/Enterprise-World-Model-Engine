# Scientific Benchmark Families (Research Instrument)

!!! info "Research Instrument — Spec §63"
    The EWM Engine benchmark suite is a **measurement instrument** designed to probe whether candidate world models can simulate structural change correctly.
    Results reflect properties of **artificial, synthetic environments** only. They do **not** imply real-world empirical superiority or automated decision readiness.

---

## 1. Motivation: Probing Structural Generalization

Standard predictive machine learning evaluates models via in-distribution accuracy metrics ($R^2$, MSE). However, organizational world models operate under **interventions, regime shifts, and hard constraints** where observational correlations collapse.

Following the empirical consensus of Physics-IQ ([arXiv:2501.09038](https://arxiv.org/abs/2501.09038)) and the Lucas Critique in economics, the EWM Engine implements **five canonical benchmark families**. Rather than ranking models to declare an automated "winner", each family probes a specific systemic vulnerability.

```
Candidate Dynamics Model (Structural, Linear, or Neural)
                           │
 ┌─────────────────────────┼─────────────────────────┬─────────────────────────┐
 ▼                         ▼                         ▼                         ▼
InterventionShift      RuleShift             ConstraintStress         LongHorizon / MultiAgent
(Policy Shift Gap)    (Governance Shock)     (Boundary Invalidation)  (Compounding Drift / Cascades)
```

---

## 2. The Five Benchmark Families

### 1. `InterventionShift`
- **Focus:** Generalization under held-out action distributions.
- **Probe:** Compares models trained on narrow observational data ($A_t \in [2, 10]$) against structural policy interventions ($A_t \in [40, 100]$) under logistical congestion.
- **Surfaces:** The **Interventional Generalization Gap** ($MAE_{\text{out}} - MAE_{\text{in}}$). Unconstrained neural and linear models hallucinate continued linear scaling, whereas structural models capture saturation.

### 2. `RuleShift`
- **Focus:** Adaptation to mid-horizon governance and regulatory changes.
- **Probe:** A statutory tax, platform fee, or quota rule activates at step $t \ge T/2$.
- **Surfaces:** **Rule Shift Vulnerability**. Empirical dynamics models trained under static policies experience instant predictive failure upon structural policy intervention (the enterprise Lucas Critique).

### 3. `ConstraintStress`
- **Focus:** Boundary safety, violation handling, and rollout invalidation under load.
- **Probe:** Escalates demand from normal operating load (50% capacity) to boundary stress (95%) and severe overload (150%).
- **Surfaces:** **Constraint Safety**. Unconstrained dynamics silently fabricate physically impossible phantom resources ($violations > 0$), while clamped and verified models correctly enforce boundaries or invalidate rollouts.

### 4. `LongHorizon`
- **Focus:** Multi-step compounding drift and autoregressive error accumulation.
- **Probe:** Simulates trajectories over extended horizons ($H = 40$ steps) comparing models with minimal single-step error ($MAE < 0.05$).
- **Surfaces:** **Compounding Autoregressive Drift**. Demonstrates that small single-step errors compound exponentially over time, disproving claims that one-step accuracy justifies multi-step decision horizons.

### 5. `MultiAgentCascade`
- **Focus:** Second-order ripple effects, propagation delays, and systemic trace depth.
- **Probe:** An upstream supplier shock halts parts flow in a three-tier supply ecosystem (`Supplier` $\to$ `Processor` $\to$ `Distributor`).
- **Surfaces:** **Multi-Agent Cascade Feedback**. Evaluates whether the dynamics model captures delayed downstream stockouts and systemic feedback loops vs myopic decoupled predictions.

---

## 3. Baselines Compared

Every benchmark compares across canonical dynamics paradigms:
1. **Deterministic Structural Reference:** Explicit physical rules and conservation laws.
2. **`LinearResidualDynamics`:** Ordinary least-squares regression baseline.
3. **`TorchNeuralResidualDynamics` (`[ml]` extra):** Multi-layer perceptron residual estimator with DreamerV3 symlog scaling.

---

## 4. Reproducibility & Provenance

Every benchmark run produces a **cryptographic provenance fingerprint**:
$$\text{Fingerprint} = \text{SHA-256}(\text{BenchmarkName} \parallel \text{Family} \parallel \text{Seed} \parallel \text{EngineVersion} \parallel \text{Parameters})$$

This guarantees that benchmark results can be audited, reproduced bit-for-bit from seed, and cited across research iterations.

---

## 5. Running the Benchmarks

### Via Command-Line Interface

```bash
# Run fast smoke-test configuration
python -m benchmarks.suite --fast

# Run full benchmark suite and export machine-readable JSON
python -m benchmarks.suite --json benchmarks_report.json

# Run specific families only
python -m benchmarks.suite --family intervention horizon
```

### Via PyTest (CI Non-Blocking Guard)

```bash
# Run the benchmark test suite under pytest
pytest -m benchmark
```

---

## 6. Example Output

```text
======================================================================
EPISTEMIC DISCLAIMER: SCIENTIFIC BENCHMARK INSTRUMENT
----------------------------------------------------------------------
These benchmarks evaluate models against artificial, synthetic worlds.
Results reflect properties of specific synthetic setups only, NOT
empirical real-world superiority. The engine forbids auto-declaring
a 'winning' model without user-specified utility and risk preferences.
======================================================================

Benchmark Suite: EWM_Scientific_Benchmark_Suite (Executed: 2026-10-05T18:10:18Z)
============================================================================================
Benchmark Family       | Model Name                   | Invariants | Key Metric / Gap      
--------------------------------------------------------------------------------------------
InterventionShift      | GroundTruthCongestedDynamics | PASS       | Gap: +0.0000 (0.0x)   
                       | LinearResidualDynamics       | PASS       | Gap: +38.2189 (38.2x)
                       | TorchNeuralResidualDynamics  | PASS       | Gap: +26.9124 (8.7x)  
--------------------------------------------------------------------------------------------
RuleShift              | StructuralRuleAwareDynamics  | PASS       | PostMAE: 0.0000 (0.0x)
                       | LinearResidualDynamics       | PASS       | PostMAE: 7.2071 (7.2x)
                       | TorchNeuralResidualDynamics  | PASS       | PostMAE: 15.5449 (15.5x)
--------------------------------------------------------------------------------------------
ConstraintStress       | ClampedCapacityDynamics      | PASS       | ViolRate: 0.0%        
                       | UnclampedPhantomDynamics     | FAIL (10)  | ViolRate: 100.0%      
--------------------------------------------------------------------------------------------
LongHorizon            | DampedInventoryDynamics      | PASS       | Drift: +0.0000/step   
                       | SlightlyBiasedEmpiricalDynamics | PASS       | Drift: +1.0690/step   
--------------------------------------------------------------------------------------------
MultiAgentCascade      | CascadeSupplyChainDynamics   | PASS       | Unserved: 75.0        
                       | DecoupledMyopicDynamics      | PASS       | Unserved: 0.0         
--------------------------------------------------------------------------------------------
```
