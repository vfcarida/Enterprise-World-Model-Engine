# FEAT-006: Experimentation Suite — DoE, Backtesting, Sensitivity, Calibration, and Optimization

- **Status:** Approved / Implemented
- **Horizon:** v1.4.0 "Experimentation Suite"
- **Authors:** Research Methods & Decision Optimization Team
- **Governance:** `01_EXPANDED_ROADMAP.md` (Track T4), ADR-028, ACP-007
- **Dependencies:** Core engine (Zero new dependencies: stdlib, NumPy, Pydantic); Optional extras: `[sensitivity]` (SALib), `[calibrate]` (SciPy, scikit-learn), `[opt-evolutionary]` (pycma, nevergrad), `[opt-pareto]` (pymoo), `[surrogate]` (scikit-learn)

---

## 1. Motivation & Context

In real-world enterprise operations and scientific inquiry, running isolated simulation rollouts is insufficient. Users must move from *"simulate once"* to **"systematically explore, validate, calibrate, and optimize"** across vast decision and scenario spaces:
1. **Design of Experiments (DoE):** Systematic parameter exploration requires structured sampling (full factorial, One-At-a-Time, Latin Hypercube, Halton low-discrepancy sequences) rather than ad-hoc loops, paired with sensitivity diagnostics (tornado diagrams) to identify high-impact levers.
2. **Backtesting & Validation:** Before taking actions based on simulation models, organizations must validate simulation ensembles against historical real-world observations using proper probabilistic scoring (RMSE, method-of-moments distance, Continuous Ranked Probability Score [CRPS], frequency-domain spectral Fourier distance, and empirical coverage/SBC concept) across walk-forward rolling origins.
3. **Global Sensitivity Analysis:** Identifying which parameters drive variance requires global sensitivity analysis (Sobol variance decomposition, Morris screening) via SALib.
4. **Parameter Calibration:** Historical observational data must calibrate latent parameters without introducing restrictive AGPL-3.0 dependencies (re-implementing clean Method of Simulated Moments [MSM] loss).
5. **Multi-Objective Policy Optimization:** Decision makers rarely face a single objective; they require Pareto frontiers of non-dominated policies with explicit trade-offs and cryptographic policy fingerprints.

---

## 2. Requirements & Acceptance Criteria

### Part A: Core Substrate (Track T4, Core, Zero-Dep)
- **AC-046 (Parameter Space & DoE Harness):**
  - Define `ParameterDef` (continuous, integer, categorical) and `ParameterSpace` with canonical SHA-256 `space_hash`.
  - Provide pure-NumPy design generators: `generate_full_factorial`, `generate_oat`, `generate_lhs`, and `generate_halton`.
  - Provide `run_doe_sweep` mapping each design point to a reproducible rollout via `np.random.SeedSequence(seed).spawn()`.
  - Produce tidy tabular results and `TornadoData` ranking parameters by sensitivity swing $|y_{\text{high}} - y_{\text{low}}|$.
  - Emit machine-readable JSON Schema (`schemas/parameter-space.schema.json`).

- **AC-047 (Validation & Backtest Harness):**
  - Implement pure-NumPy scorers: `score_rmse`, `score_moments_distance`, `score_empirical_coverage` (SBC/TARP concept), `score_crps`, and `score_spectral_distance` (1D FFT magnitude comparison).
  - Implement `run_walk_forward_backtest` evaluating rolling-origin ensembles against historical series, aggregating coverage and forecast metrics into `BacktestReport`.

- **AC-048 (Protocols & Built-ins):**
  - `Sampler` protocol with built-ins: `RandomSampler`, `LatinHypercubeSampler`, `HaltonSampler`.
  - `Surrogate` protocol for simulation emulators (`fit`, `predict`).
  - `ObjectiveVector` supporting scalarization and Pareto dominance (`dominates`).
  - `Optimizer` protocol with zero-dep built-ins: `RandomSearchOptimizer` and `HillClimbingOptimizer`.
  - `ParetoFront` filtering candidates down to non-dominated frontier with policy fingerprints.

### Part B: Statistical & Optimization Extras (Track T4, Optional Backends)
- **AC-049 (Global Sensitivity via SALib):**
  - Implement `GlobalSensitivityAnalyzer` behind optional `[sensitivity]` extra (`SALib`).
  - Support Sobol variance decomposition ($S_1, S_T, S_2$) and Morris screening ($\mu, \mu^*, \sigma$).
  - Lazy import: core remains completely dependency-free when SALib is absent.

- **AC-050 (Simulation Calibration via MSM):**
  - Implement `DistanceCalibrator` behind optional `[calibrate]` extra (`scipy`).
  - Implement permissive Method of Simulated Moments (`msm_distance_loss`, `compute_msm_moments`) without vendoring AGPL-3.0 code (no `black-it`).
  - Support recovery of known parameters from synthetic targets.
  - Document `pyabc` ABC-SMC Bayesian stub adapter.

- **AC-051 (Evolutionary & Pareto Optimization):**
  - Implement `PyCMAOptimizer` (CMA-ES via `cma`) and `NevergradOptimizer` behind `[opt-evolutionary]`.
  - Implement `PymooParetoOptimizer` (NSGA-II via `pymoo`) behind `[opt-pareto]`, returning `ParetoFront` of non-dominated policies with constraint mapping.

- **AC-052 (Gaussian Process Emulator):**
  - Implement `GaussianProcessSurrogate` behind `[surrogate]` extra (`scikit-learn`), enabling fast surrogate evaluations for parameter sweeps.

### Part C: Heavy Adapters (Quarantined)
- **AC-053 (Quarantined Neural & Bayesian Adapters):**
  - Provide `SBINeuralInferenceAdapter` (torch/sbi) and `AxBayesianOptimizerAdapter` (ax/botorch).
  - Explicitly document the "SBI trust crisis" (neural posterior overconfidence under model mismatch).
  - Ensure zero top-level imports of torch/sbi/ax in core; raise informative configuration guidance when missing.

---

## 3. Epistemic Principles & Guardrails

1. **Simulated Properties, Not Unconditional Forecasts:** All backtest scores, coverage percentages, and sensitivity indices are formal properties of the *computational simulation model* evaluated against provided empirical data. They are never reported as unconditional real-world predictions.
2. **No Automated "Best Policy" Without Stated Objectives:** Multi-objective searches return the full non-dominated `ParetoFront` rather than arbitrarily picking a single winner.
3. **Zero Core Dependencies:** Core parameter spaces, design generators, backtesting scorers, samplers, and optimizers use strictly standard library, NumPy, and Pydantic. All external solvers, Bayesian inference engines, and ML libraries are quarantined behind optional extras.
