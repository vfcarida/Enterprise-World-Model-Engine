# ADR-028: Experimentation Suite — DoE, Backtesting, Calibration, and Intervention Optimization

## Status
Accepted

## Date
2026-10-05

## Context
Up to version 1.3, the Enterprise World Model Engine focused on executing single trajectories or Monte Carlo rollouts under fixed scenario specifications. However, decision evaluation and scientific modeling require systematic exploration across multi-dimensional parameter spaces:
1. **Design of Experiments (DoE):** Users need structured sampling (full factorial, OAT, LHS, Halton) and sensitivity ranking (tornado diagrams).
2. **Backtesting & Verification:** Models must be validated against historical empirical observations using proper scoring rules (RMSE, method-of-moments distance, empirical coverage, CRPS, spectral Fourier distance) across rolling-origin walk-forward cross-validation.
3. **Global Sensitivity Analysis:** Identifying dominant drivers of variance requires Sobol variance decomposition and Morris elementary effects.
4. **Calibration:** Parameter fitting to empirical data must avoid restrictive copyleft licenses like AGPL-3.0 (e.g. `black-it`).
5. **Optimization:** Multi-objective policy optimization must return non-dominated Pareto frontiers with auditable policy fingerprints.

## Decision

### 1. Dedicated Namespace (`ewm_engine.experimentation`)
We introduce a dedicated subpackage `ewm_engine.experimentation`. Root `ewm_engine.__all__` remains strictly locked to the v1.0.0 Stable surface (`LOCKED_V1_STABLE_SURFACE`), ensuring zero breaking changes.

### 2. Zero-Dependency Core Substrate (Pure NumPy / Pydantic)
- `ParameterSpace` & `ParameterDef`: Declarative parameter space supporting continuous, integer, and categorical dimensions with canonical SHA-256 `space_hash`.
- Design Generators: `generate_full_factorial`, `generate_oat`, `generate_lhs`, and `generate_halton` (van der Corput radical inverse in pure NumPy).
- DoE Sweep Harness: `run_doe_sweep` using `SeedSequence.spawn()` to derive child seeds for bitwise reproducibility; generates tidy results tables and `TornadoData` ranking parameters by sensitivity swing.
- Backtest Harness: `run_walk_forward_backtest` supporting pure-NumPy scorers:
  - `score_rmse`
  - `score_moments_distance` (mean, variance, bias, z-score)
  - `score_empirical_coverage` (SBC/TARP concept)
  - `score_crps` (Continuous Ranked Probability Score)
  - `score_spectral_distance` (1D FFT magnitude comparison)
- Protocols & Built-ins:
  - `Sampler` protocol: `RandomSampler`, `LatinHypercubeSampler`, `HaltonSampler`.
  - `Surrogate` protocol: Fast emulation interface (`fit`, `predict`).
  - `ObjectiveVector`: Multi-objective evaluation with scalarization and Pareto dominance (`dominates`).
  - `Optimizer` protocol: `RandomSearchOptimizer` and `HillClimbingOptimizer`.
  - `ParetoFront`: Non-dominated policy filtering with cryptographic policy fingerprints.

### 3. Statistical & Optimization Extras (No Torch in Core)
- `[sensitivity]` via SALib: `GlobalSensitivityAnalyzer` executing Sobol ($S_1, S_T, S_2$) and Morris ($\mu, \mu^*, \sigma$) analysis.
- `[calibrate]` via SciPy: `DistanceCalibrator` with permissive quadratic Method of Simulated Moments (`msm_distance_loss`). License stance: does not vendor `black-it` (AGPL-3.0). Documented `pyabc` ABC-SMC adapter stub.
- `[opt-evolutionary]` via pycma & Nevergrad: `PyCMAOptimizer` and `NevergradOptimizer`.
- `[opt-pareto]` via pymoo: `PymooParetoOptimizer` running NSGA-II with hard constraint mapping to solver constraints and soft constraints to penalized objectives.
- `[surrogate]` via scikit-learn: `GaussianProcessSurrogate` emulator.

### 4. Quarantined Heavy Adapters
- `SBINeuralInferenceAdapter`: Amortized neural posterior estimation via `sbi` (torch). Explicitly addresses the "SBI trust crisis" in docs (neural posteriors can exhibit false overconfidence under model misspecification).
- `AxBayesianOptimizerAdapter`: Gaussian Process Bayesian optimization via `ax-platform` and `botorch`.
- Quarantined: Zero top-level imports in core; imports are strictly lazy.

## Consequences
- **Positive:** Moves the engine from "simulate once" to systematic exploration, validation, and optimization.
- **Positive:** Core substrate is 100% zero-dependency (NumPy, Pydantic, stdlib only).
- **Positive:** Every sweep and optimization run produces deterministic, fingerprinted provenance trails.
- **Positive:** Avoids AGPL-3.0 contamination through a clean, permissive MSM loss implementation.
- **Negative:** Advanced sensitivity (Sobol), evolutionary algorithms (CMA-ES), and multi-objective Pareto search (NSGA-II) require optional extras.
