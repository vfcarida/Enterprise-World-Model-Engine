# Experimentation Suite API Reference

This module implements the scientific experimentation layer (Track T4): parameter space declaration, DoE sweeps, walk-forward backtesting, global sensitivity, and intervention optimization.

---

## Parameter Spaces & Declarations

::: ewm_engine.experimentation.params
    options:
      show_root_heading: true
      show_source: false
      members:
        - ParameterDef
        - ParameterSpace
        - ParameterType

---

## Design of Experiments (DoE) & Sweeps

::: ewm_engine.experimentation.doe
    options:
      show_root_heading: true
      show_source: false
      members:
        - DesignType
        - SweepResult
        - TornadoData
        - generate_lhs
        - generate_full_factorial
        - generate_oat
        - run_doe_sweep

---

## Backtesting & Validation

::: ewm_engine.experimentation.backtest
    options:
      show_root_heading: true
      show_source: false
      members:
        - BacktestReport
        - run_walk_forward_backtest
        - score_rmse
        - score_crps
        - score_moments_distance

---

## Sensitivity & Calibration

::: ewm_engine.experimentation.sensitivity
    options:
      show_root_heading: true
      show_source: false
      members:
        - SensitivityReport
        - compute_sobol_indices
        - compute_morris_indices

::: ewm_engine.experimentation.calibration
    options:
      show_root_heading: true
      show_source: false
      members:
        - DistanceCalibrator
        - CalibrationResult
