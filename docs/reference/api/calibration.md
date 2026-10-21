# Amortized Calibration & Coverage Certification API Reference

This module provides Simulation-Based Inference (SBI) posterior parameter estimation, Test of Accuracy with Random Points (TARP) Empirical Coverage Probability (ECP) auditing, and split-conformal post-hoc recalibration.

---

## Calibration Engine & Posterior Estimators

::: ewm_engine.calibration.amortized
    options:
      show_root_heading: true
      show_source: false
      members:
        - CalibrationEngine
        - CalibrationReport
        - PosteriorDistribution
        - ParameterBounds
        - TrajectorySummaryExtractor

---

## Certified Coverage (TARP) & Conformal Recalibration

::: ewm_engine.calibration.coverage
    options:
      show_root_heading: true
      show_source: false
      members:
        - TarpCoverageResult
        - ConformalCalibrator
        - compute_tarp_coverage
        - recalibrate_posterior_conformal
