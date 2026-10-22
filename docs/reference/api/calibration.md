# Amortized Calibration & Coverage Certification API Reference

This module provides Simulation-Based Inference (SBI) posterior parameter estimation, Tests of Accuracy with Random Points (TARP) Empirical Coverage Probability (ECP) auditing (Lemos et al., ICML 2023), and misspecification-robust diagnostics.

---

## Calibration Workflow, Priors & Posterior Estimators

::: ewm_engine.calibration
    options:
      show_root_heading: true
      show_source: false
      members:
        - CalibrationWorkflow
        - AmortizedPosteriorEstimator
        - PosteriorDistribution
        - CalibrationPrior
        - PriorSet
        - ForwardSimulatorWrapper
        - CalibrationReport

---

## Certified Coverage (TARP, SBC) & Diagnostics

::: ewm_engine.calibration.certification
    options:
      show_root_heading: true
      show_source: false
      members:
        - CoverageCertificate
        - certify_coverage
        - compute_tarp_coverage
        - compute_posterior_sbc
        - compute_misspecification_diagnostic
