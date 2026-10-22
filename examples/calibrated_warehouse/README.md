# Calibrated Warehouse Example (`ewm calibrate`)

Demonstrates amortized Simulation-Based Inference (SBI) and mandatory coverage certification in EWM Engine.

## Background & Scientific Anchors

Hand-tuning simulation parameters is unscientific, error-prone, and hides model uncertainty. Simulation-Based Inference (SBI) solves this by treating the simulator as a stochastic or deterministic forward model $\theta \to x$ and learning an amortized posterior $p(\theta \mid x)$ over parameters.

Crucially, **EWM Engine refuses to report uncertified calibrations for downstream decision use**.

Anchors:
- **Simulation Intelligence Motif 3**: Lavin et al., arXiv:2112.03235.
- **Amortized SBI**: Boelts et al., "sbi reloaded", arXiv:2411.17337; Gloeckler et al., Simformer, ICML 2024 (arXiv:2404.09636).
- **Mandatory Coverage Certification**:
  - **TARP** (Tests of Accuracy with Random Points; Lemos et al., arXiv:2302.03026, ICML 2023).
  - **Posterior-SBC** (Simulation-Based Calibration conditional on observed data; arXiv:2502.03279).
- **Misspecification Robustness**: Huang et al., NeurIPS 2023 (arXiv:2305.15871).

## Running the Example

Run directly with Python:
```bash
python examples/calibrated_warehouse/run.py
```

Or execute via the `ewm` CLI:
```bash
ewm calibrate examples/calibrated_warehouse/spec.yaml \
  --data examples/calibrated_warehouse/empirical_data.yaml \
  --priors examples/calibrated_warehouse/priors.yaml \
  --out calib_report.json \
  --markdown-out calib_report.md
```

## Mandatory Certification Gate

1. **Certified Calibration**: When the simulator can express the observed series and empirical TARP coverage matches nominal confidence levels, the calibration is marked `is_certified=True` and `decision_usable=True`.
2. **Refusal Gate**: If the observed data falls outside the simulator's generative manifold (misspecification) or empirical coverage is statistically defective (overconfident or undercovered), `ewm calibrate` refuses the calibration (`decision_usable=False`). Downstream decision tools (`ewm decide`) reject uncertified models.
