# Simulation Calibration: Method of Simulated Moments (MSM)

This guide demonstrates how to calibrate latent simulation parameters to match empirical observed data using `DistanceCalibrator` and the Method of Simulated Moments (MSM) in `ewm_engine.experimentation`.

---

## 1. Prerequisites & Installation

Simulation calibration requires the optional `[calibrate]` extra:

```bash
pip install 'ewm-engine[calibrate]'
```

> [!NOTE]
> **License Compliance & Intellectual Honesty:** To protect users against viral copyleft constraints, EWM Engine does **not** vendor or import `black-it` (AGPL-3.0). It provides an independent, clean, and permissive MIT/Apache quadratic distance loss.

---

## 2. Parameter Calibration Workflow

The calibration pipeline:
1. Extract summary moments (mean, variance, quantiles) from empirical historical logs using `compute_msm_moments`.
2. Propose parameter vectors $\theta \in \Theta$.
3. Execute simulation rollouts at $\theta$.
4. Compute MSM quadratic distance loss $L(\theta)$.
5. Iterate via Nelder-Mead or bounded gradient-free solvers until convergence.

```python
from ewm_engine.experimentation import (
    ParameterDef,
    ParameterSpace,
    DistanceCalibrator,
    compute_msm_moments,
)

# 1. Define parameter space of unknown model constants
space = ParameterSpace(
    name="LogisticsCalibrationSpace",
    parameters=(
        ParameterDef(name="service_rate", type="continuous", bounds=(5.0, 50.0), default=15.0),
        ParameterDef(name="congestion_factor", type="continuous", bounds=(0.5, 3.0), default=1.0),
    ),
)

# 2. Historical empirical observed timeseries (e.g. observed queue depths)
historical_data = [22.4, 25.1, 28.3, 21.0, 26.5, 29.8, 24.2]
target_moments = compute_msm_moments(historical_data)
print(f"Target Moments: {target_moments}")

# 3. Simulation runner callback (maps point dict -> simulated series)
def simulate_queue(point: dict[str, float]) -> list[float]:
    rate = point["service_rate"]
    congestion = point["congestion_factor"]
    # Simulated response
    base_queue = 500.0 / rate * congestion
    return [base_queue - 2.0, base_queue, base_queue + 2.0]

# 4. Calibrate parameters using DistanceCalibrator
calibrator = DistanceCalibrator(
    space=space,
    simulator_fn=simulate_queue,
    target_moments=target_moments,
)

result = calibrator.calibrate(max_evaluations=50, method="Nelder-Mead")

print(f"Convergence Success: {result.success}")
print(f"Final MSM Loss: {result.loss_value:.6f}")
print("Calibrated Parameters:")
for param, val in result.calibrated_parameters.items():
    print(f"  {param:<20}: {val:.3f}")
print("Moment Errors:")
for k, err in result.moment_errors.items():
    print(f"  {k:<10}: {err:.4f}")
```
