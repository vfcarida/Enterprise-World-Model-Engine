# Backtesting: Rolling-Origin Validation & Probabilistic Scoring

This guide demonstrates how to validate simulation ensembles against empirical observed timeseries using walk-forward rolling-origin cross-validation and proper probabilistic scoring functions in `ewm_engine.experimentation`.

---

## 1. Probabilistic Scoring Functions (Zero-Dependency)

The engine provides five pure-NumPy scoring functions for comparing predictions and ensembles against observations:

```python
from ewm_engine.experimentation import (
    score_rmse,
    score_moments_distance,
    score_empirical_coverage,
    score_crps,
    score_spectral_distance,
)

predictions = [10.0, 12.0, 14.0]
observed = [10.5, 11.8, 14.2]

# 1. Point-forecast error: Root Mean Squared Error (RMSE)
rmse = score_rmse(predictions, observed)
print(f"RMSE: {rmse:.4f}")

# 2. Method-of-Moments Distance (Bias, variance, z-score)
ensemble_samples = [9.8, 10.1, 10.3, 10.7, 11.2]
moments = score_moments_distance(ensemble_samples, observed_value=10.5)
print(f"Mean: {moments['mean']:.2f}, Bias: {moments['bias']:.2f}, Z-score: {moments['z_score']:.2f}")

# 3. Continuous Ranked Probability Score (CRPS)
crps = score_crps(ensemble_samples, observed_value=10.5)
print(f"CRPS: {crps:.4f}")

# 4. Empirical Coverage (SBC / TARP concept across credible intervals)
import numpy as np
ensemble_matrix = np.random.normal(loc=10.0, scale=1.0, size=(50, 3))
observed_series = [10.2, 9.8, 10.5]
coverage = score_empirical_coverage(ensemble_matrix, observed_series, credible_intervals=(0.50, 0.80, 0.95))
print(f"Empirical Coverage (95% CI): {coverage['ci_95']:.1%}")

# 5. Spectral Fourier Distance (Frequency-domain cyclic dynamics comparison)
spec_dist = score_spectral_distance(predictions, observed)
print(f"Spectral Distance: {spec_dist:.4f}")
```

---

## 2. Rolling-Origin Walk-Forward Backtesting

In walk-forward cross-validation, the simulation engine is initialized from historical state snapshots at rolling origins $t_0, t_0+s, t_0+2s, \dots$, runs $N$ Monte Carlo rollouts forward to $t+h$, and scores forecast distributions against ground-truth data:

```python
from ewm_engine.core.world import World
from ewm_engine.core.state import WorldState
from ewm_engine.core.resources import Resource
from ewm_engine.experimentation import run_walk_forward_backtest

# Build historical world state sequence and observed target signal
total_history_length = 20
world_history = [
    World(initial_state=WorldState(step=t, resources=[Resource(id="demand", current=100.0 + 2.0 * t)]))
    for t in range(total_history_length)
]
observed_series = [100.0 + 2.0 * t for t in range(total_history_length)]

# Execute walk-forward evaluation
report = run_walk_forward_backtest(
    world_history=world_history,
    observed_signal_history=observed_series,
    target_signal_name="demand",
    horizon=4,      # Forecast 4 steps ahead from each origin
    step_size=2,    # Advance origin by 2 steps per window
    samples=10,     # 10 stochastic Monte Carlo rollouts per origin
    seed=42,
)

print(f"Target Evaluated: {report.target_signal}")
print(f"Windows Evaluated: {report.window_count}")
print(f"Mean Walk-Forward RMSE: {report.mean_rmse:.2f}")
print(f"Mean CRPS: {report.mean_crps:.4f}")
print(f"Aggregate Coverage (80% CI): {report.aggregate_coverage['ci_80']:.1%}")
print(f"Aggregate Coverage (95% CI): {report.aggregate_coverage['ci_95']:.1%}")
print(f"Evaluation Provenance Fingerprint: {report.fingerprint}")
```
