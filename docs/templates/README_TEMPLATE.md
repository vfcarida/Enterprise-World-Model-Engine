# Enterprise World Model Engine: Financial & AIOps Verticals

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-success)]()

**Enterprise World Model Engine** is an open-source state-transition simulation framework for
mission-critical enterprise systems. Unlike autoregressive next-token predictors, this engine
learns the **underlying dynamics** (causal mechanisms, conservation laws, topological dependencies)
to simulate `S_t + A_t → S_{t+1}`.

## Verticals

### Financial World Model (FWM)
Simulate systemic market contagion, risk rebalancing, and portfolio impact from exogenous shocks
(news, macro policy, supply-chain disruptions) using **real public data** (Market-1T 1Hz limit
order book + SEC EDGAR fundamentals).

**Key Features:**
- JEPA-based latent state representation (no pixel reconstruction overhead)
- Continuous-Time Temporal Graph Neural Networks (EvolveGCN) for dynamic financial topology
- VICReg regularization to prevent dimensional collapse
- Backtested on 2020 crash, 2022 Fed tightening (Rank IC, MAE, coverage calibration)

### AIOps World Model
Predict cascading microservice failures from isolated anomalies (memory leaks, config changes,
traffic spikes) **30-60 minutes before user-facing downtime**. Perform Root Cause Analysis via
causal inference, generate Kubernetes remediation manifests, and validate in physical
chaos simulations (Chaos Mesh).

**Key Features:**
- Spatial-Temporal Graph Convolutional Network (STGCN) for microservice topology + temporal evolution
- Causal RCA using PC algorithm + Pearl's do-calculus
- OPA Gatekeeper policy enforcement (PodDisruptionBudget, blast-radius constraints)
- Chaos Mesh validation harness (shadow namespace fault injection → measure TTR/blast-radius)

## Installation

```bash
# Clone repository
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine

# Install dependencies (Python 3.11+)
pip install -e .

# Install optional extras
pip install -e ".[dev,all]"
```

## Quickstart: Financial World Model

```python
import torch
from ewm_engine.financial import FinancialWorldModel
from ewm_engine.routing import CausalInterventionVector

# Initialize Financial World Model for 8 equities and 14 features (8 Market-1T + 6 EDGAR)
fwm = FinancialWorldModel(
    num_tickers=8,
    num_features=14,
    latent_dim=64,
    node_dim=16,
    tickers=["NVDA", "AMD", "TSM", "AAPL", "MSFT", "JPM", "XOM", "WMT"],
)

# Encode 1Hz microstructure + fundamental window (B=1, T=10, N=8, F=14) and simulate shock
x_t = torch.randn(1, 10, 8, 14)
shock = CausalInterventionVector(
    primary_target_entities=["TSM"],
    directional_sentiment=-0.90,
    magnitude=0.95,
    contagion_pathways=[{"source": "TSM", "target": "NVDA", "weight": 0.90}],
    confidence=0.92,
)
z_hat_t1, prediction = fwm(x_t, intervention=shock)
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{carida2026ewm,
  author = {Caridá, Vinícius},
  title = {Enterprise World Model Engine: Financial and AIOps State-Transition Simulations},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/vfcarida/Enterprise-World-Model-Engine},
  license = {Apache-2.0}
}
```

## License

Apache License 2.0. See [LICENSE](../../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for setup, test/lint commands, and PR guidelines.

## Author

**Vinícius Caridá** — Sole author and maintainer.
