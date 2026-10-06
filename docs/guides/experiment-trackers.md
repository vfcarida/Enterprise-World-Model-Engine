# Experiment Trackers: Local JSON, OmegaConf & MLflow / W&B

This guide demonstrates how to track parameters, metrics, cards, and artifacts using `LocalJsonTracker`, manage hierarchical YAML configurations with `OmegaConf`, and integrate with MLflow and Weights & Biases.

---

## 1. Zero-Dependency Local JSON Tracker

`LocalJsonTracker` logs experiment runs atomically to disk under `.ewm_runs/{fingerprint}/run.json` without any third-party dependencies:

```python
from ewm_engine.trackers import LocalJsonTracker
from ewm_engine.cards.models import ScenarioCard

tracker = LocalJsonTracker(base_dir=".ewm_runs")

# 1. Set canonical fingerprint
fp = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
tracker.log_fingerprint(fp)

# 2. Log hyper-parameters
tracker.log_params({"horizon": 20, "policy": "reorder_point", "threshold": 50})

# 3. Log step metrics
for t in range(1, 21):
    tracker.log_metrics({"inventory": 100 - t * 2, "unmet_demand": 0.0}, step=t)

# 4. Log reproducibility cards
card = ScenarioCard(
    artifact_fingerprint=fp,
    scenario_id="stress_q4",
    horizon=20,
    samples=10,
    seed=42,
    description="Q4 Peak Demand Stress Test",
)
tracker.log_card(card)

# 5. Log artifact references
tracker.log_artifact("artifacts/model_weights.pt", artifact_type="weights")
```

The run file `.ewm_runs/{fingerprint}/run.json` is updated atomically using re-entrant locks and safe tempfile replacement.

---

## 2. Hierarchical Configurations with OmegaConf

Load YAML configuration trees with `${...}` variable interpolation and compute canonical configuration hashes using `[config]` (OmegaConf):

```python
from ewm_engine.trackers import OmegaConfConfigLoader

loader = OmegaConfConfigLoader()
config = loader.load_from_yaml_string("""
simulation:
  name: supply_chain_v1
  horizon: 30
  warehouse:
    capacity: 1000
    target_inventory: ${simulation.warehouse.capacity}
""")

print("Resolved target inventory:", config["simulation"]["warehouse"]["target_inventory"])
print("Config fingerprint:", loader.fingerprint(config))
```

---

## 3. Enterprise Trackers: MLflow & Weights & Biases

For teams running managed tracking servers, wrap your runs with native adapters behind `[trackers]`:

```python
from ewm_engine.trackers.adapters import MLflowTracker, WandbTracker

# MLflow
mlflow_tracker = MLflowTracker(experiment_name="world_model_validation")
mlflow_tracker.log_fingerprint(fp)
mlflow_tracker.log_params({"seed": 42})
mlflow_tracker.log_metrics({"coverage": 0.96})

# Weights & Biases
wandb_tracker = WandbTracker(project="enterprise-simulations")
wandb_tracker.log_fingerprint(fp)
wandb_tracker.log_params({"seed": 42})
wandb_tracker.log_metrics({"coverage": 0.96})
```
