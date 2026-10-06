# Reproducibility Cards & Croissant 1.1 Guide

In high-stakes enterprise decision-making, simulation outputs, learned dynamics models, and synthetic datasets require transparent, auditable provenance. The Enterprise World Model Engine provides native, Pydantic-based documentation cards in `ewm_engine.cards`:
- **Model Cards** (Mitchell et al., arXiv:1810.03993): Document dynamics mechanisms, intended use, structural assumptions, out-of-scope operating regimes, and empirical calibration scores (arXiv:2411.10109).
- **Scenario Cards**: Document experiment parameters, environmental context, interventions, and boundary conditions.
- **Dataset Cards** (Gebru et al., arXiv:1803.09010): Document simulated trajectory datasets, recorded feature schemas, collection methodology, and upstream provenance fingerprints.
- **MLCommons Croissant 1.1**: Standard-library JSON-LD export enabling FAIR scientific dataset discovery without external dependencies.

---

## 1. Native Model Cards

The Google Model Card Toolkit was archived in 2024. EWM Engine fills this gap with native, strongly typed Pydantic models that render to JSON, canonical YAML, and publication-ready GitHub-flavored Markdown.

```python
from ewm_engine.cards import ModelCard

card = ModelCard(
    model_id="linear-supply-v1",
    model_name="Linear Multi-Echelon Inventory Dynamics",
    version="1.1.0",
    artifact_fingerprint="8f4b23c91d8e...",
    maturity="production",
    description="First-order linear resource flow dynamics modeling replenishment and consumption.",
    intended_use=[
        "Evaluating supply buffer safety stock reallocations",
        "Simulating supplier lead-time contractions under stable demand",
    ],
    assumptions=[
        "Deterministic delivery transit times without stochastic delay spikes",
        "Total inventory mass conservation across transfer nodes",
    ],
    out_of_scope=[
        "High-frequency intra-day stockout shocks",
        "Regimes where competitor stock availability influences consumer demand",
    ],
    limitations=[
        "Assumes static demand distribution parameters; re-calibration required quarterly",
    ],
    constraints_exercised=["capacity_ceiling", "non_negative_balance"],
    metrics={"rmse": 0.042, "r2": 0.985},
    calibration_score=0.978,
)

# Export to Markdown, YAML, or JSON
markdown_report = card.to_markdown()
yaml_spec = card.to_yaml()
json_manifest = card.to_json()
```

### Generated Markdown Preview

```markdown
# Model Card: Linear Multi-Echelon Inventory Dynamics (`linear-supply-v1`)

> [!IMPORTANT]
> **Maturity Level:** `PRODUCTION` | **Version:** `v1.1.0`

### Cryptographic Provenance
- **Artifact Fingerprint (SHA-256):** `8f4b23c91d8e...`
- **Schema Version:** `1.0.0`

### Intended Use
- Evaluating supply buffer safety stock reallocations
- Simulating supplier lead-time contractions under stable demand

### Out-of-Scope Regimes
> [!WARNING]
> The following operating conditions are outside model validity:
> - High-frequency intra-day stockout shocks
> - Regimes where competitor stock availability influences consumer demand

### Evaluation & Calibration Metrics
| Metric | Value |
|---|---|
| `population_calibration_score` | **0.9780** |
| `r2` | **0.985** |
| `rmse` | **0.042** |
```

---

## 2. Scenario Cards

A `ScenarioCard` documents the setup of a simulation experiment, parameter sweep, or stress test:

```python
from ewm_engine.cards import ScenarioCard

scen_card = ScenarioCard(
    scenario_id="tariff_stress_2026",
    description="Evaluates 25% component import tariffs combined with a 10% domestic demand surge.",
    horizon=24,
    samples=1000,
    seed=20261005,
    artifact_fingerprint="3a8e91bc7d...",
    interventions=["tariff_hike_25pct", "domestic_subsidy_5pct"],
    assumptions=["Exchange rates remain range-bound within ±3%"],
    environmental_context={"interest_rate": 0.045, "inflation": 0.028},
    metrics={"p50_margin": 0.18, "cvar_05_loss": -0.06},
)

print(scen_card.to_markdown())
```

---

## 3. Dataset Cards & Croissant 1.1 JSON-LD

When simulation rollouts are saved for offline reinforcement learning, causal discovery, or dynamics training, a `DatasetCard` captures the dataset's features, sampling process, and upstream cryptographic hashes.

```python
from ewm_engine.cards import DatasetCard, to_croissant_dataset, to_croissant_json

dataset_card = DatasetCard(
    dataset_id="supply-chain-benchmark-v1",
    name="Enterprise Supply Network Stress Traces",
    version="1.0.0",
    artifact_fingerprint="e5b12a88c...",
    description="50,000 synthetic Monte Carlo trajectories across 5 shift benchmark families.",
    num_trajectories=50000,
    num_steps=600000,
    features=["inventory_level", "cash_balance", "backlog_count", "unserved_demand"],
    intended_use=["Offline policy optimization", "Neural dynamics baseline evaluation"],
    license="Apache-2.0",
    provenance_fingerprints=["initial_state_hash_abc", "scenario_hash_def"],
)

# Render human-readable documentation
md_card = dataset_card.to_markdown()

# Hand-emit standard MLCommons Croissant 1.1 JSON-LD
croissant_jsonld = to_croissant_json(dataset_card, indent=2)
```

### Croissant 1.1 JSON-LD Structure

The emitted metadata conforms to the [MLCommons Croissant 1.1 specification](http://mlcommons.org/croissant/) without requiring external Python dependencies:

```json
{
  "@context": {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    ...
  },
  "@type": "sc:Dataset",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "name": "Enterprise Supply Network Stress Traces",
  "version": "1.0.0",
  "license": "Apache-2.0",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "trajectories-json",
      "name": "trajectories-json",
      "contentUrl": "trajectories.json",
      "encodingFormat": "application/json",
      "sha256": "e5b12a88c..."
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "steps",
      "name": "steps",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "steps/sample_id",
          "name": "sample_id",
          "dataType": "sc:Integer"
        },
        {
          "@type": "cr:Field",
          "@id": "steps/inventory_level",
          "name": "inventory_level",
          "dataType": "sc:Float"
        }
      ]
    }
  ]
}
```
