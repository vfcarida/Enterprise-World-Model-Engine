# Committed JSON Schemas Reference

EWM Engine validates declarative specifications, simulation trajectories, provenance manifests, and governance cards against **14 versioned Draft 2020-12 JSON Schemas** (`schema_version = "1.0.0"`) committed under [`schemas/`](https://github.com/vfcarida/Enterprise-World-Model-Engine/tree/main/schemas).

All schemas are generated deterministically from their governing Pydantic models via [`scripts/generate_schemas.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/scripts/generate_schemas.py) and verified for zero drift in CI (`python scripts/generate_schemas.py --check` and `tests/contract/test_api_compatibility.py`).

---

## Complete Schema Registry (`schemas/*.schema.json`)

| Schema Name (`ewm schema show <name>`) | Committed File | Canonical `$id` URI | Governing Pydantic Model | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `action` | `action.schema.json` | `https://ewm-engine.org/schemas/v1/action.schema.json` | `ewm_engine.core.actions.Action` | Declarative actor or interventional action payload (`actor_id`, `action_type`, `target_id`, `parameters`). |
| `world-state` | `world-state.schema.json` | `https://ewm-engine.org/schemas/v1/world-state.schema.json` | `ewm_engine.core.state.WorldState` | Immutable ontic snapshot $S_t$ (`entities`, `relationships`, `resources`, `memory`, `context`). |
| `world` | `world.schema.json` | `https://ewm-engine.org/schemas/v1/world.schema.json` | `ewm_engine.core.spec.WorldSpec` | Declarative world specification (`world`, `entities`, `relationships`, `resources`, `constraints`, `dynamics`, `actors`). |
| `scenario` | `scenario.schema.json` | `https://ewm-engine.org/schemas/v1/scenario.schema.json` | `ewm_engine.simulation.scenario.Scenario` | Simulation rollout configuration (`name`, `horizon`, `seed`, `samples`, `scheduled_actions`, `exogenous_events`). |
| `trajectory` | `trajectory.schema.json` | `https://ewm-engine.org/schemas/v1/trajectory.schema.json` | `ewm_engine.simulation.trajectory.Trajectory` | Complete rollout trajectory record (`steps`, `initial_state`, `final_state`, `status`, `systemic_trace`). |
| `provenance` | `provenance.schema.json` | `https://ewm-engine.org/schemas/v1/provenance.schema.json` | `ewm_engine.provenance.metadata.Provenance` | Cryptographic run metadata (`engine_version`, `seed`, `initial_state_fingerprint`, `python_version`, `platform_info`). |
| `transition-event` | `transition-event.schema.json` | `https://ewm-engine.org/schemas/v1/transition-event.schema.json` | `ewm_engine.durability.events.TransitionEvent` | Event-sourced `TraceLog` transition envelope (`run_id`, `step`, `pre_state_fp`, `post_state_fp`, `actions`, `deltas`). |
| `scenario-card` | `scenario-card.schema.json` | `https://ewm-engine.org/schemas/v1/scenario-card.schema.json` | `ewm_engine.cards.models.ScenarioCard` | Standardized scenario governance card (`assumptions`, `scope`, `limitations`, `scenario_fingerprint`). |
| `model-card` | `model-card.schema.json` | `https://ewm-engine.org/schemas/v1/model-card.schema.json` | `ewm_engine.cards.models.ModelCard` | Mitchell et al. (2019) compliant model card (`intended_use`, `out_of_scope_uses`, `evaluation_metrics`, `ethical_considerations`). |
| `dataset-card` | `dataset-card.schema.json` | `https://ewm-engine.org/schemas/v1/dataset-card.schema.json` | `ewm_engine.cards.models.DatasetCard` | Gebru et al. (2021) / Croissant JSON-LD dataset card (`collection_process`, `temporal_cutoff`, `known_biases`). |
| `property-spec` | `property-spec.schema.json` | `https://ewm-engine.org/schemas/v1/property-spec.schema.json` | `ewm_engine.verification.spec.PropertySpec` | Signal Temporal Logic (STL) and oracle-graph trajectory verification specification. |
| `parameter-space` | `parameter-space.schema.json` | `https://ewm-engine.org/schemas/v1/parameter-space.schema.json` | `ewm_engine.experimentation.params.ParameterSpace` | Parameter space bounds and distributions for DoE sweeps, sensitivity analysis, and PRIM scenario discovery. |
| `report-model` | `report-model.schema.json` | `https://ewm-engine.org/schemas/v1/report-model.schema.json` | `ewm_engine.reporting.model.ReportModel` | Renderer-neutral simulation report summary (`quantiles`, `violations`, `fingerprint`, `forecast_results`). |
| `wmif-manifest` | `wmif-manifest.schema.json` | `https://ewm-engine.org/schemas/v1/wmif-manifest.schema.json` | `ewm_engine.serialization.wmif.WMIFManifest` | World Model Interchange Format (WMIF v1.0) portable package manifest and conformance envelope. |

---

## CLI & Programmatic Inspection

```bash
# List all 14 schemas via the CLI
ewm schema list

# Inspect a specific schema (e.g., world, world-state, wmif-manifest)
ewm schema show world

# Verify that all committed schemas match current Pydantic models
python scripts/generate_schemas.py --check
```
