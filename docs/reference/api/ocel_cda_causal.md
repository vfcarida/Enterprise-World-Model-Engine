# API Reference: `StateEncoder`, `RetrieveAndReasonEngine`, and `EngineCausalInterface`

This reference documents the three foundational pillars of the realigned Enterprise World Model (EWM) Engine core architecture:

1. **Object-Centric Event Log (OCEL 2.0) State Layer** (`ewm_engine.core.ocel.StateEncoder`)
2. **Continual Discovery Agent (CDA) Retrieve-Then-Reason Dynamics** (`ewm_engine.core.dynamics.RetrieveAndReasonEngine`)
3. **DoWhy-GCM Structural Causal Model Interface** (`ewm_engine.core.causal.EngineCausalInterface`)

---

## 1. `StateEncoder` (`ewm_engine.core.ocel`)

`StateEncoder` persists discrete system events into a normalized SQLite database strictly adhering to the 24 constraints of the OCEL 2.0 relational specification ($N:M$ Event-to-Object `event_object` and Object-to-Object `object_object` mappings), eliminating case-centric convergence and divergence errors.

### Class Signature & Key Methods

- `StateEncoder(db_path: str | Path = ":memory:")`
- `encode_event(event: Mapping[str, Any] | OcelEventRecord) -> OcelEventRecord`
- `encode_events(events: Iterable[Mapping[str, Any] | OcelEventRecord]) -> List[OcelEventRecord]`
- `encode_object(object_id: str, object_type: str, attributes: Mapping[str, Any] | None = None, ...)`
- `encode_o2o(source_id: str, target_id: str, qualifier: str = "relates_to") -> None`
- `export_to_ocel_json(filepath: str | Path, *, max_payload_bytes: int = 67108864, chunk_size: int = 2048) -> Path`
- `export_to_ocel_xml(filepath: str | Path, *, max_payload_bytes: int = 67108864) -> Path`
- `ingest_csv(csv_input: str | Path | Iterable[str], *, object_columns: Mapping[str, str] | None = None, o2o_rules: Sequence[Tuple[str, str, str]] | None = None, batch_size: int = 500) -> int`
- `ingest_xes(xes_input: str | Path, *, case_object_type: str = "Case", resource_object_type: str = "Resource") -> int`
- `read_ocel2(validate: bool = True) -> OCEL`
- `validate_schema() -> Tuple[List[str], List[str]]`
- `to_canonical_snapshot() -> Dict[str, Any]`
- `equals(other: Any) -> bool`

### Executable Example

```python
import tempfile
from pathlib import Path
from ewm_engine.core.ocel import (
    OcelEventRecord,
    OcelObjectRelation,
    OcelObjectTarget,
    StateEncoder,
    validate_ocel2_json,
)

with tempfile.TemporaryDirectory() as tmp_dir:
    db_path = Path(tmp_dir) / "state.sqlite"
    json_path = Path(tmp_dir) / "state.jsonocel"

    with StateEncoder(db_path) as encoder:
        encoder.encode_event(
            OcelEventRecord(
                event_id="evt_101",
                event_type="ConfirmOrder",
                timestamp="2026-11-17T12:00:00.123456Z",
                objects=(
                    OcelObjectTarget(object_id="usr_1", object_type="User", qualifier="buyer"),
                    OcelObjectTarget(object_id="ord_1", object_type="Order", qualifier="subject"),
                    OcelObjectTarget(object_id="inv_1", object_type="Invoice", qualifier="billed"),
                ),
                attributes={"amount": 1250.0, "currency": "USD"},
                object_relations=(
                    OcelObjectRelation(source_id="usr_1", target_id="ord_1", qualifier="places"),
                ),
            )
        )
        satisfied, unsatisfied = encoder.validate_schema()
        assert len(satisfied) == 24 and not unsatisfied
        encoder.export_to_ocel_json(json_path)

    assert validate_ocel2_json(json_path) is True
```

---

## 2. `RetrieveAndReasonEngine` (`ewm_engine.core.dynamics`)

`RetrieveAndReasonEngine` replaces static offline latent rollout predictors ($z_{t+1} = f_\theta(z_t, a_t)$) with runtime configuration discovery:

$$s_{t+1} \sim P(s_{t+1} \mid s_t, a_t, c)$$

where $c = \text{Retrieve}(s_t, a_t)$ is the set of active `BusinessRule` specifications retrieved from a `ConfigurationDatastore`. Every predicted state modification in `StateDiff` ($\Delta s_{t+1}$) carries an explicit `RuleAttribution` linking the change to a retrieved `rule_id` and `rule_version`.

### Class Signature & Key Methods

- `RetrieveAndReasonEngine(datastore: ConfigurationDatastore | None = None, reasoning_client: Any = None, prompt_harness: CDAPromptHarness | None = None, name: str = "RetrieveAndReasonEngine")`
- `predict_diff(state: WorldState | Mapping[str, Any], action: Action | Mapping[str, Any], retriever: Any = None) -> StateDiff`
- `step(state: WorldState, action: Action | Mapping[str, Any], retriever: Any = None) -> Tuple[WorldState, StateDiff]`
- `transition(state: WorldState, actions: Sequence[Action], exogenous_events: Sequence[ExogenousEvent], rng: RandomGenerator, runtime_context: ContextSnapshot | None = None) -> TransitionResult`
- `telemetry_summary -> Dict[str, int | float]`

### Executable Example

```python
from ewm_engine.core.actions import Action
from ewm_engine.core.dynamics import (
    BusinessRule,
    ConfigurationDatastore,
    RetrieveAndReasonEngine,
)
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState

state = WorldState(
    step=0,
    timestamp=1700000000.0,
    entities={"order_1": Entity(id="order_1", type="Order", attributes={"status": "pending"})},
    resources={"inventory": Resource(id="inventory", current=80.0, unit="units")},
)
action = Action(id="act_1", type="fulfill_order", parameters={"quantity": 20})

datastore = ConfigurationDatastore(
    rules=[
        BusinessRule(
            rule_id="RULE-FULFILL-V1",
            action_types=("fulfill_order",),
            rule_text="resource:inventory -= 20.0; entity:order_1.status = 'shipped_v1'",
            version="1.0.0",
        )
    ]
)
engine = RetrieveAndReasonEngine(datastore=datastore)
next_state_v1, diff_v1 = engine.step(state, action)
assert next_state_v1.resources["inventory"].current == 60.0
assert next_state_v1.entities["order_1"].attributes["status"] == "shipped_v1"

# Mutate rule at runtime without retraining
datastore.update_rule(
    "RULE-FULFILL-V1",
    rule_text="IF resource:inventory <= 100.0 THEN resource:inventory -= 20.0; entity:order_1.status = 'rationed_v2'",
    version="2.0.0",
)
next_state_v2, diff_v2 = engine.step(state, action)
assert next_state_v2.entities["order_1"].attributes["status"] == "rationed_v2"
assert diff_v2.rule_attributions[0].rule_version == "2.0.0"
```

---

## 3. `EngineCausalInterface` (`ewm_engine.core.causal`)

`EngineCausalInterface` wraps `dowhy.gcm.StructuralCausalModel` over an application-provided `networkx.DiGraph` to distinguish Rung-1 observational prediction (`.predict()`) from Rung-2 causal intervention (`.simulate_intervention(target_node, value, n_samples)`).

### Class Signature & Key Methods

- `EngineCausalInterface(dag: networkx.DiGraph, *, seed: int = 42)`
- `assign_mechanisms(data: pd.DataFrame | str | Path, *, override_models: bool = True, fit: bool = True, ocel_table: str | None = None) -> Dict[str, str]`
- `predict(observed_data: pd.DataFrame | None = None, *, n_samples: int = 100) -> pd.DataFrame`
- `simulate_intervention(target_node: str | Mapping[str, Any], value: Any = None, n_samples: int = 1000) -> pd.DataFrame`

### Executable Example

```python
import networkx as nx
import numpy as np
import pandas as pd
from ewm_engine.core.causal import EngineCausalInterface

rng = np.random.default_rng(42)
x = rng.normal(10.0, 1.0, size=300)
y = 2.0 * x + 5.0
z = 3.0 * y - 4.0
df = pd.DataFrame({"X": x, "Y": y, "Z": z})

dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
causal_engine = EngineCausalInterface(dag=dag, seed=42)
mechanisms = causal_engine.assign_mechanisms(df)

samples = causal_engine.simulate_intervention("Y", 20.0, n_samples=200)
assert float(samples["Y"].var(ddof=0)) == 0.0
assert np.allclose(samples["Z"].to_numpy(), 56.0, atol=1e-5)
```
