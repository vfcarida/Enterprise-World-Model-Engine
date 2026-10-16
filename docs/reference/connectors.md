# Enterprise Data Connectors & Schema Induction

> **Discipline**: Relational Deep Learning & Point-in-Time Correctness  
> **Theoretical Anchors**: RelBench ([arXiv:2407.20060](https://arxiv.org/abs/2407.20060)), Relational Deep Learning ([arXiv:2312.04615](https://arxiv.org/abs/2312.04615)).

EWM Engine provides a pluggable suite of enterprise connectors that ingest tabular data, relational SQL databases, and DDL definitions directly into typed world model graphs with **zero temporal leakage**.

---

## 1. Enterprise Connector Matrix

| Source Type | Interface / Protocol | Ingestion Method | Point-in-Time Correctness | Core Dependency Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Apache Arrow** | `pyarrow.Table`, `__arrow_c_stream__` | `WorldSpec.from_dataframe` | Filtered on `as_of` timestamp | Zero (Optional Extra `[pyarrow]`) |
| **Pandas / Polars** | `pd.DataFrame`, `pl.DataFrame` | `WorldSpec.from_dataframe` | Filtered on `as_of` timestamp | Zero (Standard DataFrames supported) |
| **DuckDB** | `duckdb.PyConnection` | `WorldSpec.from_sql` | Temporal WHERE injection | Zero (Optional Extra `[duckdb]`) |
| **SQLite / DB-API** | `sqlite3.Connection`, PEP 249 | `WorldSpec.from_sql` | Subquery / DataFrame cutoff | Standard Library (`sqlite3`) |
| **Relational DDL** | SQL `CREATE TABLE` scripts | `induce_schema(ddl=...)` | Static schema skeleton | Pure Python / Regular Expressions |
| **OCEL 2.0 Logs** | JSON, JSONL | `WorldSpec.from_ocel` | Event timestamp ordering | Pure Python / NumPy |

---

## 2. Point-in-Time Correctness (The RelBench Discipline)

In organizational simulation, a catastrophic source of error is **temporal data leakage**—incorporating records, entity updates, or event outcomes that occurred strictly after the decision epoch. 

EWM Engine strictly enforces the RelBench discipline:

```python
from ewm_engine.core.spec import WorldSpec
from ewm_engine.induction.connectors import DataFrameSchemaMapping

# All records strictly after as_of=100 are purged prior to state compilation
spec = WorldSpec.from_dataframe(
    df=telemetry_df,
    as_of=100,
    schema=DataFrameSchemaMapping(
        entity_id_col="warehouse_id",
        timestamp_col="step",
        resource_cols=["inventory_units", "operating_cash"],
    ),
)
```

1. **Cutoff Filtering**: Any row where `timestamp > as_of` is discarded before entity state aggregation or resource snapshotting.
2. **Deterministic State Fingerprint**: Initial resource levels reflect the state of the enterprise precisely at the `as_of` milestone.

---

## 3. Relational Schema Induction (`ewm induce schema`)

A relational database schema is inherently a **temporal heterogeneous typed graph**:
- **Tables** $\implies$ `EntitySpec` types.
- **Foreign Key Constraints** $\implies$ `RelationshipSpec` edges tagged as `ASSERTED` (ground-truth physical or transactional links).
- **CHECK Constraints & NOT NULL** $\implies$ Candidate hard constraint boundaries (e.g. `credit_limit >= 0`).
- **Numeric Quantities** $\implies$ `ResourceSpec` states (inventory, balance, capacity).

### Link Prediction & Provisional Latent Edges

In enterprise data warehouses, explicit foreign keys are frequently omitted or denormalized. EWM incorporates a Relational Deep Learning heuristic to propose latent links:
- Proposes unstated joins based on primary-key pattern matching (`<entity>_id`) and high-overlap attribute Jaccard similarity.
- Tagged with `EvidenceLevel.PREDICTIVE` and `provisional: True`.
- Carries an explicit confidence score ($c \in [0, 1]$).
- Must clear the acceptance threshold before promotion into hard simulation constraints.

```bash
ewm induce schema \
    --ddl schema.sql \
    --link-prediction \
    --threshold 0.75 \
    --out induced_schema.yaml \
    --report schema_report.json
```
