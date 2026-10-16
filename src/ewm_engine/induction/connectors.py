"""DataFrame and SQL connectors for world-model induction with point-in-time correctness."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import (
    EntitySpec,
    RelationshipSpec,
    ResourceSpec,
    WorldMetadataSpec,
    WorldSpec,
)
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.induction.report import InductionReport
from ewm_engine.provenance.evidence import EvidenceLevel


class DataFrameSchemaMapping(BaseModel):
    """Declarative mapping instructions for converting tabular data into a WorldSpec."""

    model_config = ConfigDict(extra="forbid")

    entity_id_col: str | None = Field(
        default=None,
        description="Column identifying unique entity instances. Defaults to auto-generated or index.",
    )
    entity_type: str = Field(
        default="Entity",
        description="Default entity type name if entity_type_col is not provided.",
    )
    entity_type_col: str | None = Field(
        default=None,
        description="Optional column indicating polymorphic entity types.",
    )
    timestamp_col: str | None = Field(
        default=None,
        description="Timestamp or step column for point-in-time cutoff filtering.",
    )
    resource_cols: list[str] | dict[str, dict[str, Any]] = Field(
        default_factory=list,
        description="Columns representing measurable continuous or discrete resources.",
    )
    attribute_cols: list[str] = Field(
        default_factory=list,
        description="Columns to store as entity attributes.",
    )
    relationship_cols: list[tuple[str, str, str]] = Field(
        default_factory=list,
        description="List of (source_col, target_col, relation_type) triples defining edges.",
    )


def _to_records(df_or_table: Any) -> list[dict[str, Any]]:
    """Convert an arbitrary tabular input (Arrow, pandas, polars, or records) into Python dict records."""
    # 1. Pure list of dicts
    if isinstance(df_or_table, list):
        if not df_or_table or isinstance(df_or_table[0], dict):
            return df_or_table
        raise SimulationConfigurationError("List input must contain dictionaries.")

    # 2. PyArrow Table
    try:
        import pyarrow as pa  # type: ignore[import-not-found]

        if isinstance(df_or_table, pa.Table):
            return df_or_table.to_pylist()
    except ImportError:
        pass

    # 3. Arrow PyCapsule / Stream Interface
    if hasattr(df_or_table, "__arrow_c_stream__"):
        try:
            import pyarrow as pa  # type: ignore[import-not-found]

            table = pa.table(df_or_table)
            return table.to_pylist()
        except Exception:
            pass

    # 4. Pandas DataFrame
    if hasattr(df_or_table, "to_dict") and hasattr(df_or_table, "columns"):
        return df_or_table.to_dict(orient="records")

    # 5. Polars DataFrame
    if hasattr(df_or_table, "to_dicts"):
        return df_or_table.to_dicts()

    raise SimulationConfigurationError(
        f"Unsupported tabular data input of type '{type(df_or_table).__name__}'. "
        "Expected pyarrow.Table, pandas.DataFrame, polars.DataFrame, or list[dict]."
    )


def induce_from_dataframe(
    df: Any,
    schema: DataFrameSchemaMapping | None = None,
    as_of: Any | None = None,
    name: str = "InducedFromDataFrame",
    seed: int = 42,
) -> tuple[WorldSpec, InductionReport]:
    """Propose a validated, point-in-time correct WorldSpec from a tabular DataFrame or Arrow Table.

    Point-in-Time Correctness Guarantee (RelBench discipline arXiv:2407.20060):
        If `as_of` is specified, all records strictly after `as_of` are purged prior to
        world model induction, preventing temporal leakage from future observations.
    """
    records = _to_records(df)
    mapping = schema or DataFrameSchemaMapping()

    if not records:
        empty_spec = WorldSpec(world=WorldMetadataSpec(name=name))
        empty_report = InductionReport(
            source=name,
            source_type="dataframe",
            seed=seed,
            point_in_time_cutoff=str(as_of) if as_of is not None else None,
        )
        return empty_spec, empty_report

    # Infer mapping heuristics if unspecified
    first_record = records[0]
    keys = list(first_record.keys())

    # Detect timestamp column
    ts_col = mapping.timestamp_col
    if ts_col is None:
        candidate_ts = [
            k for k in keys if k.lower() in ("timestamp", "time", "date", "created_at", "step", "t")
        ]
        if candidate_ts:
            ts_col = candidate_ts[0]

    # Enforce Point-in-Time Cutoff
    cutoff_str = str(as_of) if as_of is not None else None
    if as_of is not None:
        if ts_col and ts_col in first_record:
            records = [r for r in records if r[ts_col] is not None and r[ts_col] <= as_of]
        else:
            if isinstance(as_of, int) and as_of < len(records):
                records = records[: as_of + 1]

    if not records:
        empty_spec = WorldSpec(world=WorldMetadataSpec(name=name))
        empty_report = InductionReport(
            source=name,
            source_type="dataframe",
            seed=seed,
            point_in_time_cutoff=cutoff_str,
        )
        return empty_spec, empty_report

    # Detect entity ID column
    id_col = mapping.entity_id_col
    if id_col is None:
        candidate_ids = [
            k for k in keys if k.lower() in ("entity_id", "id", "node_id", "uuid", "key")
        ]
        id_col = candidate_ids[0] if candidate_ids else None

    # Detect resource columns
    resource_config: dict[str, dict[str, Any]] = {}
    if isinstance(mapping.resource_cols, dict):
        resource_config = mapping.resource_cols
    elif isinstance(mapping.resource_cols, list) and mapping.resource_cols:
        for r_col in mapping.resource_cols:
            resource_config[r_col] = {}
    else:
        # Heuristic: numeric float/int columns that aren't ID or timestamp
        for k in keys:
            if k == id_col or k == ts_col:
                continue
            val = first_record[k]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                resource_config[k] = {}

    # Extract Entities and Resources
    entity_specs: dict[str, EntitySpec] = {}
    resource_specs: dict[str, ResourceSpec] = {}
    relationships: list[RelationshipSpec] = []

    for idx, row in enumerate(records):
        ent_id = str(row[id_col]) if id_col and id_col in row else f"entity_{idx}"
        ent_type = (
            str(row[mapping.entity_type_col])
            if mapping.entity_type_col and mapping.entity_type_col in row
            else mapping.entity_type
        )

        # Entity Attributes
        attrs: dict[str, Any] = {}
        for k, v in row.items():
            if k == id_col or k in resource_config:
                continue
            if not mapping.attribute_cols or k in mapping.attribute_cols:
                attrs[k] = v

        if ent_id not in entity_specs:
            entity_specs[ent_id] = EntitySpec(id=ent_id, type=ent_type, attributes=attrs)
        else:
            entity_specs[ent_id].attributes.update(attrs)

        # Track Resources
        for r_col, cfg in resource_config.items():
            if r_col in row and row[r_col] is not None:
                val = float(row[r_col])
                res_id = f"{ent_id}_{r_col}" if id_col else r_col
                min_v = cfg.get("min_value", 0.0)
                max_v = cfg.get("max_value", 1e9)
                unit = cfg.get("unit", "units")
                if res_id not in resource_specs:
                    resource_specs[res_id] = ResourceSpec(
                        id=res_id,
                        current=val,
                        min_value=min_v,
                        max_value=max_v,
                        unit=unit,
                        entity_id=ent_id,
                        tags=["induced", "dataframe"],
                    )
                else:
                    # Update to latest point-in-time value
                    resource_specs[res_id].current = val

        # Relationships
        for src_col, tgt_col, rel_type in mapping.relationship_cols:
            if src_col in row and tgt_col in row:
                src_val = str(row[src_col])
                tgt_val = str(row[tgt_col])
                if src_val and tgt_val:
                    relationships.append(
                        RelationshipSpec(
                            source=src_val,
                            target=tgt_val,
                            type=rel_type,
                            attributes={"evidence_level": EvidenceLevel.STRUCTURAL.value},
                        )
                    )

    spec = WorldSpec(
        world=WorldMetadataSpec(
            name=name, description=f"Induced from DataFrame ({len(records)} records)"
        ),
        entities=list(entity_specs.values()),
        relationships=relationships,
        resources=list(resource_specs.values()),
    )

    report = InductionReport(
        source=name,
        source_type="dataframe",
        seed=seed,
        point_in_time_cutoff=cutoff_str,
        entities_count=len(spec.entities),
        relationships_count=len(spec.relationships),
        resources_count=len(spec.resources),
        constraints_count=0,
        evidence_distribution={
            EvidenceLevel.STRUCTURAL.value: len(spec.relationships) + len(spec.resources),
        },
        metadata={"record_count": len(records), "timestamp_column": ts_col},
    )

    return spec, report


def induce_from_sql(
    query: str,
    con: Any,
    as_of: Any | None = None,
    schema: DataFrameSchemaMapping | None = None,
    name: str = "InducedFromSQL",
    seed: int = 42,
) -> tuple[WorldSpec, InductionReport]:
    """Execute SQL query with point-in-time correctness and induce a WorldSpec.

    Supports:
        - DuckDB connections (`duckdb.PyConnection`)
        - SQLite connections (`sqlite3.Connection`)
        - SQLAlchemy engines/connections
        - Any standard Python DB-API 2.0 cursor or connection
    """
    records: list[dict[str, Any]] = []

    # DuckDB check
    if hasattr(con, "query"):
        rel = con.query(query)
        if hasattr(rel, "to_df"):
            records = rel.to_df().to_dict(orient="records")
        elif hasattr(rel, "arrow"):
            records = rel.arrow().to_pylist()
        else:
            records = [dict(zip(rel.columns, row, strict=False)) for row in rel.fetchall()]
    elif hasattr(con, "execute"):
        cursor = con.execute(query)
        if hasattr(cursor, "description") and cursor.description:
            col_names = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            records = [dict(zip(col_names, row, strict=False)) for row in rows]
        elif hasattr(con, "cursor"):
            cur = con.cursor()
            cur.execute(query)
            col_names = [col[0] for col in cur.description]
            rows = cur.fetchall()
            records = [dict(zip(col_names, row, strict=False)) for row in rows]
    else:
        raise SimulationConfigurationError(
            f"Unsupported SQL connection type '{type(con).__name__}'. "
            "Expected DuckDB, SQLite, SQLAlchemy, or DB-API 2.0 connection."
        )

    spec, report = induce_from_dataframe(
        df=records,
        schema=schema,
        as_of=as_of,
        name=name,
        seed=seed,
    )
    report.source = query.strip()[:80] + ("..." if len(query) > 80 else "")
    report.source_type = "sql"
    report.sha256_fingerprint = report.compute_fingerprint()

    return spec, report
