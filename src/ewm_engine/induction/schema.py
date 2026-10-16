"""Relational schema induction deriving typed-graph skeletons, asserted FKs, and provisional links."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import (
    ComponentSpec,
    EntitySpec,
    RelationshipSpec,
    ResourceSpec,
    WorldMetadataSpec,
    WorldSpec,
)
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.induction.report import (
    CandidateInvariant,
    InductionReport,
    ProvisionalLink,
)
from ewm_engine.provenance.evidence import EvidenceLevel


class TableSchema(BaseModel):
    """Schema descriptor for a relational database table."""

    model_config = ConfigDict(extra="forbid")

    name: str
    columns: dict[str, str] = Field(
        default_factory=dict,
        description="Column names mapped to declared SQL types (e.g. INTEGER, FLOAT, VARCHAR).",
    )
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of FK mappings with 'from_col', 'to_table', 'to_col'.",
    )
    check_constraints: list[str] = Field(
        default_factory=list,
        description="CHECK constraint expressions extracted from DDL or metadata.",
    )
    not_null_columns: list[str] = Field(default_factory=list)


def parse_ddl_to_tables(ddl_text: str) -> dict[str, TableSchema]:
    """Parse standard SQL DDL CREATE TABLE statements into TableSchema representations."""
    tables: dict[str, TableSchema] = {}
    # Match CREATE TABLE [IF NOT EXISTS] table_name ( ... );
    table_pattern = re.compile(
        r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+([a-zA-Z0-9_]+)\s*\((.*?)\);",
        re.IGNORECASE | re.DOTALL,
    )

    for match in table_pattern.finditer(ddl_text):
        table_name = match.group(1).lower()
        body = match.group(2)

        columns: dict[str, str] = {}
        pks: list[str] = []
        fks: list[dict[str, str]] = []
        checks: list[str] = []
        not_nulls: list[str] = []

        # Split definitions by comma, taking care not to split within parentheses
        definitions: list[str] = []
        depth = 0
        current: list[str] = []
        for char in body:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                definitions.append("".join(current).strip())
                current = []
            else:
                current.append(char)
        if current:
            definitions.append("".join(current).strip())

        for defn in definitions:
            defn_clean = " ".join(defn.split())
            defn_upper = defn_clean.upper()

            # PRIMARY KEY constraint
            if defn_upper.startswith("PRIMARY KEY"):
                pk_match = re.search(r"PRIMARY\s+KEY\s*\((.*?)\)", defn_clean, re.IGNORECASE)
                if pk_match:
                    pks.extend(
                        [c.strip().strip('"').strip("`") for c in pk_match.group(1).split(",")]
                    )
                continue

            # FOREIGN KEY constraint
            if defn_upper.startswith("FOREIGN KEY"):
                fk_match = re.search(
                    r"FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+([a-zA-Z0-9_]+)\s*\((.*?)\)",
                    defn_clean,
                    re.IGNORECASE,
                )
                if fk_match:
                    from_col = fk_match.group(1).strip().strip('"').strip("`")
                    to_table = fk_match.group(2).strip().lower()
                    to_col = fk_match.group(3).strip().strip('"').strip("`")
                    fks.append({"from_col": from_col, "to_table": to_table, "to_col": to_col})
                continue

            # CHECK constraint
            if defn_upper.startswith("CHECK"):
                check_match = re.search(r"CHECK\s*\((.*)\)", defn_clean, re.IGNORECASE)
                if check_match:
                    checks.append(check_match.group(1).strip())
                continue

            # Normal Column Definition: name TYPE [CONSTRAINTS]
            parts = defn_clean.split(maxsplit=2)
            if parts:
                col_name = parts[0].strip().strip('"').strip("`")
                col_type = parts[1].upper() if len(parts) > 1 else "TEXT"
                columns[col_name] = col_type

                rest = parts[2].upper() if len(parts) > 2 else ""
                if "PRIMARY KEY" in rest:
                    pks.append(col_name)
                if "NOT NULL" in rest:
                    not_nulls.append(col_name)
                if "CHECK" in rest:
                    check_match = re.search(r"CHECK\s*\((.*)\)", parts[2], re.IGNORECASE)
                    if check_match:
                        checks.append(check_match.group(1).strip())
                if "REFERENCES" in rest:
                    ref_match = re.search(
                        r"REFERENCES\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", parts[2], re.IGNORECASE
                    )
                    if ref_match:
                        to_table = ref_match.group(1).strip().lower()
                        to_col = ref_match.group(2).strip().strip('"').strip("`")
                        fks.append({"from_col": col_name, "to_table": to_table, "to_col": to_col})

        tables[table_name] = TableSchema(
            name=table_name,
            columns=columns,
            primary_keys=pks,
            foreign_keys=fks,
            check_constraints=checks,
            not_null_columns=not_nulls,
        )

    return tables


def introspect_sqlite(con: sqlite3.Connection) -> dict[str, TableSchema]:
    """Extract schema metadata directly from a live SQLite connection."""
    cur = con.cursor()
    tables: dict[str, TableSchema] = {}

    cur.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    rows = cur.fetchall()

    for table_name, sql in rows:
        t_name = str(table_name).lower()
        if sql:
            parsed = parse_ddl_to_tables(sql)
            if t_name in parsed:
                tables[t_name] = parsed[t_name]
                continue

        # Fallback to PRAGMA introspection
        cur.execute(f"PRAGMA table_info({t_name});")
        col_info = cur.fetchall()
        cols = {str(r[1]): str(r[2]).upper() for r in col_info}
        pks = [str(r[1]) for r in col_info if r[5] > 0]
        not_nulls = [str(r[1]) for r in col_info if r[3] > 0]

        cur.execute(f"PRAGMA foreign_key_list({t_name});")
        fk_info = cur.fetchall()
        fks = [
            {"from_col": str(r[3]), "to_table": str(r[2]).lower(), "to_col": str(r[4])}
            for r in fk_info
        ]

        tables[t_name] = TableSchema(
            name=t_name,
            columns=cols,
            primary_keys=pks,
            foreign_keys=fks,
            not_null_columns=not_nulls,
        )

    return tables


def _predict_latent_links(
    tables: dict[str, TableSchema],
    threshold: float = 0.7,
) -> list[ProvisionalLink]:
    """Propose latent entity relationships absent from declared FKs (Relational Deep Learning discipline).

    Computes semantic name similarity, shared ID identifiers, and bipartite overlap
    between table column keys. Tagged as INFERRED-LINK (provisional).
    """
    proposals: list[ProvisionalLink] = []
    declared_fks = {
        (t_name, fk["to_table"]) for t_name, schema in tables.items() for fk in schema.foreign_keys
    }

    table_names = list(tables.keys())
    for i in range(len(table_names)):
        for j in range(len(table_names)):
            if i == j:
                continue
            t_src = table_names[i]
            t_tgt = table_names[j]

            if (t_src, t_tgt) in declared_fks:
                continue

            # Heuristic 1: Exact target table singular ID match (e.g. orders has 'customer_id' matching 'customers')
            src_cols = set(tables[t_src].columns.keys())
            tgt_cols = set(tables[t_tgt].columns.keys())

            singular_tgt = t_tgt
            if t_tgt.endswith("ies"):
                singular_tgt = t_tgt[:-3] + "y"
            elif t_tgt.endswith("es") and len(t_tgt) > 2 and t_tgt[:-1].endswith("e"):
                singular_tgt = t_tgt[:-1]
            elif t_tgt.endswith("s"):
                singular_tgt = t_tgt[:-1]

            candidate_fk_names = {
                f"{t_tgt}_id",
                f"{singular_tgt}_id",
                f"id_{t_tgt}",
                f"id_{singular_tgt}",
            }

            matching_cols = src_cols.intersection(candidate_fk_names)
            if matching_cols:
                matched_col = next(iter(matching_cols))
                confidence = 0.85
                proposals.append(
                    ProvisionalLink(
                        source_entity=t_src,
                        target_entity=t_tgt,
                        relationship_type=f"inferred_{singular_tgt}",
                        confidence=confidence,
                        method="foreign_key_pattern_matching",
                        evidence_level=EvidenceLevel.PREDICTIVE,
                        clears_gate=(confidence >= threshold),
                        metadata={"matched_column": matched_col},
                    )
                )
                continue

            # Heuristic 2: Shared column Jaccard similarity (excluding IDs and generic audit columns)
            shared = src_cols.intersection(tgt_cols)
            filtered_shared = {
                c
                for c in shared
                if c.lower() not in ("id", "created_at", "updated_at", "status", "name")
                and not c.lower().endswith("_id")
            }
            union = src_cols.union(tgt_cols)
            if filtered_shared and union:
                jaccard = len(filtered_shared) / len(union)
                if jaccard >= 0.15:  # Significant domain column overlap
                    conf = min(0.75, jaccard * 2.5)
                    proposals.append(
                        ProvisionalLink(
                            source_entity=t_src,
                            target_entity=t_tgt,
                            relationship_type="co_associated",
                            confidence=conf,
                            method="schema_jaccard_overlap",
                            evidence_level=EvidenceLevel.PREDICTIVE,
                            clears_gate=(conf >= threshold),
                            metadata={"shared_columns": sorted(filtered_shared)},
                        )
                    )

    return proposals


def induce_schema(
    db: Any | None = None,
    metadata: dict[str, TableSchema] | dict[str, Any] | None = None,
    ddl: str | None = None,
    link_prediction: bool = True,
    link_threshold: float = 0.7,
    name: str = "InducedSchemaWorld",
    seed: int = 42,
) -> tuple[WorldSpec, InductionReport]:
    """Derive a typed-graph world specification skeleton from relational schemas.

    Declared foreign keys are tagged as ASSERTED (EvidenceLevel.STRUCTURAL);
    latent links inferred via link prediction are tagged as provisional (EvidenceLevel.PREDICTIVE).
    Candidate hard-constraint domains are extracted from CHECK, NOT NULL, and numeric column bounds.
    """
    tables: dict[str, TableSchema] = {}

    if ddl:
        from pathlib import Path

        ddl_p = Path(ddl)
        if ddl_p.exists() and ddl_p.is_file():
            ddl_text = ddl_p.read_text(encoding="utf-8")
        else:
            ddl_text = ddl
        tables = parse_ddl_to_tables(ddl_text)
    elif isinstance(db, sqlite3.Connection):
        tables = introspect_sqlite(db)
    elif isinstance(db, str):
        # Path to sqlite database or ddl file
        from pathlib import Path

        db_p = Path(db)
        if db.strip().endswith(".sql") or (db_p.exists() and db_p.suffix.lower() == ".sql"):
            with open(db, encoding="utf-8") as f:
                tables = parse_ddl_to_tables(f.read())
        else:
            con = sqlite3.connect(db)
            try:
                tables = introspect_sqlite(con)
            finally:
                con.close()
    elif metadata:
        for t_name, t_meta in metadata.items():
            if isinstance(t_meta, TableSchema):
                tables[t_name] = t_meta
            elif isinstance(t_meta, dict):
                tables[t_name] = TableSchema.model_validate(t_meta)
    else:
        raise SimulationConfigurationError(
            "Must provide db connection, ddl string/file, or metadata dictionary for schema induction."
        )

    entities: list[EntitySpec] = []
    relationships: list[RelationshipSpec] = []
    resources: list[ResourceSpec] = []
    constraints: list[ComponentSpec] = []
    candidate_invariants: list[CandidateInvariant] = []

    resource_keywords = {
        "stock",
        "inventory",
        "balance",
        "amount",
        "cost",
        "capacity",
        "price",
        "volume",
        "quantity",
        "credit",
        "limit",
        "quota",
    }

    # 1. Build Entity Specs and Candidate Resources
    for t_name, t_schema in tables.items():
        ent_spec = EntitySpec(
            id=t_name,
            type=t_name,
            attributes={
                "columns": t_schema.columns,
                "primary_keys": t_schema.primary_keys,
                "not_null_columns": t_schema.not_null_columns,
                "asserted_schema": True,
            },
            tags=["relational_entity", "asserted_schema"],
        )
        entities.append(ent_spec)

        # Detect candidate continuous resources
        for col_name, col_type in t_schema.columns.items():
            is_numeric = any(
                n in col_type for n in ("INT", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC", "REAL")
            )
            is_keyword = any(kw in col_name.lower() for kw in resource_keywords)

            if is_numeric and is_keyword:
                res_id = f"{t_name}_{col_name}"
                res = ResourceSpec(
                    id=res_id,
                    current=100.0,
                    min_value=0.0,
                    max_value=1e9,
                    unit="units",
                    entity_id=t_name,
                    tags=["candidate_resource", "asserted_column"],
                )
                resources.append(res)

                # Candidate non-negativity constraint
                candidate_invariants.append(
                    CandidateInvariant(
                        name=f"non_negative_{res_id}",
                        target_resource=res_id,
                        invariant_type="non_negativity",
                        equation=f"{res_id} >= 0",
                        support_ratio=1.0,
                        evidence_level=EvidenceLevel.STRUCTURAL,
                        parameters={"resource_id": res_id},
                    )
                )

        # Candidate constraints from SQL CHECK
        for chk in t_schema.check_constraints:
            first_word = chk.split()[0] if chk.split() else str(len(candidate_invariants))
            candidate_invariants.append(
                CandidateInvariant(
                    name=f"check_{t_name}_{first_word}",
                    target_resource=t_name,
                    invariant_type="sql_check",
                    equation=chk,
                    support_ratio=1.0,
                    evidence_level=EvidenceLevel.STRUCTURAL,
                    parameters={"table": t_name, "check": chk},
                )
            )

        # 2. Build Declared FK Relationships (ASSERTED Ground Truth)
        for fk in t_schema.foreign_keys:
            rel = RelationshipSpec(
                source=t_name,
                target=fk["to_table"],
                type=f"fk_{fk['from_col']}_to_{fk['to_col']}",
                attributes={
                    "evidence_level": EvidenceLevel.STRUCTURAL.value,
                    "asserted": True,
                    "from_column": fk["from_col"],
                    "to_column": fk["to_col"],
                },
            )
            relationships.append(rel)

    # 3. Propose Latent Links (Relational Deep Learning)
    provisional_links: list[ProvisionalLink] = []
    if link_prediction:
        provisional_links = _predict_latent_links(tables, threshold=link_threshold)
        for pl in provisional_links:
            if pl.clears_gate:
                # Add provisionally to relationships with explicit tag
                relationships.append(
                    RelationshipSpec(
                        source=pl.source_entity,
                        target=pl.target_entity,
                        type=pl.relationship_type,
                        attributes={
                            "evidence_level": pl.evidence_level.value,
                            "inferred_link": True,
                            "confidence": pl.confidence,
                            "provisional": True,
                        },
                    )
                )

    spec = WorldSpec(
        world=WorldMetadataSpec(
            name=name,
            description=f"Induced typed-graph skeleton from {len(tables)} relational tables",
        ),
        entities=entities,
        relationships=relationships,
        resources=resources,
        constraints=constraints,
    )

    report = InductionReport(
        source=name,
        source_type="relational_schema",
        seed=seed,
        entities_count=len(spec.entities),
        relationships_count=len(spec.relationships),
        resources_count=len(spec.resources),
        constraints_count=len(candidate_invariants),
        evidence_distribution={
            EvidenceLevel.STRUCTURAL.value: len(
                [r for r in spec.relationships if not r.attributes.get("inferred_link")]
            ),
            EvidenceLevel.PREDICTIVE.value: len(
                [r for r in spec.relationships if r.attributes.get("inferred_link")]
            ),
        },
        candidate_invariants=candidate_invariants,
        provisional_links=provisional_links,
        metadata={"table_count": len(tables), "tables": sorted(tables.keys())},
    )

    return spec, report
