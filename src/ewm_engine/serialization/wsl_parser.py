"""Lark-based Domain-Specific Language (DSL) compiler for World Specification Language (WSL).

Provides a safe, bounded-time LALR(1) parser for authored .wsl documents,
compiling directly to typed Pydantic WSLDocument models without Python code execution (eval/exec).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ewm_engine.exceptions import SerializationError
from ewm_engine.serialization.wsl import (
    WSLConstraintSpec,
    WSLDocument,
    WSLEntitySpec,
    WSLMetadata,
    WSLRelationshipSpec,
    WSLResourceSpec,
    WSLScenarioSpec,
    WSLTemporalConfig,
)

WSL_LARK_GRAMMAR = r"""
?start: world_block

world_block: "world" [STRING] "{" statement* "}"

?statement: metadata_stmt
          | temporal_block
          | entity_block
          | resource_block
          | relationship_block
          | constraint_block
          | scenario_block

metadata_stmt: "id" ":" STRING -> set_id
             | "version" ":" STRING -> set_version
             | "description" ":" STRING -> set_description
             | "author" ":" STRING -> set_author

temporal_block: "temporal" "{" temporal_stmt* "}"
?temporal_stmt: "time_unit" ":" STRING -> set_time_unit
              | "step_duration" ":" NUMBER -> set_step_duration
              | "default_horizon" ":" NUMBER -> set_default_horizon

entity_block: "entity" CNAME "{" entity_stmt* "}"
?entity_stmt: "type" ":" STRING -> set_entity_type
            | "tags" ":" "[" [STRING ("," STRING)*] "]" -> set_entity_tags
            | CNAME ":" value -> set_entity_attr

resource_block: "resource" CNAME "{" resource_stmt* "}"
?resource_stmt: "entity" ":" STRING -> set_resource_entity
              | "current" ":" NUMBER -> set_resource_current
              | "min" ":" NUMBER -> set_resource_min
              | "max" ":" NUMBER -> set_resource_max
              | "unit" ":" STRING -> set_resource_unit

relationship_block: "relationship" [CNAME] "{" rel_stmt* "}"
?rel_stmt: "source" ":" STRING -> set_rel_source
         | "target" ":" STRING -> set_rel_target
         | "type" ":" STRING -> set_rel_type
         | "weight" ":" NUMBER -> set_rel_weight

constraint_block: "constraint" [CNAME] "{" constraint_stmt* "}"
?constraint_stmt: "type" ":" STRING -> set_constraint_type
                | "severity" ":" STRING -> set_constraint_severity
                | "parameters" "{" param_stmt* "}" -> set_constraint_params

param_stmt: CNAME ":" value

scenario_block: "scenario" CNAME "{" scenario_stmt* "}"
?scenario_stmt: "horizon" ":" NUMBER -> set_scen_horizon
              | "samples" ":" NUMBER -> set_scen_samples
              | "seed" ":" NUMBER -> set_scen_seed

?value: STRING -> str_val
      | NUMBER -> num_val
      | "true" -> true_val
      | "false"-> false_val

%import common.CNAME
%import common.ESCAPED_STRING -> STRING
%import common.SIGNED_NUMBER -> NUMBER
%import common.INT
%import common.WS
%import common.SH_COMMENT
%ignore WS
%ignore SH_COMMENT
"""


def is_lark_available() -> bool:
    """Check whether Lark parser library is installed."""
    return importlib.util.find_spec("lark") is not None


class WSLTransformer:
    """Transforms Lark parse tree into a typed Pydantic WSLDocument."""

    def __init__(self) -> None:
        pass

    def _strip_quotes(self, token_str: str) -> str:
        s = str(token_str)
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def transform(self, tree: Any) -> WSLDocument:
        meta_id = "enterprise_world"
        meta_name = "Enterprise World"
        meta_version = "2.0.0"
        meta_desc = ""
        meta_author = ""

        temporal = WSLTemporalConfig()
        entities: list[WSLEntitySpec] = []
        resources: list[WSLResourceSpec] = []
        relationships: list[WSLRelationshipSpec] = []
        constraints: list[WSLConstraintSpec] = []
        scenarios: list[WSLScenarioSpec] = []

        if tree.data == "world_block":
            for child in tree.children:
                if child is None:
                    continue
                if hasattr(child, "type") and child.type == "STRING":
                    meta_name = self._strip_quotes(child.value)
                elif hasattr(child, "data"):
                    d = child.data
                    if d == "set_id":
                        meta_id = self._strip_quotes(child.children[0].value)
                    elif d == "set_version":
                        meta_version = self._strip_quotes(child.children[0].value)
                    elif d == "set_description":
                        meta_desc = self._strip_quotes(child.children[0].value)
                    elif d == "set_author":
                        meta_author = self._strip_quotes(child.children[0].value)
                    elif d == "temporal_block":
                        t_kwargs: dict[str, Any] = {}
                        for t_child in child.children:
                            if t_child.data == "set_time_unit":
                                t_kwargs["time_unit"] = self._strip_quotes(
                                    t_child.children[0].value
                                )
                            elif t_child.data == "set_step_duration":
                                t_kwargs["step_duration"] = float(t_child.children[0].value)
                            elif t_child.data == "set_default_horizon":
                                t_kwargs["default_horizon"] = int(t_child.children[0].value)
                        temporal = WSLTemporalConfig(**t_kwargs)
                    elif d == "entity_block":
                        ent_id = str(child.children[0].value)
                        ent_type = "entity"
                        ent_tags: list[str] = []
                        ent_attrs: dict[str, Any] = {}
                        for stmt in child.children[1:]:
                            if stmt.data == "set_entity_type":
                                ent_type = self._strip_quotes(stmt.children[0].value)
                            elif stmt.data == "set_entity_tags":
                                ent_tags = [
                                    self._strip_quotes(tok.value)
                                    for tok in stmt.children
                                    if hasattr(tok, "value")
                                ]
                            elif stmt.data == "set_entity_attr":
                                attr_key = str(stmt.children[0].value)
                                attr_val = self._parse_val(stmt.children[1])
                                ent_attrs[attr_key] = attr_val
                        entities.append(
                            WSLEntitySpec(
                                id=ent_id, type=ent_type, attributes=ent_attrs, tags=ent_tags
                            )
                        )
                    elif d == "resource_block":
                        res_id = str(child.children[0].value)
                        res_entity = ""
                        res_curr = 0.0
                        res_min = 0.0
                        res_max = 1e9
                        res_unit = ""
                        for stmt in child.children[1:]:
                            if stmt.data == "set_resource_entity":
                                res_entity = self._strip_quotes(stmt.children[0].value)
                            elif stmt.data == "set_resource_current":
                                res_curr = float(stmt.children[0].value)
                            elif stmt.data == "set_resource_min":
                                res_min = float(stmt.children[0].value)
                            elif stmt.data == "set_resource_max":
                                res_max = float(stmt.children[0].value)
                            elif stmt.data == "set_resource_unit":
                                res_unit = self._strip_quotes(stmt.children[0].value)
                        resources.append(
                            WSLResourceSpec(
                                id=res_id,
                                current=res_curr,
                                min_value=res_min,
                                max_value=res_max,
                                unit=res_unit,
                                entity_id=res_entity or None,
                            )
                        )
                    elif d == "relationship_block":
                        rel_src = ""
                        rel_tgt = ""
                        rel_type = "connected"
                        rel_wt = 1.0
                        for stmt in child.children:
                            if hasattr(stmt, "data"):
                                if stmt.data == "set_rel_source":
                                    rel_src = self._strip_quotes(stmt.children[0].value)
                                elif stmt.data == "set_rel_target":
                                    rel_tgt = self._strip_quotes(stmt.children[0].value)
                                elif stmt.data == "set_rel_type":
                                    rel_type = self._strip_quotes(stmt.children[0].value)
                                elif stmt.data == "set_rel_weight":
                                    rel_wt = float(stmt.children[0].value)
                        relationships.append(
                            WSLRelationshipSpec(
                                source=rel_src, target=rel_tgt, type=rel_type, weight=rel_wt
                            )
                        )
                    elif d == "constraint_block":
                        c_id = (
                            str(child.children[0].value)
                            if child.children
                            and hasattr(child.children[0], "value")
                            and child.children[0].value
                            else f"constraint_{len(constraints)}"
                        )
                        c_type = ""
                        c_sev = "HARD"
                        c_params: dict[str, Any] = {}
                        for stmt in child.children:
                            if hasattr(stmt, "data"):
                                if stmt.data == "set_constraint_type":
                                    c_type = self._strip_quotes(stmt.children[0].value)
                                elif stmt.data == "set_constraint_severity":
                                    c_sev = self._strip_quotes(stmt.children[0].value).upper()
                                elif stmt.data == "set_constraint_params":
                                    for p in stmt.children:
                                        p_k = str(p.children[0].value)
                                        p_v = self._parse_val(p.children[1])
                                        c_params[p_k] = p_v
                        constraints.append(
                            WSLConstraintSpec(
                                id=c_id, type_id=c_type, severity=c_sev, parameters=c_params
                            )
                        )
                    elif d == "scenario_block":
                        scen_id = str(child.children[0].value)
                        scen_h = 10
                        scen_seed = 42
                        for stmt in child.children[1:]:
                            if stmt.data == "set_scen_horizon":
                                scen_h = int(float(stmt.children[0].value))
                            elif stmt.data == "set_scen_seed":
                                scen_seed = int(float(stmt.children[0].value))
                        scenarios.append(
                            WSLScenarioSpec(
                                id=scen_id, name=scen_id, horizon=scen_h, seed=scen_seed
                            )
                        )

        metadata = WSLMetadata(
            id=meta_id,
            name=meta_name,
            version=meta_version,
            description=meta_desc,
            author=meta_author,
        )

        return WSLDocument(
            schema_version="wsl/2.0.0",
            metadata=metadata,
            temporal=temporal,
            entities=entities,
            resources=resources,
            relationships=relationships,
            constraints=constraints,
            scenarios=scenarios,
        )

    def _parse_val(self, node: Any) -> Any:
        if not hasattr(node, "data"):
            val_str = str(getattr(node, "value", node))
            return self._strip_quotes(val_str)
        d = node.data
        if d == "str_val":
            return self._strip_quotes(node.children[0].value)
        elif d == "num_val":
            return float(node.children[0].value)
        elif d == "int_val":
            return int(node.children[0].value)
        elif d == "true_val":
            return True
        elif d == "false_val":
            return False
        return str(node)


def parse_wsl_authored(dsl_text: str) -> WSLDocument:
    """Parse an authored .wsl DSL string using the Lark LALR(1) parser.

    Parameters:
        dsl_text: Content of the authored .wsl document.

    Returns:
        Validated WSLDocument.

    Raises:
        SerializationError: On syntax errors or missing Lark parser.
    """
    if not is_lark_available():
        raise SerializationError(
            "Lark parser is required to compile authored .wsl files. "
            "Install via `pip install ewm-engine[dsl]`."
        )

    from lark import Lark, UnexpectedInput

    try:
        parser = Lark(WSL_LARK_GRAMMAR, parser="lalr", lexer="basic")
        tree = parser.parse(dsl_text)
        transformer = WSLTransformer()
        return transformer.transform(tree)
    except UnexpectedInput as exc:
        raise SerializationError(
            f"WSL DSL Syntax Error at line {exc.line}, col {exc.column}: {exc}"
        ) from exc
    except Exception as exc:
        raise SerializationError(f"Failed to parse WSL DSL document: {exc}") from exc
