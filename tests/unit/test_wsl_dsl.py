"""Unit tests for WSL DSL Lark parser and Language Server Protocol diagnostics."""

from __future__ import annotations

import pytest

from ewm_engine.exceptions import SerializationError
from ewm_engine.serialization.lsp_server import (
    create_wsl_server,
    is_pygls_available,
    validate_wsl_text,
)
from ewm_engine.serialization.wsl_parser import parse_wsl_authored

VALID_WSL_SNIPPET = """
world "DistributionNetwork" {
    id: "dist_net_01"
    version: "2.0.0"
    description: "Regional distribution world"

    temporal {
        time_unit: "day"
        step_duration: 1.0
        default_horizon: 10
    }

    entity factory_a {
        type: "factory"
        tier: 1
    }

    entity depot_b {
        type: "depot"
        tier: 2
    }

    resource stock_a {
        entity: "factory_a"
        current: 500.0
        min: 0.0
        max: 2000.0
        unit: "pallets"
    }

    relationship supply_line {
        source: "factory_a"
        target: "depot_b"
        type: "freight"
        weight: 1.0
    }

    constraint capacity_guard {
        type: "ResourceCapacityConstraint"
        severity: "HARD"
        parameters {
            resource_id: "stock_a"
        }
    }

    scenario base_rollout {
        horizon: 10
        samples: 5
        seed: 42
    }
}
"""


def test_wsl_lark_parser_compiles_valid_dsl() -> None:
    """Verify Lark LALR(1) parser compiles valid authored DSL into WSLDocument."""
    doc = parse_wsl_authored(VALID_WSL_SNIPPET)

    assert doc.schema_version == "wsl/2.0.0"
    assert doc.metadata.name == "DistributionNetwork"
    assert doc.metadata.id == "dist_net_01"
    assert doc.temporal.default_horizon == 10
    assert len(doc.entities) == 2
    assert len(doc.resources) == 1
    assert len(doc.relationships) == 1
    assert len(doc.constraints) == 1
    assert len(doc.scenarios) == 1

    assert doc.entities[0].id == "factory_a"
    assert doc.resources[0].current == 500.0
    assert doc.constraints[0].type_id == "ResourceCapacityConstraint"


def test_wsl_lark_parser_raises_on_syntax_error() -> None:
    """Verify syntax error in authored WSL document raises SerializationError with line/col."""
    invalid_dsl = "world { invalid_keyword_without_block }"

    with pytest.raises(SerializationError, match="WSL DSL Syntax Error"):
        parse_wsl_authored(invalid_dsl)


def test_wsl_lsp_diagnostics_validation() -> None:
    """Verify live diagnostics emitter detects syntax and semantic errors."""
    # 1. Valid text produces 0 diagnostics
    diags_valid = validate_wsl_text(VALID_WSL_SNIPPET)
    assert len(diags_valid) == 0

    # 2. Syntax error produces Error diagnostic (severity 1)
    syntax_err = "world { broken syntax without brackets"
    diags_syntax = validate_wsl_text(syntax_err)
    assert len(diags_syntax) > 0
    assert diags_syntax[0]["severity"] == 1

    # 3. Semantic error: dangling relationship entity produces Warning (severity 2)
    dangling_dsl = """
    world {
        entity depot_1 {
            type: "depot"
        }
        relationship rel_1 {
            source: "depot_1"
            target: "non_existent_depot"
        }
    }
    """
    diags_dangling = validate_wsl_text(dangling_dsl)
    assert any("non_existent_depot" in d["message"] for d in diags_dangling)

    # 4. Semantic error: inverted resource bounds produces Error (severity 1)
    inverted_res_dsl = """
    world {
        entity depot_1 {
            type: "depot"
        }
        resource broken_res {
            entity: "depot_1"
            current: 100.0
            min: 500.0
            max: 200.0
        }
    }
    """
    diags_inverted = validate_wsl_text(inverted_res_dsl)
    assert any("exceeds max_value" in d["message"] for d in diags_inverted)


def test_wsl_lsp_server_factory() -> None:
    """Verify LSP server factory initializes or raises appropriate ImportError."""
    if is_pygls_available():
        server = create_wsl_server()
        assert server is not None
    else:
        with pytest.raises(ImportError, match="pygls"):
            create_wsl_server()
