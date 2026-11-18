"""Verification suite for Prompt E8: Scientific Evaluation, Baselines, and Benchmarks (EnterpriseOps-Gym).

Verifies:
1. `MCPClientAdapter` simulating the EnterpriseOps-Gym environment and routing tasks
   through the E3 Continual Discovery Agent (`RetrieveAndReasonEngine`) pipeline.
2. Execution of the mock task: `"Approve a purchase order exceeding $10k."`, confirming
   retrieval of the held-out secondary approval rule (`HELD-OUT-RULE-PO-10K-SECONDARY`),
   state mutation, OCEL 2.0 SQLite persistence, and objective verification via a `SELECT`
   statement on the SQLite database.
3. Execution of the representative 5-task EnterpriseOps-Gym sample, outputting a dictionary
   with `"Avg Success Rate"` and `"Avg Verifier Pass"` equal to `1.0` based strictly on
   final SQLite environment state.
4. Zero training/evaluation leakage: confirming held-out business rules are absent from
   all system prompt definitions.
5. Operational safeguards: `ReadOnlySQLVerifier` runs strictly in read-only mode (`?mode=ro`
   + `PRAGMA query_only = ON`) and rejects any `UPDATE`, `DELETE`, `DROP`, `INSERT`, or
   multi-statement SQL query with `ReadOnlySQLVerifierError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ewm_engine.core.dynamics import BusinessRule
from ewm_engine.core.ocel import validate_ocel2_sqlite
from ewm_engine.core.types import EngineError
from ewm_engine.evaluation.enterpriseops_gym import (
    EnterpriseOpsAgentEndpoint,
    EnterpriseOpsGymError,
    EnterpriseOpsTaskSpec,
    MCPClientAdapter,
    MCPServerProtocol,
    ReadOnlySQLVerifier,
    ReadOnlySQLVerifierError,
    build_representative_5_task_sample,
    run_enterpriseops_gym_benchmark,
    verify_zero_held_out_rule_leakage,
)


def test_approve_purchase_order_exceeding_10k_via_mcp_and_sql_verifier(
    tmp_path: Path,
) -> None:
    """Run mock task 'Approve a purchase order exceeding $10k.' and verify via SQL SELECT."""
    db_path = tmp_path / "po_10k_eval.sqlite"
    adapter = MCPClientAdapter(db_path=db_path)
    endpoint = EnterpriseOpsAgentEndpoint(mcp_adapter=adapter)

    tools = adapter.list_tools()
    assert len(tools) == 3

    passed = endpoint.execute_task("Approve a purchase order exceeding $10k.")
    assert passed is True

    report = endpoint.last_report
    assert report is not None
    assert report.task_id == "task_1_po_over_10k"
    assert report.action_Success is True
    assert report.sql_verifier_passed is True
    assert report.retrieved_rule_ids == ("HELD-OUT-RULE-PO-10K-SECONDARY",)
    assert report.prompt_latency_ms >= 0.0

    # Verify via direct read-only SELECT statement on the OCEL 2.0 SQLite database
    verifier = ReadOnlySQLVerifier(db_path)
    rows = verifier.execute_select(
        'SELECT "ent_status", "ent_secondary_approval", "res_budget_reserve", "res_encumbered_funds" '
        'FROM "event_approve_purchase_order" WHERE "task_id" = ?',
        params=("task_1_po_over_10k",),
    )
    assert len(rows) == 1
    assert rows[0] == (
        "approved_with_secondary_signoff",
        "vp_finance_verified",
        35000.0,
        15000.0,
    )

    # Confirm the underlying OCEL 2.0 SQLite database satisfies all 24 relational constraints
    sat, unsat = validate_ocel2_sqlite(db_path)
    assert len(sat) == 24
    assert unsat == []


def test_representative_5_task_benchmark_suite_and_zero_leakage(
    tmp_path: Path,
) -> None:
    """Run the 5-task EnterpriseOps-Gym sample; assert Avg Success Rate and Avg Verifier Pass."""
    tasks = build_representative_5_task_sample()
    assert len(tasks) == 5
    assert verify_zero_held_out_rule_leakage(tasks) is True

    db_path = tmp_path / "benchmark_5_tasks.sqlite"
    summary = run_enterpriseops_gym_benchmark(tasks=tasks, db_path=db_path)

    assert "Avg Success Rate" in summary
    assert "Avg Verifier Pass" in summary
    assert summary["Avg Success Rate"] == pytest.approx(1.0)
    assert summary["Avg Verifier Pass"] == pytest.approx(1.0)
    assert summary["Total Tasks"] == 5
    assert summary["SQL Verifier Failures"] == []
    assert summary["Zero Leakage Verified"] is True
    assert len(summary["Task Results"]) == 5


def test_read_only_sql_verifier_blocks_mutations_and_corruptions(
    tmp_path: Path,
) -> None:
    """Operational safeguard: ReadOnlySQLVerifier blocks all non-SELECT and write operations."""
    assert issubclass(EnterpriseOpsGymError, EngineError)
    assert issubclass(ReadOnlySQLVerifierError, EnterpriseOpsGymError)

    db_path = tmp_path / "readonly_guard.sqlite"
    adapter = MCPClientAdapter(db_path=db_path)
    endpoint = EnterpriseOpsAgentEndpoint(mcp_adapter=adapter)
    assert endpoint.execute_task("Approve a purchase order exceeding $10k.") is True

    verifier = ReadOnlySQLVerifier(db_path)

    for forbidden_sql in (
        'DROP TABLE "event";',
        'DELETE FROM "event";',
        'UPDATE "event" SET "ocel_type" = "hacked";',
        'INSERT INTO "event" ("ocel_id", "ocel_type") VALUES ("x", "y");',
        'SELECT * FROM "event"; DELETE FROM "event";',
        "PRAGMA writable_schema = ON;",
    ):
        with pytest.raises(ReadOnlySQLVerifierError):
            verifier.execute_select(forbidden_sql)

    # Confirm the event table is completely intact after blocked mutation attempts
    rows = verifier.execute_select('SELECT Count(*) FROM "event"')
    assert rows == [(1,)]


def test_custom_researcher_mcp_server_protocol_integration(tmp_path: Path) -> None:
    """Verify researchers can plug in a custom MCPServerProtocol implementation."""

    class _MockResearchMCPServer(MCPServerProtocol):
        def list_tools(self) -> List[Dict[str, Any]]:
            return [{"name": "custom_ping", "description": "Researcher custom MCP tool"}]

        def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
            return {"tool": name, "echo": dict(arguments)}

    custom_srv = _MockResearchMCPServer()
    assert isinstance(custom_srv, MCPServerProtocol)

    adapter = MCPClientAdapter(db_path=tmp_path / "custom_mcp.sqlite", custom_server=custom_srv)
    assert adapter.list_tools() == [
        {"name": "custom_ping", "description": "Researcher custom MCP tool"}
    ]
    assert adapter.call_tool("custom_ping", {"k": "v"}) == {
        "tool": "custom_ping",
        "echo": {"k": "v"},
    }

    # Verify leakage detector catches leaked rule in system prompt
    tasks = build_representative_5_task_sample()
    leaked_task = EnterpriseOpsTaskSpec(
        task_id="leaked",
        task_prompt="leaked",
        action_type="noop",
        action_parameters={},
        initial_state=tasks[0].initial_state,
        held_out_rules=(
            BusinessRule(
                rule_id="Continual Discovery Agent",
                action_types=("noop",),
                rule_text="noop",
            ),
        ),
        target_object_id="po_1001",
        target_object_type="PurchaseOrder",
        sql_verifier_query='SELECT 1 FROM "event"',
    )
    with pytest.raises(EnterpriseOpsGymError, match="Evaluation leakage detected"):
        verify_zero_held_out_rule_leakage([leaked_task])
