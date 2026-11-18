"""EnterpriseOps-Gym Scientific Evaluation Harness, MCP Client Adapter & Read-Only SQL Oracle (E8).

Integrates the Enterprise World Model (EWM) Engine with an EnterpriseOps-Gym style
execution benchmark, routing tasks through the E3 Continual Discovery Agent (CDA)
retrieve-then-reason pipeline (`RetrieveAndReasonEngine`), persisting state transitions
into the E2 OCEL 2.0 relational SQLite database (`StateEncoder`), and evaluating pass/fail
objectively via read-only SQL oracle queries (`ReadOnlySQLVerifier`) rather than brittle
textual string matching.

How Researchers Can Bring Their Own MCP Server (E8 Requirement 12)
------------------------------------------------------------------
Researchers can connect any external or custom Model Context Protocol (MCP) server to
`MCPClientAdapter` by implementing `MCPServerProtocol` (or passing an MCP SDK client
session wrapped with `list_tools()` and `call_tool(name, arguments)`):

```python
from pathlib import Path
from typing import Any, Dict, List
from ewm_engine.evaluation.enterpriseops_gym import (
    EnterpriseOpsAgentEndpoint,
    MCPClientAdapter,
    MCPServerProtocol,
)

class CustomResearchMCPServer(MCPServerProtocol):
    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "get_task_spec", "description": "Fetch task state, action, and verifier"},
            {"name": "retrieve_business_rules", "description": "Retrieve held-out rules for action"},
            {"name": "commit_state_transition", "description": "Persist mutated state to OCEL 2.0 SQLite"},
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # Dispatch tool calls to your live MCP server or EnterpriseOps-Gym environment
        ...

adapter = MCPClientAdapter(db_path=Path("eval_state.sqlite"), custom_server=CustomResearchMCPServer())
endpoint = EnterpriseOpsAgentEndpoint(mcp_adapter=adapter)
passed: bool = endpoint.execute_task("Approve a purchase order exceeding $10k.")
```
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from ewm_engine.core.actions import Action
from ewm_engine.core.dynamics import (
    BusinessRule,
    CDAPromptHarness,
    ConfigurationDatastore,
    RetrieveAndReasonEngine,
    StateDiff,
)
from ewm_engine.core.entities import Entity
from ewm_engine.core.ocel import OcelEventRecord, OcelObjectTarget, StateEncoder
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import EngineError

_FORBIDDEN_SQL_MUTATION_RE = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|VACUUM|PRAGMA)\b",
    re.IGNORECASE,
)


class EnterpriseOpsGymError(EngineError):
    """Base exception for EnterpriseOps-Gym benchmark adapter and verification errors."""


class ReadOnlySQLVerifierError(EnterpriseOpsGymError):
    """Raised when a SQL verifier query attempts a write operation or fails read-only execution."""


@runtime_checkable
class MCPServerProtocol(Protocol):
    """Protocol for connecting custom or external MCP servers to `MCPClientAdapter`."""

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions exposed by the server."""
        ...

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Invoke an MCP tool by `name` with structured `arguments`."""
        ...


@dataclass(frozen=True)
class EnterpriseOpsTaskSpec:
    """Specification of a single held-out EnterpriseOps-Gym benchmark task.

    All `held_out_rules` are strictly isolated from system prompt definitions to
    guarantee zero training/prompt-engineering leakage (E8 Requirement 8).
    """

    task_id: str
    task_prompt: str
    action_type: str
    action_parameters: Dict[str, Any]
    initial_state: WorldState
    held_out_rules: Tuple[BusinessRule, ...]
    target_object_id: str
    target_object_type: str
    sql_verifier_query: str
    sql_verifier_params: Tuple[Any, ...] = ()
    expected_sql_row: Tuple[Any, ...] = (1,)


class ReadOnlySQLVerifier:
    """Objective SQL state verifier executing queries strictly in SQLite read-only mode (E8 Req 11).

    Operational Safeguards:
    1. Rejects any SQL query string that does not start with `SELECT` or contains mutation
       keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, etc.).
    2. Opens the SQLite database using the URI `?mode=ro` flag (`SQLITE_OPEN_READONLY`) so the
       OS/SQLite engine blocks any write attempt at the file-descriptor level.
    3. Executes `PRAGMA query_only = ON;` on the connection before running the `SELECT` query.
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path: Path = Path(db_path)

    @staticmethod
    def validate_select_only_sql(sql_query: str) -> str:
        """Ensure `sql_query` is a single read-only `SELECT` statement."""
        cleaned = sql_query.strip()
        if not cleaned:
            raise ReadOnlySQLVerifierError("SQL verifier query cannot be empty.")
        # Strip trailing semicolon if present, then forbid multiple statements
        body = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
        if ";" in body:
            raise ReadOnlySQLVerifierError(
                "Multi-statement SQL queries are forbidden in ReadOnlySQLVerifier."
            )
        if not body.upper().startswith("SELECT"):
            raise ReadOnlySQLVerifierError(
                f"ReadOnlySQLVerifier permits only SELECT queries; got: {cleaned!r}"
            )
        if _FORBIDDEN_SQL_MUTATION_RE.search(body):
            raise ReadOnlySQLVerifierError(
                f"Forbidden mutation keyword detected in SQL verifier query: {cleaned!r}"
            )
        return cleaned

    def execute_select(
        self,
        sql_query: str,
        params: Sequence[Any] = (),
    ) -> List[Tuple[Any, ...]]:
        """Execute a validated read-only `SELECT` query against the OCEL 2.0 SQLite database."""
        validated_sql = self.validate_select_only_sql(sql_query)
        if not self.db_path.exists():
            raise ReadOnlySQLVerifierError(
                f"Benchmark OCEL 2.0 SQLite database does not exist: {self.db_path}"
            )

        ro_uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(ro_uri, uri=True)
        try:
            conn.execute("PRAGMA query_only = ON;")
            cur = conn.execute(validated_sql, tuple(params))
            return list(cur.fetchall())
        except sqlite3.Error as exc:
            raise ReadOnlySQLVerifierError(
                f"Read-only SQL verifier execution failed: {exc}"
            ) from exc
        finally:
            conn.close()

    def verify(
        self,
        sql_query: str,
        params: Sequence[Any] = (),
        *,
        expected_first_row: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[bool, List[Tuple[Any, ...]]]:
        """Run the read-only `SELECT` query and return `(passed, rows)`."""
        rows = self.execute_select(sql_query, params=params)
        if not rows:
            return False, rows
        if expected_first_row is not None:
            actual_first = rows[0]
            if len(actual_first) != len(expected_first_row):
                return False, rows
            for act_val, exp_val in zip(actual_first, expected_first_row, strict=True):
                if isinstance(exp_val, float) and isinstance(act_val, (int, float)):
                    if abs(float(act_val) - exp_val) > 1e-6:
                        return False, rows
                elif act_val != exp_val:
                    return False, rows
            return True, rows
        return bool(rows[0][0]), rows


class MCPClientAdapter:
    """Model Context Protocol (MCP) Client Adapter simulating the EnterpriseOps-Gym environment (E8).

    Manages:
    - Held-out task specifications and runtime `ConfigurationDatastore` business rules,
    - Tool discovery (`list_tools()`) and tool execution (`call_tool()`),
    - Persistence of CDA-mutated `WorldState` transitions into an E2 OCEL 2.0 SQLite database, and
    - Optional delegation to a researcher-supplied `MCPServerProtocol` implementation.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        *,
        tasks: Optional[Sequence[EnterpriseOpsTaskSpec]] = None,
        custom_server: Optional[MCPServerProtocol] = None,
    ) -> None:
        self.db_path: Path = Path(db_path)
        self.custom_server: Optional[MCPServerProtocol] = custom_server
        self._tasks_by_prompt: Dict[str, EnterpriseOpsTaskSpec] = {}
        self._tasks_by_id: Dict[str, EnterpriseOpsTaskSpec] = {}
        self.datastore: ConfigurationDatastore = ConfigurationDatastore()

        for task in tasks or build_representative_5_task_sample():
            self.register_task(task)

        # Ensure OCEL 2.0 SQLite schema is initialized
        with StateEncoder(self.db_path):
            pass

    def register_task(self, task: EnterpriseOpsTaskSpec) -> None:
        """Register an EnterpriseOps-Gym task and its held-out business rules."""
        self._tasks_by_prompt[task.task_prompt.strip().lower()] = task
        self._tasks_by_id[task.task_id] = task
        for rule in task.held_out_rules:
            self.datastore.register_rule(rule)

    def resolve_task(self, task_input: str) -> EnterpriseOpsTaskSpec:
        """Resolve a task specification by natural-language prompt or `task_id`."""
        key = task_input.strip().lower()
        if key in self._tasks_by_prompt:
            return self._tasks_by_prompt[key]
        if task_input.strip() in self._tasks_by_id:
            return self._tasks_by_id[task_input.strip()]
        for prompt_key, spec in self._tasks_by_prompt.items():
            if prompt_key in key or key in prompt_key:
                return spec
        raise EnterpriseOpsGymError(f"Unknown EnterpriseOps-Gym task: {task_input!r}")

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools in the EnterpriseOps-Gym environment."""
        if self.custom_server is not None:
            return list(self.custom_server.list_tools())
        return [
            {
                "name": "get_task_context",
                "description": "Retrieve initial state and proposed action for an EnterpriseOps-Gym task.",
            },
            {
                "name": "retrieve_held_out_rules",
                "description": "Query the runtime ConfigurationDatastore for held-out business rules.",
            },
            {
                "name": "commit_ocel_transition",
                "description": "Commit the post-transition WorldState and StateDiff into the OCEL 2.0 SQLite store.",
            },
        ]

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool call against the EnterpriseOps-Gym environment."""
        if self.custom_server is not None:
            return dict(self.custom_server.call_tool(name, arguments))

        if name == "get_task_context":
            task_str = str(arguments["task"])
            spec = self.resolve_task(task_str)
            return {
                "task_id": spec.task_id,
                "task_prompt": spec.task_prompt,
                "action_type": spec.action_type,
                "action_parameters": dict(spec.action_parameters),
                "initial_state": spec.initial_state,
                "target_object_id": spec.target_object_id,
                "target_object_type": spec.target_object_type,
                "sql_verifier_query": spec.sql_verifier_query,
                "sql_verifier_params": spec.sql_verifier_params,
                "expected_sql_row": spec.expected_sql_row,
            }

        if name == "retrieve_held_out_rules":
            act_type = str(arguments["action_type"])
            matched = self.datastore.retrieve_rules(act_type)
            return {
                "rules": matched,
                "rule_ids": [r.rule_id for r in matched],
            }

        if name == "commit_ocel_transition":
            task_id = str(arguments["task_id"])
            action_type = str(arguments["action_type"])
            next_state: WorldState = arguments["next_state"]
            diff: StateDiff = arguments["state_diff"]
            target_oid = str(arguments["target_object_id"])
            target_otype = str(arguments["target_object_type"])

            entity_attrs: Dict[str, Any] = {}
            if target_oid in next_state.entities:
                entity_attrs.update(next_state.entities[target_oid].attributes)

            event_attrs: Dict[str, Any] = {
                "task_id": task_id,
                "is_noop": int(diff.is_noop),
                "attributed_rules": ",".join(a.rule_id for a in diff.rule_attributions),
            }
            for r_id, res_obj in next_state.resources.items():
                event_attrs[f"res_{r_id}"] = float(res_obj.current)
            for k, v in entity_attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    event_attrs[f"ent_{k}"] = v

            with StateEncoder(self.db_path) as encoder:
                ocel_rec = OcelEventRecord(
                    event_id=f"ev_{task_id}_step_{next_state.step}",
                    event_type=action_type,
                    timestamp=f"2026-11-17T14:00:{min(59, int(next_state.step)):02d}Z",
                    objects=(
                        OcelObjectTarget(
                            object_id=target_oid,
                            object_type=target_otype,
                            qualifier="target",
                            attributes=entity_attrs,
                        ),
                    ),
                    attributes=event_attrs,
                )
                encoder.encode_event(ocel_rec)
            return {"committed_event_id": ocel_rec.event_id, "status": "ok"}

        raise EnterpriseOpsGymError(f"Unsupported MCP tool name: {name!r}")


@dataclass(frozen=True)
class TaskExecutionReport:
    """Detailed execution telemetry and SQL verification outcome for a single benchmark task."""

    task_id: str
    task_prompt: str
    action_Success: bool
    sql_verifier_passed: bool
    prompt_latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_rule_ids: Tuple[str, ...]
    sql_rows: Tuple[Tuple[Any, ...], ...]


class EnterpriseOpsAgentEndpoint:
    """Standardized agent endpoint routing tasks through the E3 CDA pipeline and SQL verifier (E8).

    Accepts a textual task prompt, interacts with `MCPClientAdapter`, executes retrieve-then-reason
    dynamics via `RetrieveAndReasonEngine`, commits the mutated state to the OCEL 2.0 SQLite
    database, and returns the objective SQL state verification boolean.
    """

    def __init__(
        self,
        mcp_adapter: MCPClientAdapter,
        *,
        cda_engine: Optional[RetrieveAndReasonEngine] = None,
    ) -> None:
        self.mcp_adapter: MCPClientAdapter = mcp_adapter
        self.cda_engine: RetrieveAndReasonEngine = cda_engine or RetrieveAndReasonEngine(
            datastore=mcp_adapter.datastore
        )
        self.verifier: ReadOnlySQLVerifier = ReadOnlySQLVerifier(mcp_adapter.db_path)
        self._last_report: Optional[TaskExecutionReport] = None

    @property
    def last_report(self) -> Optional[TaskExecutionReport]:
        """Return the detailed `TaskExecutionReport` from the most recent `execute_task` call."""
        return self._last_report

    def execute_task(self, task_prompt: str) -> bool:
        """Execute a textual benchmark task via MCP + E3 CDA and return the SQL verification boolean."""
        ctx = self.mcp_adapter.call_tool("get_task_context", {"task": task_prompt})
        task_id = str(ctx["task_id"])
        action_type = str(ctx["action_type"])
        action_params: Dict[str, Any] = dict(ctx["action_parameters"])
        initial_state: WorldState = ctx["initial_state"]

        action = Action(
            id=f"act_{task_id}",
            type=action_type,
            actor_id="enterpriseops_cda_agent",
            parameters=action_params,
        )

        # Route through the E3 Continual Discovery Agent (RetrieveAndReasonEngine) pipeline
        next_state, diff = self.cda_engine.step(
            initial_state,
            action,
            retriever=self.mcp_adapter.datastore,
        )
        action_succeeded = not diff.is_noop and diff.parse_error is None

        # Persist resulting state transition into the OCEL 2.0 SQLite database via MCP tool
        self.mcp_adapter.call_tool(
            "commit_ocel_transition",
            {
                "task_id": task_id,
                "action_type": action_type,
                "next_state": next_state,
                "state_diff": diff,
                "target_object_id": ctx["target_object_id"],
                "target_object_type": ctx["target_object_type"],
            },
        )

        # Objective verification via ReadOnlySQLVerifier against the OCEL 2.0 SQLite database
        sql_passed, rows = self.verifier.verify(
            str(ctx["sql_verifier_query"]),
            params=ctx["sql_verifier_params"],
            expected_first_row=ctx["expected_sql_row"],
        )

        self._last_report = TaskExecutionReport(
            task_id=task_id,
            task_prompt=str(ctx["task_prompt"]),
            action_Success=action_succeeded,
            sql_verifier_passed=sql_passed,
            prompt_latency_ms=diff.latency_ms,
            prompt_tokens=diff.prompt_tokens,
            completion_tokens=diff.completion_tokens,
            retrieved_rule_ids=diff.retrieved_rule_ids,
            sql_rows=tuple(rows),
        )
        return bool(sql_passed)


def build_representative_5_task_sample() -> Tuple[EnterpriseOpsTaskSpec, ...]:
    """Construct the representative 5-task EnterpriseOps-Gym sample with held-out business rules.

    Includes the mandatory E8 task:
    1. `"Approve a purchase order exceeding $10k."` (Secondary VP approval & budget reserve rule)
    plus 4 cross-domain enterprise tasks (Treasury dual-control escrow, Warehouse capacity throttling,
    Microservice circuit-breaker isolation, and Cross-border wire compliance hold).
    """
    # Task 1 (Mandatory E8 Task): Approve a purchase order exceeding $10k
    state_t1 = WorldState(
        step=0,
        timestamp=1700000000.0,
        entities={
            "po_1001": Entity(
                id="po_1001",
                type="PurchaseOrder",
                attributes={
                    "amount": 15000.0,
                    "status": "pending_review",
                    "secondary_approval": "none",
                },
            )
        },
        resources={
            "budget_reserve": Resource(id="budget_reserve", current=50000.0, unit="usd"),
            "encumbered_funds": Resource(id="encumbered_funds", current=0.0, unit="usd"),
        },
    )
    rule_t1 = BusinessRule(
        rule_id="HELD-OUT-RULE-PO-10K-SECONDARY",
        action_types=("approve_purchase_order",),
        rule_text=(
            "IF po_amount > 10000.0 THEN "
            "resource:budget_reserve -= 15000.0; "
            "resource:encumbered_funds += 15000.0; "
            "entity:po_1001.status = 'approved_with_secondary_signoff'; "
            "entity:po_1001.secondary_approval = 'vp_finance_verified'"
        ),
        version="1.0.0",
        priority=100,
    )
    task_1 = EnterpriseOpsTaskSpec(
        task_id="task_1_po_over_10k",
        task_prompt="Approve a purchase order exceeding $10k.",
        action_type="approve_purchase_order",
        action_parameters={"po_id": "po_1001", "po_amount": 15000.0},
        initial_state=state_t1,
        held_out_rules=(rule_t1,),
        target_object_id="po_1001",
        target_object_type="PurchaseOrder",
        sql_verifier_query=(
            'SELECT "ent_status", "ent_secondary_approval", "res_budget_reserve", "res_encumbered_funds" '
            'FROM "event_approve_purchase_order" WHERE "task_id" = ?'
        ),
        sql_verifier_params=("task_1_po_over_10k",),
        expected_sql_row=(
            "approved_with_secondary_signoff",
            "vp_finance_verified",
            35000.0,
            15000.0,
        ),
    )

    # Task 2: Settle a high-value vendor invoice of $25k with dual-control treasury escrow
    state_t2 = WorldState(
        step=0,
        timestamp=1700000100.0,
        entities={
            "inv_2002": Entity(
                id="inv_2002",
                type="VendorInvoice",
                attributes={"settlement_state": "unpaid", "escrow_cleared": "false"},
            )
        },
        resources={
            "operating_cash": Resource(id="operating_cash", current=100000.0, unit="usd"),
            "escrow_hold": Resource(id="escrow_hold", current=25000.0, unit="usd"),
        },
    )
    rule_t2 = BusinessRule(
        rule_id="HELD-OUT-RULE-TREASURY-ESCROW-25K",
        action_types=("settle_vendor_invoice",),
        rule_text=(
            "IF invoice_amount >= 20000.0 THEN "
            "resource:escrow_hold -= 25000.0; "
            "entity:inv_2002.settlement_state = 'settled_via_dual_escrow'; "
            "entity:inv_2002.escrow_cleared = 'true'"
        ),
        version="1.0.0",
        priority=90,
    )
    task_2 = EnterpriseOpsTaskSpec(
        task_id="task_2_vendor_escrow_25k",
        task_prompt="Settle a high-value vendor invoice of $25k with dual-control treasury escrow.",
        action_type="settle_vendor_invoice",
        action_parameters={"invoice_id": "inv_2002", "invoice_amount": 25000.0},
        initial_state=state_t2,
        held_out_rules=(rule_t2,),
        target_object_id="inv_2002",
        target_object_type="VendorInvoice",
        sql_verifier_query=(
            'SELECT "ent_settlement_state", "ent_escrow_cleared", "res_escrow_hold" '
            'FROM "event_settle_vendor_invoice" WHERE "task_id" = ?'
        ),
        sql_verifier_params=("task_2_vendor_escrow_25k",),
        expected_sql_row=("settled_via_dual_escrow", "true", 0.0),
    )

    # Task 3: Reroute an east-coast fulfillment order during warehouse capacity throttling
    state_t3 = WorldState(
        step=0,
        timestamp=1700000200.0,
        entities={
            "ord_3003": Entity(
                id="ord_3003",
                type="FulfillmentOrder",
                attributes={"hub": "us-east", "routing_mode": "standard"},
            )
        },
        resources={
            "east_stock": Resource(id="east_stock", current=40.0, unit="units"),
            "central_stock": Resource(id="central_stock", current=300.0, unit="units"),
        },
    )
    rule_t3 = BusinessRule(
        rule_id="HELD-OUT-RULE-FULFILLMENT-REROUTE",
        action_types=("reroute_fulfillment_order",),
        rule_text=(
            "IF resource:east_stock < 50.0 THEN "
            "resource:central_stock -= 30.0; "
            "entity:ord_3003.hub = 'us-central-backup'; "
            "entity:ord_3003.routing_mode = 'throttled_failover'"
        ),
        version="1.0.0",
        priority=85,
    )
    task_3 = EnterpriseOpsTaskSpec(
        task_id="task_3_fulfillment_reroute",
        task_prompt="Reroute an east-coast fulfillment order during warehouse capacity throttling.",
        action_type="reroute_fulfillment_order",
        action_parameters={"order_id": "ord_3003", "units": 30},
        initial_state=state_t3,
        held_out_rules=(rule_t3,),
        target_object_id="ord_3003",
        target_object_type="FulfillmentOrder",
        sql_verifier_query=(
            'SELECT "ent_hub", "ent_routing_mode", "res_central_stock" '
            'FROM "event_reroute_fulfillment_order" WHERE "task_id" = ?'
        ),
        sql_verifier_params=("task_3_fulfillment_reroute",),
        expected_sql_row=("us-central-backup", "throttled_failover", 270.0),
    )

    # Task 4: Isolate and mitigate a payment microservice retry storm cascade
    state_t4 = WorldState(
        step=0,
        timestamp=1700000300.0,
        entities={
            "svc_payment": Entity(
                id="svc_payment",
                type="MicroserviceNode",
                attributes={"circuit_breaker": "closed", "retry_policy": "unbounded"},
            )
        },
        resources={
            "active_retry_threads": Resource(
                id="active_retry_threads", current=450.0, unit="threads"
            ),
            "shed_load_counter": Resource(id="shed_load_counter", current=0.0, unit="requests"),
        },
    )
    rule_t4 = BusinessRule(
        rule_id="HELD-OUT-RULE-SRE-CIRCUIT-BREAKER",
        action_types=("mitigate_retry_storm",),
        rule_text=(
            "IF resource:active_retry_threads > 200.0 THEN "
            "resource:active_retry_threads = 25.0; "
            "resource:shed_load_counter += 425.0; "
            "entity:svc_payment.circuit_breaker = 'half_open_isolated'; "
            "entity:svc_payment.retry_policy = 'exponential_backoff_capped'"
        ),
        version="1.0.0",
        priority=95,
    )
    task_4 = EnterpriseOpsTaskSpec(
        task_id="task_4_sre_retry_storm",
        task_prompt="Isolate and mitigate a payment microservice retry storm cascade.",
        action_type="mitigate_retry_storm",
        action_parameters={"service_id": "svc_payment"},
        initial_state=state_t4,
        held_out_rules=(rule_t4,),
        target_object_id="svc_payment",
        target_object_type="MicroserviceNode",
        sql_verifier_query=(
            'SELECT "ent_circuit_breaker", "ent_retry_policy", "res_active_retry_threads" '
            'FROM "event_mitigate_retry_storm" WHERE "task_id" = ?'
        ),
        sql_verifier_params=("task_4_sre_retry_storm",),
        expected_sql_row=("half_open_isolated", "exponential_backoff_capped", 25.0),
    )

    # Task 5: Execute a compliance audit hold on cross-border wire transfer exceeding $50k
    state_t5 = WorldState(
        step=0,
        timestamp=1700000400.0,
        entities={
            "wire_5005": Entity(
                id="wire_5005",
                type="WireTransfer",
                attributes={"compliance_status": "unscreened", "hold_flag": "none"},
            )
        },
        resources={
            "compliance_queue_depth": Resource(
                id="compliance_queue_depth", current=3.0, unit="items"
            ),
            "held_wire_volume": Resource(id="held_wire_volume", current=0.0, unit="usd"),
        },
    )
    rule_t5 = BusinessRule(
        rule_id="HELD-OUT-RULE-WIRE-COMPLIANCE-50K",
        action_types=("screen_cross_border_wire",),
        rule_text=(
            "IF wire_amount > 50000.0 THEN "
            "resource:compliance_queue_depth += 1.0; "
            "resource:held_wire_volume += 75000.0; "
            "entity:wire_5005.compliance_status = 'quarantined_for_aml_audit'; "
            "entity:wire_5005.hold_flag = 'mandatory_50k_hold'"
        ),
        version="1.0.0",
        priority=100,
    )
    task_5 = EnterpriseOpsTaskSpec(
        task_id="task_5_wire_compliance_50k",
        task_prompt="Execute a compliance audit hold on cross-border wire transfer exceeding $50k.",
        action_type="screen_cross_border_wire",
        action_parameters={"wire_id": "wire_5005", "wire_amount": 75000.0},
        initial_state=state_t5,
        held_out_rules=(rule_t5,),
        target_object_id="wire_5005",
        target_object_type="WireTransfer",
        sql_verifier_query=(
            'SELECT "ent_compliance_status", "ent_hold_flag", "res_held_wire_volume" '
            'FROM "event_screen_cross_border_wire" WHERE "task_id" = ?'
        ),
        sql_verifier_params=("task_5_wire_compliance_50k",),
        expected_sql_row=("quarantined_for_aml_audit", "mandatory_50k_hold", 75000.0),
    )

    return (task_1, task_2, task_3, task_4, task_5)


def verify_zero_held_out_rule_leakage(
    tasks: Sequence[EnterpriseOpsTaskSpec],
    prompt_harness: Optional[CDAPromptHarness] = None,
) -> bool:
    """Verify that none of the held-out benchmark business rules appear in the system prompt (E8 Req 8)."""
    harness = prompt_harness or CDAPromptHarness()
    sys_prompt = harness.build_system_prompt()
    for task in tasks:
        for rule in task.held_out_rules:
            if rule.rule_id in sys_prompt or rule.rule_text in sys_prompt:
                raise EnterpriseOpsGymError(
                    f"Evaluation leakage detected: held-out rule '{rule.rule_id}' found in system prompt!"
                )
    return True


def run_enterpriseops_gym_benchmark(
    tasks: Optional[Sequence[EnterpriseOpsTaskSpec]] = None,
    *,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the 5-task EnterpriseOps-Gym benchmark suite and return objective SQL verifier metrics.

    Acceptance Criteria (E8 Requirement 10):
    Returns a dictionary containing `"Avg Success Rate"` and `"Avg Verifier Pass"` based
    strictly on final environment state in the OCEL 2.0 SQLite database.
    """
    sample_tasks = tuple(tasks) if tasks is not None else build_representative_5_task_sample()
    verify_zero_held_out_rule_leakage(sample_tasks)

    def _run_with_db(target_db: Path) -> Dict[str, Any]:
        adapter = MCPClientAdapter(db_path=target_db, tasks=sample_tasks)
        endpoint = EnterpriseOpsAgentEndpoint(mcp_adapter=adapter)

        task_results: List[Dict[str, Any]] = []
        success_count = 0
        verifier_pass_count = 0
        total_latency_ms = 0.0
        sql_failures: List[str] = []

        for spec in sample_tasks:
            passed = endpoint.execute_task(spec.task_prompt)
            rep = endpoint.last_report
            if rep is None:
                raise EnterpriseOpsGymError(
                    f"Missing TaskExecutionReport after executing task '{spec.task_id}'."
                )
            if rep.action_Success:
                success_count += 1
            if passed and rep.sql_verifier_passed:
                verifier_pass_count += 1
            else:
                sql_failures.append(spec.task_id)
            total_latency_ms += rep.prompt_latency_ms

            task_results.append(
                {
                    "task_id": rep.task_id,
                    "task_prompt": rep.task_prompt,
                    "action_success": rep.action_Success,
                    "verifier_pass": rep.sql_verifier_passed,
                    "prompt_latency_ms": round(rep.prompt_latency_ms, 4),
                    "prompt_tokens": rep.prompt_tokens,
                    "completion_tokens": rep.completion_tokens,
                    "retrieved_rule_ids": list(rep.retrieved_rule_ids),
                    "sql_rows": [list(r) for r in rep.sql_rows],
                }
            )

        n = len(sample_tasks)
        avg_success = success_count / n if n > 0 else 0.0
        avg_verifier = verifier_pass_count / n if n > 0 else 0.0
        mean_latency = total_latency_ms / n if n > 0 else 0.0

        return {
            "Avg Success Rate": avg_success,
            "Avg Verifier Pass": avg_verifier,
            "Total Tasks": n,
            "Mean Prompt Latency (ms)": round(mean_latency, 4),
            "Total Prompt Latency (ms)": round(total_latency_ms, 4),
            "SQL Verifier Failures": sql_failures,
            "Zero Leakage Verified": True,
            "Task Results": task_results,
        }

    if db_path is not None:
        return _run_with_db(Path(db_path))

    with tempfile.TemporaryDirectory() as tmp_dir:
        return _run_with_db(Path(tmp_dir) / "enterpriseops_gym_eval.sqlite")


_TYPING_CHECK = (typing.Dict, typing.List, typing.Optional)

__all__ = [
    "EnterpriseOpsAgentEndpoint",
    "EnterpriseOpsGymError",
    "EnterpriseOpsTaskSpec",
    "MCPClientAdapter",
    "MCPServerProtocol",
    "ReadOnlySQLVerifier",
    "ReadOnlySQLVerifierError",
    "TaskExecutionReport",
    "build_representative_5_task_sample",
    "run_enterpriseops_gym_benchmark",
    "verify_zero_held_out_rule_leakage",
]
