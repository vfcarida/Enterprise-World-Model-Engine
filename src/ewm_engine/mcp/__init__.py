"""Model Context Protocol (MCP) server integration for EWM Engine."""

from __future__ import annotations

from ewm_engine.mcp.provenance_strict import AuditEvent, ProvenanceStrictPolicy
from ewm_engine.mcp.safety import (
    CheckConstraintsInput,
    CounterfactualInput,
    DecideInput,
    ExplainInput,
    SimulateInput,
)
from ewm_engine.mcp.server import EwmMcpServer, ServerConfig, create_mcp_server, is_mcp_available

__all__ = [
    "AuditEvent",
    "CheckConstraintsInput",
    "CounterfactualInput",
    "DecideInput",
    "EwmMcpServer",
    "ExplainInput",
    "ProvenanceStrictPolicy",
    "ServerConfig",
    "SimulateInput",
    "create_mcp_server",
    "is_mcp_available",
]
