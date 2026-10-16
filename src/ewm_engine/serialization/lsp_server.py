"""Language Server Protocol (LSP) server for World Specification Language (WSL).

Provides real-time diagnostics, autocompletion, and hover documentation
for authored .wsl documents via pygls.

Reuses the safe Pydantic validator and Lark grammar for exact diagnostic parity
with the simulation CLI and CI pipeline.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ewm_engine.exceptions import SerializationError
from ewm_engine.serialization.wsl_parser import parse_wsl_authored


def is_pygls_available() -> bool:
    """Check whether pygls is installed."""
    return importlib.util.find_spec("pygls") is not None


def validate_wsl_text(text: str) -> list[dict[str, Any]]:
    """Validate WSL source text and produce LSP-compatible diagnostic records."""
    diagnostics: list[dict[str, Any]] = []

    # 1. Parse using Lark LALR(1) parser
    try:
        doc = parse_wsl_authored(text)
    except SerializationError as exc:
        msg = str(exc)
        line = 0
        col = 0
        if "line " in msg and ", col " in msg:
            try:
                parts = msg.split("line ")[1].split(", col ")
                line = max(0, int(parts[0]) - 1)
                col = max(0, int(parts[1].split(":")[0]) - 1)
            except Exception:
                pass

        diagnostics.append(
            {
                "range": {
                    "start": {"line": line, "character": col},
                    "end": {"line": line, "character": col + 10},
                },
                "message": msg,
                "severity": 1,  # Error
                "source": "wsl-validator",
            }
        )
        return diagnostics

    # 2. Semantic validations
    entity_ids = {e.id for e in doc.entities}

    # Check relationship source and target references
    for rel in doc.relationships:
        if rel.source and rel.source not in entity_ids:
            diagnostics.append(
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 20},
                    },
                    "message": f"Relationship source entity '{rel.source}' is not defined in entities.",
                    "severity": 2,  # Warning
                    "source": "wsl-linter",
                }
            )
        if rel.target and rel.target not in entity_ids:
            diagnostics.append(
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 20},
                    },
                    "message": f"Relationship target entity '{rel.target}' is not defined in entities.",
                    "severity": 2,  # Warning
                    "source": "wsl-linter",
                }
            )

    # Check resource min/max consistency
    for res in doc.resources:
        if res.min_value > res.max_value:
            diagnostics.append(
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 20},
                    },
                    "message": f"Resource '{res.id}' min_value ({res.min_value}) exceeds max_value ({res.max_value}).",
                    "severity": 1,  # Error
                    "source": "wsl-linter",
                }
            )

    return diagnostics


def create_wsl_server() -> Any:
    """Create a pygls LanguageServer instance configured for WSL authoring."""
    if not is_pygls_available():
        raise ImportError(
            "The 'pygls' package is required to run the WSL Language Server. "
            "Install via `pip install ewm-engine[dsl]`."
        )

    from lsprotocol.types import (
        TEXT_DOCUMENT_COMPLETION,
        TEXT_DOCUMENT_DID_CHANGE,
        TEXT_DOCUMENT_DID_OPEN,
        TEXT_DOCUMENT_DID_SAVE,
        TEXT_DOCUMENT_HOVER,
        CompletionItem,
        CompletionItemKind,
        CompletionList,
        CompletionParams,
        Diagnostic,
        DiagnosticSeverity,
        DidChangeTextDocumentParams,
        DidOpenTextDocumentParams,
        DidSaveTextDocumentParams,
        Hover,
        HoverParams,
        MarkupContent,
        MarkupKind,
        Position,
        Range,
    )
    from pygls.server import LanguageServer

    server = LanguageServer("wsl-language-server", "v1.0.0")

    def _publish_diagnostics(ls: LanguageServer, uri: str, text: str) -> None:
        raw_diags = validate_wsl_text(text)
        lsp_diags: list[Diagnostic] = []
        for d in raw_diags:
            diag = Diagnostic(
                range=Range(
                    start=Position(
                        line=d["range"]["start"]["line"], character=d["range"]["start"]["character"]
                    ),
                    end=Position(
                        line=d["range"]["end"]["line"], character=d["range"]["end"]["character"]
                    ),
                ),
                message=d["message"],
                severity=DiagnosticSeverity(d["severity"]),
                source=d["source"],
            )
            lsp_diags.append(diag)
        ls.publish_diagnostics(uri, lsp_diags)

    @server.feature(TEXT_DOCUMENT_DID_OPEN)
    def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams) -> None:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(ls, doc.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_DID_CHANGE)
    def did_change(ls: LanguageServer, params: DidChangeTextDocumentParams) -> None:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(ls, doc.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_DID_SAVE)
    def did_save(ls: LanguageServer, params: DidSaveTextDocumentParams) -> None:
        doc = ls.workspace.get_text_document(params.text_document.uri)
        _publish_diagnostics(ls, doc.uri, doc.source)

    @server.feature(TEXT_DOCUMENT_COMPLETION)
    def completions(ls: LanguageServer, params: CompletionParams) -> CompletionList:
        items = [
            CompletionItem(
                label="world", kind=CompletionItemKind.Keyword, detail="World declaration block"
            ),
            CompletionItem(
                label="entity", kind=CompletionItemKind.Keyword, detail="Entity participant block"
            ),
            CompletionItem(
                label="resource",
                kind=CompletionItemKind.Keyword,
                detail="Measurable state resource",
            ),
            CompletionItem(
                label="relationship",
                kind=CompletionItemKind.Keyword,
                detail="Relational graph edge",
            ),
            CompletionItem(
                label="constraint",
                kind=CompletionItemKind.Keyword,
                detail="Constraint rule specification",
            ),
            CompletionItem(
                label="scenario", kind=CompletionItemKind.Keyword, detail="Simulation scenario"
            ),
            CompletionItem(
                label="ResourceCapacityConstraint",
                kind=CompletionItemKind.Class,
                detail="Built-in capacity guard",
            ),
            CompletionItem(
                label="ActionTransferAvailabilityConstraint",
                kind=CompletionItemKind.Class,
                detail="Transfer availability constraint",
            ),
        ]
        return CompletionList(is_incomplete=False, items=items)

    @server.feature(TEXT_DOCUMENT_HOVER)
    def hover(ls: LanguageServer, params: HoverParams) -> Hover | None:
        content = (
            "### World Specification Language (WSL)\n"
            "Declarative, safe DSL for enterprise simulation models.\n"
            "Conforms to Draft 2020-12 schema with bit-exact fingerprint reproducibility."
        )
        return Hover(contents=MarkupContent(kind=MarkupKind.Markdown, value=content))

    return server


def start_server_stdio() -> None:
    """Start the WSL Language Server communicating over standard I/O."""
    server = create_wsl_server()
    server.start_io()


if __name__ == "__main__":
    start_server_stdio()
