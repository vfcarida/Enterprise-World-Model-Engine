"""Induction report, audit metadata, and deterministic fingerprinting for induced world models."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.spec import ComponentSpec
from ewm_engine.provenance.evidence import EvidenceLevel


class CandidateInvariant(BaseModel):
    """Candidate hard conservation law or bounded-resource constraint discovered during induction."""

    model_config = ConfigDict(extra="forbid")

    name: str
    target_resource: str
    invariant_type: str = "conservation"
    equation: str
    support_ratio: float = Field(ge=0.0, le=1.0)
    max_observed_violation: float = 0.0
    evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL
    parameters: dict[str, Any] = Field(default_factory=dict)

    def to_constraint_spec(self) -> ComponentSpec:
        """Compile candidate invariant into an official WorldSpec ComponentSpec."""
        is_hard = self.evidence_level == EvidenceLevel.STRUCTURAL
        return ComponentSpec(
            type="capacity",
            id=f"invariant_{self.name}",
            parameters={
                "resource_id": self.target_resource,
                "severity": "hard" if is_hard else "soft",
                "equation": self.equation,
                **self.parameters,
            },
        )


class ProvisionalLink(BaseModel):
    """Provisional latent relationship proposed via link prediction / relational learning."""

    model_config = ConfigDict(extra="forbid")

    source_entity: str
    target_entity: str
    relationship_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = "jaccard_co_occurrence"
    evidence_level: EvidenceLevel = EvidenceLevel.PREDICTIVE
    clears_gate: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConformanceScore(BaseModel):
    """Trace replay alignment metrics quantifying process model fidelity."""

    model_config = ConfigDict(extra="forbid")

    object_type: str
    fitness: float = Field(ge=0.0, le=1.0, description="Fraction of log traces replayable by model")
    precision: float = Field(
        ge=0.0, le=1.0, description="Fraction of model behavior witnessed in log"
    )
    traces_analyzed: int = 0
    replay_errors: int = 0
    evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL


class CausalEquivalenceClassSummary(BaseModel):
    """Summary of discovered causal Markov equivalence class (CPDAG/PAG)."""

    model_config = ConfigDict(extra="forbid")

    variables: list[str]
    is_dag: bool = False
    markov_equivalent_dags_count: int = 1
    asserted_edges_count: int = 0
    directional_uncertain_edges_count: int = 0
    confounded_edges_count: int = 0
    edges: list[dict[str, Any]] = Field(default_factory=list)


class InducedRateLawSummary(BaseModel):
    """Summary of discovered governing differential or difference rate law."""

    model_config = ConfigDict(extra="forbid")

    target_variable: str
    expression: str
    r_squared: float
    complexity: int
    evidence_level: str


class InductionReport(BaseModel):
    """Auditable, reproducible report produced during world-model induction.

    Contains empirical conformance scores, candidate conservation laws,
    provisional links, causal equivalence classes, induced rate laws,
    and a cryptographic SHA-256 fingerprint guaranteeing that two runs
    on the same input data with the same seed yield identical results.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    source: str
    source_type: Literal[
        "dataframe", "sql", "relational_schema", "ocel_2_0", "causal", "dynamics", "invariants"
    ]
    induced_at: str = "1970-01-01T00:00:00+00:00"
    seed: int = 42
    point_in_time_cutoff: str | None = None

    entities_count: int = 0
    relationships_count: int = 0
    resources_count: int = 0
    constraints_count: int = 0

    evidence_distribution: dict[str, int] = Field(default_factory=dict)
    conformance: dict[str, ConformanceScore] = Field(default_factory=dict)
    candidate_invariants: list[CandidateInvariant] = Field(default_factory=list)
    provisional_links: list[ProvisionalLink] = Field(default_factory=list)
    causal_equivalence_class: CausalEquivalenceClassSummary | None = None
    induced_rate_laws: list[InducedRateLawSummary] = Field(default_factory=list)

    # Induction Honesty and Refusal Gate fields
    validation_passed: bool | None = None
    gate_outcome: str | None = None
    refused_elements_count: int = 0
    open_questions_count: int = 0
    refused_elements: list[dict[str, Any]] = Field(default_factory=list)
    open_questions: list[dict[str, Any]] = Field(default_factory=list)
    validation_fingerprint: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    sha256_fingerprint: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute SHA-256 fingerprint if not explicitly provided."""
        if not self.sha256_fingerprint:
            self.sha256_fingerprint = self.compute_fingerprint()

    def compute_fingerprint(self) -> str:
        """Calculate canonical SHA-256 hash over the deterministic fields."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "source_type": self.source_type,
            "seed": self.seed,
            "point_in_time_cutoff": self.point_in_time_cutoff,
            "entities_count": self.entities_count,
            "relationships_count": self.relationships_count,
            "resources_count": self.resources_count,
            "constraints_count": self.constraints_count,
            "evidence_distribution": sorted(self.evidence_distribution.items()),
            "conformance": {
                k: {
                    "fitness": round(v.fitness, 4),
                    "precision": round(v.precision, 4),
                    "evidence_level": v.evidence_level.value,
                }
                for k, v in sorted(self.conformance.items())
            },
            "candidate_invariants": [
                {
                    "name": inv.name,
                    "target_resource": inv.target_resource,
                    "equation": inv.equation,
                    "support_ratio": round(inv.support_ratio, 4),
                }
                for inv in sorted(self.candidate_invariants, key=lambda x: x.name)
            ],
            "provisional_links": [
                {
                    "source": pl.source_entity,
                    "target": pl.target_entity,
                    "confidence": round(pl.confidence, 4),
                }
                for pl in sorted(
                    self.provisional_links, key=lambda x: (x.source_entity, x.target_entity)
                )
            ],
        }
        if self.causal_equivalence_class is not None:
            payload["causal_equivalence_class"] = {
                "variables": sorted(self.causal_equivalence_class.variables),
                "is_dag": self.causal_equivalence_class.is_dag,
                "edges": sorted(
                    [
                        f"{e.get('source')}->{e.get('target')}:{e.get('type')}:{e.get('level')}"
                        for e in self.causal_equivalence_class.edges
                    ]
                ),
            }
        if self.induced_rate_laws:
            payload["induced_rate_laws"] = sorted(
                [
                    f"{r.target_variable}:{r.expression}:{round(r.r_squared, 4)}"
                    for r in self.induced_rate_laws
                ]
            )
        if self.validation_fingerprint:
            payload["validation_fingerprint"] = self.validation_fingerprint
        if self.refused_elements:
            payload["refused_elements"] = sorted(
                [
                    f"{r.get('element_type')}:{r.get('identifier')}:{r.get('refusal_reason')}"
                    for r in self.refused_elements
                ]
            )
        if self.open_questions:
            payload["open_questions"] = sorted(
                [f"{q.get('id')}:{q.get('topic')}" for q in self.open_questions]
            )
        canonical_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate human-readable Markdown summary of the induction audit."""
        lines = [
            "# World Model Induction Audit Report",
            "",
            f"- **Source**: `{self.source}` ({self.source_type})",
            f"- **Induced At**: `{self.induced_at}`",
            f"- **Seed**: `{self.seed}`",
            f"- **Point-in-Time Cutoff**: `{self.point_in_time_cutoff or 'None (Full Horizon)'}`",
            f"- **SHA-256 Fingerprint**: `{self.sha256_fingerprint}`",
            "",
            "## Proposed World Topology",
            "",
            "| Element | Count |",
            "| :--- | :--- |",
            f"| **Entities** | {self.entities_count} |",
            f"| **Relationships** | {self.relationships_count} |",
            f"| **Resources** | {self.resources_count} |",
            f"| **Candidate Constraints** | {self.constraints_count} |",
            "",
            "## Conformance & Replay Alignment",
            "",
            "| Object Type | Fitness | Precision | Evidence Level | Traces Replayed |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]
        for obj_type, conf in self.conformance.items():
            lines.append(
                f"| `{obj_type}` | {conf.fitness:.3f} | {conf.precision:.3f} | `{conf.evidence_level.value}` | {conf.traces_analyzed} |"
            )
        if not self.conformance:
            lines.append("| *None* | - | - | - | - |")

        lines.extend(
            [
                "",
                "## Discovered Conservation Invariants",
                "",
                "| Invariant | Target Resource | Candidate Equation | Support Ratio | Evidence |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for inv in self.candidate_invariants:
            lines.append(
                f"| `{inv.name}` | `{inv.target_resource}` | `{inv.equation}` | {inv.support_ratio:.1%} | `{inv.evidence_level.value}` |"
            )
        if not self.candidate_invariants:
            lines.append("| *None* | - | - | - | - |")

        lines.extend(
            [
                "",
                "## Provisional Inferred Links (Validation Gating Required)",
                "",
                "| Source Entity | Target Entity | Relationship | Confidence | Status |",
                "| :--- | :--- | :--- | :--- | :--- |",
            ]
        )
        for pl in self.provisional_links:
            status = "APPROVED" if pl.clears_gate else "PENDING_VALIDATION_GATE"
            lines.append(
                f"| `{pl.source_entity}` | `{pl.target_entity}` | `{pl.relationship_type}` | {pl.confidence:.2f} | `{status}` |"
            )
        if not self.provisional_links:
            lines.append("| *None* | - | - | - | - |")

        if self.causal_equivalence_class:
            lines.extend(
                [
                    "",
                    "## Discovered Causal Structure (CPDAG / PAG)",
                    "",
                    f"- **Variables**: {len(self.causal_equivalence_class.variables)}",
                    f"- **Estimated Markov DAGs**: {self.causal_equivalence_class.markov_equivalent_dags_count}",
                    f"- **Asserted Edges**: {self.causal_equivalence_class.asserted_edges_count}",
                    f"- **Directional-Uncertain Edges**: {self.causal_equivalence_class.directional_uncertain_edges_count}",
                    f"- **Confounded Edges**: {self.causal_equivalence_class.confounded_edges_count}",
                    "",
                    "| Source | Target | Edge Type | Support | Evidence Level | Interventional |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for e in self.causal_equivalence_class.edges:
                lines.append(
                    f"| `{e.get('source')}` | `{e.get('target')}` | `{e.get('type')}` | {e.get('support', 0.0):.2f} | `{e.get('level')}` | {e.get('interventional', False)} |"
                )
            if not self.causal_equivalence_class.edges:
                lines.append("| *None* | - | - | - | - | - |")

        if self.induced_rate_laws:
            lines.extend(
                [
                    "",
                    "## Induced Dynamics Rate Laws (SINDy / Symbolic Regression)",
                    "",
                    "| Target Variable | Rate Law Equation | R² | Complexity | Evidence Level |",
                    "| :--- | :--- | :--- | :--- | :--- |",
                ]
            )
            for r in self.induced_rate_laws:
                lines.append(
                    f"| `{r.target_variable}` | `{r.expression}` | {r.r_squared:.3f} | {r.complexity} | `{r.evidence_level}` |"
                )

        if self.validation_passed is not None:
            gate_status = (
                "PASSED (Clean Induction)"
                if self.validation_passed
                else "REFUSALS PRESENT (Fails Closed)"
            )
            lines.extend(
                [
                    "",
                    "## Induction Honesty & Refusal Gate",
                    "",
                    f"- **Gate Status**: `{gate_status}`",
                    f"- **Refused Elements**: {self.refused_elements_count}",
                    f"- **Surfaced Open Questions**: {self.open_questions_count}",
                ]
            )
            if self.refused_elements:
                lines.extend(
                    [
                        "",
                        "### Refused Elements",
                        "",
                        "| Type | Identifier | Reason | Alternatives |",
                        "| :--- | :--- | :--- | :--- |",
                    ]
                )
                for ref_item in self.refused_elements:
                    alts = (
                        "; ".join(ref_item.get("candidate_alternatives", []))
                        if ref_item.get("candidate_alternatives")
                        else "-"
                    )
                    lines.append(
                        f"| `{ref_item.get('element_type')}` | `{ref_item.get('identifier')}` | {ref_item.get('refusal_reason')} | {alts} |"
                    )
            if self.open_questions:
                lines.extend(
                    [
                        "",
                        "### Open Questions for Human Modeler",
                        "",
                    ]
                )
                for oq_item in self.open_questions:
                    lines.append(
                        f"- **{oq_item.get('id')} ({oq_item.get('topic')})**: {oq_item.get('description')}"
                    )

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Render a rich interactive HTML card for Jupyter and Marimo notebooks."""
        fp_short = self.sha256_fingerprint[:12] if self.sha256_fingerprint else "uncomputed"
        conf_rows = "".join(
            f"<tr><td style='padding: 6px 10px;'><code>{k}</code></td>"
            f"<td style='padding: 6px 10px; text-align: right;'>{v.fitness:.3f}</td>"
            f"<td style='padding: 6px 10px; text-align: right;'>{v.precision:.3f}</td>"
            f"<td style='padding: 6px 10px;'><span style='background: #f0fdf4; color: #166534; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>{v.evidence_level.value}</span></td></tr>"
            for k, v in self.conformance.items()
        )
        conf_table = (
            (
                f"<table style='width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px;'>"
                f"<thead><tr style='border-bottom: 1px solid #cbd5e1; text-align: left; color: #64748b; font-size: 11px;'>"
                f"<th style='padding: 6px 10px;'>Type</th><th style='padding: 6px 10px; text-align: right;'>Fitness</th>"
                f"<th style='padding: 6px 10px; text-align: right;'>Precision</th><th style='padding: 6px 10px;'>Evidence</th></tr></thead>"
                f"<tbody>{conf_rows}</tbody></table>"
            )
            if self.conformance
            else "<p style='color: #94a3b8; font-size: 12px; margin: 4px 0;'>No process conformance data.</p>"
        )

        causal_snippet = ""
        if self.causal_equivalence_class:
            causal_snippet = (
                f"<div style='margin-top: 10px; padding: 8px; background: #f8fafc; border-radius: 6px; font-size: 12px;'>"
                f"<strong>Causal Structure:</strong> {self.causal_equivalence_class.asserted_edges_count} ASSERTED, "
                f"{self.causal_equivalence_class.directional_uncertain_edges_count} UNCERTAIN, "
                f"{self.causal_equivalence_class.confounded_edges_count} CONFOUNDED "
                f"(~{self.causal_equivalence_class.markov_equivalent_dags_count} Markov DAGs)</div>"
            )

        dynamics_snippet = ""
        if self.induced_rate_laws:
            dynamics_snippet = (
                f"<div style='margin-top: 10px; padding: 8px; background: #f8fafc; border-radius: 6px; font-size: 12px;'>"
                f"<strong>Induced Rate Laws:</strong> {len(self.induced_rate_laws)} discovered governing equations</div>"
            )

        return (
            f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
            f"border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff; "
            f"box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-width: 680px; margin: 8px 0; color: #1e293b;'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;'>"
            f"<div>"
            f"<span style='font-weight: 700; font-size: 15px; color: #0f172a;'>InductionReport</span>"
            f"<span style='background: #eff6ff; color: #1d4ed8; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 8px;'>{self.source_type}</span>"
            + (
                "<span style='background: #f0fdf4; color: #166534; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 6px;'>GATE: PASSED</span>"
                if self.validation_passed is True
                else (
                    f"<span style='background: #fef2f2; color: #991b1b; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 6px;'>GATE: {self.refused_elements_count} REFUSED</span>"
                    if self.validation_passed is False
                    else ""
                )
            )
            + f"</div>"
            f"<code style='font-size: 11px; color: #64748b;'>{fp_short}...</code>"
            f"</div>"
            f"<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0; font-size: 12px; text-align: center;'>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Entities</strong><br><span style='font-size: 16px; font-weight: 700;'>{self.entities_count}</span></div>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Relations</strong><br><span style='font-size: 16px; font-weight: 700;'>{self.relationships_count}</span></div>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Resources</strong><br><span style='font-size: 16px; font-weight: 700;'>{self.resources_count}</span></div>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Invariants</strong><br><span style='font-size: 16px; font-weight: 700;'>{len(self.candidate_invariants)}</span></div>"
            f"</div>"
            f"{causal_snippet}"
            f"{dynamics_snippet}"
            f"<div style='margin-top: 10px;'><strong>Process Conformance Replay:</strong>{conf_table}</div>"
            f"<div style='margin-top: 8px; font-size: 11px; color: #64748b;'>Source: <code>{self.source}</code> | Seed: <code>{self.seed}</code></div>"
            f"</div>"
        )
