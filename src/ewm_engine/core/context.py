"""First-class, typed, versioned, provenance-aware runtime context (K_t).

Provides the formal context component K of the Enterprise World Model 10-tuple
W = <S, A, E, K, O, T, C, U, Q, P>, enabling context-conditioned transitions
S_{t+1} ~ T(S_t, A_t, E_t | K_t) and vendor-neutral runtime context discovery.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.provenance.evidence import EvidenceLevel

ScalarParameter = float | int | str | bool


class WorkflowConfig(BaseModel):
    """Typed configuration for an enterprise workflow definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    version: str = "1.0.0"
    enabled: bool = True
    max_concurrency: int = Field(default=100, ge=1)
    sla_target_seconds: float = Field(default=3600.0, gt=0.0)
    routing_policy: str = "fifo"
    parameters: dict[str, ScalarParameter] = Field(default_factory=dict)


class AutomationRuleConfig(BaseModel):
    """Typed configuration for an automated policy or workflow rule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    automation_id: str
    version: str = "1.0.0"
    enabled: bool = True
    trigger_condition: str = "always"
    priority: int = 0
    parameters: dict[str, ScalarParameter] = Field(default_factory=dict)


class ServiceEndpointConfig(BaseModel):
    """Typed configuration for an external or internal service dependency."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_id: str
    endpoint_uri: str = "local://default"
    timeout_ms: int = Field(default=1000, ge=1)
    rate_limit_rps: float = Field(default=100.0, gt=0.0)
    degraded_mode: bool = False


class ContractTermConfig(BaseModel):
    """Typed configuration for an active SLA or commercial contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    version: str = "1.0.0"
    active: bool = True
    penalty_rate: float = Field(default=0.0, ge=0.0)
    sla_threshold: float = Field(default=0.99, ge=0.0, le=1.0)


class ContextSnapshot(BaseModel):
    """Immutable, versioned, provenance-aware deployment/tenant/runtime context K_t.

    Never uses an untyped dict[str, Any]; all configuration domains are strictly
    typed and canonically fingerprinted via SHA-256.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    context_id: str = "default_context"
    version: str = "1.0.0"
    tenant_id: str = "default_tenant"
    environment_regime: str = "nominal"
    effective_from: datetime | None = None
    effective_until: datetime | None = None

    workflow_definitions: dict[str, WorkflowConfig] = Field(default_factory=dict)
    enabled_automations: dict[str, AutomationRuleConfig] = Field(default_factory=dict)
    policy_versions: dict[str, str] = Field(default_factory=dict)
    feature_flags: dict[str, bool] = Field(default_factory=dict)
    topology_config: dict[str, str] = Field(default_factory=dict)
    external_services: dict[str, ServiceEndpointConfig] = Field(default_factory=dict)
    active_contracts: dict[str, ContractTermConfig] = Field(default_factory=dict)
    parameters: dict[str, ScalarParameter] = Field(default_factory=dict)

    provenance_source: str = "static_declaration"
    evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL
    parent_version: str | None = None

    @property
    def fingerprint(self) -> str:
        """Compute canonical SHA-256 hex digest of the full ContextSnapshot."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def project(self, required_keys: Sequence[str] | set[str]) -> ContextSnapshot:
        """Project context onto a mechanism's declared required_context_keys.

        Supports both top-level domain names (e.g. ``'feature_flags'``,
        ``'environment_regime'``) and dotted/colon paths (e.g.
        ``'parameters.tax_rate'``, ``'feature_flags.fast_track'``,
        ``'policy_versions.approval'``, ``'workflow_definitions.incident'``).
        Also matches bare keys in ``parameters``, ``feature_flags``, or
        ``policy_versions`` for convenience.

        Unreferenced keys and domains are excluded so that unrelated context
        changes cannot affect mechanisms that do not depend on them.
        """
        key_set = set(required_keys)

        workflows: dict[str, WorkflowConfig] = {}
        automations: dict[str, AutomationRuleConfig] = {}
        policies: dict[str, str] = {}
        flags: dict[str, bool] = {}
        topology: dict[str, str] = {}
        services: dict[str, ServiceEndpointConfig] = {}
        contracts: dict[str, ContractTermConfig] = {}
        params: dict[str, ScalarParameter] = {}

        keep_tenant = "tenant_id" in key_set
        keep_regime = "environment_regime" in key_set
        keep_dates = "effective_from" in key_set or "effective_until" in key_set

        for raw_key in sorted(key_set):
            sep = "." if "." in raw_key else (":" if ":" in raw_key else None)
            if sep is not None:
                domain, subkey = raw_key.split(sep, 1)
                if domain == "workflow_definitions" and subkey in self.workflow_definitions:
                    workflows[subkey] = self.workflow_definitions[subkey]
                elif domain == "enabled_automations" and subkey in self.enabled_automations:
                    automations[subkey] = self.enabled_automations[subkey]
                elif domain == "policy_versions" and subkey in self.policy_versions:
                    policies[subkey] = self.policy_versions[subkey]
                elif domain == "feature_flags" and subkey in self.feature_flags:
                    flags[subkey] = self.feature_flags[subkey]
                elif domain == "topology_config" and subkey in self.topology_config:
                    topology[subkey] = self.topology_config[subkey]
                elif domain == "external_services" and subkey in self.external_services:
                    services[subkey] = self.external_services[subkey]
                elif domain == "active_contracts" and subkey in self.active_contracts:
                    contracts[subkey] = self.active_contracts[subkey]
                elif domain == "parameters" and subkey in self.parameters:
                    params[subkey] = self.parameters[subkey]
            else:
                if raw_key == "workflow_definitions":
                    workflows.update(self.workflow_definitions)
                elif raw_key == "enabled_automations":
                    automations.update(self.enabled_automations)
                elif raw_key == "policy_versions":
                    policies.update(self.policy_versions)
                elif raw_key == "feature_flags":
                    flags.update(self.feature_flags)
                elif raw_key == "topology_config":
                    topology.update(self.topology_config)
                elif raw_key == "external_services":
                    services.update(self.external_services)
                elif raw_key == "active_contracts":
                    contracts.update(self.active_contracts)
                elif raw_key == "parameters":
                    params.update(self.parameters)
                else:
                    # Bare key fallback across scalar/config dictionaries
                    if raw_key in self.parameters:
                        params[raw_key] = self.parameters[raw_key]
                    if raw_key in self.feature_flags:
                        flags[raw_key] = self.feature_flags[raw_key]
                    if raw_key in self.policy_versions:
                        policies[raw_key] = self.policy_versions[raw_key]
                    if raw_key in self.workflow_definitions:
                        workflows[raw_key] = self.workflow_definitions[raw_key]
                    if raw_key in self.enabled_automations:
                        automations[raw_key] = self.enabled_automations[raw_key]
                    if raw_key in self.topology_config:
                        topology[raw_key] = self.topology_config[raw_key]
                    if raw_key in self.external_services:
                        services[raw_key] = self.external_services[raw_key]
                    if raw_key in self.active_contracts:
                        contracts[raw_key] = self.active_contracts[raw_key]

        return ContextSnapshot(
            context_id=self.context_id,
            version=self.version,
            tenant_id=self.tenant_id if keep_tenant else "default_tenant",
            environment_regime=self.environment_regime if keep_regime else "nominal",
            effective_from=self.effective_from if keep_dates else None,
            effective_until=self.effective_until if keep_dates else None,
            workflow_definitions=workflows,
            enabled_automations=automations,
            policy_versions=policies,
            feature_flags=flags,
            topology_config=topology,
            external_services=services,
            active_contracts=contracts,
            parameters=params,
            provenance_source=self.provenance_source,
            evidence_level=self.evidence_level,
            parent_version=self.parent_version,
        )

    def domain_fingerprint(self, required_keys: Sequence[str] | set[str] | None = None) -> str:
        """Compute canonical SHA-256 digest over only the mechanism-relevant domain.

        When ``required_keys`` is provided, administrative version metadata
        (``version``, ``parent_version``, ``context_id``, ``provenance_source``)
        is excluded unless explicitly listed in ``required_keys``, ensuring
        that evolving unrelated context fields leaves the domain fingerprint
        strictly invariant.
        """
        if required_keys is None:
            return self.fingerprint
        projected = self.project(required_keys)
        payload = projected.model_dump(mode="json")
        key_set = set(required_keys)
        for meta_field in ("context_id", "version", "parent_version", "provenance_source"):
            if meta_field not in key_set:
                payload.pop(meta_field, None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def evolve(self, *, new_version: str, **updates: Any) -> ContextSnapshot:
        """Return a new ContextSnapshot with updated fields without mutating self."""
        data = copy.deepcopy(self.model_dump())
        data["parent_version"] = self.version
        data["version"] = new_version
        for k, v in updates.items():
            if k not in ContextSnapshot.model_fields:
                raise ValueError(f"Unknown ContextSnapshot field: {k!r}")
            data[k] = copy.deepcopy(v)
        return ContextSnapshot.model_validate(data)

    def is_effective_at(self, timestamp: datetime) -> bool:
        """Return True if this context snapshot is effective at ``timestamp``."""
        if self.effective_from is not None and timestamp < self.effective_from:
            return False
        return not (self.effective_until is not None and timestamp > self.effective_until)

    def to_dict(self) -> dict[str, Any]:
        """Serialize context snapshot to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextSnapshot:
        """Deserialize a ContextSnapshot from a dictionary."""
        return cls.model_validate(dict(data))


@runtime_checkable
class ContextDiscoveryProvider(Protocol):
    """Vendor-neutral extension point for runtime context discovery.

    Adapters inspecting external audit logs, configuration registries, or
    runtime metadata implement this protocol to produce a typed
    ``ContextSnapshot`` without coupling the EWM core to any vendor (SAP,
    ServiceNow, Salesforce, etc.).
    """

    def discover_context(self, raw_metadata: Mapping[str, Any]) -> ContextSnapshot:
        """Inspect runtime or audit metadata and produce a typed ContextSnapshot."""
        ...


class AuditMetadataContextDiscovery:
    """Vendor-neutral runtime context discovery from structured audit metadata.

    Enforces fail-closed causal evidence semantics: runtime-discovered context
    defaults to ``EvidenceLevel.PREDICTIVE`` and cannot claim
    ``INTERVENTIONAL``, ``QUASI_CAUSAL``, or ``COUNTERFACTUAL`` evidence without
    an explicit ``identification_certificate_id`` in the metadata.
    """

    def __init__(self, default_source: str = "runtime_audit_discovery") -> None:
        self.default_source = default_source

    def discover_context(self, raw_metadata: Mapping[str, Any]) -> ContextSnapshot:
        """Parse vendor-neutral audit metadata into a validated ContextSnapshot."""
        from ewm_engine.core.trust import UnqualifiedCausalClaimError

        data = dict(raw_metadata)
        cert_id = data.pop("identification_certificate_id", None)

        raw_evidence = data.get("evidence_level", EvidenceLevel.PREDICTIVE)
        evidence = (
            raw_evidence
            if isinstance(raw_evidence, EvidenceLevel)
            else EvidenceLevel(str(raw_evidence))
        )

        # Fail-closed causal evidence enforcement
        if evidence in (
            EvidenceLevel.INTERVENTIONAL,
            EvidenceLevel.QUASI_CAUSAL,
            EvidenceLevel.COUNTERFACTUAL,
        ) and not (isinstance(cert_id, str) and cert_id.strip()):
            raise UnqualifiedCausalClaimError(
                f"Fail-closed EvidenceLevel violation: runtime context discovery cannot "
                f"claim {evidence.value!r} without a non-empty "
                f"'identification_certificate_id'."
            )

        data["evidence_level"] = evidence
        data.setdefault("provenance_source", self.default_source)
        return ContextSnapshot.model_validate(data)
