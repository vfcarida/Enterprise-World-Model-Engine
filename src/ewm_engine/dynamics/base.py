"""Base protocols, mechanism taxonomy, and transition types for pluggable dynamics."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.trust import UnqualifiedCausalClaimError
from ewm_engine.core.types import RandomGenerator
from ewm_engine.provenance.evidence import EvidenceLevel


class MechanismKind(StrEnum):
    """Taxonomy of enterprise world-model transition mechanisms."""

    KNOWN_DETERMINISTIC = "known_deterministic"
    KNOWN_STOCHASTIC = "known_stochastic"
    LEARNED = "learned"
    DISCOVERED_AT_RUNTIME = "discovered_at_runtime"
    HYBRID = "hybrid"


class InfeasibleResidualPolicy(StrEnum):
    """Explicit handling policy when a learned residual proposes an infeasible state.

    Silent repair (such as unlogged clamping) is strictly prohibited:
    - PROJECT: Project onto the feasible set via conservation/box-constraint projection
      and log the exact L2 projection distance and violated bounds.
    - REJECT: Reject the learned residual component and fall back to the pure
      structural transition.
    - ABSTAIN: Refuse state evolution (returning S_t unchanged) and flag abstention.
    - INVALIDATE: Mark the transition as infeasible so the simulation engine halts
      or marks the trajectory INVALID.
    """

    PROJECT = "project"
    REJECT = "reject"
    ABSTAIN = "abstain"
    INVALIDATE = "invalidate"


class ValidityDomain(BaseModel):
    """Declared operating regime and state-space support for a transition mechanism."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_regimes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Permitted ContextSnapshot.environment_regime values (empty = all).",
    )
    resource_bounds: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Required [min, max] operating envelope per resource ID.",
    )
    max_feature_distance: float | None = Field(
        default=None,
        description="Maximum OOD feature distance before leaving the validity domain.",
    )
    description: str = Field(
        default="unrestricted",
        description="Human-readable specification of the validity domain.",
    )

    def check_validity(
        self,
        state: WorldState,
        runtime_context: ContextSnapshot | None = None,
    ) -> tuple[bool, list[str]]:
        """Check whether (state, runtime_context) lies inside this validity domain."""
        reasons: list[str] = []
        if (
            self.allowed_regimes
            and runtime_context is not None
            and runtime_context.environment_regime not in self.allowed_regimes
        ):
            reasons.append(
                f"Context regime {runtime_context.environment_regime!r} not in "
                f"allowed_regimes {self.allowed_regimes}."
            )
        for rid, (lo, hi) in self.resource_bounds.items():
            if rid in state.resources:
                val = state.resources[rid].current
                if val < lo or val > hi:
                    reasons.append(
                        f"Resource {rid!r} level {val} outside validity domain [{lo}, {hi}]."
                    )
        return (len(reasons) == 0, reasons)


class MechanismProvenance(BaseModel):
    """Formal metadata and epistemic contract for a transition mechanism.

    Every mechanism exposes its identifier, version, mechanism kind, validity
    domain, required context keys, fail-closed causal evidence level, provenance
    source, uncertainty contract, affected invariants, and failure/refusal policy.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism_id: str
    version: str = "1.0.0"
    kind: MechanismKind = MechanismKind.KNOWN_DETERMINISTIC
    validity_domain: ValidityDomain = Field(default_factory=ValidityDomain)
    required_context_keys: tuple[str, ...] = Field(default_factory=tuple)
    evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL
    source: str = "explicit_specification"
    uncertainty_contract: str = "point_deterministic"
    invariants_affected: tuple[str, ...] = Field(default_factory=tuple)
    failure_semantics: InfeasibleResidualPolicy = InfeasibleResidualPolicy.PROJECT
    identification_certificate_id: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._validate_fail_closed_evidence()

    def _validate_fail_closed_evidence(self) -> None:
        """Enforce fail-closed causal semantics on mechanism declarations."""
        if self.kind in (MechanismKind.LEARNED, MechanismKind.DISCOVERED_AT_RUNTIME):
            if self.evidence_level == EvidenceLevel.STRUCTURAL:
                raise UnqualifiedCausalClaimError(
                    f"Mechanism {self.mechanism_id!r} of kind {self.kind.value!r} cannot "
                    f"claim STRUCTURAL evidence level."
                )
            if self.evidence_level in (
                EvidenceLevel.COUNTERFACTUAL,
                EvidenceLevel.INTERVENTIONAL,
                EvidenceLevel.QUASI_CAUSAL,
            ) and not (
                isinstance(self.identification_certificate_id, str)
                and self.identification_certificate_id.strip()
            ):
                raise UnqualifiedCausalClaimError(
                    f"Fail-closed EvidenceLevel violation: mechanism {self.mechanism_id!r} "
                    f"({self.kind.value}) cannot claim {self.evidence_level.value!r} "
                    f"without a non-empty 'identification_certificate_id'."
                )


class ResidualFeasibilityRecord(BaseModel):
    """Audit record documenting how a learned residual proposal was handled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_feasible: bool
    policy: InfeasibleResidualPolicy
    action_taken: str
    raw_deltas: dict[str, float] = Field(default_factory=dict)
    applied_deltas: dict[str, float] = Field(default_factory=dict)
    l2_projection_distance: float = 0.0
    violations: tuple[str, ...] = Field(default_factory=tuple)


class TransitionResult(BaseModel):
    """The result of executing a dynamics transition step.

    Contains the resulting next world state, a summary of applied changes,
    the epistemic evidence tier, and execution diagnostics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    next_state: WorldState = Field(
        description="The evolved world state after dynamics application."
    )
    applied_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Key deltas, resource flows, and state modifications enacted during transition.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Epistemic status of the transition mechanism.",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata, internal model states, and performance metrics.",
    )
    model_name: str = Field(
        default="", description="Name of the dynamics model producing this result."
    )

    @property
    def mechanism_provenance(self) -> MechanismProvenance | None:
        """Typed accessor for mechanism provenance stored in diagnostics."""
        raw = self.diagnostics.get("mechanism_provenance")
        if raw is None:
            return None
        if isinstance(raw, MechanismProvenance):
            return raw
        if isinstance(raw, dict):
            return MechanismProvenance.model_validate(raw)
        return None

    @property
    def context_metadata(self) -> dict[str, Any] | None:
        """Accessor for runtime context version and fingerprint metadata."""
        raw = self.diagnostics.get("context_metadata")
        return dict(raw) if isinstance(raw, dict) else None

    @property
    def residual_feasibility(self) -> ResidualFeasibilityRecord | None:
        """Typed accessor for hybrid residual feasibility diagnostics."""
        raw = self.diagnostics.get("residual_feasibility")
        if raw is None:
            return None
        if isinstance(raw, ResidualFeasibilityRecord):
            return raw
        if isinstance(raw, dict):
            return ResidualFeasibilityRecord.model_validate(raw)
        return None


@runtime_checkable
class DynamicsModel(Protocol):
    """Pluggable interface for defining how a world state evolves over time.

    Dynamics models accept the current WorldState, accepted Actions, exogenous shocks,
    and a seeded random generator, returning a TransitionResult.
    """

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        """Compute the next state given actions, events, and randomness."""
        ...


@runtime_checkable
class ContextualDynamicsModel(Protocol):
    """Protocol for context-conditioned transition mechanisms T(S_t, A_t, E_t | K_t)."""

    @property
    def mechanism_provenance(self) -> MechanismProvenance:
        """Return the formal provenance and metadata contract of this mechanism."""
        ...

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: ContextSnapshot | None = None,
    ) -> TransitionResult:
        """Compute the next state conditioned on runtime_context K_t."""
        ...
