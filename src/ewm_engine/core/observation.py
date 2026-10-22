"""Partial observability (O_t) and belief state (B_t) semantics.

Distinguishes the true ontic world state S_t from actor-specific observations
O_t = O(S_t, K_t, a) and estimated belief state B_t while keeping the default
full-observability deterministic path simple and zero-overhead.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


class ObservationSpec(BaseModel):
    """Declarative visibility specification for an actor or observer.

    Any filter field set to ``None`` means unrestricted visibility on that
    dimension; a ``frozenset`` (or set/sequence coerced by Pydantic) restricts
    visibility strictly to matching identifiers or types.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str = "*"
    observable_entity_ids: frozenset[str] | None = None
    observable_entity_types: frozenset[str] | None = None
    observable_resource_ids: frozenset[str] | None = None
    observable_relationship_types: frozenset[str] | None = None
    observable_memory_keys: frozenset[str] | None = None
    observable_rule_keys: frozenset[str] | None = None
    observable_context_keys: frozenset[str] | None = None
    allow_dangling_relationships: bool = False

    @classmethod
    def full_observability(cls, actor_id: str = "*") -> ObservationSpec:
        """Return an unrestricted ObservationSpec (MDP full observability)."""
        return cls(actor_id=actor_id)

    @property
    def is_full_observability(self) -> bool:
        """Return True if no visibility restrictions are active."""
        return (
            self.observable_entity_ids is None
            and self.observable_entity_types is None
            and self.observable_resource_ids is None
            and self.observable_relationship_types is None
            and self.observable_memory_keys is None
            and self.observable_rule_keys is None
            and self.observable_context_keys is None
        )


class ObservationFilter:
    """Projects true ontic WorldState S_t into a leak-free observed WorldState O_t.

    Guarantees that partially observed actors cannot access hidden entities,
    resources, relationships, memory, rules, or context via accidental API
    leakage or shared object references.
    """

    def __init__(
        self,
        specs: ObservationSpec | Mapping[str, ObservationSpec] | None = None,
    ) -> None:
        if specs is None:
            self._default_spec = ObservationSpec.full_observability()
            self._actor_specs: dict[str, ObservationSpec] = {}
        elif isinstance(specs, ObservationSpec):
            if specs.actor_id == "*":
                self._default_spec = specs
                self._actor_specs = {}
            else:
                self._default_spec = ObservationSpec.full_observability()
                self._actor_specs = {specs.actor_id: specs}
        else:
            self._actor_specs = dict(specs)
            self._default_spec = self._actor_specs.get("*", ObservationSpec.full_observability())

    def spec_for_actor(self, actor_id: str) -> ObservationSpec:
        """Resolve the ObservationSpec applicable to ``actor_id``."""
        return self._actor_specs.get(actor_id, self._default_spec)

    def observe(self, state: WorldState, actor_id: str = "*") -> WorldState:
        """Construct an isolated, filtered WorldState view O_t for ``actor_id``."""
        spec = self.spec_for_actor(actor_id)
        if spec.is_full_observability:
            return state.to_ontic_state()

        visible_entities: dict[str, Entity] = {}
        for eid, entity in state.entities.items():
            if spec.observable_entity_ids is not None and eid not in spec.observable_entity_ids:
                continue
            if (
                spec.observable_entity_types is not None
                and entity.type not in spec.observable_entity_types
            ):
                continue
            visible_entities[eid] = copy.deepcopy(entity)

        visible_resources: dict[str, Resource] = {}
        for rid, resource in state.resources.items():
            if spec.observable_resource_ids is not None and rid not in spec.observable_resource_ids:
                continue
            if (
                spec.observable_entity_ids is not None
                and resource.entity_id is not None
                and resource.entity_id not in visible_entities
                and spec.observable_resource_ids is None
            ):
                continue
            visible_resources[rid] = copy.deepcopy(resource)

        visible_relationships: list[Relationship] = []
        for rel in state.relationships:
            if (
                spec.observable_relationship_types is not None
                and rel.type not in spec.observable_relationship_types
            ):
                continue
            if not spec.allow_dangling_relationships and (
                rel.source not in visible_entities or rel.target not in visible_entities
            ):
                continue
            visible_relationships.append(copy.deepcopy(rel))

        visible_memory: dict[str, Any] = {}
        for mkey, mval in state.memory.items():
            if str(mkey).startswith("_epistemic_"):
                continue
            if spec.observable_memory_keys is not None and mkey not in spec.observable_memory_keys:
                continue
            visible_memory[mkey] = copy.deepcopy(mval)

        visible_rules: dict[str, Any] = {}
        for rkey, rval in state.active_rules.items():
            if spec.observable_rule_keys is not None and rkey not in spec.observable_rule_keys:
                continue
            visible_rules[rkey] = copy.deepcopy(rval)

        visible_context: dict[str, Any] = {}
        for ckey, cval in state.context.items():
            if str(ckey).startswith("_epistemic_"):
                continue
            if (
                spec.observable_context_keys is not None
                and ckey not in spec.observable_context_keys
            ):
                continue
            visible_context[ckey] = copy.deepcopy(cval)

        return WorldState(
            schema_version=state.schema_version,
            step=state.step,
            timestamp=state.timestamp,
            entities=visible_entities,
            resources=visible_resources,
            relationships=tuple(visible_relationships),
            memory=visible_memory,
            active_rules=visible_rules,
            context=visible_context,
        )


class BeliefState(BaseModel):
    """Actor's epistemic belief / estimated state B_t under partial observability.

    Optional abstraction for POMDP-style actors that track point estimates and
    epistemic uncertainty over unobserved resources or state variables.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str
    step: int
    observed_state: WorldState
    estimated_hidden_resources: dict[str, float] = Field(default_factory=dict)
    epistemic_uncertainty: dict[str, float] = Field(default_factory=dict)
    observation_history_digests: tuple[str, ...] = Field(default_factory=tuple)


@runtime_checkable
class BeliefUpdater(Protocol):
    """Protocol for updating an actor's BeliefState B_t from new observations O_t."""

    def update_belief(
        self,
        prior_belief: BeliefState | None,
        observation: WorldState,
        actor_id: str,
    ) -> BeliefState:
        """Compute posterior BeliefState B_t given prior B_{t-1} and observation O_t."""
        ...


class ExponentialSmoothingBeliefUpdater:
    """Lightweight deterministic belief updater for partially observed actors."""

    def __init__(
        self,
        prior_resource_estimates: Mapping[str, float] | None = None,
        prior_uncertainty: float = 1.0,
        uncertainty_growth_rate: float = 0.05,
    ) -> None:
        self.prior_resource_estimates = dict(prior_resource_estimates or {})
        self.prior_uncertainty = prior_uncertainty
        self.uncertainty_growth_rate = uncertainty_growth_rate

    def update_belief(
        self,
        prior_belief: BeliefState | None,
        observation: WorldState,
        actor_id: str,
    ) -> BeliefState:
        """Update belief state with observed resource values and growing uncertainty on hidden ones."""
        if prior_belief is None:
            estimates = dict(self.prior_resource_estimates)
            uncertainties = dict.fromkeys(estimates, self.prior_uncertainty)
            history: tuple[str, ...] = ()
        else:
            estimates = dict(prior_belief.estimated_hidden_resources)
            uncertainties = {
                k: v + self.uncertainty_growth_rate
                for k, v in prior_belief.epistemic_uncertainty.items()
            }
            history = prior_belief.observation_history_digests

        for rid, res in observation.resources.items():
            estimates[rid] = res.current
            uncertainties[rid] = 0.0

        new_history = (*history, observation.fingerprint)
        return BeliefState(
            actor_id=actor_id,
            step=observation.step,
            observed_state=observation,
            estimated_hidden_resources=estimates,
            epistemic_uncertainty=uncertainties,
            observation_history_digests=new_history,
        )
