"""Synthetic defect and fault-injection dynamics models for mutation testing.

Used exclusively by metamorphic evaluation suites to inject non-equivalent
mutants and verify mutation adequacy score (Tarp & Chen et al.).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel


class UnmeteredLeakageDynamics:
    """Deliberately leaks resources without logging to applied_changes (violates MR-2 Conservation)."""

    def __init__(
        self,
        resource_id: str = "warehouse_a",
        leak_rate: float = 15.0,
        name: str = "UnmeteredLeakageDynamics",
    ) -> None:
        self.resource_id = resource_id
        self.leak_rate = leak_rate
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        res = dict(state.resources)
        target_id = self.resource_id if self.resource_id in res else next(iter(res), None)
        if target_id is not None and target_id in res:
            r = res[target_id]
            res[target_id] = r.model_copy(update={"current": max(0.0, r.current - self.leak_rate)})
        new_state = state.model_copy(update={"resources": res})
        return TransitionResult(
            next_state=new_state,
            applied_changes={"unmetered_leak": True},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class UnseededJitterDynamics:
    """Deliberately injects non-deterministic variation (violates MR-1 Determinism)."""

    def __init__(
        self,
        resource_id: str = "warehouse_a",
        name: str = "UnseededJitterDynamics",
    ) -> None:
        self.resource_id = resource_id
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        jitter = float((time.time_ns() % 1000) + 1) * 0.05
        res = dict(state.resources)
        target_id = self.resource_id if self.resource_id in res else next(iter(res), None)
        if target_id is not None and target_id in res:
            r = res[target_id]
            res[target_id] = r.model_copy(update={"current": float(r.current + jitter)})
        new_state = state.model_copy(update={"resources": res})
        return TransitionResult(
            next_state=new_state,
            applied_changes={},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class PermutationBiasDynamics:
    """Deliberately introduces bias based on entity naming (violates MR-5 Permutation)."""

    def __init__(
        self,
        resource_id: str = "warehouse_a",
        name: str = "PermutationBiasDynamics",
    ) -> None:
        self.resource_id = resource_id
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        is_permuted = any(str(e_id).startswith("permuted_") for e_id in state.entities)
        bias = 50.0 if is_permuted else 0.0
        res = dict(state.resources)
        target_id = self.resource_id if self.resource_id in res else next(iter(res), None)
        if target_id is not None and target_id in res:
            r = res[target_id]
            res[target_id] = r.model_copy(update={"current": float(r.current + bias)})
        new_state = state.model_copy(update={"resources": res})
        return TransitionResult(
            next_state=new_state,
            applied_changes={},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )
