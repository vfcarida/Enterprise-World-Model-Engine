"""Deterministic Game-Master bridge for LLM multi-agent simulations (Concordia/Smallville pattern).

Scientific Grounding:
    - Vafa et al. (NeurIPS 2024, arXiv:2406.03689): "Evaluating the World Model Implicit in a Generative Model."
      Demonstrates that LLMs hallucinate state and fail to preserve physical and operational invariants.
    - Kang et al. (ICML 2025, arXiv:2411.02385): "How Far is Video Generation from World Model."
    - Gkountouras et al. (arXiv:2410.19923): "Language Agents Meet Causality."
    - DeepMind Concordia (github.com/google-deepmind/concordia): The Game-Master pattern where agents
      propose natural language intent, and a world arbiter evaluates physical consequences.

EWM Engine serves as the hard, deterministic, auditable WorldArbiter that Concordia lacks:
    LLM Free-Text Intent -> Typed Action Mutation -> Constraint Validation (V_pre) ->
    Physics/Accounting Forward Transition -> Grounded Fact Observation + SHA-256 Fingerprint.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action
from ewm_engine.core.world import World
from ewm_engine.provenance.evidence import EvidenceLevel


class ArbiterVerdict(BaseModel):
    """Immutable outcome of an arbitration step by the deterministic EWM Game-Master."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(description="Decision status: 'APPROVED', 'REJECTED', or 'MODIFIED'.")
    agent_id: str = Field(description="Identifier of the proposing agent.")
    action: Action = Field(description="The validated or modified typed action.")
    is_valid: bool = Field(description="Whether the action passed all hard pre-action constraints.")
    violations: list[dict[str, Any]] = Field(
        default_factory=list, description="Constraint violations detected."
    )
    state_fingerprint_before: str = Field(
        description="SHA-256 state fingerprint prior to transition."
    )
    state_fingerprint_after: str = Field(
        description="SHA-256 state fingerprint following transition."
    )
    grounded_observation: str = Field(
        description="Deterministic factual observation returned to agent."
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Epistemic evidence level of the resulting state observation.",
    )
    trace_id: str = Field(description="Cryptographic trace identifier for auditability.")


class EWMWorldArbiter:
    """Deterministic World Arbiter grounding generative agent simulations in explicit conservation rules."""

    def __init__(self, world: World) -> None:
        self.world = world

    def parse_intent_to_action(
        self,
        agent_id: str,
        intent: str | dict[str, Any],
    ) -> Action:
        """Parse natural language intent or dictionary payload into a structured Action."""
        if isinstance(intent, dict):
            return Action(
                id=intent.get("id", f"act_{uuid.uuid4().hex[:6]}"),
                type=intent.get("type", "agent_action"),
                parameters=intent.get("parameters", {}),
                actor_id=agent_id,
            )

        # Regex heuristic parser for natural language transfer intent: e.g., "transfer 30 from stock_north to stock_south"
        text = str(intent).strip()
        transfer_match = re.search(
            r"transfer\s+([\d.]+)\s+(?:from\s+)?([a-zA-Z0-9_]+)\s+to\s+([a-zA-Z0-9_]+)",
            text,
            re.IGNORECASE,
        )
        if transfer_match:
            qty = float(transfer_match.group(1))
            src = transfer_match.group(2)
            tgt = transfer_match.group(3)
            return Action(
                id=f"nl_act_{uuid.uuid4().hex[:6]}",
                type="transfer_resource",
                parameters={"quantity": qty, "source_resource": src, "target_resource": tgt},
                actor_id=agent_id,
            )

        # Default generic action with intent text stored in parameters
        return Action(
            id=f"nl_act_{uuid.uuid4().hex[:6]}",
            type="generic_intent",
            parameters={"raw_intent": text},
            actor_id=agent_id,
        )

    def arbitrate_intent(
        self,
        agent_id: str,
        intent: str | dict[str, Any],
    ) -> ArbiterVerdict:
        """Evaluate agent intent against world constraints and step the state if valid."""
        fp_before = self.world.initial_state.fingerprint
        action = self.parse_intent_to_action(agent_id, intent)

        # 1. Pre-action constraint check
        violations = []
        for c in self.world.constraints:
            if hasattr(c, "evaluate"):
                from ewm_engine.constraints.results import ConstraintPhase

                viol = c.evaluate(self.world.initial_state, phase=ConstraintPhase.PRE_ACTION)
            elif hasattr(c, "check"):
                viol = c.check(self.world.initial_state)
            else:
                viol = None

            if viol is not None:
                if hasattr(viol, "satisfied") and viol.satisfied:
                    continue
                sev = getattr(viol, "severity", getattr(c, "severity", ConstraintSeverity.HARD))
                sev_val = sev.value if hasattr(sev, "value") else str(sev)
                violations.append(
                    {
                        "constraint_id": getattr(
                            viol,
                            "constraint_id",
                            getattr(c, "constraint_id", getattr(c, "id", c.__class__.__name__)),
                        ),
                        "severity": sev_val,
                        "message": getattr(viol, "message", str(viol)),
                    }
                )

        has_hard_violation = any(v["severity"] == ConstraintSeverity.HARD.value for v in violations)

        if has_hard_violation:
            trace_id = canonical_sha256(
                {"fp": fp_before, "act": action.model_dump(), "approved": False}
            )
            return ArbiterVerdict(
                status="REJECTED",
                agent_id=agent_id,
                action=action,
                is_valid=False,
                violations=violations,
                state_fingerprint_before=fp_before,
                state_fingerprint_after=fp_before,
                grounded_observation=f"Action '{action.type}' rejected by World Arbiter due to constraint violations: {violations}",
                evidence_level=EvidenceLevel.STRUCTURAL,
                trace_id=trace_id,
            )

        # 2. Execute transition if dynamics are attached
        if self.world.dynamics is not None:
            if hasattr(self.world.dynamics, "transition"):
                import numpy as np

                rng = np.random.default_rng(42)
                trans_res = self.world.dynamics.transition(
                    state=self.world.initial_state,
                    actions=[action],
                    exogenous_events=[],
                    rng=rng,
                )
                next_state = trans_res.next_state
            elif hasattr(self.world.dynamics, "step"):
                next_state = self.world.dynamics.step(self.world.initial_state, [action])
            else:
                next_state = self.world.initial_state.advance_time(1.0)
        else:
            # Default state advance
            next_state = self.world.initial_state.advance_time(1.0)

        fp_after = next_state.fingerprint
        # Update world state
        self.world.initial_state = next_state
        trace_id = canonical_sha256(
            {"fp_before": fp_before, "fp_after": fp_after, "action": action.model_dump()}
        )

        observation = (
            f"Action '{action.type}' executed successfully. World advanced to step {next_state.step}. "
            f"Fingerprint: {fp_after[:16]}."
        )

        return ArbiterVerdict(
            status="APPROVED",
            agent_id=agent_id,
            action=action,
            is_valid=True,
            violations=violations,
            state_fingerprint_before=fp_before,
            state_fingerprint_after=fp_after,
            grounded_observation=observation,
            evidence_level=EvidenceLevel.STRUCTURAL,
            trace_id=trace_id,
        )
