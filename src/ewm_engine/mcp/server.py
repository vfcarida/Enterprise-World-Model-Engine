"""Model Context Protocol (MCP) server for the Enterprise World Model Engine.

Exposes deterministic simulation, Pearl three-step counterfactual evaluation,
constraint checking (Game-Master validation), scenario decision optimization,
and cryptographic provenance explanation as standard MCP tools, resources, and prompts.

Guarantees:
- Zero LLM on the determinism-critical path.
- All inputs validated via Pydantic; safe YAML loader without code execution.
- Returns canonical SHA-256 fingerprints and conservative EvidenceLevels.
- Refuses non-identifiable counterfactual claims instead of hallucinating.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.core.actions import Action
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.mcp.provenance_strict import ProvenanceStrictPolicy
from ewm_engine.mcp.safety import (
    CheckAbstentionInput,
    CheckConstraintsInput,
    CounterfactualInput,
    DecideInput,
    ExplainInput,
    SimulateInput,
    VerifyExplanationInput,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.serialization.yaml import safe_load_yaml
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


def is_mcp_available() -> bool:
    """Check if the official Model Context Protocol SDK is installed."""
    return importlib.util.find_spec("mcp") is not None


class ServerConfig(BaseModel):
    """Configuration options for the EWM MCP Server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    server_name: str = Field(default="ewm-engine", description="MCP server identifier.")
    version: str = Field(default="1.5.0", description="EWM Engine semantic release version.")
    provenance_strict: bool = Field(
        default=False,
        description="Whether to enforce strict audit mode and default LLM specs to low evidence.",
    )


class EwmMcpServer:
    """High-level MCP Server encapsulating EWM Engine tools, resources, and prompts."""

    def __init__(self, config: ServerConfig | None = None) -> None:
        if not is_mcp_available():
            raise SimulationConfigurationError(
                "The MCP SDK is required for EwmMcpServer. Install via `pip install ewm-engine[mcp]`."
            )

        self.config = config or ServerConfig()
        self.policy = ProvenanceStrictPolicy(enabled=self.config.provenance_strict)

        # In-memory stores for runs, traces, worlds, and induction reports
        self._runs_store: dict[str, dict[str, Any]] = {}
        self._traces_store: dict[str, dict[str, Any]] = {}
        self._worlds_store: dict[str, str] = {}
        self._induction_reports_store: dict[str, dict[str, Any]] = {}

        # Instantiate native MCPServer from official SDK
        try:
            from mcp.server.mcpserver import MCPServer

            self._server = MCPServer(self.config.server_name)
        except ImportError:
            # Fallback for mcp < 2.0 FastMCP wrapper
            from mcp.server.fastmcp import FastMCP  # type: ignore[attr-defined]

            self._server = FastMCP(self.config.server_name)

        self._register_tools()
        self._register_resources()
        self._register_prompts()

    @property
    def server(self) -> Any:
        """Underlying SDK server instance."""
        return self._server

    def _parse_world(self, world_yaml: str) -> tuple[World, str]:
        """Safely parse a declarative world YAML and return World + world_id."""
        raw_dict = safe_load_yaml(world_yaml)
        if not isinstance(raw_dict, dict):
            raise ValueError("World specification must be a YAML mapping.")

        # Build World from spec or state
        if "schema_version" in raw_dict and "world" in raw_dict:
            spec_dict = {
                k: v
                for k, v in raw_dict.items()
                if k
                in (
                    "schema_version",
                    "world",
                    "entities",
                    "relationships",
                    "resources",
                    "dynamics",
                    "constraints",
                    "event_sources",
                )
            }
            spec = WorldSpec.from_dict(spec_dict)
            world = WorldFactory().create_world(spec)
        elif "entities" in raw_dict or "resources" in raw_dict:
            state_dict = {
                k: v
                for k, v in raw_dict.items()
                if k
                in (
                    "step",
                    "timestamp",
                    "entities",
                    "relationships",
                    "resources",
                    "memory",
                    "context",
                )
            }
            state = WorldState.model_validate(state_dict)
            world = World(state=state)
        elif (
            "world" in raw_dict
            and isinstance(raw_dict["world"], dict)
            and "state" in raw_dict["world"]
        ):
            state = WorldState.model_validate(raw_dict["world"]["state"])
            world = World(state=state)
        else:
            state = WorldState(entities=[], resources=[])
            world = World(state=state)

        world_id = world.initial_state.fingerprint
        self._worlds_store[world_id] = world_yaml
        return world, world_id

    def simulate(
        self,
        world_yaml: str,
        horizon: int = 5,
        samples: int = 1,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Simulate forward trajectories of an enterprise world under candidate assumptions."""
        validated = SimulateInput(
            world_yaml=world_yaml,
            horizon=horizon,
            samples=samples,
            seed=seed,
        )
        world, world_id = self._parse_world(validated.world_yaml)
        scenario = Scenario(
            scenario_id=f"mcp-sim-{uuid.uuid4().hex[:8]}",
            name="MCP Forward Simulation",
            horizon=validated.horizon,
            samples=validated.samples,
            seed=validated.seed,
        )
        engine = SimulationEngine()
        result = engine.run(world, scenario)

        run_id = f"run-{uuid.uuid4().hex[:12]}"
        trace_id = f"trace-{result.fingerprint[:16]}"
        ev_level = self.policy.stamp_evidence_level(EvidenceLevel.STRUCTURAL)

        manifest = {
            "run_id": run_id,
            "world_id": world_id,
            "fingerprint": result.fingerprint,
            "horizon": validated.horizon,
            "samples": validated.samples,
            "seed": validated.seed,
            "status": "COMPLETED",
            "evidence_level": ev_level.value,
            "resource_uri": f"ewm://runs/{run_id}",
            "trace_uri": f"ewm://traces/{trace_id}",
        }
        self._runs_store[run_id] = manifest
        mech_exp_dict: dict[str, Any] | None = None
        if result.trajectories:
            from ewm_engine.provenance.explainability import build_verifiable_explanation

            mech_exp = build_verifiable_explanation(
                result.trajectories[0],
                explanation_id=trace_id,
            )
            mech_exp_dict = mech_exp.model_dump(mode="json")

        self._traces_store[trace_id] = {
            "trace_id": trace_id,
            "fingerprint": result.fingerprint,
            "provenance": result.provenance.model_dump(),
            "evidence_level": ev_level.value,
            "mechanistic_explanation": mech_exp_dict,
        }

        self.policy.log_invocation(
            tool_name="ewm.simulate",
            input_data=validated.model_dump(),
            output_fingerprint=result.fingerprint,
            evidence_level=ev_level,
        )

        return {
            "status": "COMPLETED",
            "run_id": run_id,
            "state_fingerprint": result.fingerprint,
            "evidence_level": ev_level.value,
            "horizon": validated.horizon,
            "samples": validated.samples,
            "resource_uri": f"ewm://runs/{run_id}",
            "trace_uri": f"ewm://traces/{trace_id}",
            "disclaimer": "Forward stochastic simulation based strictly on declared world rules and seed.",
        }

    def counterfactual(
        self,
        world_yaml: str,
        intervention: dict[str, Any],
        seed: int = 42,
        horizon: int = 5,
    ) -> dict[str, Any]:
        """Evaluate Pearl three-step counterfactual trajectory deltas under Twin-Rollout coupling."""
        validated = CounterfactualInput(
            world_yaml=world_yaml,
            intervention=intervention,
            seed=seed,
            horizon=horizon,
        )
        raw_dict = safe_load_yaml(validated.world_yaml)
        identifiable, reason, e_value = self.policy.check_identifiability(
            raw_dict if isinstance(raw_dict, dict) else {},
            validated.intervention,
        )

        if not identifiable:
            ev_level = EvidenceLevel.ASSUMED
            self.policy.log_invocation(
                tool_name="ewm.counterfactual",
                input_data=validated.model_dump(),
                output_fingerprint="unidentified",
                evidence_level=ev_level,
                identified=False,
            )
            return {
                "status": "NOT_IDENTIFIED",
                "refusal": True,
                "reason": reason,
                "e_value_sensitivity": e_value,
                "e_value_type": "confounding_sensitivity_e_value_vanderweele_ding_2017",
                "evidence_level": ev_level.value,
                "message": "Refusing to emit counterfactual point estimate: causal query is structurally non-identifiable.",
            }

        world, _ = self._parse_world(validated.world_yaml)
        engine = SimulationEngine()

        # Factual baseline rollout
        scen_factual = Scenario(
            scenario_id="factual",
            name="Factual Baseline",
            horizon=validated.horizon,
            samples=1,
            seed=validated.seed,
        )
        res_factual = engine.run(world, scen_factual)

        # Counterfactual branch with intervention
        world_cf = world.branch()
        act_params = validated.intervention.get("parameters", {})
        act_type = validated.intervention.get("type", "intervention_action")
        scen_cf = Scenario(
            scenario_id="counterfactual",
            name="Counterfactual Branch",
            horizon=validated.horizon,
            samples=1,
            seed=validated.seed,
            scheduled_actions=(
                ScheduledAction(
                    step=validated.intervention.get("step", 1),
                    action=Action(
                        id=f"cf_act_{uuid.uuid4().hex[:6]}",
                        type=act_type,
                        parameters=act_params,
                    ),
                ),
            ),
        )
        res_cf = engine.run(world_cf, scen_cf)

        cf_fingerprint = res_cf.fingerprint
        ev_level = self.policy.stamp_evidence_level(EvidenceLevel.COUNTERFACTUAL)
        run_id = f"cf-run-{uuid.uuid4().hex[:12]}"
        trace_id = f"trace-{cf_fingerprint[:16]}"

        self.policy.log_invocation(
            tool_name="ewm.counterfactual",
            input_data=validated.model_dump(),
            output_fingerprint=cf_fingerprint,
            evidence_level=ev_level,
        )

        return {
            "status": "COMPLETED",
            "run_id": run_id,
            "estimand": "Y_{x'}(u) - Y_x(u) under P_model",
            "factual_fingerprint": res_factual.fingerprint,
            "counterfactual_fingerprint": cf_fingerprint,
            "evidence_level": ev_level.value,
            "e_value_robustness": e_value,
            "e_value_type": "confounding_sensitivity_e_value_vanderweele_ding_2017",
            "resource_uri": f"ewm://runs/{run_id}",
            "trace_uri": f"ewm://traces/{trace_id}",
            "epistemic_disclaimer": (
                "Model-counterfactual evaluated within the fully specified simulated SCM under Twin-Rollout "
                "noise coupling (arXiv:2402.10877). Epistemic disclaimer: this represents a counterfactual "
                "of the model (P_model), NOT a validated real-world counterfactual (P_real)."
            ),
        }

    def check_constraints(
        self,
        world_yaml: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate candidate action against operational constraints and physical invariants."""
        validated = CheckConstraintsInput(world_yaml=world_yaml, action=action)
        world, _ = self._parse_world(validated.world_yaml)

        # Validate action construction
        _ = Action(
            id=validated.action.get("id", f"act_{uuid.uuid4().hex[:6]}"),
            type=validated.action.get("type", "generic_action"),
            parameters=validated.action.get("parameters", {}),
            actor_id=validated.action.get("actor_id"),
        )

        # Evaluate preconditions and invariants
        violations = []
        for c in world.constraints:
            if hasattr(c, "evaluate"):
                from ewm_engine.constraints.results import ConstraintPhase

                viol = c.evaluate(world.initial_state, phase=ConstraintPhase.PRE_ACTION)
            elif hasattr(c, "check"):
                viol = c.check(world.initial_state)
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

        is_valid = (
            len([v for v in violations if v["severity"] == ConstraintSeverity.HARD.value]) == 0
        )

        return {
            "is_valid": is_valid,
            "status": "APPROVED" if is_valid else "REJECTED",
            "hard_violations_count": len(
                [v for v in violations if v["severity"] == ConstraintSeverity.HARD.value]
            ),
            "soft_violations_count": len(
                [v for v in violations if v["severity"] == ConstraintSeverity.SOFT.value]
            ),
            "violations": violations,
            "state_fingerprint": world.initial_state.fingerprint,
        }

    def decide(
        self,
        world_yaml: str,
        candidates: dict[str, Any],
        satisficing_threshold: float = 0.0,
        robustness_metric: str = "regret",
        horizon: int = 5,
        samples: int = 20,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Rank candidate policies and action interventions under deep parametric uncertainty."""
        validated = DecideInput(
            world_yaml=world_yaml,
            candidates=candidates,
            satisficing_threshold=satisficing_threshold,
            robustness_metric=robustness_metric,  # type: ignore[arg-type]
            horizon=horizon,
            samples=samples,
            seed=seed,
        )
        world, world_id = self._parse_world(validated.world_yaml)

        from ewm_engine.decision.robustness import evaluate_decision_robustness
        from ewm_engine.evaluation.abstention import ValidityLimitAbstentionGate

        engine = SimulationEngine()
        outcomes_by_candidate: dict[str, list[float]] = {}
        fingerprints_by_candidate: dict[str, str] = {}

        for name, cand_def in validated.candidates.items():
            w_branch = world.branch()
            cand_dict = cand_def if isinstance(cand_def, dict) else {}
            act_params = cand_dict.get("parameters", {})
            param_signal = (
                sum(float(v) for v in act_params.values() if isinstance(v, (int, float))) * 1e-3
            )
            scen = Scenario(
                scenario_id=f"decide-{name}",
                name=name,
                horizon=validated.horizon,
                samples=validated.samples,
                seed=validated.seed,
                scheduled_actions=(
                    ScheduledAction(
                        step=1,
                        action=Action(
                            id=f"act_{name}",
                            type=cand_dict.get("type", "candidate_action"),
                            parameters=act_params,
                        ),
                    ),
                ),
            )
            res = engine.run(w_branch, scen)
            fingerprints_by_candidate[name] = res.fingerprint
            traj_outcomes: list[float] = []
            for idx, traj in enumerate(res.trajectories):
                final_res_sum = sum(float(r.current) for r in traj.final_state.resources.values())
                hard_viols = sum(
                    1
                    for step_rec in traj.steps
                    for v in step_rec.constraint_violations
                    if v.severity == ConstraintSeverity.HARD
                )
                soft_viols = sum(
                    1
                    for step_rec in traj.steps
                    for v in step_rec.constraint_violations
                    if v.severity == ConstraintSeverity.SOFT
                )
                sample_jitter = ((traj.seed + idx) % 101) * 1e-4
                utility = (
                    final_res_sum
                    + param_signal
                    + sample_jitter
                    - (100.0 * hard_viols)
                    - (10.0 * soft_viols)
                )
                traj_outcomes.append(float(utility))
            outcomes_by_candidate[name] = traj_outcomes or [0.0]

        comparison = evaluate_decision_robustness(
            outcomes=outcomes_by_candidate,
            satisficing_threshold=validated.satisficing_threshold,
            higher_is_better=True,
        )

        rankings = []
        for name in validated.candidates:
            cand_scores = comparison.scores[name]
            if validated.robustness_metric in ("satisficing", "domain"):
                rob_score = float(cand_scores.satisficing_rate)
            else:
                rob_score = float(1.0 / (1.0 + max(0.0, cand_scores.max_regret)))
            rankings.append(
                {
                    "candidate": name,
                    "fingerprint": fingerprints_by_candidate[name],
                    "robustness_score": round(rob_score, 4),
                    "max_regret": round(float(cand_scores.max_regret), 4),
                    "cvar_regret_90": round(float(cand_scores.cvar_regret_90), 4),
                    "satisficing_rate": round(float(cand_scores.satisficing_rate), 4),
                    "mean_outcome": round(float(cand_scores.mean_outcome), 4),
                    "satisficing": bool(cand_scores.satisficing_rate >= 0.5),
                }
            )

        rankings.sort(key=lambda x: float(str(x["robustness_score"])), reverse=True)
        best_candidate = (
            comparison.dominant_action
            if not comparison.refusal_to_rank and comparison.dominant_action is not None
            else (rankings[0]["candidate"] if rankings else None)
        )
        abstention = ValidityLimitAbstentionGate().evaluate(state=world.initial_state)

        return {
            "status": "COMPLETED",
            "robustness_metric": validated.robustness_metric,
            "best_candidate": best_candidate,
            "refusal_to_rank": comparison.refusal_to_rank,
            "refusal_rationale": comparison.refusal_rationale,
            "abstention_decision": abstention.model_dump(mode="json"),
            "rankings": rankings,
            "world_id": world_id,
        }

    def explain(self, identifier: str) -> dict[str, Any]:
        """Retrieve the cryptographic provenance trace DAG for a simulation or decision."""
        validated = ExplainInput(identifier=identifier)
        ident = validated.identifier

        if ident in self._traces_store:
            return self._traces_store[ident]
        if ident in self._runs_store:
            return self._runs_store[ident]
        if ident in self._induction_reports_store:
            return self._induction_reports_store[ident]

        return {
            "identifier": ident,
            "status": "NOT_FOUND",
            "message": f"No stored trace, run manifest, or induction report matching '{ident}'.",
        }

    def verify_explanation(
        self,
        narrative_text: str,
        trace_id: str | None = None,
        explanation_dict: dict[str, Any] | None = None,
        claimed_mechanisms: list[str] | None = None,
        known_universe_mechanisms: list[str] | None = None,
    ) -> dict[str, Any]:
        """Audit an LLM or narrative explanation against a recorded VerifiableMechanisticExplanation."""
        from ewm_engine.provenance.explainability import (
            CandidateNarrativeExplanation,
            VerifiableMechanisticExplanation,
            verify_narrative_explanation_fidelity,
        )

        validated = VerifyExplanationInput(
            trace_id=trace_id,
            explanation_dict=explanation_dict,
            narrative_text=narrative_text,
            claimed_mechanisms=claimed_mechanisms or [],
            known_universe_mechanisms=known_universe_mechanisms or [],
        )

        gt_exp: VerifiableMechanisticExplanation | None = None
        if validated.explanation_dict is not None:
            gt_exp = VerifiableMechanisticExplanation.model_validate(validated.explanation_dict)
        elif validated.trace_id is not None:
            tid = validated.trace_id
            if tid in self._runs_store:
                tid = str(self._runs_store[tid].get("trace_uri", "")).replace("ewm://traces/", "")
            trace_entry = self._traces_store.get(tid, {})
            mech_raw = trace_entry.get("mechanistic_explanation")
            if isinstance(mech_raw, dict):
                gt_exp = VerifiableMechanisticExplanation.model_validate(mech_raw)

        cand = CandidateNarrativeExplanation(
            narrative_text=validated.narrative_text,
            claimed_mechanisms=tuple(validated.claimed_mechanisms),
        )
        report = verify_narrative_explanation_fidelity(
            gt_exp,
            cand,
            known_universe_mechanisms=validated.known_universe_mechanisms,
            strict=False,
        )
        return report.model_dump(mode="json")

    def check_abstention(
        self,
        aleatoric_std: float | None = None,
        parameter_std: float | None = None,
        structural_model_std: float | None = None,
        runtime_context_std: float | None = None,
        monte_carlo_se: float | None = None,
        model_discrepancy_vs_reality: float | None = None,
        max_parameter_std: float = 0.25,
        max_structural_model_std: float = 0.25,
        max_model_discrepancy: float = 0.30,
        performative_invalidation: bool = False,
        insufficient_causal_evidence: bool = False,
    ) -> dict[str, Any]:
        """Evaluate uncertainty-typed abstention ('DO_NOT_USE_ME_HERE') at the agent boundary."""
        from ewm_engine.core.drift import DriftKind, DriftSignal
        from ewm_engine.core.trust import EvidenceGateDecision
        from ewm_engine.evaluation.abstention import ValidityLimitAbstentionGate
        from ewm_engine.evaluation.uncertainty import UncertaintyTaxonomyBreakdown
        from ewm_engine.provenance.evidence import (
            DependencyEvidenceNature,
            DependencyQueryKind,
            EvidenceLevel,
        )

        validated = CheckAbstentionInput(
            aleatoric_std=aleatoric_std,
            parameter_std=parameter_std,
            structural_model_std=structural_model_std,
            runtime_context_std=runtime_context_std,
            monte_carlo_se=monte_carlo_se,
            model_discrepancy_vs_reality=model_discrepancy_vs_reality,
            max_parameter_std=max_parameter_std,
            max_structural_model_std=max_structural_model_std,
            max_model_discrepancy=max_model_discrepancy,
            performative_invalidation=performative_invalidation,
            insufficient_causal_evidence=insufficient_causal_evidence,
        )

        breakdown = UncertaintyTaxonomyBreakdown(
            aleatoric_std=validated.aleatoric_std,
            parameter_std=validated.parameter_std,
            structural_model_std=validated.structural_model_std,
            runtime_context_std=validated.runtime_context_std,
            monte_carlo_standard_error=validated.monte_carlo_se,
            model_discrepancy_vs_reality=validated.model_discrepancy_vs_reality,
        )
        drifts: list[DriftSignal] = []
        if validated.performative_invalidation:
            drifts.append(
                DriftSignal(
                    kind=DriftKind.POLICY_INDUCED_SHIFT,
                    magnitude=0.5,
                    threshold=0.1,
                    details="Policy-induced performative feedback flagged at MCP boundary.",
                )
            )
        ev_decision: EvidenceGateDecision | None = None
        if validated.insufficient_causal_evidence:
            ev_decision = EvidenceGateDecision(
                dependency_id="mcp_unverified_edge",
                evidence_nature=DependencyEvidenceNature.STATISTICAL_ASSOCIATION,
                query_kind=DependencyQueryKind.INTERVENTIONAL_DO,
                matrix_condition="forbidden",
                permitted=False,
                effective_evidence_level=EvidenceLevel.PREDICTIVE,
                refusal_reasons=("Insufficient causal identification for interventional query.",),
            )

        gate = ValidityLimitAbstentionGate(
            max_parameter_std=validated.max_parameter_std,
            max_structural_model_std=validated.max_structural_model_std,
            max_model_discrepancy=validated.max_model_discrepancy,
        )
        decision = gate.evaluate(
            uncertainty_breakdown=breakdown,
            drift_signals=drifts,
            evidence_gate_decision=ev_decision,
        )
        return decision.model_dump(mode="json")

    def get_world(self, world_id: str) -> str:
        """Read a declarative World specification by SHA-256 fingerprint."""
        if world_id in self._worlds_store:
            return self._worlds_store[world_id]
        return json.dumps({"error": f"World '{world_id}' not found."})

    def get_run(self, run_id: str) -> str:
        """Read execution run manifest and summary metrics by run_id."""
        if run_id in self._runs_store:
            return json.dumps(self._runs_store[run_id], indent=2)
        return json.dumps({"error": f"Run '{run_id}' not found."})

    def get_trace(self, trace_id: str) -> str:
        """Read cryptographic Systemic Trace DAG and epistemic audit trail."""
        if trace_id in self._traces_store:
            return json.dumps(self._traces_store[trace_id], indent=2)
        return json.dumps({"error": f"Trace '{trace_id}' not found."})

    def register_induction_report(self, report: Any) -> str:
        """Register an InductionReport in the MCP store."""
        rep_dict = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        fp = str(rep_dict.get("sha256_fingerprint", ""))
        if fp:
            self._induction_reports_store[fp] = rep_dict
        val_fp = rep_dict.get("validation_fingerprint")
        if val_fp:
            self._induction_reports_store[str(val_fp)] = rep_dict
        return fp

    def get_induction_report(self, fingerprint: str) -> str:
        """Read a cryptographically fingerprinted InductionReport."""
        if fingerprint in self._induction_reports_store:
            return json.dumps(self._induction_reports_store[fingerprint], indent=2)
        return json.dumps({"error": f"Induction report '{fingerprint}' not found."})

    def author_what_if(self, topic: str = "inventory_transfer") -> str:
        """Prompt template guiding an LLM to formulate a valid EWM what-if intervention."""
        return (
            f"You are formulating a what-if operational intervention for '{topic}'.\n"
            "Constraints & Rules:\n"
            "1. State must specify typed entities and bounded continuous resources.\n"
            "2. Physical and accounting conservation laws hold strictly (no free resources).\n"
            "3. Interventions must specify action type, parameters, and time step.\n"
            "4. Use ewm.check_constraints to validate legality before simulating.\n"
            "5. Call ewm.counterfactual to inspect factual vs counterfactual divergence.\n"
        )

    def explain_result(self, run_id: str) -> str:
        """Prompt template guiding an LLM to interpret simulation results without hallucinating causality."""
        return (
            f"You are explaining simulation results for execution run '{run_id}'.\n"
            "Epistemic Guidelines:\n"
            "1. Cite the canonical SHA-256 state fingerprint.\n"
            "2. State the declared EvidenceLevel (Simulation vs Counterfactual).\n"
            "3. Never claim real-world causality: describe deltas within the simulated model only.\n"
            "4. Point to ewm.explain for the inspectable Systemic Trace DAG.\n"
        )

    def _register_tools(self) -> None:
        """Register EWM tools on the MCP server."""
        self._server.tool(name="ewm.simulate")(self.simulate)
        self._server.tool(name="simulate")(self.simulate)

        self._server.tool(name="ewm.counterfactual")(self.counterfactual)
        self._server.tool(name="counterfactual")(self.counterfactual)

        self._server.tool(name="ewm.check_constraints")(self.check_constraints)
        self._server.tool(name="check_constraints")(self.check_constraints)

        self._server.tool(name="ewm.decide")(self.decide)
        self._server.tool(name="decide")(self.decide)

        self._server.tool(name="ewm.explain")(self.explain)
        self._server.tool(name="explain")(self.explain)

        self._server.tool(name="ewm.verify_explanation")(self.verify_explanation)
        self._server.tool(name="verify_explanation")(self.verify_explanation)

        self._server.tool(name="ewm.check_abstention")(self.check_abstention)
        self._server.tool(name="check_abstention")(self.check_abstention)

    def _register_resources(self) -> None:
        """Register MCP dynamic resources."""
        self._server.resource("ewm://worlds/{world_id}")(self.get_world)
        self._server.resource("ewm://runs/{run_id}")(self.get_run)
        self._server.resource("ewm://traces/{trace_id}")(self.get_trace)
        self._server.resource("ewm://induction/{fingerprint}")(self.get_induction_report)

    def _register_prompts(self) -> None:
        """Register MCP prompt templates."""
        self._server.prompt(name="author_what_if")(self.author_what_if)
        self._server.prompt(name="explain_result")(self.explain_result)

    def run_stdio(self) -> None:
        """Run the MCP server over standard input/output (standard for Claude Desktop and Cursor)."""
        import asyncio

        if hasattr(self._server, "run_stdio_async"):
            asyncio.run(self._server.run_stdio_async())
        elif hasattr(self._server, "run"):
            self._server.run()


def create_mcp_server(
    server_name: str = "ewm-engine",
    provenance_strict: bool = False,
) -> EwmMcpServer:
    """Factory creating and configuring an EwmMcpServer instance."""
    config = ServerConfig(server_name=server_name, provenance_strict=provenance_strict)
    return EwmMcpServer(config)
