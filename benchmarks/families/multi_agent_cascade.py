"""MultiAgentCascade benchmark family: probes second-order ripple effects and cascading feedback loops.

Tests whether world models capture multi-actor systemic feedback, propagation delays,
and amplification (e.g. the organizational Bullwhip effect) across interdependent entities.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from benchmarks.protocol import (
    Benchmark,
    BenchmarkFamily,
    BenchmarkResult,
    ModelEvaluationSummary,
    compute_benchmark_fingerprint,
)
from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.core import Action, Resource, World, WorldState
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


class UpstreamSupplierActor:
    """Upstream supplier that produces raw parts unless disrupted by an outage."""

    def __init__(self, actor_id: str = "supplier_upstream", production_rate: float = 20.0) -> None:
        self.actor_id = actor_id
        self.production_rate = production_rate

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        # Disrupted from step 1 onward (outage shock)
        if context.step >= 1:
            return []
        return [
            Action(
                id=f"supply_{context.step}",
                type="produce_raw",
                parameters={"value": self.production_rate},
            )
        ]


class IntermediateProcessorActor:
    """Processes raw buffer into finished stock."""

    def __init__(self, actor_id: str = "processor_mid", batch_size: float = 15.0) -> None:
        self.actor_id = actor_id
        self.batch_size = batch_size

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        raw_res = state.get_resource("raw_buffer")
        # Only process if raw buffer has stock
        amount = min(self.batch_size, raw_res.current)
        if amount > 0.0:
            return [
                Action(
                    id=f"process_{context.step}",
                    type="convert_stock",
                    parameters={"value": amount},
                )
            ]
        return []


class DownstreamDistributorActor:
    """Fulfills external consumer demand from finished stock."""

    def __init__(self, actor_id: str = "distributor_downstream", demand: float = 15.0) -> None:
        self.actor_id = actor_id
        self.demand = demand

    def act(self, state: WorldState, context: ActorContext) -> list[Action]:
        finished_res = state.get_resource("finished_stock")
        fulfilled = min(self.demand, finished_res.current)
        unserved = self.demand - fulfilled
        return [
            Action(
                id=f"fulfill_{context.step}",
                type="fulfill_order",
                parameters={"fulfilled": fulfilled, "unserved": unserved},
            )
        ]


class CascadeSupplyChainDynamics(DynamicsModel):
    """Fully coupled systemic dynamics simulating multi-echelon resource flows."""

    def __init__(self, name: str = "CascadeSupplyChainDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        raw_add = 0.0
        raw_sub = 0.0
        finished_add = 0.0
        finished_sub = 0.0
        unserved_demand = 0.0

        for act in actions:
            if act.type == "produce_raw":
                raw_add += float(act.parameters.get("value", 0.0))
            elif act.type == "convert_stock":
                val = float(act.parameters.get("value", 0.0))
                raw_sub += val
                finished_add += val
            elif act.type == "fulfill_order":
                finished_sub += float(act.parameters.get("fulfilled", 0.0))
                unserved_demand += float(act.parameters.get("unserved", 0.0))

        s1 = state.update_resource("raw_buffer", delta=(raw_add - raw_sub), clamp=True)
        s2 = s1.update_resource("finished_stock", delta=(finished_add - finished_sub), clamp=True)
        s3 = s2.update_resource("unserved_total", delta=unserved_demand, clamp=True)

        return TransitionResult(
            next_state=s3,
            applied_changes={"unserved_step": unserved_demand},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class DecoupledMyopicDynamics(DynamicsModel):
    """Decoupled model that ignores intermediate buffer depletion and assumes static downstream flow."""

    def __init__(self, name: str = "DecoupledMyopicDynamics") -> None:
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        # Myopic: assumes finished stock is magically refreshed and never experiences stockouts
        next_state = state.update_resource("finished_stock", new_value=50.0, clamp=True)
        return TransitionResult(
            next_state=next_state,
            applied_changes={"myopic": True},
            evidence_level=EvidenceLevel.PREDICTIVE,
            model_name=self.name,
        )


class MultiAgentCascadeBenchmark(Benchmark):
    """Evaluates multi-agent feedback propagation, ripple delays, and systemic trace depth."""

    def __init__(
        self,
        name: str = "MultiAgentCascade_ThreeTierBullwhip",
        seed: int = 42,
        horizon: int = 8,
    ) -> None:
        self._name = name
        self.seed = seed
        self.horizon = horizon

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> BenchmarkFamily:
        return BenchmarkFamily.MULTI_AGENT_CASCADE

    def _build_world(self, dynamics: DynamicsModel) -> World:
        s0 = WorldState(
            resources={
                "raw_buffer": Resource(
                    id="raw_buffer", current=10.0, min_value=0.0, max_value=200.0
                ),
                "finished_stock": Resource(
                    id="finished_stock", current=15.0, min_value=0.0, max_value=200.0
                ),
                "unserved_total": Resource(
                    id="unserved_total", current=0.0, min_value=0.0, max_value=500.0
                ),
            }
        )
        actors: list[Actor] = [
            UpstreamSupplierActor(production_rate=20.0),
            IntermediateProcessorActor(batch_size=15.0),
            DownstreamDistributorActor(demand=20.0),
        ]
        return World(state=s0, dynamics=dynamics, actors=actors)

    def run(
        self,
        models: Sequence[DynamicsModel] | None = None,
    ) -> BenchmarkResult:
        scenario = Scenario(
            scenario_id="cascade_sim",
            name="Cascade Simulation",
            horizon=self.horizon,
            samples=1,
            seed=self.seed,
        )

        eval_models: list[DynamicsModel] = list(models or [])
        if not eval_models:
            coupled = CascadeSupplyChainDynamics()
            decoupled = DecoupledMyopicDynamics()
            eval_models = [coupled, decoupled]

        engine = SimulationEngine()
        summaries: list[ModelEvaluationSummary] = []

        for m in eval_models:
            m_name = getattr(m, "name", type(m).__name__)
            world = self._build_world(dynamics=m)
            sim_res = engine.run(world, scenario)
            traj = sim_res.trajectories[0]

            final_unserved = traj.final_state.get_resource("unserved_total").current
            trace_edges_count = len(traj.systemic_trace.edges)

            # Measure shock propagation delay: step when unserved demand first occurs (> 0)
            propagation_delay = self.horizon
            for st in traj.steps:
                cur_unserved = st.transition_result.next_state.get_resource(
                    "unserved_total"
                ).current
                if cur_unserved > 0.0:
                    propagation_delay = st.step
                    break

            summaries.append(
                ModelEvaluationSummary(
                    model_name=m_name,
                    metrics={
                        "final_unserved_demand": final_unserved,
                        "cascade_propagation_delay_steps": float(propagation_delay),
                        "systemic_trace_edges": float(trace_edges_count),
                        "final_raw_buffer": traj.final_state.get_resource("raw_buffer").current,
                        "final_finished_stock": traj.final_state.get_resource(
                            "finished_stock"
                        ).current,
                    },
                    invariants_passed=traj.hard_violations == 0,
                    invariant_violations=traj.hard_violations,
                    divergence_or_gap=round(final_unserved, 4),
                    diagnostics={
                        "propagation_delay": propagation_delay,
                        "trace_edges": trace_edges_count,
                    },
                )
            )

        params = {
            "horizon": self.horizon,
            "actors": ["supplier", "processor", "distributor"],
            "disruption_step": 2,
        }
        fp = compute_benchmark_fingerprint(self.name, self.family.value, self.seed, params)

        return BenchmarkResult(
            benchmark_name=self.name,
            family=self.family,
            seed=self.seed,
            provenance_fingerprint=fp,
            timestamp=datetime.now(UTC).isoformat(),
            parameters=params,
            models_evaluated=summaries,
            phenomenon_surfaced=(
                "Multi-Agent Cascade Feedback: Disruption in upstream actors ripples downstream with "
                "delayed multi-echelon stockouts, exposing the failure of myopic decoupled models."
            ),
        )
