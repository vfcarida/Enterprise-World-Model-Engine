"""Integration tests for planning layer controller, re-grounding, and constraint guards (P08)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from ewm_engine.actors.base import ActorContext
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core import Action, Resource, WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult
from ewm_engine.experimental.planning import (
    ExpectedObjectiveScorer,
    PlanningActor,
    PlanningCandidate,
    RolloutPlanner,
)
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.mpc import RecedingHorizonSimulator
from ewm_engine.simulation.scenario import Scenario


class WarehouseInventoryDynamics(DynamicsModel):
    """Synthetic inventory dynamics with constant customer demand and restock actions."""

    def __init__(self, step_demand: float = 10.0, name: str = "WarehouseInventoryDynamics") -> None:
        self.step_demand = step_demand
        self.name = name

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[Any],
        rng: Any,
    ) -> TransitionResult:
        stock = state.get_resource("stock").current
        restock_amount = 0.0
        for act in actions:
            if act.type == "restock":
                restock_amount += float(act.parameters.get("quantity", 0.0))

        # Stock after arrival
        available = stock + restock_amount
        served = min(available, self.step_demand)
        new_unserved = self.step_demand - served
        remaining_stock = available - served

        s1 = state.update_resource("stock", new_value=remaining_stock, clamp=True)
        s2 = s1.update_resource("unserved_demand", delta=new_unserved, clamp=True)

        return TransitionResult(
            next_state=s2,
            applied_changes={"demand_served": served, "restock": restock_amount},
            evidence_level=EvidenceLevel.STRUCTURAL,
            model_name=self.name,
        )


class StrictStorageCapacityConstraint:
    """Hard invariant: warehouse stock must never exceed 60.0 units."""

    def __init__(self, max_capacity: float = 60.0) -> None:
        self.max_capacity = max_capacity

    @property
    def constraint_id(self) -> str:
        return "strict_storage_capacity"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return "Storage capacity upper bound"

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        if phase != ConstraintPhase.PRE_ACTION:
            return ConstraintResult(
                satisfied=True,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
            )
        current_stock = state.get_resource("stock").current
        proposed_restock = sum(
            float(a.parameters.get("quantity", 0.0)) for a in actions if a.type == "restock"
        )
        if (current_stock + proposed_restock) > self.max_capacity:
            return ConstraintResult(
                satisfied=False,
                constraint_id=self.constraint_id,
                severity=self.severity,
                phase=phase,
                message=f"Total stock {current_stock + proposed_restock} exceeds maximum {self.max_capacity}",
            )
        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


def _build_test_warehouse_world() -> World:
    s0 = WorldState(
        resources={
            "stock": Resource(id="stock", current=25.0, min_value=0.0, max_value=100.0),
            "unserved_demand": Resource(
                id="unserved_demand", current=0.0, min_value=0.0, max_value=500.0
            ),
        }
    )
    return World(
        state=s0,
        dynamics=WarehouseInventoryDynamics(step_demand=10.0),
        constraints=ConstraintRegistry([StrictStorageCapacityConstraint(max_capacity=60.0)]),
    )


def test_planning_controller_beats_no_plan_baseline() -> None:
    """Controller uses lookahead planning to avoid stockouts, beating a no-order baseline."""
    world = _build_test_warehouse_world()

    # Candidates:
    # 0: Do nothing (0 restock)
    # 1: Replenish 10 units
    # 2: Replenish 20 units
    cand_none = PlanningCandidate.from_action(
        Action(id="c_none", type="restock", parameters={"quantity": 0.0}), id="no_order"
    )
    cand_small = PlanningCandidate.from_action(
        Action(id="c_small", type="restock", parameters={"quantity": 10.0}), id="order_10"
    )
    cand_med = PlanningCandidate.from_action(
        Action(id="c_med", type="restock", parameters={"quantity": 20.0}), id="order_20"
    )
    candidates = [cand_none, cand_small, cand_med]

    # Baseline: no planning (orders 0 every step)
    baseline_sim = RecedingHorizonSimulator(
        lookahead_horizon=3,
        samples_per_candidate=2,
        objective_metric="resource:unserved_demand",
        minimize=True,
        seed=42,
    )
    res_baseline = baseline_sim.run(
        world=world,
        total_steps=5,
        candidates=[cand_none],
    )
    # Baseline starts with 25 stock:
    # Step 0: 25 - 10 = 15
    # Step 1: 15 - 10 = 5
    # Step 2: 5 - 10 = 0 (unserved = 5)
    # Step 3: 0 - 10 = 0 (unserved = 10)
    # Step 4: 0 - 10 = 0 (unserved = 10)
    # Total unserved demand = 25.0
    baseline_unserved = res_baseline.final_state.get_resource("unserved_demand").current
    assert baseline_unserved == pytest.approx(25.0)

    # Planning controller with lookahead=3:
    # Anticipates stock depletion and proactively chooses replenishment!
    controller = RecedingHorizonSimulator(
        lookahead_horizon=3,
        samples_per_candidate=3,
        objective_metric="resource:unserved_demand",
        minimize=True,
        seed=42,
    )
    res_controller = controller.run(
        world=world,
        total_steps=5,
        candidates=candidates,
    )
    controller_unserved = res_controller.final_state.get_resource("unserved_demand").current

    # Controller successfully maintains stock and avoids stockout!
    assert controller_unserved == pytest.approx(0.0, abs=1e-6)
    assert controller_unserved < baseline_unserved

    # Controller selected proactive restocks
    selected_actions = [d.selected_candidate for d in res_controller.decision_history]
    assert any(act in ("order_10", "order_20") for act in selected_actions)


def test_receding_horizon_controller_determinism_under_fixed_seed() -> None:
    """Planning controller execution is strictly reproducible under fixed seed."""
    world = _build_test_warehouse_world()
    cands = [
        PlanningCandidate.from_action(
            Action(id="c0", type="restock", parameters={"quantity": 0.0}), id="idle"
        ),
        PlanningCandidate.from_action(
            Action(id="c1", type="restock", parameters={"quantity": 15.0}), id="restock_15"
        ),
    ]

    sim1 = RecedingHorizonSimulator(lookahead_horizon=3, samples_per_candidate=3, seed=777)
    res1 = sim1.run(world=world, total_steps=4, candidates=cands)

    sim2 = RecedingHorizonSimulator(lookahead_horizon=3, samples_per_candidate=3, seed=777)
    res2 = sim2.run(world=world, total_steps=4, candidates=cands)

    assert len(res1.decision_history) == len(res2.decision_history) == 4
    for d1, d2 in zip(res1.decision_history, res2.decision_history, strict=True):
        assert d1.selected_candidate == d2.selected_candidate
        assert d1.candidate_scores == d2.candidate_scores
        assert d1.state_hash_before == d2.state_hash_before
        assert d1.state_hash_after == d2.state_hash_after

    assert res1.final_state.state_hash == res2.final_state.state_hash


def test_decision_provenance_audit_record() -> None:
    """PlanningDecision captures complete audit metadata and serializes to JSON cleanly."""
    world = _build_test_warehouse_world()
    cands = [
        PlanningCandidate.from_action(
            Action(id="c0", type="restock", parameters={"quantity": 0.0}), id="idle"
        ),
        PlanningCandidate.from_action(
            Action(id="c1", type="restock", parameters={"quantity": 10.0}), id="restock_10"
        ),
    ]

    sim = RecedingHorizonSimulator(lookahead_horizon=2, samples_per_candidate=2, seed=888)
    res = sim.run(world=world, total_steps=2, candidates=cands)

    assert len(res.planning_decisions) == 2
    for p_dec in res.planning_decisions:
        assert p_dec.chosen_candidate_id in ("idle", "restock_10")
        assert "idle" in p_dec.candidate_scores
        assert "restock_10" in p_dec.candidate_scores
        assert p_dec.lookahead_horizon == 2
        assert p_dec.samples_per_candidate == 2
        assert p_dec.seed == 888 + p_dec.step
        assert len(p_dec.state_hash_before) == 64

        # Verify JSON round-trip
        json_data = p_dec.model_dump_json()
        assert "chosen_candidate_id" in json_data


def test_constraint_pipeline_never_bypassed_by_planner() -> None:
    """Actions proposed by planner pass through the constraint engine and cannot breach invariants."""
    world = _build_test_warehouse_world()

    # Propose an illegal action: restock 100 units (violates StrictStorageCapacity max 60.0)
    illegal_cand = PlanningCandidate.from_action(
        Action(id="illegal", type="restock", parameters={"quantity": 100.0}),
        id="illegal_overload",
    )

    sim = RecedingHorizonSimulator(
        lookahead_horizon=2,
        samples_per_candidate=2,
        objective_metric="violations_count",
        minimize=True,
        seed=999,
    )

    # Force selection of the illegal candidate
    res = sim.run(world=world, total_steps=1, candidates=[illegal_cand])

    # The 1-step realization in the world evaluated the constraint pipeline!
    step_record = res.realized_trajectory.steps[0]
    # The action was rejected or generated hard violation
    assert len(step_record.constraint_violations) > 0
    assert any(
        not v.satisfied and v.constraint_id == "strict_storage_capacity"
        for v in step_record.constraint_violations
    )


def test_planning_actor_autonomous_decision_in_simulation() -> None:
    """PlanningActor operates as an autonomous decision agent within standard simulation."""
    world = _build_test_warehouse_world()

    def candidate_gen(state: WorldState, ctx: ActorContext) -> Sequence[PlanningCandidate]:
        current_stock = state.get_resource("stock").current
        # Propose order if stock below 20
        if current_stock < 20.0:
            return [
                PlanningCandidate.from_action(
                    Action(id="ord", type="restock", parameters={"quantity": 15.0}), id="order"
                ),
            ]
        return [
            PlanningCandidate.from_action(
                Action(id="wait", type="restock", parameters={"quantity": 0.0}), id="wait"
            ),
        ]

    planner = RolloutPlanner()
    scorer = ExpectedObjectiveScorer(metric="resource:unserved_demand", minimize=True)
    actor = PlanningActor(
        actor_id="autonomous_receding_actor",
        world_model=world,
        candidate_generator=candidate_gen,
        scorer=scorer,
        lookahead_horizon=2,
        samples_per_candidate=2,
        planner=planner,
    )

    world.add_actor(actor)
    engine = SimulationEngine()
    scenario = Scenario(name="AutonomousActorRun", horizon=4, samples=1, seed=42)

    sim_res = engine.run(world=world, scenario=scenario)
    assert len(sim_res.trajectories) == 1
    assert len(actor.decision_history) == 4
    assert actor.decision_history[0].chosen_candidate_id in ("order", "wait")
