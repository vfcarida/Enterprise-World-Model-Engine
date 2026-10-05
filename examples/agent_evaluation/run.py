"""Agent-evaluation example demonstrating verifiable state-based scoring and OR auditability.

Inspired by ARE / Gaia2 (arXiv:2509.17158):
- External decision agents propose actions at each step.
- The EWM Engine simulates dynamics and strictly enforces physical constraints.
- Evaluation is based on verifiable, structured state transitions rather than
  subjective LLM-judge grading (ARE oracle agreement: 0.98 vs 0.72 for LLM judges).
- Demonstrates mathematical auditability: extracts the unsatisfiable core when
  allocations exceed physical limits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.evaluation import compare_scenarios
from ewm_engine.integrations.ortools import CPSATAllocationPlanner
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from examples.agent_evaluation.agents import (
    CPSATOptimizationAgent,
    GreedyMyopicAgent,
    ProportionalHeuristicAgent,
)
from examples.agent_evaluation.environment import create_hospital_world


def run_agent_evaluation() -> dict[str, Any]:
    """Execute end-to-end agent evaluation and auditability demonstration."""
    engine = SimulationEngine()
    horizon = 3
    seed = 42

    # 1. Baseline Scenario: Status Quo (no replenishment actions)
    world_baseline = create_hospital_world()
    scen_baseline = Scenario(
        scenario_id="baseline_status_quo",
        name="Status Quo (No Transfers)",
        horizon=horizon,
        samples=1,
        seed=seed,
    )
    res_baseline = engine.run(world_baseline, scen_baseline)

    # 2. Greedy Myopic Agent: ignores link capacities, triggers constraint rejections
    agent_greedy = GreedyMyopicAgent()
    world_greedy = create_hospital_world(actors=[agent_greedy])
    scen_greedy = Scenario(
        scenario_id="agent_greedy",
        name="Greedy Myopic Agent",
        horizon=horizon,
        samples=1,
        seed=seed,
    )
    res_greedy = engine.run(world_greedy, scen_greedy)

    # 3. Proportional Heuristic Agent: equal split, feasible but inefficient
    agent_prop = ProportionalHeuristicAgent()
    world_prop = create_hospital_world(actors=[agent_prop])
    scen_prop = Scenario(
        scenario_id="agent_proportional",
        name="Proportional Heuristic Agent",
        horizon=horizon,
        samples=1,
        seed=seed,
    )
    res_prop = engine.run(world_prop, scen_prop)

    # 4. CP-SAT Optimization Agent: joint optimization respecting all constraints
    agent_cpsat = CPSATOptimizationAgent()
    world_cpsat = create_hospital_world(actors=[agent_cpsat])
    scen_cpsat = Scenario(
        scenario_id="agent_cpsat",
        name="CP-SAT Optimization Agent",
        horizon=horizon,
        samples=1,
        seed=seed,
    )
    res_cpsat = engine.run(world_cpsat, scen_cpsat)

    # 5. Systemic Outcome Comparison (engine != agent)
    comparison = compare_scenarios(
        baseline=res_baseline,
        candidates=[res_greedy, res_prop, res_cpsat],
        metrics=[
            "resource_cumulative_unserved",
            "violations_count",
            "resource_equity_disparity",
        ],
    )

    # 6. ARE / Gaia2-Inspired Verifiable State Oracle Scoring
    # Computes deterministic state verification directly on structured trajectory states:
    # - Invariant Preservation: total_violations == 0 and stocks >= 0
    # - Service Level: 1.0 - (cumulative_unserved / total_potential_demand)
    # - Regional Equity: 1.0 / (1.0 + equity_disparity)
    total_demand = (30.0 + 45.0 + 25.0) * horizon  # 300.0 units
    oracle_scores: dict[str, dict[str, Any]] = {}

    for name, res in [
        ("Status Quo", res_baseline),
        ("Greedy Myopic", res_greedy),
        ("Proportional", res_prop),
        ("CP-SAT Optimizer", res_cpsat),
    ]:
        traj = res.trajectories[0]
        final_state = traj.final_state
        unserved = final_state.get_resource("cumulative_unserved").current
        disparity = final_state.get_resource("equity_disparity").current
        violations = traj.total_violations

        service_level = max(0.0, 1.0 - (unserved / total_demand))
        equity_score = 1.0 / (1.0 + disparity)
        invariants_preserved = violations == 0

        composite = (service_level * 0.6 + equity_score * 0.4) if invariants_preserved else 0.0

        oracle_scores[name] = {
            "invariants_preserved": invariants_preserved,
            "violations": violations,
            "unserved_demand": unserved,
            "service_level": round(service_level, 4),
            "equity_score": round(equity_score, 4),
            "composite_verifiable_score": round(composite, 4),
        }

    # 7. Auditability Demonstration: Unsat Core / Infeasibility Certificate
    # Introduce an extreme crisis shock exceeding physical link and depot capacity
    planner_audit = CPSATAllocationPlanner(
        actor_id="audit_planner",
        sources=["hub_stock"],
        destinations=["hospital_north", "hospital_central", "hospital_south"],
        demands={"hospital_north": 250, "hospital_central": 200},  # Total 450
        capacities={
            ("hub_stock", "hospital_north"): 60,  # Link limit 60 < demand 250
            ("hub_stock", "hospital_central"): 60,
        },
        time_limit_seconds=5.0,
    )

    crisis_state = WorldState(
        resources=[
            Resource(id="hub_stock", current=100.0, min_value=0.0, max_value=500.0),
            Resource(id="hospital_north", current=0.0, min_value=0.0, max_value=500.0),
            Resource(
                id="hospital_central",
                current=0.0,
                min_value=0.0,
                max_value=500.0,
            ),
            Resource(id="hospital_south", current=0.0, min_value=0.0, max_value=500.0),
        ]
    )

    proposed_actions = planner_audit.propose(state=crisis_state)
    audit_result = planner_audit.last_result
    unsat_core = audit_result.unsat_core if audit_result is not None else ()

    return {
        "comparison": comparison,
        "oracle_scores": oracle_scores,
        "crisis_actions": proposed_actions,
        "crisis_audit_result": audit_result,
        "unsat_core": unsat_core,
    }


def main() -> None:
    """CLI entrypoint for running the agent evaluation example."""
    print("=" * 80)
    print("EWM Engine - Verifiable Agent Evaluation & Auditability Example")
    print("Reference: ARE / Gaia2 (arXiv:2509.17158) - 'Engine != Agent' Thesis")
    print("=" * 80)

    results = run_agent_evaluation()

    print("\n--- Systemic Scenario Comparison (compare_scenarios) ---")
    print(results["comparison"].summary_table())

    print("\n--- ARE Oracle-Graph Verifiable State-Based Scoring ---")
    print(
        f"{'Agent':<25} | {'Invariants':<10} | {'Violations':<10} | {'Service Level':<14} | {'Equity':<8} | {'Score':<8}"
    )
    print("-" * 88)
    for agent_name, sc in results["oracle_scores"].items():
        inv_str = "PASS" if sc["invariants_preserved"] else "FAIL"
        print(
            f"{agent_name:<25} | {inv_str:<10} | {sc['violations']:<10} | "
            f"{sc['service_level'] * 100:>11.1f}% | {sc['equity_score']:>8.3f} | {sc['composite_verifiable_score']:>8.3f}"
        )

    print("\n--- Operations Research Auditability: Infeasibility Certificate ---")
    core = results["unsat_core"]
    print(f"Crisis Shock Allocation Infeasible: {len(core) > 0}")
    print(f"Surfaced Unsat Core Assumptions ({len(core)} constraints):")
    for item in core:
        print(f"  * {item}")

    print("\nEvaluation successfully completed without subjective LLM scoring.")


if __name__ == "__main__":
    main()
