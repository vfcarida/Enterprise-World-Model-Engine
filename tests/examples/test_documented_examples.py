"""Execution tests for documented runnable examples (AC-014 and AC-021)."""

from __future__ import annotations

from typing import Any

import pytest

from ewm_engine import (
    Action,
    Entity,
    Relationship,
    Resource,
    Scenario,
    SimulationEngine,
    World,
    WorldState,
    compare_scenarios,
)
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.scenario import ScheduledAction
from examples.civicflow.run import run_civicflow_simulation
from examples.minimal_warehouse.run import run_minimal_warehouse
from examples.minimal_world.run import run_minimal_world


@pytest.mark.example
def test_readme_quickstart_executes() -> None:
    """AC-021: Ensure the exact 5-minute README Quickstart executes end-to-end and produces stated comparison."""
    # 1. Initialize World State S_0
    state = WorldState(
        entities=[
            Entity(id="wh_north", type="warehouse", attributes={"region": "north"}),
            Entity(id="wh_south", type="warehouse", attributes={"region": "south"}),
        ],
        relationships=[
            Relationship(source="wh_north", target="wh_south", type="connected_to"),
        ],
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ],
    )

    # 2. Attach Constraints and Dynamics
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(resource_id="stock_north"),
            ResourceCapacityConstraint(resource_id="stock_south"),
        ],
    )

    # 3. Simulate Baseline Scenario (Status Quo)
    engine = SimulationEngine()
    scenario_baseline = Scenario(
        scenario_id="baseline",
        name="Status Quo",
        horizon=2,
        samples=1,
        seed=42,
    )
    res_baseline = engine.run(world, scenario_baseline)

    # 4. Branch and Simulate Scenario Intervention
    world_alt = world.branch()
    scenario_intervention = Scenario(
        scenario_id="intervention",
        name="Transfer 30 North -> South",
        horizon=2,
        samples=1,
        seed=42,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(
                    id="act_transfer",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_north",
                        "target_resource": "stock_south",
                        "quantity": 30.0,
                    },
                ),
            ),
        ),
    )
    res_intervention = engine.run(world_alt, scenario_intervention)

    # 5. Evaluate and Compare Scenarios
    comparison = compare_scenarios(
        baseline=res_baseline,
        candidates=[res_intervention],
        metrics=["resource_stock_north", "resource_stock_south", "violations_count"],
    )

    # Assert correct trajectory progression
    traj_base = res_baseline.trajectories[0]
    traj_alt = res_intervention.trajectories[0]

    assert traj_base.final_state.get_resource("stock_north").current == 100.0
    assert traj_base.final_state.get_resource("stock_south").current == 20.0

    assert traj_alt.final_state.get_resource("stock_north").current == 70.0
    assert traj_alt.final_state.get_resource("stock_south").current == 50.0

    table = comparison.summary_table()
    assert "Status Quo" in table
    assert "Transfer 30 North -> South" in table
    assert "-30.00" in table
    assert "+30.00" in table


@pytest.mark.example
def test_readme_scenario_branching_executes() -> None:
    """AC-014: Ensure the documented scenario branching example executes without mutation."""
    state = WorldState(
        resources=[
            Resource(id="stock_main", current=50.0, min_value=0.0, max_value=100.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())
    engine = SimulationEngine()

    policy_branch = world.branch()
    branch_scenario = Scenario(
        scenario_id="branch_policy_b",
        name="Aggressive Reorder",
        horizon=3,
        samples=2,
        seed=101,
    )

    branch_results = engine.run(policy_branch, branch_scenario)
    assert len(branch_results.trajectories) == 2
    assert world.initial_state.fingerprint == state.fingerprint


@pytest.mark.example
def test_readme_constraint_rejection_executes() -> None:
    """AC-014: Ensure documented constraint pre-action rejection snippet executes properly."""
    state = WorldState(
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_b", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[ActionTransferAvailabilityConstraint()],
    )

    invalid_action_scenario = Scenario(
        scenario_id="reject_excess",
        horizon=1,
        scheduled_actions=(
            ScheduledAction(
                step=0,
                action=Action(
                    id="act_overflow",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_a",
                        "target_resource": "stock_b",
                        "quantity": 150.0,
                    },
                ),
            ),
        ),
    )

    res = SimulationEngine().run(world, invalid_action_scenario)
    step = res.trajectories[0].steps[0]

    assert len(step.actions_proposed) == 1
    assert len(step.actions_accepted) == 0
    assert res.trajectories[0].final_state.get_resource("stock_a").current == 100.0


@pytest.mark.example
def test_readme_systemic_trace_executes() -> None:
    """AC-014: Ensure documented systemic trace export snippet executes properly."""
    state = WorldState(
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())
    scenario = Scenario(
        scenario_id="trace_scen",
        horizon=2,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(
                    id="act_transfer",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_north",
                        "target_resource": "stock_south",
                        "quantity": 30.0,
                    },
                ),
            ),
        ),
    )

    res = SimulationEngine().run(world, scenario)
    trajectory = res.trajectories[0]
    trace = trajectory.systemic_trace

    mermaid_markup = trace.to_mermaid()
    assert "flowchart TD" in mermaid_markup
    assert "transfer_resource" in mermaid_markup
    assert "drives" in mermaid_markup
    assert "causes" not in mermaid_markup

    nx_graph = trace.to_networkx()
    assert nx_graph.number_of_nodes() >= 2
    assert nx_graph.number_of_edges() >= 1


@pytest.mark.example
def test_minimal_warehouse_run_script_executes() -> None:
    """Ensure minimal_warehouse run.py script executes end-to-end and returns comparison."""
    comparison = run_minimal_warehouse()
    assert comparison is not None
    assert comparison.baseline_name is not None
    assert len(comparison.results_by_scenario) >= 2
    table = comparison.summary_table()
    assert "served_demand" in table
    assert "unserved_demand" in table


@pytest.mark.example
def test_civicflow_run_script_executes() -> None:
    """Ensure civicflow run.py script executes end-to-end and returns comparison."""
    comparison = run_civicflow_simulation()
    assert comparison is not None
    assert comparison.baseline_name is not None
    assert len(comparison.results_by_scenario) >= 2
    table = comparison.summary_table()
    assert "cumulative_unserved" in table or "unserved" in table


@pytest.mark.example
def test_minimal_world_backward_compatibility() -> None:
    """Ensure minimal_world example continues to execute for backward compatibility."""
    comparison = run_minimal_world()
    assert comparison is not None
    assert comparison.baseline_name is not None


@pytest.mark.example
def test_documented_ortools_cpsat_planner_example_executes() -> None:
    """Ensure documented OR-Tools CP-SAT discrete allocation planner snippet executes."""
    from ewm_engine.integrations.ortools import CPSATAllocationPlanner

    state = WorldState(
        resources=[
            Resource(id="depot_1", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="depot_2", current=80.0, min_value=0.0, max_value=200.0),
            Resource(id="clinic_north", current=0.0, min_value=0.0, max_value=100.0),
            Resource(id="clinic_south", current=0.0, min_value=0.0, max_value=100.0),
        ]
    )

    planner = CPSATAllocationPlanner(
        actor_id="cpsat_logistics",
        sources=["depot_1", "depot_2"],
        destinations=["clinic_north", "clinic_south"],
        demands={"clinic_north": 50, "clinic_south": 40},
        capacities={
            ("depot_1", "clinic_north"): 60,
            ("depot_2", "clinic_south"): 50,
        },
        costs={
            ("depot_1", "clinic_north"): 2,
            ("depot_2", "clinic_south"): 3,
        },
        time_limit_seconds=5.0,
    )

    actions = planner.propose(state=state)
    assert len(actions) == 2
    assert planner.last_result is not None
    assert planner.last_result.satisfied


@pytest.mark.example
def test_documented_scipy_allocation_planner_example_executes() -> None:
    """Ensure documented SciPy continuous allocation planner snippet executes."""
    from ewm_engine.integrations.scipy_planner import SciPyAllocationPlanner

    state = WorldState(
        resources=[
            Resource(id="treasury_reserve", current=250.0, min_value=0.0, max_value=1000.0),
            Resource(id="region_north", current=10.0, min_value=0.0, max_value=500.0),
            Resource(id="region_south", current=5.0, min_value=0.0, max_value=500.0),
        ]
    )

    planner = SciPyAllocationPlanner(
        actor_id="scipy_budget_allocator",
        sources=["treasury_reserve"],
        destinations=["region_north", "region_south"],
        demands={"region_north": 75.5, "region_south": 60.0},
        costs={
            ("treasury_reserve", "region_north"): 1.1,
            ("treasury_reserve", "region_south"): 1.4,
        },
        capacities={
            ("treasury_reserve", "region_north"): 100.0,
            ("treasury_reserve", "region_south"): 80.0,
        },
        time_limit_seconds=5.0,
        method="highs",
    )

    actions = planner.propose(state=state)
    assert len(actions) == 2
    assert planner.last_result is not None
    assert planner.last_result.satisfied


@pytest.mark.example
def test_documented_z3_smt_constraint_example_executes() -> None:
    """Ensure documented Z3 SMT constraint adapter snippet executes."""
    from ewm_engine.constraints.results import ConstraintSeverity
    from ewm_engine.integrations.solvers import Z3ConstraintAdapter

    def verify_budget_bounds(
        state: WorldState, action: Action | None, z3: Any, timeout_ms: int = 5000
    ) -> tuple[bool, str, dict[str, Any]]:
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        cash = z3.Real("cash")
        spend = z3.Real("spend")
        p_nonneg = z3.Bool("p_non_negative_cash")
        current_cash = state.get_resource("cash").current
        solver.add(cash == current_cash)
        spend_qty = action.parameters.get("amount", 0.0) if action else 0.0
        solver.add(spend == spend_qty)
        solver.assert_and_track(cash - spend < 0, p_nonneg)
        check_result = solver.check()
        if check_result == z3.sat:
            return False, "Symbolic invariant violated", {"deficit": True}
        elif check_result == z3.unsat:
            return True, "Symbolic invariant proven", {}
        return False, "Timed out", {"timed_out": True}

    constraint = Z3ConstraintAdapter(
        constraint_id="symbolic_cash_reserve",
        solver_fn=verify_budget_bounds,
        severity=ConstraintSeverity.HARD,
        timeout_ms=3000,
    )

    state = WorldState(
        resources=[Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)]
    )

    res = constraint.check(state=state)
    assert res.satisfied


@pytest.mark.example
def test_documented_gymnasium_snippet_executes() -> None:
    """Ensure documented Gymnasium integration snippet executes."""
    from ewm_engine.integrations.gym import EnterpriseGymEnv

    state = WorldState(
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(resource_id="stock_north"),
            ResourceCapacityConstraint(resource_id="stock_south"),
            ActionTransferAvailabilityConstraint(),
        ],
    )
    candidate_actions = [
        Action(id="noop", type="no_operation"),
        Action(
            id="transfer_10",
            type="transfer_resource",
            parameters={
                "source_resource": "stock_north",
                "target_resource": "stock_south",
                "quantity": 10.0,
            },
        ),
    ]
    env = EnterpriseGymEnv(
        world=world,
        max_steps=5,
        action_mapping=candidate_actions,
        violation_penalty=50.0,
    )
    _obs, info = env.reset(seed=42)
    _obs, _reward, terminated, _truncated, info = env.step(1)
    assert not terminated
    assert info["violations_count"] == 0


@pytest.mark.example
def test_documented_graph_state_snippet_executes() -> None:
    """Ensure documented HeterogeneousGraphView and schema migration snippets execute."""
    from ewm_engine.core.migration import migrate_v1_to_v2, migrate_v2_to_v1
    from ewm_engine.experimental.graph_state import HeterogeneousGraphView

    state = WorldState(
        schema_version="1.0.0",
        entities=[
            Entity(id="wh_north", type="warehouse", attributes={"sqft": 50000.0}),
            Entity(id="retail_south", type="retail", attributes={"demand": 120.0}),
        ],
        relationships=[
            Relationship(
                source="wh_north",
                target="retail_south",
                type="supplies",
                attributes={
                    "transit_days": 2.0,
                    "temporal_valid_from": 0.0,
                    "temporal_valid_until": 20.0,
                },
            ),
        ],
        resources=[
            Resource(
                id="stock_north",
                current=100.0,
                min_value=0.0,
                max_value=200.0,
                entity_id="wh_north",
            ),
            Resource(
                id="stock_south",
                current=20.0,
                min_value=0.0,
                max_value=200.0,
                entity_id="retail_south",
            ),
        ],
    )

    # Heterogeneous graph view
    graph = HeterogeneousGraphView.from_world_state(state)
    assert "warehouse" in graph.node_types
    assert "retail" in graph.node_types

    features = graph.get_node_features("warehouse")
    assert features.shape[0] == 1

    src_idx, dst_idx = graph.get_edge_index("warehouse__supplies__retail")
    assert len(src_idx) == 1
    assert len(dst_idx) == 1

    active_graph = graph.filter_temporal(current_time=15.0)
    assert active_graph.num_edges() >= 1

    # Schema migration v1 <-> v2
    state_v2 = migrate_v1_to_v2(state)
    assert state_v2.schema_version == "2.0.0"
    restored_v1 = migrate_v2_to_v1(state_v2)
    assert restored_v1.schema_version == "1.0.0"


@pytest.mark.example
def test_documented_wsl_snippet_executes() -> None:
    """Ensure documented WSL parsing, compilation, and export snippets execute."""
    from ewm_engine.serialization import (
        compile_wsl,
        dump_wsl_yaml,
        export_wsl,
        parse_wsl_yaml,
    )

    sample_wsl = """
schema_version: "wsl/2.0.0"
metadata:
  id: "test_doc_network"
  name: "Documented Test Network"
  version: "2.0.0"
  author: "Docs Engineer"
  description: "Test network for documented example execution."
temporal:
  time_unit: "step"
  step_duration: 1.0
  default_horizon: 5
entities:
  - id: "hub_1"
    type: "hub"
    attributes:
      capacity: 500.0
  - id: "station_1"
    type: "station"
    attributes:
      rate: 10.0
relationships:
  - source: "hub_1"
    target: "station_1"
    type: "delivers_to"
resources:
  - id: "res_hub"
    current: 100.0
    min_value: 0.0
    max_value: 500.0
    entity_id: "hub_1"
  - id: "res_station"
    current: 10.0
    min_value: 0.0
    max_value: 100.0
    entity_id: "station_1"
dynamics:
  type_id: "deterministic_transfer"
  parameters:
    action_type: "transfer_resource"
constraints:
  - id: "cap_hub"
    type_id: "resource_capacity"
    parameters:
      resource_id: "res_hub"
scenarios:
  - id: "scen_test"
    name: "Doc Scenario"
    seed: 42
    horizon: 2
"""
    doc = parse_wsl_yaml(sample_wsl)
    assert doc.metadata.id == "test_doc_network"

    world = compile_wsl(doc)
    assert "res_hub" in world.initial_state.resources

    exported = export_wsl(world, metadata=doc.metadata)
    yaml_out = dump_wsl_yaml(exported)
    assert "test_doc_network" in yaml_out


@pytest.mark.example
def test_documented_ood_snippet_executes() -> None:
    """Ensure documented OOD support and covariance detection snippets execute."""
    from ewm_engine.experimental.ood import SupportBoundaryOODDetector

    baseline_states = [
        WorldState(
            step=i,
            resources=[
                Resource(id="demand", current=100.0 + i, min_value=0.0, max_value=1000.0),
                Resource(id="inventory", current=500.0 - i, min_value=0.0, max_value=1000.0),
            ],
        )
        for i in range(50)
    ]

    detector = SupportBoundaryOODDetector(tolerance_fraction=0.05)
    detector.fit(baseline_states)

    normal_state = WorldState(
        step=50,
        resources=[
            Resource(id="demand", current=125.0, min_value=0.0, max_value=1000.0),
            Resource(id="inventory", current=475.0, min_value=0.0, max_value=1000.0),
        ],
    )
    assert not detector.is_ood(normal_state)

    ood_state = WorldState(
        step=51,
        resources=[
            Resource(id="demand", current=800.0, min_value=0.0, max_value=1000.0),
            Resource(id="inventory", current=50.0, min_value=0.0, max_value=1000.0),
        ],
    )
    assert detector.is_ood(ood_state)


@pytest.mark.example
def test_documented_causal_diagnostics_snippet_executes() -> None:
    """Ensure documented Causal diagnostics (identifiability, positivity, twin rollout) execute."""
    from ewm_engine.experimental.causal import (
        CausalGraph,
        check_backdoor_identifiability,
        report_confounding_sensitivity,
    )

    # 1. Structural graph backdoor check
    graph = CausalGraph()
    graph.add_node("Z_weather")
    graph.add_node("X_dispatch")
    graph.add_node("Y_delivery_delay")
    graph.add_edge("Z_weather", "X_dispatch")
    graph.add_edge("Z_weather", "Y_delivery_delay")
    graph.add_edge("X_dispatch", "Y_delivery_delay")

    res_unadjusted = check_backdoor_identifiability(
        graph, treatment="X_dispatch", outcome="Y_delivery_delay", conditioning_set=()
    )
    assert not res_unadjusted.is_identifiable

    res_adjusted = check_backdoor_identifiability(
        graph, treatment="X_dispatch", outcome="Y_delivery_delay", conditioning_set=("Z_weather",)
    )
    assert res_adjusted.is_identifiable

    # 2. Confounding sensitivity
    sens_report = report_confounding_sensitivity(
        treated_outcomes=[10.0, 12.0, 14.0],
        control_outcomes=[5.0, 6.0, 7.0],
        gammas=[1.0, 1.5, 2.0],
    )
    assert sens_report.gamma_breakdown >= 1.0

    assert sens_report.gamma_breakdown >= 1.0


@pytest.mark.example
def test_documented_planning_snippet_executes() -> None:
    """Ensure documented planning and receding-horizon simulation snippet executes."""
    from ewm_engine.experimental.planning import (
        CVaRScorer,
        PlanningCandidate,
    )
    from ewm_engine.simulation.mpc import RecedingHorizonSimulator

    state = WorldState(
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())

    candidates = [
        PlanningCandidate.from_action(
            Action(
                id="act_idle",
                type="transfer_resource",
                parameters={
                    "source_resource": "stock_north",
                    "target_resource": "stock_south",
                    "quantity": 0.0,
                },
            ),
            id="idle",
        ),
        PlanningCandidate.from_action(
            Action(
                id="act_transfer_10",
                type="transfer_resource",
                parameters={
                    "source_resource": "stock_north",
                    "target_resource": "stock_south",
                    "quantity": 10.0,
                },
            ),
            id="transfer_10",
        ),
    ]

    scorer = CVaRScorer(metric="resource:stock_south", alpha=0.20, minimize=False)
    simulator = RecedingHorizonSimulator(
        lookahead_horizon=2,
        samples_per_candidate=2,
        scorer=scorer,
        seed=42,
    )

    result = simulator.run(world=world, total_steps=2, candidates=candidates)
    assert len(result.decision_history) == 2
