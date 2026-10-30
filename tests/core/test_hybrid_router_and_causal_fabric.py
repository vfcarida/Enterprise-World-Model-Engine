"""Verification suite for EWM-002: Hybrid Dynamics Router & Causal Fabric.

Validates:
1. `HybridTransitionRouter` (`Pre-rules -> ML -> Post-rules`) separation of concerns.
2. `TransitionEngine.step(state, action)` yielding deterministic outcomes when only structural
   rules apply and calibrated distributions (`CalibratedStateDistribution`) when ML is used.
3. 100% rejection and rollback of out-of-bounds ML predictions and hallucinated invalid actions.
4. Operational safeguard catching numerical `NaN` / `Inf` from learned modules before
   structural `Post-rules` evaluation.
5. `engine.branch().do(action)` severing inbound edges $(u \\to X_t)$ in the causal graph
   for $Y_{t+1} = f(X_t, U_t)$ and disabling observational confounding mechanisms.
6. Negative control: intervening on an independent variable $W_t$ does not affect target $Y_{t+1}$.
7. `Intervention` context manager forcing state nodes and overriding standard & ML dynamics.
8. Backwards compatibility (`legacy_step`, `DynamicsModel.transition`, `World.branch().do()`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pytest

from ewm_engine.causal import (
    CalibratedPrediction,
    CausalGraph,
    HybridTransitionRouter,
    Intervention,
    InterventionContext,
    NumericalInstabilityError,
    TransitionEngine,
    get_active_interventions,
    normalize_node_name,
    parse_intervention_targets,
)
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicDynamics
from ewm_engine.dynamics.learned import LinearResidualDynamics
from ewm_engine.experimental.grey_box import CallableResidualEstimator, GreyBoxDynamics
from ewm_engine.simulation.engine import SimulationEngine


class _MaxDeltaActionConstraint:
    """Hard PRE_ACTION constraint rejecting actions with `delta > max_delta`."""

    def __init__(self, max_delta: float = 50.0) -> None:
        self.max_delta = max_delta

    @property
    def constraint_id(self) -> str:
        return "max_action_delta_limit"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return f"Action delta must not exceed {self.max_delta}."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        if phase == ConstraintPhase.PRE_ACTION:
            for act in actions:
                d = act.parameters.get("delta", 0.0)
                if isinstance(d, (int, float)) and abs(float(d)) > self.max_delta:
                    return ConstraintResult(
                        satisfied=False,
                        constraint_id=self.constraint_id,
                        severity=self.severity,
                        phase=phase,
                        message=f"Action delta {d} exceeds limit {self.max_delta}",
                    )
        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


class _MinimumReservePostConstraint:
    """Hard POST_TRANSITION constraint requiring `Y >= min_reserve`."""

    def __init__(self, min_reserve: float = 10.0) -> None:
        self.min_reserve = min_reserve

    @property
    def constraint_id(self) -> str:
        return "minimum_y_reserve"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return f"Resource Y must stay >= {self.min_reserve}."

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        if phase == ConstraintPhase.POST_TRANSITION and "Y" in state.resources:
            y_val = state.get_resource("Y").current
            if y_val < self.min_reserve:
                return ConstraintResult(
                    satisfied=False,
                    constraint_id=self.constraint_id,
                    severity=self.severity,
                    phase=phase,
                    message=f"Y={y_val:.2f} fell below reserve {self.min_reserve:.2f}",
                    violating_resources=("Y",),
                )
        return ConstraintResult(
            satisfied=True,
            constraint_id=self.constraint_id,
            severity=self.severity,
            phase=phase,
        )


def _build_causal_state() -> WorldState:
    """Create an initial immutable WorldState with confounder Z, treatment X, outcome Y, and independent W."""
    return WorldState(
        step=0,
        timestamp=0.0,
        entities={
            "ent_1": Entity(id="ent_1", type="service", attributes={"mode": "auto"})
        },
        resources={
            "Z": Resource(id="Z", current=10.0, min_value=0.0, max_value=500.0),
            "X": Resource(id="X", current=20.0, min_value=0.0, max_value=200.0),
            "Y": Resource(id="Y", current=50.0, min_value=0.0, max_value=300.0),
            "W": Resource(id="W", current=5.0, min_value=0.0, max_value=1000.0),
        },
    )


def _build_confounded_causal_graph() -> CausalGraph:
    """Construct a causal DAG with confounder Z -> X, Z -> Y, U_X -> X, causal edge X -> Y, and independent W."""
    graph = CausalGraph()
    for node in ("Z", "U_X", "X", "Y", "U_W", "W"):
        graph.add_node(node)
    graph.add_edge("Z", "X")
    graph.add_edge("U_X", "X")
    graph.add_edge("Z", "Y")
    graph.add_edge("X", "Y")
    graph.add_edge("U_W", "W")
    return graph


def test_deterministic_vs_calibrated_distribution_contract() -> None:
    """Verify TransitionEngine.step() returns deterministic outcome without ML and calibrated distribution with ML."""
    state = _build_causal_state()
    graph = _build_confounded_causal_graph()

    structural_eqs = {
        "X": lambda parents, actions, u: 2.0 * parents.get("Z", 0.0) + u.get("U_X", 0.0),
        "Y": lambda parents, actions, u: 1.5 * parents.get("X", 0.0) + 0.5 * parents.get("Z", 0.0) + u.get("U_Y", 0.0),
        "W": lambda parents, actions, u: parents.get("W", 0.0) + 1.0 + u.get("U_W", 0.0),
    }

    # 1. Purely structural engine -> deterministic outcome
    det_engine = TransitionEngine(
        initial_state=state,
        causal_graph=graph,
        structural_equations=structural_eqs,
    )
    det_outcome = det_engine.step(
        state,
        Action(id="act_noop", type="observe"),
        u_noise={"U_X": 3.0, "U_Y": 2.0, "U_W": 0.5},
    )
    assert det_outcome.is_deterministic is True
    assert det_outcome.distribution is None
    # X = 2 * 10 + 3 = 23.0; Y = 1.5 * 23 + 0.5 * 10 + 2 = 41.5; W = 5 + 1 + 0.5 = 6.5
    assert det_outcome.next_state.get_resource("X").current == pytest.approx(23.0)
    assert det_outcome.next_state.get_resource("Y").current == pytest.approx(41.5)
    assert det_outcome.next_state.get_resource("W").current == pytest.approx(6.5)

    # 2. Hybrid engine with ML -> calibrated distribution
    ml_engine = TransitionEngine(
        initial_state=state,
        causal_graph=graph,
        structural_equations=structural_eqs,
        ml_predictor=lambda s, a, struct_s: CalibratedPrediction(
            mean_deltas={"Y": 4.5},
            std_deltas={"Y": 2.0, "X": 1.0},
            calibration_temperature=1.2,
        ),
    )
    ml_outcome = ml_engine.step(
        state,
        Action(id="act_obs", type="observe"),
        u_noise={"U_X": 3.0, "U_Y": 2.0, "U_W": 0.5},
    )
    assert ml_outcome.is_deterministic is False
    assert ml_outcome.distribution is not None
    assert ml_outcome.distribution.is_calibrated is True
    assert ml_outcome.next_state.get_resource("Y").current == pytest.approx(46.0)
    assert ml_outcome.distribution.mean["Y"] == pytest.approx(46.0)
    assert ml_outcome.distribution.std["Y"] == pytest.approx(2.4)
    assert ml_outcome.distribution.lower_95["Y"] < 46.0 < ml_outcome.distribution.upper_95["Y"]

    sample_draw = ml_outcome.distribution.sample(np.random.default_rng(42))
    assert set(sample_draw.keys()) == {"Z", "X", "Y", "W"}


def test_causal_do_severs_inbound_edges_and_negative_control() -> None:
    """Verify `engine.branch().do(action)` severs inbound edges to X_t and passes negative control on W_t."""
    state = _build_causal_state()
    graph = _build_confounded_causal_graph()

    structural_eqs = {
        "X": lambda parents, actions, u: 2.0 * parents.get("Z", 0.0) + u.get("U_X", 0.0),
        "Y": lambda parents, actions, u: 1.5 * parents.get("X", 0.0) + 0.5 * parents.get("Z", 0.0) + u.get("U_Y", 0.0),
        "W": lambda parents, actions, u: parents.get("W", 0.0) + 2.0 + u.get("U_W", 0.0),
    }

    engine = TransitionEngine(
        initial_state=state,
        causal_graph=graph,
        structural_equations=structural_eqs,
    )

    # 1. Intervene on X via `engine.branch().do(action)`
    do_x_action = Action(id="act_do_x", type="do", parameters={"target": "X", "value": 100.0})
    branched_do_x = engine.branch("counterfactual_do_X").do(do_x_action)

    # Verify inbound edges to X (("Z", "X") and ("U_X", "X")) were severed in the branched graph!
    assert ("Z", "X") not in branched_do_x.causal_graph.edges
    assert ("U_X", "X") not in branched_do_x.causal_graph.edges
    # Outbound edge ("X", "Y") and other edges remain intact
    assert ("X", "Y") in branched_do_x.causal_graph.edges
    assert ("Z", "Y") in branched_do_x.causal_graph.edges
    assert ("U_W", "W") in branched_do_x.causal_graph.edges

    # Origin engine's graph must remain untouched (no cross-branch contamination)
    assert ("Z", "X") in engine.causal_graph.edges
    assert ("U_X", "X") in engine.causal_graph.edges

    # Check causal execution log entry for `do_surgery`
    surgery_log = branched_do_x.causal_execution_log[0]
    assert surgery_log.operation == "do_surgery"
    assert surgery_log.intervened_variables == ("X",)
    assert surgery_log.severed_inbound_edges == (("U_X", "X"), ("Z", "X"))
    assert surgery_log.confounding_disabled is True

    # Step the branched engine with large U_X noise: because inbound edges to X are severed,
    # U_X and Z->X confounding have ZERO effect on X, which stays at 100.0, driving Y = 1.5*100 + 0.5*10 = 155.0
    cf_outcome = branched_do_x.step(u_noise={"U_X": 9999.0, "U_Y": 0.0, "U_W": 0.0})
    assert cf_outcome.next_state.get_resource("X").current == pytest.approx(100.0)
    assert cf_outcome.next_state.get_resource("Y").current == pytest.approx(155.0)
    assert cf_outcome.confounding_disabled is True

    # 2. Negative Control: Intervening on independent variable W must NOT affect target Y
    baseline_outcome = engine.branch("baseline").step(
        u_noise={"U_X": 3.0, "U_Y": 1.0, "U_W": 0.0}
    )
    branched_do_w = engine.branch("negative_control_do_W").do(
        Action(id="act_do_w", type="do", parameters={"W": 777.0})
    )
    assert ("U_W", "W") not in branched_do_w.causal_graph.edges
    neg_ctrl_outcome = branched_do_w.step(
        u_noise={"U_X": 3.0, "U_Y": 1.0, "U_W": 0.0}
    )

    assert neg_ctrl_outcome.next_state.get_resource("W").current == pytest.approx(777.0)
    # Target Y and treatment X are completely unaffected by intervening on independent W
    assert neg_ctrl_outcome.next_state.get_resource("X").current == pytest.approx(
        baseline_outcome.next_state.get_resource("X").current
    )
    assert neg_ctrl_outcome.next_state.get_resource("Y").current == pytest.approx(
        baseline_outcome.next_state.get_resource("Y").current
    )


def test_router_rejects_100_percent_of_out_of_bounds_and_hallucinated_ml_predictions() -> None:
    """Verify Post-rules catch and roll back 100% of out-of-bounds ML predictions and invalid ML actions."""
    state = _build_causal_state()  # Y is at 50.0, bounds [0.0, 300.0]
    rng = np.random.default_rng(2026)

    # 1. Test 200 randomized out-of-bounds ML predictions (100 above max_value, 100 below min_value)
    rejected_count = 0
    total_trials = 200

    def _make_oob_predictor(
        delta_val: float,
    ) -> Callable[[WorldState, Sequence[Action], WorldState], dict[str, float]]:
        return lambda s, a, struct_s: {"Y": delta_val}

    for i in range(total_trials):
        if i % 2 == 0:
            # Exceeds upper bound 300.0 (needs delta > 250.0)
            oob_delta = float(rng.uniform(250.01, 5000.0))
        else:
            # Violates lower bound 0.0 (needs delta < -50.0)
            oob_delta = float(rng.uniform(-5000.0, -50.01))

        router = HybridTransitionRouter(
            ml_predictor=_make_oob_predictor(oob_delta),
            constraints=[_MaxDeltaActionConstraint(max_delta=50.0), _MinimumReservePostConstraint(min_reserve=10.0)],
        )
        outcome = router.step(state, Action(id=f"act_{i}", type="noop"))
        if outcome.post_rule_rejected and outcome.ml_rejected and outcome.rolled_back:
            rejected_count += 1
        # Verify state was rolled back to valid structural state (Y == 50.0)
        assert outcome.next_state.get_resource("Y").current == pytest.approx(50.0)

    assert rejected_count == total_trials  # 100% rejection rate!

    # 2. Rollback test: Invalid actions triggered by ML hallucination must be caught by Post-rules
    hallucinating_router = HybridTransitionRouter(
        ml_predictor=lambda s, a, struct_s: CalibratedPrediction(
            mean_deltas={"Y": 5.0},
            std_deltas={"Y": 1.0},
            proposed_actions=(
                Action(
                    id="ml_hallucinated_act",
                    type="unauthorized_transfer",
                    parameters={"resource_id": "Y", "delta": 999.0, "invalid": True},
                ),
            ),
        ),
        constraints=[_MaxDeltaActionConstraint(max_delta=50.0)],
    )
    hallucinated_outcome = hallucinating_router.step(state, Action(id="valid_act", type="observe"))
    assert hallucinated_outcome.post_rule_rejected is True
    assert hallucinated_outcome.ml_rejected is True
    assert hallucinated_outcome.rolled_back is True
    assert "ml_hallucinated_act" in hallucinated_outcome.rejected_actions
    assert hallucinated_outcome.next_state.get_resource("Y").current == pytest.approx(50.0)


def test_nan_and_inf_operational_safeguard_before_structural_evaluation() -> None:
    """Verify numerical NaN and Inf from learned modules are caught BEFORE structural evaluation."""
    state = _build_causal_state()
    post_rule_called = False

    def _spy_post_rule(
        struct_s: WorldState, cand_s: WorldState, actions: Sequence[Action]
    ) -> tuple[bool, str]:
        nonlocal post_rule_called
        post_rule_called = True
        return True, "ok"

    # 1. Soft catch mode (`raise_on_nan=False`): catches NaN before _spy_post_rule is ever called
    nan_router = HybridTransitionRouter(
        ml_predictor=lambda s, a, struct_s: {"Y": float("nan")},
        post_rules=[_spy_post_rule],
        raise_on_nan=False,
    )
    outcome = nan_router.step(state)
    assert outcome.nan_caught_before_post_rules is True
    assert outcome.ml_rejected is True
    assert outcome.rolled_back is True
    assert post_rule_called is False  # Caught BEFORE post-rule structural evaluation!
    assert outcome.next_state.get_resource("Y").current == pytest.approx(50.0)

    # 2. Strict mode (`raise_on_nan=True`): raises NumericalInstabilityError immediately
    strict_nan_router = HybridTransitionRouter(
        ml_predictor=lambda s, a, struct_s: CalibratedPrediction(
            mean_deltas={"Y": 1.0},
            std_deltas={"Y": float("inf")},
        ),
        raise_on_nan=True,
    )
    with pytest.raises(NumericalInstabilityError, match="Numerical instability detected"):
        strict_nan_router.step(state)

    # 3. Verify GreyBoxDynamics and LinearResidualDynamics also catch NaN before structural evaluation
    known_dyn = DeterministicDynamics()
    grey_nan = GreyBoxDynamics(
        known_model=known_dyn,
        residual_estimator=CallableResidualEstimator(lambda s, a, e, k: {"Y": float("nan")}),
    )
    with pytest.raises(NumericalInstabilityError, match="Numerical instability detected"):
        grey_nan.transition(state, (), (), np.random.default_rng(0))

    lin_dyn = LinearResidualDynamics(target_resource="Y", action_type="boost")
    lin_dyn.action_weight = float("nan")
    with pytest.raises(NumericalInstabilityError, match="Numerical instability detected"):
        lin_dyn.transition(state, (Action(id="a1", type="boost"),), (), np.random.default_rng(0))


def test_intervention_context_manager_forces_nodes_and_overrides_dynamics() -> None:
    """Verify `with Intervention(...):` and `with engine.intervene(...):` override standard and ML dynamics."""
    state = _build_causal_state()
    graph = _build_confounded_causal_graph()

    engine = TransitionEngine(
        initial_state=state,
        causal_graph=graph,
        structural_equations={
            "X": lambda parents, actions, u: 2.0 * parents.get("Z", 0.0),
            "Y": lambda parents, actions, u: 1.5 * parents.get("X", 0.0),
        },
        ml_predictor=lambda s, a, struct_s: {"X": 15.0, "Y": 10.0},
    )

    # 1. Direct `with Intervention(...):` context manager
    iv = Intervention(
        id="ctx_force_x",
        description="Force X=80 and ent_1.mode=manual",
        target_type="node_override",
        parameters={"X": 80.0, "entity:ent_1:mode": "manual", "memory:override_flag": True},
    )
    assert not iv.is_active
    with iv:
        active_flag = bool(iv.is_active)
        assert active_flag is True
        assert len(get_active_interventions()) == 1
        forced_state = iv.apply(state)
        assert forced_state.get_resource("X").current == pytest.approx(80.0)
        assert forced_state.get_entity("ent_1").attributes["mode"] == "manual"
        assert forced_state.memory["override_flag"] is True

        # Stepping the engine inside `with iv:` overrides both structural equation (2*Z=20)
        # and ML prediction (+15.0) for X, keeping X locked at 80.0 and driving Y = 1.5*80 + 10 = 130.0
        ctx_outcome = engine.step(state)
        assert ctx_outcome.next_state.get_resource("X").current == pytest.approx(80.0)
        assert ctx_outcome.next_state.get_resource("Y").current == pytest.approx(130.0)
        assert ctx_outcome.confounding_disabled is True
        assert ("Z", "X") in ctx_outcome.severed_edges

    after_flag = bool(iv.is_active)
    assert after_flag is False
    assert get_active_interventions() == ()

    # 2. Engine-bound `with engine.intervene(...):` context manager restores causal graph on exit
    assert ("Z", "X") in engine.causal_graph.edges
    with engine.intervene({"resource:X": 40.0}) as ctx:
        assert isinstance(ctx, InterventionContext)
        assert ctx.forced_nodes == {"resource:X": 40.0}
        assert ("Z", "X") not in engine.causal_graph.edges
        s_mod = ctx.apply_to_state(state)
        assert s_mod.get_resource("X").current == pytest.approx(40.0)
        step_in_ctx = engine.step(state)
        assert step_in_ctx.next_state.get_resource("X").current == pytest.approx(40.0)

    # Causal graph edges restored after exiting context manager
    assert ("Z", "X") in engine.causal_graph.edges


def test_pre_rules_and_backwards_compatibility_surfaces() -> None:
    """Exercise all Pre-rules rejection paths, predictors, and backwards-compatible step() wrappers."""
    state = _build_causal_state()

    # 1. Pre-rules rejection: invalid action flag, unknown target resource, non-finite delta, out-of-bounds delta
    router = HybridTransitionRouter(
        constraints=[_MaxDeltaActionConstraint(max_delta=50.0)],
    )
    bad_actions = [
        Action(id="bad_flag", type="op", parameters={"invalid": True}),
        Action(id="bad_target", type="op", parameters={"resource_id": "NON_EXISTENT", "delta": 5.0}),
        Action(id="bad_nan", type="op", parameters={"resource_id": "X", "delta": float("nan")}),
        Action(id="bad_bounds", type="op", parameters={"resource_id": "X", "delta": 999.0}),
    ]
    pre_rej_outcome = router.step(state, bad_actions)
    assert pre_rej_outcome.pre_rule_rejected is True
    assert pre_rej_outcome.rolled_back is True
    assert set(pre_rej_outcome.rejected_actions) == {"bad_flag", "bad_target", "bad_nan", "bad_bounds"}

    # Custom pre_rule function & raise_on_pre_rule_violation
    strict_pre_router = HybridTransitionRouter(
        pre_rules=[lambda s, acts: (False, "freeze_window_active")],
        raise_on_pre_rule_violation=True,
    )
    with pytest.raises(ValueError, match="freeze_window_active"):
        strict_pre_router.step(state, Action(id="a1", type="op"))

    # 2. DynamicsModel protocol compatibility (`router.transition(...)`) and `legacy_step(...)`
    engine = TransitionEngine(initial_state=state)
    next_s = engine.legacy_step(
        state,
        Action(id="good_act", type="adjust", parameters={"resource_id": "X", "delta": 10.0}),
    )
    assert next_s.get_resource("X").current == pytest.approx(30.0)

    trans_res = router.transition(
        state,
        [Action(id="good_act2", type="adjust", parameters={"resource_id": "X", "delta": 5.0})],
        (),
        np.random.default_rng(0),
    )
    assert trans_res.next_state.get_resource("X").current == pytest.approx(25.0)
    assert router.mechanism_provenance.mechanism_id == "HybridTransitionRouter"

    # 3. World.branch().do() and SimulationEngine.branch().do() compatibility
    world = World(state=state)
    cf_world = world.branch().do({"X": 75.0})
    w_outcome = cf_world.step()
    assert w_outcome.next_state.get_resource("X").current == pytest.approx(75.0)

    sim_engine = SimulationEngine()
    sim_branch = sim_engine.branch().do({"X": 88.0})
    sim_outcome = sim_branch.step(state)
    assert sim_outcome.next_state.get_resource("X").current == pytest.approx(88.0)
    sim_direct = sim_engine.step(state, Action(id="s_act", type="adjust", parameters={"resource_id": "X", "delta": 2.0}))
    assert sim_direct.next_state.get_resource("X").current == pytest.approx(22.0)

    # 4. Additional helper and validation branches for 100% coverage
    assert parse_intervention_targets(
        Action(id="a_t", type="do", parameters={"target": "X", "value": 12.0, "targets": {"Y": 34.0}})
    ) == {"X": 12.0, "Y": 34.0}
    assert parse_intervention_targets(
        Action(id="a_t2", type="do", parameters={"targets": {"X": 15.0}})
    ) == {"X": 15.0}
    with pytest.raises(TypeError, match="Expected Action, Intervention, or Mapping"):
        parse_intervention_targets(12345)  # type: ignore[arg-type]

    g_prefixed = CausalGraph(nodes={"resource:X"}, edges=set())
    assert normalize_node_name("X", g_prefixed) == "resource:X"
    assert normalize_node_name("resource:X", g_prefixed) == "resource:X"
    assert normalize_node_name("X", None) == "X"
    assert parse_intervention_targets(Intervention(id="iv_p", description="d", parameters={"X": 9.0})) == {"X": 9.0}

    # 5. Exercise LearnedPredictorProtocol, ResidualEstimator, DynamicsModel ml_predictor, and error guards
    import ewm_engine.causal as causal_pkg
    from ewm_engine.dynamics.router import validate_learned_prediction_no_nan

    with pytest.raises(AttributeError):
        _ = causal_pkg.non_existent_symbol

    with pytest.raises(NumericalInstabilityError, match="non-numeric"):
        validate_learned_prediction_no_nan({"X": "not_a_float"})
    with pytest.raises(NumericalInstabilityError, match="negative std"):
        validate_learned_prediction_no_nan(
            CalibratedPrediction(mean_deltas={"X": 1.0}, std_deltas={"X": -0.5})
        )
    with pytest.raises(NumericalInstabilityError, match="calibration_temperature"):
        validate_learned_prediction_no_nan(
            CalibratedPrediction(mean_deltas={"X": 1.0}, calibration_temperature=0.0)
        )
    with pytest.raises(NumericalInstabilityError, match="proposed_action"):
        validate_learned_prediction_no_nan(
            CalibratedPrediction(
                mean_deltas={"X": 1.0},
                proposed_actions=(Action(id="nan_a", type="op", parameters={"v": float("nan")}),),
            )
        )
    dict_pred = validate_learned_prediction_no_nan(
        {
            "mean_deltas": {"X": 2.0},
            "std_deltas": {"X": 0.5},
            "proposed_actions": [Action(id="ok_a", type="op", parameters={"v": 1.0})],
            "calibration_temperature": 1.5,
        }
    )
    assert dict_pred.mean_deltas["X"] == 2.0
    with pytest.raises(NumericalInstabilityError, match="negative std"):
        validate_learned_prediction_no_nan(
            {"mean_deltas": {"X": 1.0}, "std_deltas": {"X": -1.0}}
        )
    with pytest.raises(TypeError, match="Unsupported learned prediction type"):
        validate_learned_prediction_no_nan(123)  # type: ignore[arg-type]

    class _ProtoPredictor:
        def predict(
            self, s: WorldState, actions: Sequence[Action], struct_s: WorldState
        ) -> CalibratedPrediction:
            return CalibratedPrediction(
                mean_deltas={"X": 3.0, "UNKNOWN_RES": 1.0},
                std_deltas={"X": 0.0},
                proposed_actions=(
                    Action(
                        id="unk_act",
                        type="op",
                        parameters={"resource_id": "UNKNOWN_RES", "delta": 1.0},
                    ),
                ),
            )

    proto_router = HybridTransitionRouter(
        structural_model=DeterministicDynamics(),
        ml_predictor=_ProtoPredictor(),
    )
    proto_out = proto_router.step(state)
    assert proto_out.post_rule_rejected is True

    res_est_router = HybridTransitionRouter(
        ml_predictor=CallableResidualEstimator(lambda s, a, e, k: {"X": 2.0}),
        post_rules=[lambda s0, s1, acts: (False, "custom_post_rule_failed")],
    )
    res_est_out = res_est_router.step(state)
    assert res_est_out.post_rule_rejected is True
    assert any("custom_post_rule_failed" in v for v in res_est_out.post_rule_violations)

    dyn_ml_router = HybridTransitionRouter(
        ml_predictor=DeterministicDynamics(),
    )
    dyn_ml_out = dyn_ml_router.step(state)
    assert dyn_ml_out.is_deterministic is False
    assert dyn_ml_out.distribution is not None
    drawn_zero_std = dyn_ml_out.distribution.sample()
    assert "X" in drawn_zero_std

    bad_ml_router = HybridTransitionRouter(ml_predictor=12345)
    with pytest.raises(TypeError, match="Unsupported ml_predictor type"):
        bad_ml_router.step(state)

    # Standalone InterventionContext without engine and TransitionEngine properties
    with InterventionContext(Intervention(id="standalone_iv", description="d", parameters={"X": 5.0})) as s_ctx:
        assert s_ctx.forced_nodes == {"X": 5.0}

    empty_engine = TransitionEngine(router=proto_router, causal_graph=_build_confounded_causal_graph())
    assert empty_engine.current_state is None
    assert empty_engine.current_step == 0
    with pytest.raises(ValueError, match="requires a `state` argument"):
        empty_engine.step()
    empty_engine.causal_graph = CausalGraph()
    empty_engine.do({"X": 11.0, "memory:m_key": 7.0})
    assert empty_engine.active_do_interventions["X"] == 11.0
    step_with_mem = empty_engine.step(state.with_memory("m_key", 1.0))
    assert step_with_mem.next_state.memory["m_key"] == 7.0

