"""Contract and property tests for Formal Enterprise World-Model Semantics (ADR-053 / RFC-0002).

Verifies all eight required executable properties:
1. Deterministic mechanisms remain deterministic under identical state/context/seed.
2. Context changes can alter relevant dynamics without mutating past snapshots.
3. Unrelated context changes do not affect unrelated mechanisms.
4. Branch isolation still holds (across state, runtime_context, and observation_spec).
5. Hard invariants cannot be violated by learned residuals without explicit
   projection/rejection/abstention/invalidation — never silent repair.
6. Provenance explains which mechanism and context version produced a transition.
7. Partially observed actors cannot access hidden state via accidental API leakage.
8. Serialization round-trips preserve semantics and fingerprints per the declared contract.

Plus tests for ontic vs. epistemic state separation, fail-closed causal evidence levels,
vendor-neutral runtime context discovery, and the 6-category drift taxonomy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from ewm_engine.actors.base import ActorContext
from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core import (
    Action,
    AuditMetadataContextDiscovery,
    AutomationRuleConfig,
    ContextDiscoveryProvider,
    ContextSnapshot,
    ContractTermConfig,
    DriftKind,
    Entity,
    EpistemicState,
    ExponentialSmoothingBeliefUpdater,
    ObservationSpec,
    Relationship,
    Resource,
    ServiceEndpointConfig,
    UnqualifiedCausalClaimError,
    WorkflowConfig,
    World,
    WorldState,
    detect_world_drift,
    resolve_fail_closed_evidence_level,
    separate_ontic_epistemic,
)
from ewm_engine.dynamics import (
    CompositeDynamics,
    DeterministicDemandDynamics,
    DeterministicDynamics,
    DeterministicTransferDynamics,
    HybridDynamics,
    InfeasibleResidualPolicy,
    LinearResidualDynamics,
    MechanismKind,
    MechanismProvenance,
    ValidityDomain,
)
from ewm_engine.experimental.conservation import (
    ConservationProjectionLayer,
    LinearConservationLaw,
)
from ewm_engine.experimental.grey_box import CallableResidualEstimator
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.trajectory import TrajectoryStatus


def _make_base_state() -> WorldState:
    return WorldState(
        entities=[
            Entity(id="wh_public", type="warehouse", attributes={"tier": "gold"}),
            Entity(id="wh_secret", type="vault", attributes={"classified_code": "X99"}),
        ],
        relationships=[
            Relationship(source="wh_public", target="wh_secret", type="internal_link"),
            Relationship(source="wh_public", target="wh_public", type="self_audit"),
        ],
        resources=[
            Resource(
                id="stock_a",
                entity_id="wh_public",
                current=100.0,
                min_value=0.0,
                max_value=200.0,
            ),
            Resource(
                id="stock_b",
                entity_id="wh_secret",
                current=50.0,
                min_value=10.0,
                max_value=100.0,
            ),
        ],
        memory={"public_metric": 10.0, "secret_audit_note": "restricted"},
        active_rules={"public_rule": True, "secret_override": False},
        context={"region": "us-east", "internal_token": "hidden"},
    )


def _make_context_v1() -> ContextSnapshot:
    return ContextSnapshot(
        context_id="ctx_prod",
        version="1.0.0",
        tenant_id="acme_corp",
        environment_regime="nominal",
        workflow_definitions={
            "fulfillment": WorkflowConfig(
                workflow_id="fulfillment",
                version="1.0.0",
                max_concurrency=50,
                sla_target_seconds=1800.0,
            )
        },
        enabled_automations={
            "auto_replenish": AutomationRuleConfig(
                automation_id="auto_replenish",
                version="1.0.0",
                enabled=True,
                priority=1,
            )
        },
        policy_versions={"inventory_policy": "v1"},
        feature_flags={"fast_lane": False},
        topology_config={"primary_hub": "wh_public"},
        external_services={
            "carrier_api": ServiceEndpointConfig(
                service_id="carrier_api",
                timeout_ms=500,
                rate_limit_rps=200.0,
            )
        },
        active_contracts={
            "enterprise_sla": ContractTermConfig(
                contract_id="enterprise_sla",
                penalty_rate=50.0,
                sla_threshold=0.995,
            )
        },
        parameters={"demand_scale": 1.0, "unrelated_tax_rate": 0.05},
    )


# ==============================================================================
# Property 1: Deterministic mechanisms remain deterministic under identical
#             state / context / seed
# ==============================================================================


def test_property_1_determinism_under_identical_state_context_seed() -> None:
    """Property 1: Deterministic mechanisms produce bit-identical states & fingerprints."""
    state = _make_base_state()
    ctx = _make_context_v1()
    dynamics = DeterministicDemandDynamics(
        resource_id="stock_a",
        demand=12.5,
        version="2.1.0",
        context_demand_multiplier_key="demand_scale",
    )

    world_1 = World(state=state, dynamics=dynamics, runtime_context=ctx)
    world_2 = World(
        state=state.clone(), dynamics=dynamics, runtime_context=ctx.from_dict(ctx.to_dict())
    )

    res_1 = world_1.simulate(horizon=5, samples=1, seed=12345)
    res_2 = world_2.simulate(horizon=5, samples=1, seed=12345)

    traj_1 = res_1.trajectories[0]
    traj_2 = res_2.trajectories[0]

    assert traj_1.final_state.fingerprint == traj_2.final_state.fingerprint
    assert [s.state_hash for s in traj_1.steps] == [s.state_hash for s in traj_2.steps]
    assert (
        traj_1.epistemic_state.epistemic_fingerprint == traj_2.epistemic_state.epistemic_fingerprint
    )


# ==============================================================================
# Property 2: Context changes can alter relevant dynamics without mutating
#             past snapshots
# ==============================================================================


def test_property_2_context_changes_alter_dynamics_without_mutating_past_snapshots() -> None:
    """Property 2: Evolving K_t alters context-dependent transitions while keeping prior K_t and S_t immutable."""
    initial_state = _make_base_state()
    initial_state_fp = initial_state.fingerprint

    ctx_v1 = _make_context_v1()
    ctx_v1_fp = ctx_v1.fingerprint

    dynamics = DeterministicDemandDynamics(
        resource_id="stock_a",
        demand=10.0,
        version="1.2.0",
        context_demand_multiplier_key="demand_scale",
    )

    world_v1 = World(state=initial_state, dynamics=dynamics, runtime_context=ctx_v1)
    res_v1 = world_v1.simulate(horizon=3, samples=1, seed=42)
    step1_snapshot_v1 = res_v1.trajectories[0].steps[0].transition_result.next_state
    step1_fp_v1 = step1_snapshot_v1.fingerprint

    # Evolve context to v2 with 2.5x demand scale
    ctx_v2 = ctx_v1.evolve(
        new_version="2.0.0",
        parameters={"demand_scale": 2.5, "unrelated_tax_rate": 0.05},
    )
    assert ctx_v2.parent_version == "1.0.0"
    assert ctx_v2.version == "2.0.0"
    assert ctx_v1.fingerprint == ctx_v1_fp  # Past context snapshot unmutated

    world_v2 = world_v1.with_runtime_context(ctx_v2)
    res_v2 = world_v2.simulate(horizon=3, samples=1, seed=42)

    # Dynamics changed as expected: v1 consumed 3 * 10 = 30 (leaving 70), v2 consumed 3 * 25 = 75 (leaving 25)
    assert res_v1.trajectories[0].final_state.get_resource("stock_a").current == pytest.approx(70.0)
    assert res_v2.trajectories[0].final_state.get_resource("stock_a").current == pytest.approx(25.0)

    # Past snapshots and world_v1 remain completely unmutated
    assert initial_state.fingerprint == initial_state_fp
    assert step1_snapshot_v1.fingerprint == step1_fp_v1
    assert world_v1.runtime_context is not None
    assert world_v1.runtime_context.fingerprint == ctx_v1_fp


# ==============================================================================
# Property 3: Unrelated context changes do not affect unrelated mechanisms
# ==============================================================================


def test_property_3_unrelated_context_changes_do_not_affect_unrelated_mechanisms() -> None:
    """Property 3: Changing context keys outside a mechanism's required_context_keys has zero effect."""
    state = _make_base_state()
    ctx_v1 = _make_context_v1()

    # Mechanism A only depends on parameters.demand_scale
    observed_projected_contexts: list[ContextSnapshot] = []

    def mechanism_a_fn(
        s: WorldState,
        actions: Sequence[Action],
        events: Sequence[object],
        ctx: ContextSnapshot | None,
    ) -> WorldState:
        assert ctx is not None
        observed_projected_contexts.append(ctx)
        scale = float(ctx.parameters.get("demand_scale", 1.0))
        return s.update_resource("stock_a", delta=-5.0 * scale)

    mech_a = DeterministicDynamics(
        name="MechanismA",
        transition_fn=mechanism_a_fn,
        version="1.0.0",
        required_context_keys=("parameters.demand_scale",),
    )

    # Evolve completely unrelated context fields (tax rate, feature flags, policies, topology)
    ctx_v2 = ctx_v1.evolve(
        new_version="1.1.0",
        parameters={"demand_scale": 1.0, "unrelated_tax_rate": 0.99},
        feature_flags={"fast_lane": True, "experimental_ui": True},
        policy_versions={"inventory_policy": "v99"},
        topology_config={"primary_hub": "wh_remote"},
        environment_regime="stressed",
    )

    # Full fingerprints differ, but MechanismA's domain_fingerprint is identical
    assert ctx_v1.fingerprint != ctx_v2.fingerprint
    assert ctx_v1.domain_fingerprint(mech_a.mechanism_provenance.required_context_keys) == (
        ctx_v2.domain_fingerprint(mech_a.mechanism_provenance.required_context_keys)
    )

    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    out_1 = mech_a.transition(state, [], [], rng1, runtime_context=ctx_v1)
    out_2 = mech_a.transition(state, [], [], rng2, runtime_context=ctx_v2)

    assert out_1.next_state.fingerprint == out_2.next_state.fingerprint
    # Verify that the projected context passed into mechanism_a_fn stripped unrelated_tax_rate and feature_flags
    assert "unrelated_tax_rate" not in observed_projected_contexts[0].parameters
    assert "unrelated_tax_rate" not in observed_projected_contexts[1].parameters
    assert observed_projected_contexts[0].feature_flags == {}
    assert observed_projected_contexts[1].feature_flags == {}


# ==============================================================================
# Property 4: Branch isolation still holds
# ==============================================================================


def test_property_4_branch_isolation_holds_across_state_context_and_observations() -> None:
    """Property 4: Branching a World isolates state, constraints, actors, runtime_context, and observation_spec."""
    state = _make_base_state()
    ctx_v1 = _make_context_v1()
    obs_v1 = ObservationSpec(
        actor_id="actor_1",
        observable_resource_ids=frozenset({"stock_a"}),
    )
    dynamics = DeterministicDemandDynamics(
        resource_id="stock_a",
        demand=10.0,
        context_demand_multiplier_key="demand_scale",
    )

    base_world = World(
        state=state,
        dynamics=dynamics,
        runtime_context=ctx_v1,
        observation_spec=obs_v1,
    )

    branch_ctx = ctx_v1.evolve(
        new_version="2.0.0",
        parameters={"demand_scale": 3.0, "unrelated_tax_rate": 0.05},
    )
    branched_world = base_world.branch(runtime_context=branch_ctx)

    res_branch = branched_world.simulate(horizon=2, samples=1, seed=42)
    res_base = base_world.simulate(horizon=2, samples=1, seed=42)

    assert res_branch.trajectories[0].final_state.get_resource("stock_a").current == pytest.approx(
        40.0
    )
    assert res_base.trajectories[0].final_state.get_resource("stock_a").current == pytest.approx(
        80.0
    )
    assert base_world.initial_state.get_resource("stock_a").current == pytest.approx(100.0)
    assert base_world.runtime_context is not None
    assert base_world.runtime_context.version == "1.0.0"


# ==============================================================================
# Property 5: Hard invariants cannot be violated by learned residuals without
#             explicit projection, rejection, abstention, or invalidation
#             (never silent repair)
# ==============================================================================


class _MinStockHardConstraint(Constraint):
    """Hard post-transition constraint requiring stock_a >= 85.0."""

    @property
    def constraint_id(self) -> str:
        return "min_stock_a_85"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def severity(self) -> ConstraintSeverity:
        return ConstraintSeverity.HARD

    @property
    def description(self) -> str:
        return "stock_a must remain at least 85.0"

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        val = state.get_resource("stock_a").current
        ok = val >= 85.0
        return ConstraintResult(
            satisfied=ok,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            phase=phase,
            message="" if ok else f"stock_a={val} < 85.0",
            violating_resources=() if ok else ("stock_a",),
            violating_values={} if ok else {"stock_a": val},
        )


def test_property_5_no_silent_repair_on_infeasible_learned_residuals() -> None:
    """Property 5: Infeasible learned residuals trigger explicit PROJECT, REJECT, ABSTAIN, or INVALIDATE."""
    state = _make_base_state()  # stock_a = 100 in [0, 200], stock_b = 50 in [10, 100]
    known = DeterministicTransferDynamics()
    law = LinearConservationLaw.zero_sum(resource_ids=["stock_a", "stock_b"])
    layer = ConservationProjectionLayer(laws=[law], enforce_capacity_bounds=True)

    # Estimator proposes an infeasible delta: stock_b -60 (would drop stock_b to -10 < min_value=10)
    # and violates zero-sum conservation (-60 + 20 != 0)
    infeasible_estimator = CallableResidualEstimator(
        fn=lambda s, a, e, ns: {"stock_a": 20.0, "stock_b": -60.0}
    )

    rng = np.random.default_rng(42)

    # 1. Policy: PROJECT -> projects onto conservation + box bounds and logs exact L2 distance & violations
    gb_project = HybridDynamics(
        known_model=known,
        residual_estimator=infeasible_estimator,
        conservation_layer=layer,
        infeasible_policy=InfeasibleResidualPolicy.PROJECT,
    )
    res_proj = gb_project.transition(state, [], [], rng)
    feas_proj = res_proj.residual_feasibility
    assert feas_proj is not None
    assert feas_proj.is_feasible is False
    assert feas_proj.policy == InfeasibleResidualPolicy.PROJECT
    assert feas_proj.action_taken == "projected"
    assert feas_proj.l2_projection_distance > 0.0
    assert len(feas_proj.violations) >= 1
    assert res_proj.next_state.get_resource("stock_b").current >= 10.0

    # 2. Policy: REJECT -> rejects the residual and preserves the structural transition
    gb_reject = HybridDynamics(
        known_model=known,
        residual_estimator=infeasible_estimator,
        conservation_layer=layer,
        infeasible_policy=InfeasibleResidualPolicy.REJECT,
    )
    res_rej = gb_reject.transition(state, [], [], rng)
    feas_rej = res_rej.residual_feasibility
    assert feas_rej is not None
    assert feas_rej.is_feasible is False
    assert feas_rej.action_taken == "rejected"
    assert feas_rej.applied_deltas == {}
    assert res_rej.next_state.fingerprint == state.fingerprint

    # 3. Policy: ABSTAIN -> abstains from stepping and flags abstained=True
    known_demand = DeterministicDemandDynamics(resource_id="stock_a", demand=5.0)
    gb_abstain = HybridDynamics(
        known_model=known_demand,
        residual_estimator=infeasible_estimator,
        infeasible_policy=InfeasibleResidualPolicy.ABSTAIN,
    )
    res_abs = gb_abstain.transition(state, [], [], rng)
    feas_abs = res_abs.residual_feasibility
    assert feas_abs is not None
    assert feas_abs.is_feasible is False
    assert feas_abs.action_taken == "abstained"
    assert res_abs.diagnostics.get("abstained") is True
    # Abstention returns prior state S_t unchanged
    assert res_abs.next_state.fingerprint == state.fingerprint

    # 4. Policy: INVALIDATE -> marks transition invalidated and halts trajectory with TrajectoryStatus.INVALID
    gb_invalidate = HybridDynamics(
        known_model=known_demand,
        residual_estimator=infeasible_estimator,
        infeasible_policy=InfeasibleResidualPolicy.INVALIDATE,
    )
    world_inv = World(state=state, dynamics=gb_invalidate)
    sim_inv = world_inv.simulate(horizon=3, samples=1, seed=42)
    traj_inv = sim_inv.trajectories[0]
    assert traj_inv.status == TrajectoryStatus.INVALID
    assert traj_inv.hard_violations >= 1
    assert len(traj_inv.steps) == 1
    assert (
        traj_inv.steps[0].transition_result.residual_feasibility is not None
        and traj_inv.steps[0].transition_result.residual_feasibility.action_taken == "invalidated"
    )

    # 5. Also verify LinearResidualDynamics never silently clamps out-of-bounds predictions
    lin_dyn = LinearResidualDynamics(
        target_resource="stock_b",
        action_type="drain",
        infeasible_policy=InfeasibleResidualPolicy.REJECT,
    )
    lin_dyn.action_weight = -100.0
    lin_dyn.bias = 0.0
    res_lin = lin_dyn.transition(
        state,
        [Action(id="a1", type="drain", parameters={"value": 1.0})],
        [],
        rng,
    )
    assert res_lin.residual_feasibility is not None
    assert res_lin.residual_feasibility.is_feasible is False
    assert res_lin.residual_feasibility.action_taken == "rejected"
    assert res_lin.next_state.get_resource("stock_b").current == pytest.approx(50.0)


# ==============================================================================
# Property 6: Provenance explains which mechanism and context version produced
#             a transition
# ==============================================================================


def test_property_6_provenance_records_mechanism_and_context_version() -> None:
    """Property 6: TransitionResult and SystemicTrace record mechanism ID, version, kind, and context version/fingerprint."""
    state = _make_base_state()
    ctx = _make_context_v1()
    dynamics = DeterministicDemandDynamics(
        resource_id="stock_a",
        demand=8.0,
        name="FulfillmentDemandMechanism",
        version="3.4.1",
        context_demand_multiplier_key="demand_scale",
    )

    world = World(state=state, dynamics=dynamics, runtime_context=ctx)
    sim_res = world.simulate(horizon=2, samples=1, seed=99)
    traj = sim_res.trajectories[0]

    step0_tr = traj.steps[0].transition_result
    prov = step0_tr.mechanism_provenance
    ctx_meta = step0_tr.context_metadata

    assert prov is not None
    assert prov.mechanism_id == "FulfillmentDemandMechanism"
    assert prov.version == "3.4.1"
    assert prov.kind == MechanismKind.KNOWN_DETERMINISTIC
    assert prov.required_context_keys == ("parameters.demand_scale",)

    assert ctx_meta is not None
    assert ctx_meta["context_id"] == "ctx_prod"
    assert ctx_meta["version"] == "1.0.0"
    assert ctx_meta["fingerprint"] == ctx.fingerprint

    # SystemicTrace node also records mechanism and context version
    trace_node = traj.systemic_trace.nodes["trans_step_0"]
    assert trace_node.details["mechanism_id"] == "FulfillmentDemandMechanism"
    assert trace_node.details["mechanism_version"] == "3.4.1"
    assert trace_node.details["mechanism_kind"] == "known_deterministic"
    assert trace_node.details["context_version"] == "1.0.0"
    assert trace_node.details["context_fingerprint"] == ctx.fingerprint

    # EpistemicState extracted from trajectory exposes full provenance
    epistemic = traj.epistemic_state
    assert isinstance(epistemic, EpistemicState)
    assert epistemic.context_versions == ("1.0.0", "1.0.0")
    assert epistemic.context_fingerprints == (ctx.fingerprint, ctx.fingerprint)
    assert len(epistemic.mechanism_provenances) == 2


# ==============================================================================
# Property 7: Partially observed actors cannot access hidden state via
#             accidental API leakage
# ==============================================================================


class _CuriosityInspectorActor:
    """Actor that attempts to inspect everything in the WorldState passed to act()."""

    def __init__(self, actor_id: str = "external_partner_actor") -> None:
        self._actor_id = actor_id
        self.received_states: list[WorldState] = []

    @property
    def actor_id(self) -> str:
        return self._actor_id

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        self.received_states.append(state)
        return ()


def test_property_7_partial_observability_prevents_hidden_state_leakage() -> None:
    """Property 7: Partially observed actors cannot access hidden entities, resources, relationships, memory, rules, or context."""
    state = _make_base_state()
    spec = ObservationSpec(
        actor_id="external_partner_actor",
        observable_entity_ids=frozenset({"wh_public"}),
        observable_resource_ids=frozenset({"stock_a"}),
        observable_relationship_types=frozenset({"self_audit"}),
        observable_memory_keys=frozenset({"public_metric"}),
        observable_rule_keys=frozenset({"public_rule"}),
        observable_context_keys=frozenset({"region"}),
    )

    actor = _CuriosityInspectorActor("external_partner_actor")
    dynamics = DeterministicDemandDynamics(resource_id="stock_a", demand=5.0)
    world = World(
        state=state,
        dynamics=dynamics,
        actors=[actor],
        observation_spec=spec,
    )

    world.simulate(horizon=2, samples=1, seed=42)
    assert len(actor.received_states) == 2

    obs_state = actor.received_states[0]
    # Visible items are present
    assert set(obs_state.entities.keys()) == {"wh_public"}
    assert set(obs_state.resources.keys()) == {"stock_a"}
    assert len(obs_state.relationships) == 1
    assert obs_state.relationships[0].type == "self_audit"
    assert set(obs_state.memory.keys()) == {"public_metric"}
    assert set(obs_state.active_rules.keys()) == {"public_rule"}
    assert set(obs_state.context.keys()) == {"region"}

    # Hidden items are completely absent (including relationships pointing to hidden entities)
    assert "wh_secret" not in obs_state.entities
    assert "stock_b" not in obs_state.resources
    assert "secret_audit_note" not in obs_state.memory
    assert "secret_override" not in obs_state.active_rules
    assert "internal_token" not in obs_state.context
    for rel in obs_state.relationships:
        assert rel.source != "wh_secret" and rel.target != "wh_secret"

    # Also test optional BeliefState updater
    updater = ExponentialSmoothingBeliefUpdater(
        prior_resource_estimates={"stock_a": 90.0, "stock_b": 45.0},
        prior_uncertainty=1.0,
    )
    b0 = updater.update_belief(None, actor.received_states[0], actor.actor_id)
    b1 = updater.update_belief(b0, actor.received_states[1], actor.actor_id)
    assert b1.estimated_hidden_resources["stock_a"] == pytest.approx(95.0)
    assert b1.epistemic_uncertainty["stock_a"] == 0.0
    # Unobserved stock_b retains prior point estimate while its epistemic uncertainty grows
    assert b1.estimated_hidden_resources["stock_b"] == pytest.approx(45.0)
    assert b1.epistemic_uncertainty["stock_b"] > b0.epistemic_uncertainty["stock_b"]


# ==============================================================================
# Property 8: Serialization round-trips preserve semantics and fingerprints
#             per the declared contract
# ==============================================================================


def test_property_8_serialization_round_trips_preserve_semantics_and_fingerprints() -> None:
    """Property 8: ContextSnapshot, WorldState, MechanismProvenance, and EpistemicState round-trip with identical fingerprints."""
    state = _make_base_state()
    ctx = _make_context_v1()
    prov = MechanismProvenance(
        mechanism_id="hybrid_fulfillment",
        version="2.0.0",
        kind=MechanismKind.HYBRID,
        validity_domain=ValidityDomain(
            allowed_regimes=("nominal", "peak"),
            resource_bounds={"stock_a": (0.0, 200.0)},
        ),
        required_context_keys=("parameters.demand_scale", "feature_flags.fast_lane"),
        evidence_level=EvidenceLevel.PREDICTIVE,
        source="ude_grey_box",
        invariants_affected=("mass_balance",),
        failure_semantics=InfeasibleResidualPolicy.PROJECT,
    )

    # ContextSnapshot round-trip
    ctx_restored = ContextSnapshot.from_dict(ctx.to_dict())
    assert ctx_restored == ctx
    assert ctx_restored.fingerprint == ctx.fingerprint
    assert ctx_restored.domain_fingerprint(prov.required_context_keys) == ctx.domain_fingerprint(
        prov.required_context_keys
    )

    # WorldState round-trip
    state_restored = WorldState.from_dict(state.to_dict())
    assert state_restored.fingerprint == state.fingerprint
    assert state_restored.ontic_fingerprint == state.ontic_fingerprint

    # MechanismProvenance round-trip
    prov_restored = MechanismProvenance.model_validate(prov.model_dump(mode="json"))
    assert prov_restored == prov

    # EpistemicState round-trip
    _, epistemic = separate_ontic_epistemic(
        state,
        effective_evidence_level=EvidenceLevel.PREDICTIVE,
        mechanism_provenances=[prov.model_dump(mode="json")],
        context_versions=[ctx.version],
        context_fingerprints=[ctx.fingerprint],
        assumptions=["no_unobserved_confounding"],
    )
    epistemic_restored = EpistemicState.model_validate(epistemic.model_dump(mode="json"))
    assert epistemic_restored.epistemic_fingerprint == epistemic.epistemic_fingerprint


# ==============================================================================
# Additional Architectural Contracts: Ontic vs. Epistemic, Fail-Closed Causal
# Evidence, Runtime Context Discovery, and Drift Taxonomy
# ==============================================================================


def test_ontic_vs_epistemic_separation_invariant() -> None:
    """Epistemic metadata changes never alter the ontic WorldState fingerprint."""
    s1 = _make_base_state()
    s2 = s1.with_memory("_epistemic_assumption", "linear_demand").with_context(
        "_epistemic_note", "calibrated_on_q3"
    )

    # Pure ontic fingerprint is invariant to _epistemic_* annotations
    assert s1.ontic_fingerprint == s2.ontic_fingerprint
    ontic_s2, epistemic_s2 = separate_ontic_epistemic(s2)
    assert ontic_s2.fingerprint == s1.fingerprint
    assert any("_epistemic_assumption=linear_demand" in a for a in epistemic_s2.assumptions)


def test_fail_closed_evidence_level_and_runtime_context_discovery() -> None:
    """Learned and runtime-discovered mechanisms/contexts are fail-closed on causal claims."""
    discovery: ContextDiscoveryProvider = AuditMetadataContextDiscovery()

    # Valid observational discovery
    discovered_ctx = discovery.discover_context(
        {
            "context_id": "discovered_01",
            "version": "1.0.0",
            "tenant_id": "tenant_alpha",
            "parameters": {"concurrency": 32},
        }
    )
    assert discovered_ctx.evidence_level == EvidenceLevel.PREDICTIVE
    assert discovered_ctx.provenance_source == "runtime_audit_discovery"

    # Uncertified interventional claim in runtime discovery must fail closed
    with pytest.raises(UnqualifiedCausalClaimError, match="Fail-closed EvidenceLevel violation"):
        discovery.discover_context(
            {
                "context_id": "discovered_bad",
                "version": "1.0.0",
                "evidence_level": "interventional",
            }
        )

    # Uncertified causal claim on LEARNED or DISCOVERED_AT_RUNTIME mechanism must fail closed
    with pytest.raises(UnqualifiedCausalClaimError, match="Fail-closed EvidenceLevel violation"):
        MechanismProvenance(
            mechanism_id="unidentified_neural_net",
            kind=MechanismKind.LEARNED,
            evidence_level=EvidenceLevel.INTERVENTIONAL,
        )

    with pytest.raises(UnqualifiedCausalClaimError, match="cannot claim STRUCTURAL"):
        MechanismProvenance(
            mechanism_id="sindy_runtime_rule",
            kind=MechanismKind.DISCOVERED_AT_RUNTIME,
            evidence_level=EvidenceLevel.STRUCTURAL,
        )

    # Certified interventional mechanism succeeds
    certified = MechanismProvenance(
        mechanism_id="rct_calibrated_model",
        kind=MechanismKind.LEARNED,
        evidence_level=EvidenceLevel.INTERVENTIONAL,
        identification_certificate_id="cert:backdoor_rct_2026_01",
    )
    assert certified.evidence_level == EvidenceLevel.INTERVENTIONAL

    # Composite fail-closed evidence resolution caps unidentified learned components at PREDICTIVE
    eff = resolve_fail_closed_evidence_level(
        [EvidenceLevel.STRUCTURAL, EvidenceLevel.INTERVENTIONAL],
        has_unidentified_learned_or_discovered=True,
    )
    assert eff == EvidenceLevel.PREDICTIVE


def test_six_category_drift_taxonomy_detection() -> None:
    """Verify detection of all 6 formal drift categories in detect_world_drift."""
    s_base = _make_base_state()
    s_cand = s_base.update_resource("stock_a", delta=-40.0).with_relationship(
        Relationship(source="wh_secret", target="wh_public", type="emergency_route")
    )

    ctx_base = _make_context_v1()
    ctx_cand = ctx_base.evolve(
        new_version="2.0.0",
        parameters={"demand_scale": 1.8, "unrelated_tax_rate": 0.05},
        feature_flags={"fast_lane": True},
        policy_versions={"inventory_policy": "v2"},
        topology_config={"primary_hub": "wh_secret"},
    )

    signals = detect_world_drift(
        baseline_context=ctx_base,
        candidate_context=ctx_cand,
        baseline_state=s_base,
        candidate_state=s_cand,
        baseline_mechanism_version="1.0.0",
        candidate_mechanism_version="2.0.0",
        baseline_objectives={"sla": 1.0, "cost": 0.5},
        candidate_objectives={"sla": 2.0, "cost": 0.2},
    )

    detected_kinds = {sig.kind for sig in signals if sig.is_drifted}
    assert detected_kinds == {
        DriftKind.COVARIATE_SHIFT,
        DriftKind.PARAMETER_DRIFT,
        DriftKind.STRUCTURAL_MECHANISM_CHANGE,
        DriftKind.RULE_CONSTRAINT_CHANGE,
        DriftKind.TOPOLOGY_CHANGE,
        DriftKind.OBJECTIVE_CHANGE,
    }


def test_composite_dynamics_context_isolation() -> None:
    """CompositeDynamics forwards runtime_context to child mechanisms while preserving domain isolation."""
    state = _make_base_state()
    ctx_1 = _make_context_v1()
    ctx_2 = ctx_1.evolve(
        new_version="1.1.0",
        parameters={"demand_scale": 2.0, "unrelated_tax_rate": 0.05},
    )

    m1 = DeterministicDemandDynamics(
        resource_id="stock_a",
        demand=10.0,
        name="DemandA",
        context_demand_multiplier_key="demand_scale",
    )
    m2 = DeterministicDemandDynamics(
        resource_id="stock_b",
        demand=5.0,
        name="DemandB",
    )
    comp = CompositeDynamics(models=[m1, m2], name="PipelineAB")

    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    r1 = comp.transition(state, [], [], rng1, runtime_context=ctx_1)
    r2 = comp.transition(state, [], [], rng2, runtime_context=ctx_2)

    # DemandA changed (10 vs 20), whereas DemandB (which has no context dependency) consumed 5.0 in both
    assert r1.next_state.get_resource("stock_a").current == pytest.approx(90.0)
    assert r2.next_state.get_resource("stock_a").current == pytest.approx(80.0)
    assert r1.next_state.get_resource("stock_b").current == pytest.approx(45.0)
    assert r2.next_state.get_resource("stock_b").current == pytest.approx(45.0)
