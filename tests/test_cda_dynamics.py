"""Unit and synthetic CascadeBench tests for Prompt E3: CDA Retrieve-then-Reason Dynamics.

Verifies:
1. Deprecation and isolation of neural weight-based latent rollout predictors
   (`LatentStatePredictor`, `LatentRolloutPredictor`, `NeuralLatentDynamics`),
   confirming legacy interfaces remain intact while raising `NotImplementedError`
   redirecting developers to the Continual Discovery Agent (CDA) pipeline.
2. `RetrieveAndReasonEngine` modeling $s_{t+1} \\sim P(s_{t+1} | s_t, a_t, c)$
   with explicit `RuleAttribution` of every predicted state change to retrieved
   business rules $c$.
3. Synthetic CascadeBench benchmark: defining a text-based business rule,
   programmatically mutating the rule at runtime, and verifying that transition
   predictions immediately update to align 100% with the new rule without
   retraining.
4. Sandboxed LLM output parser (`parse_sandboxed_state_diff`) safely handling
   malformed JSON, markdown fences, unattributed diffs, hallucinated rule IDs,
   and client exceptions by defaulting to a no-op state transition.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from ewm_engine.core.actions import Action
from ewm_engine.core.dynamics import (
    BusinessRule,
    CDAConfigurationError,
    CDAPromptHarness,
    ConfigurationDatastore,
    RetrieveAndReasonEngine,
    RuleAttribution,
    RuleGuidedReasoningClient,
    parse_sandboxed_state_diff,
)
from ewm_engine.core.entities import Entity
from ewm_engine.core.latent_dynamics import (
    LatentRolloutPredictor,
    LatentStatePredictor,
    NeuralLatentDynamics,
)
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import DynamicsInterface, EngineError
from ewm_engine.dynamics import (
    LatentRolloutPredictor as DynamicsLatentRolloutPredictor,
)
from ewm_engine.dynamics import (
    RetrieveAndReasonEngine as DynamicsRetrieveAndReasonEngine,
)


def _build_cascade_bench_state() -> WorldState:
    """Create a deterministic initial WorldState for CascadeBench evaluations."""
    return WorldState(
        step=0,
        timestamp=1700000000.0,
        entities={
            "order-101": Entity(
                id="order-101",
                type="PurchaseOrder",
                attributes={"status": "pending", "tier": "enterprise"},
            ),
            "warehouse-east": Entity(
                id="warehouse-east",
                type="FulfillmentNode",
                attributes={"mode": "normal", "region": "us-east"},
            ),
        },
        relationships=(),
        resources={
            "inventory": Resource(
                id="inventory",
                current=80.0,
                max_value=500.0,
                unit="stock_units",
            ),
            "revenue": Resource(
                id="revenue",
                current=5000.0,
                unit="usd",
            ),
            "cascade_penalty": Resource(
                id="cascade_penalty",
                current=0.0,
                unit="usd",
            ),
            "audit_reserve": Resource(
                id="audit_reserve",
                current=100.0,
                unit="usd",
            ),
        },
    )


def test_legacy_latent_predictors_isolated_and_redirect_to_cda() -> None:
    """Legacy neural latent predictors emit DeprecationWarning and raise NotImplementedError."""
    state = _build_cascade_bench_state()
    action = Action(id="act-1", type="process_order", actor_id="ops-agent")
    rng = np.random.default_rng(42)

    state_predictor = LatentStatePredictor(latent_dim=32)
    rollout_predictor = LatentRolloutPredictor(latent_dim=32, action_dim=16)
    neural_dynamics = NeuralLatentDynamics(model_weights_path="offline_weights.pt")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        with pytest.raises(NotImplementedError, match="RetrieveAndReasonEngine"):
            state_predictor.predict_latent([0.1, 0.2], action)
        with pytest.raises(NotImplementedError, match="RetrieveAndReasonEngine"):
            state_predictor.forward([0.1, 0.2], action)
        with pytest.raises(NotImplementedError, match="RetrieveAndReasonEngine"):
            rollout_predictor.rollout(state, [action])
        with pytest.raises(NotImplementedError, match="RetrieveAndReasonEngine"):
            neural_dynamics.transition(state, [action], [], rng)

    assert len(caught) == 4
    assert all(issubclass(w.category, DeprecationWarning) for w in caught)
    assert DynamicsLatentRolloutPredictor is LatentRolloutPredictor
    assert DynamicsRetrieveAndReasonEngine is RetrieveAndReasonEngine


def test_cascade_bench_runtime_rule_shift_updates_prediction_without_retraining() -> None:
    """Synthetic CascadeBench benchmark: runtime rule updates immediately alter state diffs."""
    state = _build_cascade_bench_state()
    action = Action(
        id="act-order-101",
        type="process_order",
        actor_id="fulfillment-controller",
        parameters={"order_id": "order-101", "quantity": 20},
    )

    datastore = ConfigurationDatastore()

    # Phase 1: Standard fulfillment rule (v1.0.0)
    rule_v1 = BusinessRule(
        rule_id="RULE-FULFILL-001",
        action_types=("process_order",),
        rule_text=(
            "WHEN action_type == 'process_order' THEN "
            "resource:inventory -= 20.0; "
            "resource:revenue += 1000.0; "
            "entity:order-101.status = 'fulfilled_standard_v1'; "
            "entity:warehouse-east.mode = 'standard_dispatch'"
        ),
        version="1.0.0",
        priority=10,
        parameters={"regime": "baseline_v1"},
    )
    datastore.register_rule(rule_v1)

    engine = RetrieveAndReasonEngine(datastore=datastore)
    assert isinstance(engine, DynamicsInterface)

    diff_v1 = engine.predict_diff(state, action)
    assert not diff_v1.is_noop
    assert diff_v1.resource_deltas == {
        "inventory": -20.0,
        "revenue": 1000.0,
    }
    assert diff_v1.entity_updates == {
        "order-101": {"status": "fulfilled_standard_v1"},
        "warehouse-east": {"mode": "standard_dispatch"},
    }
    assert len(diff_v1.rule_attributions) == 4
    assert all(attr.rule_id == "RULE-FULFILL-001" for attr in diff_v1.rule_attributions)
    assert all(attr.rule_version == "1.0.0" for attr in diff_v1.rule_attributions)
    assert {attr.target_key for attr in diff_v1.rule_attributions} == {
        "resource:inventory",
        "resource:revenue",
        "entity:order-101:status",
        "entity:warehouse-east:mode",
    }

    next_state_v1, step_diff_v1 = engine.step(state, action)
    assert not step_diff_v1.is_noop
    assert next_state_v1.step == state.step + 1
    assert next_state_v1.resources["inventory"].current == pytest.approx(60.0)
    assert next_state_v1.resources["revenue"].current == pytest.approx(6000.0)
    assert next_state_v1.resources["cascade_penalty"].current == pytest.approx(0.0)
    assert next_state_v1.entities["order-101"].attributes["status"] == "fulfilled_standard_v1"
    assert next_state_v1.entities["warehouse-east"].attributes["mode"] == "standard_dispatch"

    # Phase 2: Programmatically change the business rule at runtime (CascadeBench shift)
    # Inventory in `state` is 80.0 (<= 100.0), triggering the low-stock cascade branch!
    rule_v2 = datastore.update_rule(
        "RULE-FULFILL-001",
        rule_text=(
            "IF resource:inventory <= 100.0 THEN "
            "resource:inventory -= 20.0; "
            "resource:revenue += 850.0; "
            "resource:cascade_penalty += 150.0; "
            "resource:audit_reserve += 75.0; "
            "entity:order-101.status = 'throttled_cascade_v2'; "
            "entity:warehouse-east.mode = 'emergency_rationing' "
            "ELSE "
            "resource:inventory -= 20.0; "
            "resource:revenue += 1000.0; "
            "entity:order-101.status = 'fulfilled_standard_v2'"
        ),
        version="2.0.0",
        parameters={"regime": "cascade_shift_v2"},
    )
    assert rule_v2.version == "2.0.0"

    # Without recreating or retraining the engine, predict on the exact same (state, action)
    diff_v2 = engine.predict_diff(state, action)
    assert not diff_v2.is_noop
    assert diff_v2.resource_deltas == {
        "inventory": -20.0,
        "revenue": 850.0,
        "cascade_penalty": 150.0,
        "audit_reserve": 75.0,
    }
    assert diff_v2.entity_updates == {
        "order-101": {"status": "throttled_cascade_v2"},
        "warehouse-east": {"mode": "emergency_rationing"},
    }
    assert len(diff_v2.rule_attributions) == 6
    assert all(attr.rule_id == "RULE-FULFILL-001" for attr in diff_v2.rule_attributions)
    assert all(attr.rule_version == "2.0.0" for attr in diff_v2.rule_attributions)
    assert {attr.target_key for attr in diff_v2.rule_attributions} == {
        "resource:inventory",
        "resource:revenue",
        "resource:cascade_penalty",
        "resource:audit_reserve",
        "entity:order-101:status",
        "entity:warehouse-east:mode",
    }

    rng = np.random.default_rng(42)
    transition_res = engine.transition(state, [action], [], rng)
    next_state_v2 = transition_res.next_state
    assert next_state_v2.resources["inventory"].current == pytest.approx(60.0)
    assert next_state_v2.resources["revenue"].current == pytest.approx(5850.0)
    assert next_state_v2.resources["cascade_penalty"].current == pytest.approx(150.0)
    assert next_state_v2.resources["audit_reserve"].current == pytest.approx(175.0)
    assert next_state_v2.entities["order-101"].attributes["status"] == "throttled_cascade_v2"
    assert next_state_v2.entities["warehouse-east"].attributes["mode"] == "emergency_rationing"
    assert transition_res.applied_changes["actions_processed"] == 1
    assert len(transition_res.applied_changes["rule_attributions"]) == 6


def test_multi_rule_composition_and_rule_deactivation() -> None:
    """Multiple retrieved rules compose in priority order and respect active flags."""
    state = _build_cascade_bench_state()
    action = Action(
        id="act-order-102",
        type="process_order",
        actor_id="fulfillment-controller",
    )

    datastore = ConfigurationDatastore(
        rules=[
            BusinessRule(
                rule_id="RULE-BASE",
                action_types=("process_order",),
                rule_text="resource:inventory -= 10.0; resource:revenue += 500.0",
                priority=20,
            ),
            BusinessRule(
                rule_id="RULE-TAX-SURCHARGE",
                action_types=("*",),
                rule_text="resource:revenue -= 50.0; resource:audit_reserve += 50.0",
                priority=10,
            ),
        ]
    )
    engine = RetrieveAndReasonEngine(datastore=datastore)

    diff = engine.predict_diff(state, action)
    assert diff.resource_deltas == {
        "inventory": -10.0,
        "revenue": 450.0,
        "audit_reserve": 50.0,
    }
    assert [attr.rule_id for attr in diff.rule_attributions] == [
        "RULE-BASE",
        "RULE-BASE",
        "RULE-TAX-SURCHARGE",
        "RULE-TAX-SURCHARGE",
    ]

    # Deactivate tax surcharge rule at runtime
    datastore.update_rule("RULE-TAX-SURCHARGE", active=False)
    diff_after = engine.predict_diff(state, action)
    assert diff_after.resource_deltas == {
        "inventory": -10.0,
        "revenue": 500.0,
    }
    assert {attr.rule_id for attr in diff_after.rule_attributions} == {"RULE-BASE"}


def test_sandboxed_parser_gracefully_handles_malformed_json_and_unattributed_diffs() -> None:
    """Sandboxed LLM output parser defaults to a no-op state transition on any failure."""
    state = _build_cascade_bench_state()
    action = Action(id="act-err", type="process_order", actor_id="tester")
    rule = BusinessRule(
        rule_id="RULE-001",
        action_types=("process_order",),
        rule_text="resource:inventory -= 5.0",
    )
    datastore = ConfigurationDatastore(rules=[rule])

    # 1. Truncated / malformed JSON string
    diff_malformed = parse_sandboxed_state_diff(
        '{"resource_deltas": {"inventory": -5.0',
        [rule],
    )
    assert diff_malformed.is_noop
    assert diff_malformed.parse_error is not None
    assert "Sandboxed parser fallback to no-op" in diff_malformed.parse_error

    # 2. Non-object JSON payload (e.g., list)
    diff_list = parse_sandboxed_state_diff("[]", [rule])
    assert diff_list.is_noop
    assert diff_list.parse_error is not None
    assert "Expected JSON object" in diff_list.parse_error

    # 3. Non-finite float (NaN) in resource_deltas
    diff_nan = parse_sandboxed_state_diff(
        '{"resource_deltas": {"inventory": NaN}, "rule_attributions": [{"rule_id": "RULE-001", "target_key": "resource:inventory", "delta_or_value": -5.0}]}',
        [rule],
    )
    assert diff_nan.is_noop
    assert diff_nan.parse_error is not None
    assert "Non-finite resource_delta" in diff_nan.parse_error

    # 4. Missing rule attribution when no retrieved rules exist
    diff_unattributed = parse_sandboxed_state_diff(
        '{"resource_deltas": {"inventory": -5.0}, "entity_updates": {}, "rule_attributions": []}',
        [],
    )
    assert diff_unattributed.is_noop
    assert diff_unattributed.parse_error is not None
    assert "no business rules were retrieved" in diff_unattributed.parse_error

    # 5. Attribution referencing a hallucinated / unretrieved rule ID
    diff_hallucinated = parse_sandboxed_state_diff(
        """
        ```json
        {
          "resource_deltas": {"inventory": -5.0},
          "entity_updates": {},
          "rule_attributions": [
            {"rule_id": "HALLUCINATED-RULE-999", "target_key": "resource:inventory", "delta_or_value": -5.0}
          ]
        }
        ```
        """,
        [rule],
    )
    assert diff_hallucinated.is_noop
    assert diff_hallucinated.parse_error is not None
    assert "Unretrieved rule_id" in diff_hallucinated.parse_error

    # 6. End-to-end engine fallback when LLM client raises or returns garbage
    def _broken_llm_client(_prompt: str) -> str:
        return "<<<INVALID JSON OUTPUT FROM LLM>>>"

    engine_broken = RetrieveAndReasonEngine(
        datastore=datastore,
        reasoning_client=_broken_llm_client,
    )
    safe_next_state, broken_diff = engine_broken.step(state, action)
    assert broken_diff.is_noop
    assert broken_diff.parse_error is not None
    assert safe_next_state.step == state.step
    assert safe_next_state.resources["inventory"].current == state.resources["inventory"].current
    assert safe_next_state.resources["revenue"].current == state.resources["revenue"].current

    def _raising_llm_client(_prompt: str) -> str:
        raise RuntimeError("Simulated LLM connection timeout")

    engine_raising = RetrieveAndReasonEngine(
        datastore=datastore,
        reasoning_client=_raising_llm_client,
    )
    rng = np.random.default_rng(42)
    transition_fallback = engine_raising.transition(state, [action], [], rng)
    cda_diff = transition_fallback.diagnostics["cda_diffs"][0]
    assert cda_diff["is_noop"] is True
    assert "Reasoning client failure fallback to no-op" in str(cda_diff["parse_error"])
    assert (
        transition_fallback.next_state.resources["inventory"].current
        == state.resources["inventory"].current
    )


def test_prompt_harness_and_telemetry_metrics() -> None:
    """Verify CDAPromptHarness formatting, telemetry tracking, and error hierarchy."""
    assert issubclass(CDAConfigurationError, EngineError)

    state = _build_cascade_bench_state()
    action = Action(id="act-telemetry", type="process_order", actor_id="auditor")
    datastore = ConfigurationDatastore(
        rules=[
            BusinessRule(
                rule_id="RULE-AUDIT-01",
                action_types=("process_order",),
                rule_text=(
                    "IF action_type == 'process_order' THEN "
                    "resource:inventory = 50.0; "
                    "entity:order-101.reviewed = true; "
                    "entity:order-101.priority_score = 3"
                ),
            )
        ]
    )
    harness = CDAPromptHarness()
    rules = datastore.retrieve_rules(action, state)
    sys_prompt = harness.build_system_prompt()
    usr_prompt = harness.build_user_prompt(state, action, rules)
    assert "RULE-AUDIT-01" in usr_prompt
    assert "process_order" in usr_prompt
    assert "rule_attributions" in sys_prompt

    engine = RetrieveAndReasonEngine(
        datastore=datastore,
        prompt_harness=harness,
        reasoning_client=RuleGuidedReasoningClient(),
    )
    next_state, diff = engine.step(state, action)
    assert not diff.is_noop
    assert next_state.resources["inventory"].current == pytest.approx(50.0)
    assert next_state.entities["order-101"].attributes["reviewed"] is True
    assert next_state.entities["order-101"].attributes["priority_score"] == 3

    summary = engine.telemetry_summary
    assert summary["total_calls"] == 1
    assert summary["total_tokens"] > 0
    assert summary["mean_latency_ms"] >= 0.0

    # Verify Validation errors on invalid rule constructs
    with pytest.raises(ValidationError):
        BusinessRule(rule_id="", action_types=("process_order",), rule_text="noop")
    with pytest.raises(ValidationError):
        RuleAttribution(rule_id="", target_key="resource:inventory", delta_or_value=1.0)
