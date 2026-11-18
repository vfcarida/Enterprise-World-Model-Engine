"""Continual Discovery Agent (CDA) Retrieve-Then-Reason Dynamics Engine (E3).

Replaces static weight-based latent rollout predictors ($z_{t+1} = f_\\theta(z_t, a_t)$)
with explicit runtime discovery and conditioning over retrieved enterprise business rules:
    s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)
where:
- $s_t$ is the current immutable ``WorldState`` (or state dictionary),
- $a_t$ is the proposed operational ``Action``,
- $c = \\text{Retrieve}(s_t, a_t)$ is the set of active business rules retrieved at runtime
  from a dynamic ``ConfigurationDatastore``, and
- $\\Delta s_{t+1}$ is the structured ``StateDiff`` with mandatory per-change attribution to
  the retrieved ``BusinessRule`` identifiers in $c$.

Why Continual Discovery Replaces Latent Rollout Prediction
----------------------------------------------------------
Static neural predictors encode transition dynamics in frozen weights $\\theta$ trained on
historical logs. When enterprise policies, fee schedules, routing rules, or SLA thresholds
shift at runtime (as modeled in CascadeBench), $P(s_{t+1} \\mid s_t, a_t, c_{\\text{new}})
\\neq P(s_{t+1} \\mid s_t, a_t, c_{\\text{old}})$, causing latent predictors to collapse.
By retrieving active text-based business rules $c$ on every step and reasoning over
$(s_t, a_t, c)$ through a sandboxed JSON diff harness, ``RetrieveAndReasonEngine`` adapts
immediately to programmatic rule changes with zero retraining.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import operator
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ewm_engine.core.actions import Action
from ewm_engine.core.context import ContextSnapshot
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.latent_dynamics import (
    LatentRolloutPredictor,
    LatentStatePredictor,
    NeuralLatentDynamics,
)
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import DynamicsInterface, EngineError, RandomGenerator
from ewm_engine.provenance.evidence import EvidenceLevel

_OS_COMMAND_INJECTION_RE = re.compile(
    r"(?:"
    r"\bos\s*\.\s*(?:system|popen|spawn|exec|remove|unlink|rmdir)\b"
    r"|\bsubprocess\s*\.\s*(?:Popen|run|call|check_output|check_call)\b"
    r"|\b(?:__import__|eval|exec|compile|getattr|setattr|delattr|globals|locals)\s*\("
    r"|\bshutil\s*\.\s*rmtree\b"
    r"|\brm\s+-[rRfF]+\b"
    r"|(?:/bin/(?:sh|bash|zsh|dash)|\bcmd\.exe\b|\bpowershell(?:\.exe)?\b)"
    r"|\$\([^)]+\)"
    r"|`[^`]+`"
    r"|(?:;\s*|\|\s*|&&\s*)(?:rm|curl|wget|nc|netcat|bash|sh|chmod|chown|whoami|cat\s+/etc/passwd)\b"
    r")",
    re.IGNORECASE,
)


def _validate_no_malicious_os_command(value: Any, *, field_label: str = "payload") -> None:
    """Recursively inspect `value` and raise `ValueError` if an OS command or code-exec payload is present."""
    if isinstance(value, str):
        match = _OS_COMMAND_INJECTION_RE.search(value)
        if match:
            raise ValueError(
                f"Security violation in {field_label}: malicious OS command or code execution "
                f"pattern detected ({match.group(0)!r})."
            )
    elif isinstance(value, Mapping):
        for k, v in value.items():
            _validate_no_malicious_os_command(k, field_label=f"{field_label}.key")
            _validate_no_malicious_os_command(v, field_label=f"{field_label}[{k!r}]")
    elif isinstance(value, (list, tuple, set)):
        for idx, item in enumerate(value):
            _validate_no_malicious_os_command(item, field_label=f"{field_label}[{idx}]")


class CDAConfigurationError(EngineError):
    """Raised when a business rule or configuration datastore operation is invalid."""


class BusinessRule(BaseModel):
    """Declarative text-based enterprise business rule stored in ``ConfigurationDatastore``.

    Represents an active operational rule in $c$ governing state transitions
    $s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)$.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1, description="Unique identifier of the business rule.")
    action_types: Tuple[str, ...] = Field(
        default=("*",),
        description="Action types triggering this rule ('*' matches all action types).",
    )
    rule_text: str = Field(
        min_length=1,
        description="Human-readable and agent-interpretable business rule specification.",
    )
    version: str = Field(default="1.0.0", description="Semantic version of this rule.")
    active: bool = Field(default=True, description="Whether this rule is currently active.")
    priority: int = Field(
        default=0,
        description="Execution priority (higher priority rules are ordered first).",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured parameters referenced by the rule text.",
    )

    @model_validator(mode="after")
    def _enforce_security_sanitization(self) -> BusinessRule:
        _validate_no_malicious_os_command(self.rule_id, field_label="BusinessRule.rule_id")
        _validate_no_malicious_os_command(self.rule_text, field_label="BusinessRule.rule_text")
        _validate_no_malicious_os_command(self.version, field_label="BusinessRule.version")
        _validate_no_malicious_os_command(
            self.action_types, field_label="BusinessRule.action_types"
        )
        _validate_no_malicious_os_command(self.parameters, field_label="BusinessRule.parameters")
        return self

    def matches_action(self, action_type: str) -> bool:
        """Return True if this rule is active and applies to `action_type`."""
        if not self.active:
            return False
        return "*" in self.action_types or action_type in self.action_types


class CDAConfigurationPayload(BaseModel):
    """Strict Pydantic schema validating JSON rule configuration bundles entering the CDA loop (E9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: Tuple[BusinessRule, ...] = Field(
        min_length=1,
        description="Validated sequence of BusinessRule configurations.",
    )


class CDAInputPayload(BaseModel):
    """Strict Pydantic schema validating $(s_t, a_t, c)$ JSON payloads entering the CDA loop (E9)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: Dict[str, Any] = Field(description="Serialized state payload s_t.")
    action: Dict[str, Any] = Field(description="Serialized action payload a_t.")
    retrieved_rules: Tuple[BusinessRule, ...] = Field(
        default_factory=tuple,
        description="Validated BusinessRule configurations c.",
    )

    @model_validator(mode="after")
    def _validate_state_and_action_safety(self) -> CDAInputPayload:
        _validate_no_malicious_os_command(self.state, field_label="CDAInputPayload.state")
        _validate_no_malicious_os_command(self.action, field_label="CDAInputPayload.action")
        return self


def parse_cda_configuration_json(
    raw_input: Union[str, bytes, Mapping[str, Any], Sequence[Any]],
) -> List[BusinessRule]:
    """Safely parse and validate untrusted JSON business rule configurations via Pydantic (E9).

    Rejects malformed JSON, unknown fields (`extra="forbid"`), and any embedded OS command
    injection or code execution payloads by raising `pydantic.ValidationError`.
    """
    if isinstance(raw_input, (str, bytes)):
        text = raw_input.decode("utf-8") if isinstance(raw_input, bytes) else raw_input
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            # Trigger a Pydantic ValidationError on invalid JSON syntax
            CDAConfigurationPayload.model_validate_json(text)
            raise ValueError(f"Invalid JSON: {exc}") from exc
    else:
        decoded = raw_input

    if isinstance(decoded, Mapping):
        if "rules" in decoded:
            bundle = CDAConfigurationPayload.model_validate(dict(decoded))
            return list(bundle.rules)
        single_rule = BusinessRule.model_validate(dict(decoded))
        return [single_rule]

    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)):
        bundle = CDAConfigurationPayload.model_validate({"rules": list(decoded)})
        return list(bundle.rules)

    # Trigger Pydantic ValidationError for unsupported root types
    CDAConfigurationPayload.model_validate(decoded)
    return []


class RuleAttribution(BaseModel):
    """Explicit attribution linking a predicted state modification in $\\Delta s_{t+1}$ to a ``BusinessRule``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(
        min_length=1, description="ID of the retrieved BusinessRule causing this change."
    )
    rule_version: str = Field(
        default="1.0.0", description="Version of the attributed BusinessRule."
    )
    target_key: str = Field(
        min_length=1,
        description="State path modified (e.g., 'resource:inventory', 'memory:tier', 'entity:wh_1:status').",
    )
    delta_or_value: Any = Field(description="Numeric delta or assigned value enacted by the rule.")
    rationale: str = Field(
        default="",
        description="Explanation or rule clause justifying the state change.",
    )


class StateDiff(BaseModel):
    """Predicted state diff $\\Delta s_{t+1}$ emitted by ``RetrieveAndReasonEngine``.

    Encapsulates resource deltas, resource overrides, entity attribute updates,
    memory/context updates, and mandatory ``rule_attributions`` tracing every
    predicted change back to retrieved business rules $c$.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_deltas: Dict[str, float] = Field(
        default_factory=dict,
        description="Additive resource level changes (resource_id -> delta).",
    )
    resource_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Absolute resource level overrides (resource_id -> new_value).",
    )
    entity_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Entity attribute updates (entity_id -> {attr: value}).",
    )
    memory_updates: Dict[str, Any] = Field(
        default_factory=dict,
        description="State memory updates (key -> value).",
    )
    context_updates: Dict[str, Any] = Field(
        default_factory=dict,
        description="State context updates (key -> value).",
    )
    rule_attributions: Tuple[RuleAttribution, ...] = Field(
        default_factory=tuple,
        description="Mandatory attributions mapping every predicted change to a retrieved BusinessRule.",
    )
    retrieved_rule_ids: Tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of the active business rules retrieved for this transition.",
    )
    is_noop: bool = Field(
        default=False,
        description="True if no changes were predicted or parser defaulted to a safe no-op.",
    )
    parse_error: Optional[str] = Field(
        default=None,
        description="Error message if malformed LLM output triggered a sandboxed no-op fallback.",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Wall-clock latency of the retrieve-then-reason step in milliseconds.",
    )
    prompt_tokens: int = Field(default=0, ge=0, description="Estimated prompt token count.")
    completion_tokens: int = Field(default=0, ge=0, description="Estimated completion token count.")
    total_tokens: int = Field(default=0, ge=0, description="Total estimated token count.")

    @property
    def has_changes(self) -> bool:
        """Return True if this diff contains any state modifications."""
        return bool(
            self.resource_deltas
            or self.resource_overrides
            or self.entity_updates
            or self.memory_updates
            or self.context_updates
        )

    def apply_to_state(self, state: WorldState, *, advance_step: bool = True) -> WorldState:
        """Apply this predicted diff $\\Delta s_{t+1}$ to an immutable ``WorldState`` $s_t$.

        If `self.is_noop` is True (e.g., on sandboxed parser fallback), returns an
        unmodified clone of `state` (advancing time only if `advance_step=True` and not a
        parse-failure fallback).
        """
        if self.parse_error is not None or self.is_noop:
            return state.clone()

        next_state = state.clone()
        for rid, new_val in self.resource_overrides.items():
            if rid in next_state.resources:
                next_state = next_state.update_resource(rid, new_value=float(new_val))

        for rid, delta_val in self.resource_deltas.items():
            if rid in next_state.resources:
                next_state = next_state.update_resource(rid, delta=float(delta_val))

        for eid, attr_map in self.entity_updates.items():
            if eid in next_state.entities:
                ent = next_state.get_entity(eid)
                merged_attrs = dict(ent.attributes)
                merged_attrs.update(attr_map)
                next_state = next_state.with_entity(
                    ent.model_copy(update={"attributes": merged_attrs})
                )

        for mkey, mval in self.memory_updates.items():
            next_state = next_state.with_memory(str(mkey), mval)

        for ckey, cval in self.context_updates.items():
            next_state = next_state.with_context(str(ckey), cval)

        if advance_step:
            next_state = next_state.advance_time(delta_t=1.0, steps=1)

        return next_state


@runtime_checkable
class ContextRetrieverProtocol(Protocol):
    """Protocol for retrieving active business rules $c$ relevant to $(s_t, a_t)$."""

    def retrieve_rules(
        self,
        action: Union[Action, Mapping[str, Any], str],
        state: Optional[Union[WorldState, Mapping[str, Any]]] = None,
    ) -> List[BusinessRule]:
        """Return active business rules matching the proposed action and state."""
        ...


class ConfigurationDatastore:
    """In-memory runtime configuration datastore managing active ``BusinessRule`` specifications.

    Supports dynamic rule registration, programmatic live updates (`update_rule`),
    deactivation, and action-scoped retrieval without model retraining.
    """

    def __init__(
        self, rules: Optional[Sequence[Union[BusinessRule, Mapping[str, Any]]]] = None
    ) -> None:
        self._rules: Dict[str, BusinessRule] = {}
        for r in rules or ():
            self.register_rule(r)

    @property
    def rules(self) -> Dict[str, BusinessRule]:
        """Return a copy of all registered rules keyed by `rule_id`."""
        return dict(self._rules)

    def register_rule(self, rule: Union[BusinessRule, Mapping[str, Any], str]) -> BusinessRule:
        """Register or replace a BusinessRule in the datastore with strict Pydantic validation."""
        if isinstance(rule, str):
            parsed_list = parse_cda_configuration_json(rule)
            for item in parsed_list:
                self._rules[item.rule_id] = item
            return parsed_list[-1]
        parsed = (
            BusinessRule.model_validate(rule.model_dump())
            if isinstance(rule, BusinessRule)
            else BusinessRule.model_validate(dict(rule))
        )
        self._rules[parsed.rule_id] = parsed
        return parsed

    def load_json(
        self,
        json_input: Union[str, bytes, Mapping[str, Any], Sequence[Any]],
    ) -> List[BusinessRule]:
        """Validate and load one or more business rules from untrusted JSON via Pydantic (E9)."""
        rules = parse_cda_configuration_json(json_input)
        for r in rules:
            self._rules[r.rule_id] = r
        return rules

    def update_rule(
        self,
        rule_id: str,
        *,
        rule_text: Optional[str] = None,
        action_types: Optional[Sequence[str]] = None,
        version: Optional[str] = None,
        active: Optional[bool] = None,
        priority: Optional[int] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> BusinessRule:
        """Programmatically update an existing business rule at runtime without retraining."""
        if rule_id not in self._rules:
            raise CDAConfigurationError(
                f"BusinessRule '{rule_id}' not found in ConfigurationDatastore."
            )
        existing = self._rules[rule_id]
        updated_dict = {
            "rule_id": existing.rule_id,
            "rule_text": rule_text if rule_text is not None else existing.rule_text,
            "action_types": (
                tuple(action_types) if action_types is not None else existing.action_types
            ),
            "version": version if version is not None else existing.version,
            "active": active if active is not None else existing.active,
            "priority": priority if priority is not None else existing.priority,
            "parameters": (
                dict(parameters) if parameters is not None else dict(existing.parameters)
            ),
        }
        updated = BusinessRule.model_validate(updated_dict)
        self._rules[rule_id] = updated
        return updated

    def remove_rule(self, rule_id: str) -> None:
        """Remove a rule from the datastore by `rule_id`."""
        self._rules.pop(rule_id, None)

    def get_rule(self, rule_id: str) -> BusinessRule:
        """Fetch a registered rule by ID or raise `CDAConfigurationError`."""
        if rule_id not in self._rules:
            raise CDAConfigurationError(f"BusinessRule '{rule_id}' not found.")
        return self._rules[rule_id]

    def retrieve_rules(
        self,
        action: Union[Action, Mapping[str, Any], str],
        state: Optional[Union[WorldState, Mapping[str, Any]]] = None,
    ) -> List[BusinessRule]:
        """Extract active business rules related to `action` ($a_t$), sorted by priority."""
        if isinstance(action, Action):
            act_type = action.type
        elif isinstance(action, Mapping):
            act_type = str(action.get("type", action.get("action_type", "*")))
        else:
            act_type = str(action)

        matched = [r for r in self._rules.values() if r.matches_action(act_type)]
        matched.sort(key=lambda r: (-r.priority, r.rule_id))
        return matched


class CDAPromptHarness:
    """Prompting harness formatting $(s_t, a_t, c)$ for Continual Discovery Agent reasoning."""

    def build_system_prompt(self) -> str:
        """Return the system instruction enforcing JSON-only output with explicit rule attribution."""
        return (
            "You are the Continual Discovery Agent (CDA) reasoning engine for the Enterprise World Model.\n"
            "Given the current state s_t, proposed action a_t, and retrieved active business rules c, "
            "predict the state transition diff Delta s_{t+1} ~ P(s_{t+1} | s_t, a_t, c).\n"
            "Return ONLY a valid JSON object with keys:\n"
            '  - "resource_deltas": {resource_id: float_delta}\n'
            '  - "resource_overrides": {resource_id: float_new_value}\n'
            '  - "entity_updates": {entity_id: {attr_name: value}}\n'
            '  - "memory_updates": {key: value}\n'
            '  - "context_updates": {key: value}\n'
            '  - "rule_attributions": [{"rule_id": str, "rule_version": str, "target_key": str, '
            '"delta_or_value": any, "rationale": str}]\n'
            "Every predicted change MUST be explicitly attributed to a retrieved rule_id."
        )

    def build_user_prompt(
        self,
        state: Union[WorldState, Mapping[str, Any]],
        action: Union[Action, Mapping[str, Any]],
        retrieved_rules: Sequence[BusinessRule],
    ) -> str:
        """Serialize $s_t$, $a_t$, and retrieved rules $c$ into a structured prompt payload."""
        if isinstance(state, WorldState):
            state_payload: Dict[str, Any] = {
                "step": state.step,
                "timestamp": state.timestamp,
                "resources": {rid: res.current for rid, res in state.resources.items()},
                "entities": {eid: dict(ent.attributes) for eid, ent in state.entities.items()},
                "memory": dict(state.memory),
                "context": dict(state.context),
            }
        else:
            state_payload = dict(state)

        if isinstance(action, Action):
            action_payload: Dict[str, Any] = {
                "id": action.id,
                "type": action.type,
                "parameters": dict(action.parameters),
            }
        else:
            action_payload = dict(action)

        # Enforce strict Pydantic validation on all payloads entering the CDA loop (E9 Req 7)
        validated_input = CDAInputPayload(
            state=state_payload,
            action=action_payload,
            retrieved_rules=tuple(retrieved_rules),
        )

        rules_payload = [
            {
                "rule_id": r.rule_id,
                "version": r.version,
                "action_types": list(r.action_types),
                "rule_text": r.rule_text,
                "parameters": dict(r.parameters),
            }
            for r in validated_input.retrieved_rules
        ]

        return (
            f"<state_json>\n{json.dumps(validated_input.state, sort_keys=True)}\n</state_json>\n"
            f"<action_json>\n{json.dumps(validated_input.action, sort_keys=True)}\n</action_json>\n"
            f"<retrieved_rules_json>\n{json.dumps(rules_payload, sort_keys=True)}\n</retrieved_rules_json>"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Deterministically estimate token count (~4 characters per token, minimum 1 for non-empty)."""
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, (len(stripped) + 3) // 4)


# ==============================================================================
# Safe Arithmetic & Rule Text Evaluator for RuleGuidedReasoningClient
# ==============================================================================

_SAFE_BINARY_OPS: Dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_SAFE_UNARY_OPS: Dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_CMP_OPS: Dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _safe_eval_expr(expr_str: str, env: Mapping[str, Any]) -> Any:
    """Safely evaluate a numeric, string, or boolean expression over state and action variables."""
    cleaned = expr_str.strip()
    cleaned = re.sub(r"\b(?:resource|memory|context):([a-zA-Z0-9_]+)\b", r"\1", cleaned)
    # Convert percentage literals like '15%' -> '(15 / 100.0)'
    cleaned = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1 / 100.0)", cleaned)
    tree = ast.parse(cleaned, mode="eval")

    def _eval_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in ("true", "True"):
                return True
            if node.id in ("false", "False"):
                return False
            if node.id in ("null", "None"):
                return None
            if node.id in env:
                return env[node.id]
            return node.id
        if isinstance(node, ast.Attribute):
            base_val = _eval_node(node.value)
            if isinstance(base_val, Mapping) and node.attr in base_val:
                return base_val[node.attr]
            dotted_key = f"{base_val}.{node.attr}"
            if dotted_key in env:
                return env[dotted_key]
            raise ValueError(f"Unknown attribute '{node.attr}' in rule expression.")
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINARY_OPS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return _SAFE_BINARY_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:
            return _SAFE_UNARY_OPS[type(node.op)](_eval_node(node.operand))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            op_type = type(node.ops[0])
            if op_type in _SAFE_CMP_OPS:
                return _SAFE_CMP_OPS[op_type](
                    _eval_node(node.left), _eval_node(node.comparators[0])
                )
        if isinstance(node, ast.BoolOp):
            vals = [_eval_node(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(vals)
            if isinstance(node.op, ast.Or):
                return any(vals)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            args = [_eval_node(a) for a in node.args]
            if fn_name == "min":
                return min(*args)
            if fn_name == "max":
                return max(*args)
            if fn_name == "abs":
                return abs(args[0])
            if fn_name == "round":
                return round(*args)
        raise ValueError(f"Unsupported AST node '{type(node).__name__}' in rule expression.")

    return _eval_node(tree)


class RuleGuidedReasoningClient:
    """Deterministic text-reasoning agent that interprets retrieved business rules $c$.

    Extracts `<state_json>`, `<action_json>`, and `<retrieved_rules_json>` from the
    prompting harness, evaluates the natural-language / declarative clauses in each
    rule's `rule_text`, and returns a structured JSON diff with explicit `rule_attributions`.
    """

    @staticmethod
    def _extract_tag_json(prompt: str, tag: str) -> Any:
        pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL)
        m = pattern.search(prompt)
        if not m:
            raise ValueError(f"Missing <{tag}> section in prompt.")
        return json.loads(m.group(1))

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Reason over $(s_t, a_t, c)$ in `user_prompt` and emit a structured JSON StateDiff."""
        _ = system_prompt
        state_data: Dict[str, Any] = self._extract_tag_json(user_prompt, "state_json")
        action_data: Dict[str, Any] = self._extract_tag_json(user_prompt, "action_json")
        rules_data: List[Dict[str, Any]] = self._extract_tag_json(
            user_prompt, "retrieved_rules_json"
        )

        resources: Dict[str, float] = {
            str(k): float(v) for k, v in state_data.get("resources", {}).items()
        }
        memory: Dict[str, Any] = dict(state_data.get("memory", {}))
        context: Dict[str, Any] = dict(state_data.get("context", {}))
        entities: Dict[str, Dict[str, Any]] = {
            str(k): dict(v) for k, v in state_data.get("entities", {}).items()
        }
        action_params: Dict[str, Any] = dict(action_data.get("parameters", {}))
        action_type: str = str(action_data.get("type", action_data.get("action_type", "")))

        resource_deltas: Dict[str, float] = {}
        resource_overrides: Dict[str, float] = {}
        entity_updates: Dict[str, Dict[str, Any]] = {}
        memory_updates: Dict[str, Any] = {}
        context_updates: Dict[str, Any] = {}
        attributions: List[Dict[str, Any]] = []

        for rule in rules_data:
            rule_id = str(rule["rule_id"])
            rule_version = str(rule.get("version", "1.0.0"))
            rule_text = str(rule.get("rule_text", "")).strip()
            rule_params: Dict[str, Any] = dict(rule.get("parameters", {}))

            env: Dict[str, Any] = {
                "resources": resources,
                "memory": memory,
                "context": context,
                "entities": entities,
                "action": action_params,
                "action_type": action_type,
                **resources,
                **memory,
                **context,
                **action_params,
                **rule_params,
            }

            self._apply_rule_clauses(
                rule_id=rule_id,
                rule_version=rule_version,
                rule_text=rule_text,
                env=env,
                resources=resources,
                resource_deltas=resource_deltas,
                resource_overrides=resource_overrides,
                entity_updates=entity_updates,
                memory_updates=memory_updates,
                context_updates=context_updates,
                attributions=attributions,
            )

        response_payload = {
            "resource_deltas": resource_deltas,
            "resource_overrides": resource_overrides,
            "entity_updates": entity_updates,
            "memory_updates": memory_updates,
            "context_updates": context_updates,
            "rule_attributions": attributions,
        }
        return json.dumps(response_payload, sort_keys=True)

    def _apply_rule_clauses(
        self,
        *,
        rule_id: str,
        rule_version: str,
        rule_text: str,
        env: Dict[str, Any],
        resources: Dict[str, float],
        resource_deltas: Dict[str, float],
        resource_overrides: Dict[str, float],
        entity_updates: Dict[str, Dict[str, Any]],
        memory_updates: Dict[str, Any],
        context_updates: Dict[str, Any],
        attributions: List[Dict[str, Any]],
    ) -> None:
        """Parse and execute declarative/natural-language business rule clauses."""
        text = rule_text.strip()

        # Strip optional leading "WHEN <action condition> THEN"
        when_match = re.match(
            r"^(?:WHEN|IF)\s+action(?:\.type|_type)?\s+(?:is|==|=)\s+['\"]?([a-zA-Z0-9_.\-]+)['\"]?\s+THEN\s+(.+)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if when_match:
            expected_act = when_match.group(1).strip()
            if expected_act != "*" and expected_act != str(env.get("action_type", "")):
                return
            text = when_match.group(2).strip()

        # Handle general IF <cond> THEN <then_body> [ELSE <else_body>]
        if_match = re.match(
            r"^(?:IF|WHEN)\s+(.+?)\s+THEN\s+(.+?)(?:\s+ELSE\s+(.+))?$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if if_match:
            cond_str = if_match.group(1).strip()
            then_body = if_match.group(2).strip()
            else_body = if_match.group(3)
            cond_ok = bool(_safe_eval_expr(cond_str, env))
            active_body = then_body if cond_ok else (else_body.strip() if else_body else "")
            if not active_body:
                return
            text = active_body

        # Split compound clauses on ';' or ' AND ' (when followed by a verb/assignment)
        raw_clauses = re.split(
            r"\s*;\s*|\s+AND\s+(?=(?:deduct|subtract|decrease|add|increase|credit|transfer|set|update|delta|[a-zA-Z0-9_.]+\s*[+\-]?=)\b)",
            text,
            flags=re.IGNORECASE,
        )

        for raw_clause in raw_clauses:
            clause = raw_clause.strip().rstrip(".")
            if not clause:
                continue

            # 1. TRANSFER <expr> FROM <src> TO <dst>
            m_tx = re.match(
                r"^transfer\s+(.+?)\s+from\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?\s+to\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_tx:
                qty = float(_safe_eval_expr(m_tx.group(1), env))
                src_res = m_tx.group(2)
                dst_res = m_tx.group(3)
                resource_deltas[src_res] = resource_deltas.get(src_res, 0.0) - qty
                resource_deltas[dst_res] = resource_deltas.get(dst_res, 0.0) + qty
                env[src_res] = float(env.get(src_res, 0.0)) - qty
                env[dst_res] = float(env.get(dst_res, 0.0)) + qty
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{src_res}",
                        "delta_or_value": -qty,
                        "rationale": clause,
                    }
                )
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{dst_res}",
                        "delta_or_value": qty,
                        "rationale": clause,
                    }
                )
                continue

            # 2. DEDUCT / SUBTRACT <expr> FROM [resource] <res>
            m_ded = re.match(
                r"^(?:deduct|subtract)\s+(.+?)\s+from\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_ded:
                val = float(_safe_eval_expr(m_ded.group(1), env))
                res_id = m_ded.group(2)
                resource_deltas[res_id] = resource_deltas.get(res_id, 0.0) - val
                env[res_id] = float(env.get(res_id, 0.0)) - val
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{res_id}",
                        "delta_or_value": -val,
                        "rationale": clause,
                    }
                )
                continue

            # 3. DECREASE [resource] <res> BY <expr>
            m_dec = re.match(
                r"^decrease\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?\s+by\s+(.+)$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_dec:
                res_id = m_dec.group(1)
                val = float(_safe_eval_expr(m_dec.group(2), env))
                resource_deltas[res_id] = resource_deltas.get(res_id, 0.0) - val
                env[res_id] = float(env.get(res_id, 0.0)) - val
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{res_id}",
                        "delta_or_value": -val,
                        "rationale": clause,
                    }
                )
                continue

            # 4. ADD / CREDIT <expr> TO [resource] <res>
            m_add = re.match(
                r"^(?:add|credit)\s+(.+?)\s+to\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_add:
                val = float(_safe_eval_expr(m_add.group(1), env))
                res_id = m_add.group(2)
                resource_deltas[res_id] = resource_deltas.get(res_id, 0.0) + val
                env[res_id] = float(env.get(res_id, 0.0)) + val
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{res_id}",
                        "delta_or_value": val,
                        "rationale": clause,
                    }
                )
                continue

            # 5. INCREASE [resource] <res> BY <expr>
            m_inc = re.match(
                r"^increase\s+(?:resource\s+)?['\"]?([a-zA-Z0-9_]+)['\"]?\s+by\s+(.+)$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_inc:
                res_id = m_inc.group(1)
                val = float(_safe_eval_expr(m_inc.group(2), env))
                resource_deltas[res_id] = resource_deltas.get(res_id, 0.0) + val
                env[res_id] = float(env.get(res_id, 0.0)) + val
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{res_id}",
                        "delta_or_value": val,
                        "rationale": clause,
                    }
                )
                continue

            # 6. <target> += <expr> or <target> -= <expr> or delta <target> = <expr>
            m_aug = re.match(
                r"^(?:delta\s+)?([a-zA-Z0-9_.:]+)\s*(\+=|-=|=)\s*(.+)$",
                clause,
                flags=re.IGNORECASE,
            )
            if m_aug and (clause.lower().startswith("delta ") or m_aug.group(2) in ("+=", "-=")):
                tgt = m_aug.group(1).strip()
                op_sym = m_aug.group(2)
                val = float(_safe_eval_expr(m_aug.group(3), env))
                signed_val = -val if op_sym == "-=" else val
                if tgt.startswith("resource.") or tgt.startswith("resource:"):
                    res_id = tgt.split("." if "." in tgt else ":", 1)[1]
                else:
                    res_id = tgt
                resource_deltas[res_id] = resource_deltas.get(res_id, 0.0) + signed_val
                env[res_id] = float(env.get(res_id, 0.0)) + signed_val
                attributions.append(
                    {
                        "rule_id": rule_id,
                        "rule_version": rule_version,
                        "target_key": f"resource:{res_id}",
                        "delta_or_value": signed_val,
                        "rationale": clause,
                    }
                )
                continue

            # 7. SET [resource|memory|context|entity] <target> (=|to) <expr>
            m_set = re.match(
                r"^(?:set|update)\s+(?:(resource|memory|context|entity)\s+)?([a-zA-Z0-9_.:\-]+)\s*(?:=|to)\s*(.+)$",
                clause,
                flags=re.IGNORECASE,
            )
            if not m_set:
                m_set = re.match(
                    r"^()((?:resource|memory|context|entity)[.:][a-zA-Z0-9_.:\-]+|[a-zA-Z0-9_]+)\s*=\s*(.+)$",
                    clause,
                    flags=re.IGNORECASE,
                )
            if m_set:
                scope_hint = (m_set.group(1) or "").lower()
                raw_target = m_set.group(2).strip()
                val_eval = _safe_eval_expr(m_set.group(3), env)

                if raw_target.startswith("resource.") or raw_target.startswith("resource:"):
                    scope_hint = "resource"
                    raw_target = raw_target[len("resource.") :]
                elif raw_target.startswith("memory.") or raw_target.startswith("memory:"):
                    scope_hint = "memory"
                    raw_target = raw_target[len("memory.") :]
                elif raw_target.startswith("context.") or raw_target.startswith("context:"):
                    scope_hint = "context"
                    raw_target = raw_target[len("context.") :]
                elif raw_target.startswith("entity.") or raw_target.startswith("entity:"):
                    scope_hint = "entity"
                    raw_target = raw_target[len("entity.") :]

                if scope_hint == "entity" or ("." in raw_target and not scope_hint):
                    parts = (
                        raw_target.split(".", 1) if "." in raw_target else raw_target.split(":", 1)
                    )
                    if len(parts) == 2:
                        eid, attr_name = parts[0], parts[1]
                        entity_updates.setdefault(eid, {})[attr_name] = val_eval
                        attributions.append(
                            {
                                "rule_id": rule_id,
                                "rule_version": rule_version,
                                "target_key": f"entity:{eid}:{attr_name}",
                                "delta_or_value": val_eval,
                                "rationale": clause,
                            }
                        )
                        continue

                if scope_hint == "resource" or (not scope_hint and raw_target in resources):
                    fval = float(val_eval)
                    resource_overrides[raw_target] = fval
                    env[raw_target] = fval
                    attributions.append(
                        {
                            "rule_id": rule_id,
                            "rule_version": rule_version,
                            "target_key": f"resource:{raw_target}",
                            "delta_or_value": fval,
                            "rationale": clause,
                        }
                    )
                elif scope_hint == "context":
                    context_updates[raw_target] = val_eval
                    env[raw_target] = val_eval
                    attributions.append(
                        {
                            "rule_id": rule_id,
                            "rule_version": rule_version,
                            "target_key": f"context:{raw_target}",
                            "delta_or_value": val_eval,
                            "rationale": clause,
                        }
                    )
                else:
                    memory_updates[raw_target] = val_eval
                    env[raw_target] = val_eval
                    attributions.append(
                        {
                            "rule_id": rule_id,
                            "rule_version": rule_version,
                            "target_key": f"memory:{raw_target}",
                            "delta_or_value": val_eval,
                            "rationale": clause,
                        }
                    )


def parse_sandboxed_state_diff(
    raw_output: Any,
    retrieved_rules: Sequence[BusinessRule],
    *,
    latency_ms: float = 0.0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> StateDiff:
    """Sandboxed parser converting LLM JSON output into a validated ``StateDiff``.

    Operational Safeguard (E3 Requirement 11):
    Gracefully handles malformed JSON, truncated strings, markdown code fences,
    schema violations, or non-finite numbers by defaulting to a safe no-op
    ``StateDiff(is_noop=True, parse_error=...)`` instead of crashing the simulation.
    """
    rule_ids = tuple(r.rule_id for r in retrieved_rules)
    rule_version_map = {r.rule_id: r.version for r in retrieved_rules}
    total_tok = prompt_tokens + completion_tokens

    try:
        if isinstance(raw_output, Mapping):
            payload = dict(raw_output)
        elif isinstance(raw_output, str):
            cleaned = raw_output.strip()
            if not cleaned:
                raise ValueError("Empty LLM diff response.")
            # Strip optional ```json ... ``` fences
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if fence_match:
                cleaned = fence_match.group(1)
            payload = json.loads(cleaned)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object, got {type(payload).__name__}.")
        else:
            raise TypeError(f"Unsupported LLM output type: {type(raw_output).__name__}.")

        res_deltas: Dict[str, float] = {}
        for k, v in (payload.get("resource_deltas") or {}).items():
            fval = float(v)
            if not math.isfinite(fval):
                raise ValueError(f"Non-finite resource_delta for '{k}': {v!r}")
            res_deltas[str(k)] = fval

        res_overrides: Dict[str, float] = {}
        for k, v in (payload.get("resource_overrides") or {}).items():
            fval = float(v)
            if not math.isfinite(fval):
                raise ValueError(f"Non-finite resource_override for '{k}': {v!r}")
            res_overrides[str(k)] = fval

        ent_updates: Dict[str, Dict[str, Any]] = {}
        for eid, attrs in (payload.get("entity_updates") or {}).items():
            if not isinstance(attrs, Mapping):
                raise ValueError(f"entity_updates['{eid}'] must be a mapping.")
            ent_updates[str(eid)] = dict(attrs)

        mem_updates: Dict[str, Any] = dict(payload.get("memory_updates") or {})
        ctx_updates: Dict[str, Any] = dict(payload.get("context_updates") or {})

        raw_attrs = payload.get("rule_attributions") or []
        if not isinstance(raw_attrs, list):
            raise ValueError("rule_attributions must be a list.")

        parsed_attributions: List[RuleAttribution] = []
        for item in raw_attrs:
            if not isinstance(item, Mapping):
                raise ValueError("Each rule_attribution entry must be an object.")
            rid = str(item.get("rule_id", "")).strip()
            if not rid:
                raise ValueError("rule_attribution entry missing 'rule_id'.")
            if rule_version_map and rid not in rule_version_map:
                raise ValueError(f"Unretrieved rule_id '{rid}' in rule_attributions.")
            parsed_attributions.append(
                RuleAttribution(
                    rule_id=rid,
                    rule_version=str(item.get("rule_version", rule_version_map.get(rid, "1.0.0"))),
                    target_key=str(item.get("target_key", "state")),
                    delta_or_value=item.get("delta_or_value"),
                    rationale=str(item.get("rationale", "")),
                )
            )

        has_modifications = bool(
            res_deltas or res_overrides or ent_updates or mem_updates or ctx_updates
        )

        # Contract: Explicit attribution of predicted changes to retrieved business rules is required
        if has_modifications and not parsed_attributions:
            if not retrieved_rules:
                raise ValueError(
                    "StateDiff contains predicted changes but no business rules were retrieved or attributed."
                )
            primary_rule = retrieved_rules[0]
            for rk, dv in res_deltas.items():
                parsed_attributions.append(
                    RuleAttribution(
                        rule_id=primary_rule.rule_id,
                        rule_version=primary_rule.version,
                        target_key=f"resource:{rk}",
                        delta_or_value=dv,
                        rationale=primary_rule.rule_text,
                    )
                )
            for rk, ov in res_overrides.items():
                parsed_attributions.append(
                    RuleAttribution(
                        rule_id=primary_rule.rule_id,
                        rule_version=primary_rule.version,
                        target_key=f"resource:{rk}",
                        delta_or_value=ov,
                        rationale=primary_rule.rule_text,
                    )
                )
            for mk, mv in mem_updates.items():
                parsed_attributions.append(
                    RuleAttribution(
                        rule_id=primary_rule.rule_id,
                        rule_version=primary_rule.version,
                        target_key=f"memory:{mk}",
                        delta_or_value=mv,
                        rationale=primary_rule.rule_text,
                    )
                )

        return StateDiff(
            resource_deltas=res_deltas,
            resource_overrides=res_overrides,
            entity_updates=ent_updates,
            memory_updates=mem_updates,
            context_updates=ctx_updates,
            rule_attributions=tuple(parsed_attributions),
            retrieved_rule_ids=rule_ids,
            is_noop=not has_modifications,
            parse_error=None,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
        )
    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        KeyError,
        ValidationError,
    ) as exc:
        return StateDiff(
            is_noop=True,
            parse_error=f"Sandboxed parser fallback to no-op: {exc}",
            retrieved_rule_ids=rule_ids,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tok,
        )


class RetrieveAndReasonEngine(DynamicsInterface):
    """Continual Discovery Agent (CDA) Retrieve-Then-Reason Dynamics Engine.

    Models enterprise state transitions as:
        s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)
    by:
    1. Retrieving active business rules $c$ relevant to proposed action $a_t$ from a
       ``ConfigurationDatastore`` (or custom context retriever),
    2. Constructing a structured reasoning prompt over $(s_t, a_t, c)$ via ``CDAPromptHarness``,
    3. Invoking the reasoning agent / LLM client, and
    4. Extracting a sandboxed, rule-attributed ``StateDiff`` ($\\Delta s_{t+1}$).
    """

    def __init__(
        self,
        datastore: Optional[ConfigurationDatastore] = None,
        reasoning_client: Optional[Any] = None,
        prompt_harness: Optional[CDAPromptHarness] = None,
        name: str = "RetrieveAndReasonEngine",
    ) -> None:
        self.datastore: ConfigurationDatastore = datastore or ConfigurationDatastore()
        self.reasoning_client: Any = reasoning_client or RuleGuidedReasoningClient()
        self.prompt_harness: CDAPromptHarness = prompt_harness or CDAPromptHarness()
        self.name: str = name
        self._total_calls: int = 0
        self._total_latency_ms: float = 0.0
        self._total_tokens: int = 0

    @property
    def telemetry_summary(self) -> Dict[str, Union[int, float]]:
        """Return cumulative execution telemetry (calls, latency_ms, total_tokens)."""
        avg_lat = self._total_latency_ms / self._total_calls if self._total_calls > 0 else 0.0
        return {
            "total_calls": self._total_calls,
            "total_latency_ms": round(self._total_latency_ms, 4),
            "mean_latency_ms": round(avg_lat, 4),
            "total_tokens": self._total_tokens,
        }

    def _retrieve_active_rules(
        self,
        state: Union[WorldState, Mapping[str, Any]],
        action: Union[Action, Mapping[str, Any]],
        retriever: Optional[Any] = None,
    ) -> List[BusinessRule]:
        target_retriever = retriever if retriever is not None else self.datastore
        if hasattr(target_retriever, "retrieve_rules") and callable(
            target_retriever.retrieve_rules
        ):
            return list(target_retriever.retrieve_rules(action, state))
        if callable(target_retriever):
            raw_rules = target_retriever(state, action)
            return [
                r if isinstance(r, BusinessRule) else BusinessRule.model_validate(dict(r))
                for r in raw_rules
            ]
        raise CDAConfigurationError(
            f"Unsupported context retriever of type '{type(target_retriever).__name__}'."
        )

    def _invoke_reasoning_client(self, system_prompt: str, user_prompt: str) -> Any:
        client = self.reasoning_client
        if hasattr(client, "complete") and callable(client.complete):
            return client.complete(system_prompt, user_prompt)
        if callable(client):
            sig = inspect.signature(client)
            if len(sig.parameters) >= 2:
                return client(system_prompt, user_prompt)
            return client(f"{system_prompt}\n\n{user_prompt}")
        raise CDAConfigurationError(
            f"Reasoning client '{type(client).__name__}' must implement .complete(sys, user) or be callable."
        )

    def predict_diff(
        self,
        state: Union[WorldState, Mapping[str, Any]],
        action: Union[Action, Mapping[str, Any]],
        retriever: Optional[Any] = None,
    ) -> StateDiff:
        """Execute the retrieve-then-reason pipeline for $(s_t, a_t)$ and return $\\Delta s_{t+1}$."""
        t0 = time.perf_counter()
        retrieved_rules = self._retrieve_active_rules(state, action, retriever=retriever)

        if not retrieved_rules:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._total_calls += 1
            self._total_latency_ms += elapsed_ms
            return StateDiff(
                is_noop=True,
                retrieved_rule_ids=(),
                latency_ms=elapsed_ms,
            )

        sys_prompt = self.prompt_harness.build_system_prompt()
        usr_prompt = self.prompt_harness.build_user_prompt(state, action, retrieved_rules)
        prompt_tok = self.prompt_harness.estimate_tokens(sys_prompt + "\n" + usr_prompt)

        try:
            raw_response = self._invoke_reasoning_client(sys_prompt, usr_prompt)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._total_calls += 1
            self._total_latency_ms += elapsed_ms
            self._total_tokens += prompt_tok
            return StateDiff(
                is_noop=True,
                parse_error=f"Reasoning client failure fallback to no-op: {exc}",
                retrieved_rule_ids=tuple(r.rule_id for r in retrieved_rules),
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=0,
                total_tokens=prompt_tok,
            )

        comp_str = raw_response if isinstance(raw_response, str) else json.dumps(raw_response)
        comp_tok = self.prompt_harness.estimate_tokens(comp_str)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        diff = parse_sandboxed_state_diff(
            raw_response,
            retrieved_rules,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
        )
        self._total_calls += 1
        self._total_latency_ms += elapsed_ms
        self._total_tokens += diff.total_tokens
        return diff

    def step(
        self,
        state: WorldState,
        action: Union[Action, Mapping[str, Any]],
        retriever: Optional[Any] = None,
    ) -> Tuple[WorldState, StateDiff]:
        """Compute $\\Delta s_{t+1}$ and return `(next_state, state_diff)`."""
        diff = self.predict_diff(state, action, retriever=retriever)
        next_state = diff.apply_to_state(state, advance_step=True)
        return next_state, diff

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
        runtime_context: Optional[ContextSnapshot] = None,
    ) -> Any:
        """Pluggable ``DynamicsInterface`` implementation for ``SimulationEngine`` integration."""
        from ewm_engine.dynamics.base import TransitionResult

        _ = (exogenous_events, rng, runtime_context)
        current_state = state.clone()
        diffs: List[Dict[str, Any]] = []
        attributions: List[Dict[str, Any]] = []

        for act in actions:
            diff = self.predict_diff(current_state, act)
            current_state = diff.apply_to_state(current_state, advance_step=False)
            diffs.append(diff.model_dump(mode="json"))
            attributions.extend(a.model_dump(mode="json") for a in diff.rule_attributions)

        return TransitionResult(
            next_state=current_state,
            applied_changes={
                "actions_processed": len(actions),
                "rule_attributions": attributions,
            },
            evidence_level=EvidenceLevel.STRUCTURAL,
            diagnostics={
                "cda_diffs": diffs,
                "telemetry": self.telemetry_summary,
            },
            model_name=self.name,
        )


__all__ = [
    "BusinessRule",
    "CDAConfigurationError",
    "CDAConfigurationPayload",
    "CDAInputPayload",
    "CDAPromptHarness",
    "ConfigurationDatastore",
    "ContextRetrieverProtocol",
    "LatentRolloutPredictor",
    "LatentStatePredictor",
    "NeuralLatentDynamics",
    "RetrieveAndReasonEngine",
    "RuleAttribution",
    "RuleGuidedReasoningClient",
    "StateDiff",
    "parse_cda_configuration_json",
    "parse_sandboxed_state_diff",
]
