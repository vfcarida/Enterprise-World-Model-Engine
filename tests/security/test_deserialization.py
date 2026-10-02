"""Adversarial security test suite (AC-018): Safe deserialization and trusted registry enforcement."""

from __future__ import annotations

import pickle

import pytest
from pydantic import ValidationError

from ewm_engine.core.spec import (
    ComponentRegistry,
    ComponentSpec,
    WorldFactory,
    WorldMetadataSpec,
    WorldSpec,
)
from ewm_engine.exceptions import SerializationSecurityError, SimulationConfigurationError
from ewm_engine.serialization.yaml import load_pickle, safe_load_yaml


@pytest.mark.security
def test_yaml_rejects_python_object_tag() -> None:
    """AC-018: Strict safe loader must reject !!python/object tags with SerializationSecurityError."""
    unsafe_payload = """
    world:
      name: "CompromisedWorld"
    dynamics:
      type: !!python/object:os.system {}
    """
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        safe_load_yaml(unsafe_payload)

    with pytest.raises(SerializationSecurityError):
        WorldSpec.from_yaml(unsafe_payload)


@pytest.mark.security
def test_yaml_rejects_python_object_apply() -> None:
    """AC-018: Strict safe loader must reject !!python/object/apply code execution."""
    unsafe_payload = """
    world:
      name: !!python/object/apply:os.system ["id"]
    """
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        safe_load_yaml(unsafe_payload)

    with pytest.raises(SerializationSecurityError):
        WorldSpec.from_yaml(unsafe_payload)


@pytest.mark.security
def test_yaml_rejects_python_name_and_module_tags() -> None:
    """AC-018: Strict safe loader must reject !!python/name and !!python/module tags."""
    payload_name = """
    func: !!python/name:os.system
    """
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        safe_load_yaml(payload_name)

    payload_module = """
    mod: !!python/module:subprocess
    """
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        safe_load_yaml(payload_module)


@pytest.mark.security
def test_yaml_rejects_arbitrary_custom_tags() -> None:
    """AC-018: Strict safe loader must reject arbitrary custom tags."""
    custom_tag_payload = """
    world:
      name: !CustomConstructor "malicious_payload"
    """
    with pytest.raises(SerializationSecurityError, match="Unsafe YAML tag"):
        safe_load_yaml(custom_tag_payload)


@pytest.mark.security
def test_pickle_deserialization_is_strictly_forbidden() -> None:
    """AC-018: Any pickle deserialization attempt must raise SerializationSecurityError."""
    payload = pickle.dumps({"trusted": False})

    with pytest.raises(
        SerializationSecurityError, match="Pickle deserialization is strictly forbidden"
    ):
        load_pickle(payload)


@pytest.mark.security
def test_pickle_payload_in_yaml_stream_is_rejected() -> None:
    """AC-018: Reject accidental or malicious pickle opcode sequences in YAML input."""
    raw_pickle_str = "cos\nsystem\n(S'echo exploit'\ntR."

    with pytest.raises(SerializationSecurityError, match="Pickle opcode"):
        safe_load_yaml(raw_pickle_str)


@pytest.mark.security
def test_component_spec_rejects_importable_paths_and_unknown_fields() -> None:
    """G8 & AC-018: ComponentSpec must fail closed on unknown fields like implementation or module."""
    # Attempting to supply an import path or unknown field must fail validation
    with pytest.raises(ValidationError):
        ComponentSpec.model_validate(
            {
                "type": "transfer",
                "implementation": "ewm_engine.dynamics.deterministic.DeterministicTransferDynamics",
            }
        )

    with pytest.raises(ValidationError):
        ComponentSpec.model_validate(
            {
                "type": "transfer",
                "module": "os",
                "class": "system",
            }
        )


@pytest.mark.security
def test_world_factory_fails_closed_on_unregistered_components() -> None:
    """G8 & AC-018: WorldFactory must never dynamically import or execute unregistered type IDs."""
    spec = WorldSpec(
        world=WorldMetadataSpec(name="AttackerWorld"),
        dynamics=ComponentSpec(type="malicious_unregistered_dynamics", parameters={}),
        constraints=[
            ComponentSpec(type="malicious_unregistered_constraint", parameters={}),
        ],
        event_sources=[
            ComponentSpec(type="malicious_unregistered_event_source", parameters={}),
        ],
    )

    factory = WorldFactory(registry=ComponentRegistry())

    with pytest.raises(SimulationConfigurationError, match="Unknown dynamics component type"):
        factory.create_world(spec)

    # When dynamics is valid but constraint is unknown
    spec_valid_dynamics = spec.model_copy(update={"dynamics": None})
    with pytest.raises(SimulationConfigurationError, match="Unknown constraint type"):
        factory.create_world(spec_valid_dynamics)

    # When dynamics and constraints valid but event source is unknown
    spec_valid_constraints = spec_valid_dynamics.model_copy(update={"constraints": []})
    with pytest.raises(SimulationConfigurationError, match="Unknown event source component type"):
        factory.create_world(spec_valid_constraints)
