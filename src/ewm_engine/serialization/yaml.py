"""Safe YAML serialization and deserialization with strict tag and code-execution prevention."""

from __future__ import annotations

from typing import Any, TypeVar

import yaml
from pydantic import BaseModel
from yaml.constructor import ConstructorError

from ewm_engine.core._canonical import normalize_canonical_value
from ewm_engine.exceptions import SerializationError, SerializationSecurityError

T = TypeVar("T", bound=BaseModel)


class StrictSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """SafeLoader variant strictly prohibiting custom, Python, and external tags."""


def _forbid_custom_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> None:
    raise SerializationSecurityError(
        f"Unsafe YAML tag '{node.tag}' detected. Custom and Python object tags are strictly forbidden."
    )


# Intercept all custom tags starting with '!' and Python tags
StrictSafeLoader.add_multi_constructor("!", _forbid_custom_tag)
StrictSafeLoader.add_multi_constructor("tag:yaml.org,2002:python/", _forbid_custom_tag)


def safe_load_yaml(yaml_str: str) -> Any:
    """Parse YAML string into plain, safe Python data structures.

    Guarantees:
    - Never executes arbitrary Python objects or constructors.
    - Rejects any custom or Python tags (e.g. `!tag`, `!!python/object`).
    - Produces only standard JSON-compatible Python primitives (dict, list, str, int, float, bool, None).
    """
    if not isinstance(yaml_str, str):
        raise SerializationSecurityError(
            f"Expected string for YAML deserialization, got {type(yaml_str).__name__}"
        )

    # Detect pickle bytes passed accidentally as string or raw
    if "cos\nsystem" in yaml_str or "cposix\nsystem" in yaml_str:
        raise SerializationSecurityError("Pickle opcode sequence detected in input data.")

    try:
        data = yaml.load(yaml_str, Loader=StrictSafeLoader)
    except SerializationSecurityError:
        raise
    except ConstructorError as exc:
        raise SerializationSecurityError(f"Unsafe YAML tag or construct detected: {exc}") from exc
    except Exception as exc:
        raise SerializationError(f"Invalid YAML content: {exc}") from exc

    return normalize_canonical_value(data)


def safe_dump_yaml(obj: Any) -> str:
    """Dump any model or mapping into clean, safe YAML without custom tags."""
    if isinstance(obj, BaseModel):
        data = obj.model_dump(mode="json")
    else:
        data = normalize_canonical_value(obj)
    return str(
        yaml.dump(
            data,
            Dumper=yaml.SafeDumper,
            sort_keys=True,
            default_flow_style=False,
            allow_unicode=True,
        )
    )


def load_model_yaml(model_cls: type[T], yaml_str: str) -> T:
    """Safely parse YAML and validate into a Pydantic domain model."""
    data = safe_load_yaml(yaml_str)
    if not isinstance(data, dict):
        raise SerializationError(
            f"Expected YAML mapping for model {model_cls.__name__}, got {type(data).__name__}"
        )
    return model_cls.model_validate(data)


def dump_model_yaml(model: BaseModel) -> str:
    """Serialize a Pydantic domain model to safe YAML string."""
    return safe_dump_yaml(model)


def load_pickle(*args: Any, **kwargs: Any) -> None:
    """Explicit security barrier: pickle deserialization is strictly prohibited."""
    raise SerializationSecurityError(
        "Pickle deserialization is strictly forbidden in EWM Engine for security reasons."
    )
