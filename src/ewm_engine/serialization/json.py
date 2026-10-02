"""Canonical and safe JSON serialization for enterprise world model data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ewm_engine.core._canonical import canonical_json_dumps
from ewm_engine.exceptions import (
    InvalidWorldStateError,
    SerializationError,
    SerializationSecurityError,
)

T = TypeVar("T", bound=BaseModel)


def _reject_non_standard_constants(constant_name: str) -> None:
    """Callback for json.loads parse_constant rejecting NaN, Infinity, -Infinity."""
    raise SerializationSecurityError(
        f"Canonical JSON deserialization rejects non-standard constant '{constant_name}' (NaN or Infinity)."
    )


def canonical_dumps(obj: Any) -> str:
    """Serialize an object or dictionary to canonical RFC 8259 JSON.

    - Sorts all keys recursively.
    - Uses compact separators (',', ':').
    - Deterministically rounds floats to 6 decimal places.
    - Rejects non-finite floats (NaN, Infinity).
    """
    try:
        return canonical_json_dumps(obj)
    except InvalidWorldStateError as exc:
        raise SerializationSecurityError(str(exc)) from exc


def canonical_loads(json_str: str) -> Any:
    """Safely parse a JSON string into plain Python data.

    Strictly rejects non-standard JSON values (NaN, Infinity, -Infinity).
    """
    try:
        return json.loads(json_str, parse_constant=_reject_non_standard_constants)
    except SerializationSecurityError:
        raise
    except Exception as exc:
        raise SerializationError(f"Invalid JSON content: {exc}") from exc


def dump_model_json(model: BaseModel) -> str:
    """Serialize a Pydantic domain model to canonical JSON."""
    data = model.model_dump(mode="json")
    return canonical_dumps(data)


def load_model_json(model_cls: type[T], json_str: str) -> T:
    """Safely parse and validate a Pydantic domain model from canonical JSON."""
    data = canonical_loads(json_str)
    if not isinstance(data, dict):
        raise SerializationError(
            f"Expected JSON object for model {model_cls.__name__}, got {type(data).__name__}"
        )
    return model_cls.model_validate(data)


def to_json_file(obj: Any, path: Path | str) -> None:
    """Write an object or model to a JSON file."""
    content = canonical_dumps(
        obj if not isinstance(obj, BaseModel) else obj.model_dump(mode="json")
    )
    Path(path).write_text(content, encoding="utf-8")


def from_json_file(path: Path | str) -> Any:
    """Read and parse plain JSON data from a file."""
    content = Path(path).read_text(encoding="utf-8")
    return canonical_loads(content)
