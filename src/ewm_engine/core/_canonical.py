"""Internal canonical serialization and deterministic cryptographic hashing utilities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ewm_engine.exceptions import InvalidWorldStateError

CANONICAL_FLOAT_PRECISION = 6


def normalize_canonical_value(val: Any) -> Any:
    """Recursively normalize any Python value into a canonical, deterministic representation.

    Rules:
    - Rejects non-finite floats (NaN, +Inf, -Inf) with InvalidWorldStateError.
    - Deterministically rounds finite floats to CANONICAL_FLOAT_PRECISION (6 decimal places).
    - Converts datetime objects to UTC ISO-8601 strings. Rejects naive datetimes.
    - Converts Mappings into sorted dictionaries (keys ordered lexicographically).
    - Converts Sequences (lists, tuples) into lists of normalized elements.
    - Converts Sets into sorted lists.
    - Passes booleans, integers, strings, and None as-is.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise InvalidWorldStateError(
                f"Canonical serialization rejects non-finite float value '{val}' (NaN or Infinity)."
            )
        rounded = round(val, CANONICAL_FLOAT_PRECISION)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(val, datetime):
        if val.tzinfo is None or val.tzinfo.utcoffset(val) is None:
            dt = val.replace(tzinfo=UTC)
        else:
            dt = val.astimezone(UTC)
        return dt.isoformat()
    if isinstance(val, (str, bytes)):
        return val if isinstance(val, str) else val.decode("utf-8", errors="replace")
    if val is None:
        return None
    if isinstance(val, Mapping):
        return {
            str(k): normalize_canonical_value(v)
            for k, v in sorted(val.items(), key=lambda item: str(item[0]))
        }
    if isinstance(val, (set, frozenset)):
        normalized_items = [normalize_canonical_value(item) for item in val]
        return sorted(normalized_items, key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(val, Sequence):
        return [normalize_canonical_value(item) for item in val]
    if hasattr(val, "model_dump"):
        return normalize_canonical_value(val.model_dump())

    return str(val)


def canonical_json_dumps(obj: Any) -> str:
    """Serialize an object into canonical JSON: sorted keys, compact separators, UTF-8 compatible."""
    normalized = normalize_canonical_value(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_sha256(obj: Any) -> str:
    """Produce a deterministic SHA-256 hexadecimal digest over canonical JSON."""
    encoded = canonical_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
