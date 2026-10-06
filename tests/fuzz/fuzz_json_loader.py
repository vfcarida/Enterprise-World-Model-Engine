"""Atheris fuzzing harness for canonical and safe JSON loader.

Untrusted-input surface: ewm_engine.serialization.json.canonical_loads
Tests resilience against:
- Non-standard JSON constants (NaN, Infinity, -Infinity)
- Deeply nested recursive objects
- Malformed encodings and invalid syntax
"""

from __future__ import annotations

import sys

try:
    import atheris

    with atheris.instrument_imports():
        from ewm_engine.exceptions import SerializationError, SerializationSecurityError
        from ewm_engine.serialization.json import canonical_loads
except ImportError:
    atheris = None
    from ewm_engine.exceptions import SerializationError, SerializationSecurityError
    from ewm_engine.serialization.json import canonical_loads


def TestOneInput(data: bytes) -> None:
    """Consume raw fuzz input, decode, and execute canonical_loads."""
    try:
        json_text = data.decode("utf-8")
    except UnicodeDecodeError:
        return

    try:
        canonical_loads(json_text)
    except (SerializationError, SerializationSecurityError, RecursionError):
        # Expected safe rejections
        pass
    except Exception as exc:
        raise exc


if __name__ == "__main__":
    if atheris is not None:
        atheris.instrument_all()
        atheris.Setup(sys.argv, TestOneInput)
        atheris.Fuzz()
    else:
        print(
            "Atheris is not installed. Use scripts/run_fuzz_smoke.py for cross-platform fuzz testing."
        )
