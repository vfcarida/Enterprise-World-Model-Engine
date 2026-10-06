"""Atheris fuzzing harness for safe YAML loader.

Untrusted-input surface: ewm_engine.serialization.yaml.safe_load_yaml
Tests resilience against:
- Custom / unsafe YAML tags (!python/object, etc.)
- Billion laughs / recursive entity expansion
- Pickle byte sequence injection
- Malformed syntax and non-UTF-8 sequences
"""

from __future__ import annotations

import sys

try:
    import atheris

    with atheris.instrument_imports():
        from ewm_engine.exceptions import SerializationError, SerializationSecurityError
        from ewm_engine.serialization.yaml import safe_load_yaml
except ImportError:
    atheris = None
    from ewm_engine.exceptions import SerializationError, SerializationSecurityError
    from ewm_engine.serialization.yaml import safe_load_yaml


def TestOneInput(data: bytes) -> None:
    """Consume raw fuzz input, decode, and execute safe_load_yaml."""
    try:
        yaml_text = data.decode("utf-8")
    except UnicodeDecodeError:
        # Invalid UTF-8 is expected to be rejected at string decode stage
        return

    try:
        safe_load_yaml(yaml_text)
    except (SerializationError, SerializationSecurityError, RecursionError):
        # Expected safe rejections
        pass
    except Exception as exc:
        # Any unexpected internal crash or unhandled error is a bug
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
