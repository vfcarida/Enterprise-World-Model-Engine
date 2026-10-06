"""Atheris fuzzing harness for WSL grammar parser and compiler.

Untrusted-input surface:
- ewm_engine.serialization.wsl.parse_wsl_yaml
- ewm_engine.serialization.wsl.compile_wsl

Tests resilience against:
- Malformed schema structures, invalid entity/relationship references
- Illegal resource bounds and cycles
- Arbitrary YAML constructs in WSL specifications
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

try:
    import atheris

    with atheris.instrument_imports():
        from ewm_engine.exceptions import EWMError
        from ewm_engine.serialization.wsl import compile_wsl, parse_wsl_yaml
except ImportError:
    atheris = None
    from ewm_engine.exceptions import EWMError
    from ewm_engine.serialization.wsl import compile_wsl, parse_wsl_yaml


def TestOneInput(data: bytes) -> None:
    """Consume raw fuzz input, decode, parse WSL document, and compile to World."""
    try:
        wsl_text = data.decode("utf-8")
    except UnicodeDecodeError:
        return

    try:
        doc = parse_wsl_yaml(wsl_text)
        compile_wsl(doc)
    except (
        EWMError,
        ValidationError,
        ValueError,
        KeyError,
        TypeError,
        RecursionError,
    ):
        # Graceful, controlled rejection of invalid or malformed domain inputs
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
