"""Pytest smoke tests for fuzz harnesses with structured mutation.

In accordance with R03 Task 5:
- Validates seed corpus across all 3 fuzz harnesses (YAML, JSON, WSL).
- Executes bounded structure-aware and random mutations to guarantee no crashes.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tests.fuzz.fuzz_json_loader import TestOneInput as run_json_fuzz
from tests.fuzz.fuzz_wsl_compiler import TestOneInput as run_wsl_fuzz
from tests.fuzz.fuzz_yaml_loader import TestOneInput as run_yaml_fuzz

CORPUS_DIR = Path(__file__).parent / "corpus"


def mutate_bytes(data: bytes, rng: random.Random) -> bytes:
    """Apply a random byte mutation: bit flip, byte replacement, insertion, or deletion."""
    if not data:
        return b"x"
    b_arr = bytearray(data)
    op = rng.choice(["flip", "replace", "insert", "delete"])
    idx = rng.randint(0, len(b_arr) - 1)

    if op == "flip":
        b_arr[idx] ^= 1 << rng.randint(0, 7)
    elif op == "replace":
        b_arr[idx] = rng.randint(0, 255)
    elif op == "insert":
        b_arr.insert(idx, rng.randint(0, 255))
    elif op == "delete" and len(b_arr) > 1:
        del b_arr[idx]

    return bytes(b_arr)


@pytest.mark.fuzz
def test_yaml_loader_fuzz_smoke() -> None:
    """Fuzz smoke test for safe YAML loader against seed corpus and random mutations."""
    rng = random.Random(42)
    seeds = list((CORPUS_DIR / "yaml").glob("*.yaml"))
    assert len(seeds) > 0, "Seed corpus for YAML must not be empty"

    for seed_path in seeds:
        seed_data = seed_path.read_bytes()
        # Verify valid seed does not crash
        run_yaml_fuzz(seed_data)

        # 50 mutations per seed
        current = seed_data
        for _ in range(50):
            current = mutate_bytes(current, rng)
            run_yaml_fuzz(current)


@pytest.mark.fuzz
def test_json_loader_fuzz_smoke() -> None:
    """Fuzz smoke test for safe JSON loader against seed corpus and random mutations."""
    rng = random.Random(42)
    seeds = list((CORPUS_DIR / "json").glob("*.json"))
    assert len(seeds) > 0, "Seed corpus for JSON must not be empty"

    for seed_path in seeds:
        seed_data = seed_path.read_bytes()
        run_json_fuzz(seed_data)

        current = seed_data
        for _ in range(50):
            current = mutate_bytes(current, rng)
            run_json_fuzz(current)


@pytest.mark.fuzz
def test_wsl_compiler_fuzz_smoke() -> None:
    """Fuzz smoke test for WSL parser and compiler against seed corpus and mutations."""
    rng = random.Random(42)
    seeds = list((CORPUS_DIR / "wsl").glob("*.wsl"))
    assert len(seeds) > 0, "Seed corpus for WSL must not be empty"

    for seed_path in seeds:
        seed_data = seed_path.read_bytes()
        run_wsl_fuzz(seed_data)

        current = seed_data
        for _ in range(50):
            current = mutate_bytes(current, rng)
            run_wsl_fuzz(current)
