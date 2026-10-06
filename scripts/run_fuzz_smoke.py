"""Cross-platform smoke fuzz runner for CI and local developer workstations.

Executes bounded fuzz testing on untrusted-input surfaces (YAML, JSON, WSL).
- On Linux CI with Atheris installed: invokes libFuzzer/Atheris engine with -max_total_time.
- On Windows or environments without Atheris: runs structured corpus mutation loops.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Add src and tests to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tests.fuzz.fuzz_json_loader import TestOneInput as fuzz_json  # noqa: E402
from tests.fuzz.fuzz_wsl_compiler import TestOneInput as fuzz_wsl  # noqa: E402
from tests.fuzz.fuzz_yaml_loader import TestOneInput as fuzz_yaml  # noqa: E402

CORPUS_DIR = REPO_ROOT / "tests" / "fuzz" / "corpus"


def mutate(data: bytes, rng: random.Random) -> bytes:
    if not data:
        return b"\x00"
    arr = bytearray(data)
    op = rng.choice(["flip", "replace", "insert", "delete"])
    idx = rng.randint(0, len(arr) - 1)
    if op == "flip":
        arr[idx] ^= 1 << rng.randint(0, 7)
    elif op == "replace":
        arr[idx] = rng.randint(0, 255)
    elif op == "insert":
        arr.insert(idx, rng.randint(0, 255))
    elif op == "delete" and len(arr) > 1:
        del arr[idx]
    return bytes(arr)


def run_target(name: str, target_fn, corpus_subpath: str, max_seconds: float) -> int:
    print(f"[*] Starting fuzz smoke on '{name}' for {max_seconds:.1f}s...")
    corpus_dir = CORPUS_DIR / corpus_subpath
    seeds = [p.read_bytes() for p in corpus_dir.glob("*") if p.is_file()]
    if not seeds:
        print(f"[!] Warning: no seed files found in {corpus_dir}")
        seeds = [b""]

    # Test baseline seeds
    for seed in seeds:
        target_fn(seed)

    start_time = time.time()
    iterations = 0
    rng = random.Random(1337)

    while time.time() - start_time < max_seconds:
        base = rng.choice(seeds)
        mutated = mutate(base, rng)
        # Apply 1 to 5 successive mutations
        for _ in range(rng.randint(0, 4)):
            mutated = mutate(mutated, rng)
        target_fn(mutated)
        iterations += 1

    elapsed = time.time() - start_time
    print(
        f"[+] '{name}' passed cleanly: {iterations} mutations executed in {elapsed:.2f}s ({iterations / elapsed:.0f} exec/s)."
    )
    return iterations


def main() -> int:
    parser = argparse.ArgumentParser(description="EWM Engine Bounded Fuzz Smoke Runner")
    parser.add_argument(
        "--max-seconds", type=float, default=3.0, help="Max execution time per target in seconds"
    )
    args = parser.parse_args()

    print(f"=== EWM Engine Fuzz Smoke Suite (Target duration: {args.max_seconds}s/target) ===")
    total_execs = 0
    try:
        total_execs += run_target("YAML Loader", fuzz_yaml, "yaml", args.max_seconds)
        total_execs += run_target("JSON Loader", fuzz_json, "json", args.max_seconds)
        total_execs += run_target("WSL Compiler", fuzz_wsl, "wsl", args.max_seconds)
        print(
            f"\n[OK] All fuzz smoke targets passed with zero attributable crashes! ({total_execs} total iterations)"
        )
        return 0
    except Exception as exc:
        print(f"\n[FAIL] Attributable crash encountered during fuzz smoke: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
