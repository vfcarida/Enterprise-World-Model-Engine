"""Cross-platform smoke fuzz runner for CI and local developer workstations.

Executes bounded fuzz testing on untrusted-input surfaces (YAML, JSON, WSL):
- Exposes and records deterministic `--seed` for 100% reproducible mutation streams.
- Automatically persists any crashing input and seed metadata to `--crash-dir`
  (`tests/fuzz/crashes/`) so failures can be replayed deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

# Add src and tests to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tests.fuzz.fuzz_json_loader import TestOneInput as fuzz_json
from tests.fuzz.fuzz_wsl_compiler import TestOneInput as fuzz_wsl
from tests.fuzz.fuzz_yaml_loader import TestOneInput as fuzz_yaml

CORPUS_DIR = REPO_ROOT / "tests" / "fuzz" / "corpus"
DEFAULT_CRASH_DIR = REPO_ROOT / "tests" / "fuzz" / "crashes"


def mutate(data: bytes, rng: random.Random) -> bytes:
    """Apply a deterministic byte-level mutation using the caller-provided RNG."""
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


def persist_failing_seed(
    target_name: str,
    master_seed: int,
    iteration: int,
    payload: bytes,
    exc: BaseException,
    crash_dir: Path = DEFAULT_CRASH_DIR,
) -> Path:
    """Persist a crashing fuzz input and its reproduction metadata to disk."""
    crash_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    safe_target = target_name.lower().replace(" ", "_")
    stem = f"crash_{safe_target}_seed{master_seed}_iter{iteration}_{digest}"

    bin_path = crash_dir / f"{stem}.bin"
    meta_path = crash_dir / f"{stem}.json"

    bin_path.write_bytes(payload)
    metadata = {
        "target": target_name,
        "seed": master_seed,
        "iteration": iteration,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_file": bin_path.name,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "reproduce_command": (
            f"python scripts/run_fuzz_smoke.py --seed {master_seed} "
            f"--replay-crash {bin_path.as_posix()}"
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return meta_path


def run_target(
    name: str,
    target_fn: Callable[[bytes], None],
    corpus_subpath: str,
    max_seconds: float,
    *,
    seed: int = 1337,
    crash_dir: Path = DEFAULT_CRASH_DIR,
) -> int:
    """Run bounded fuzzing on a single target with deterministic seed and crash persistence."""
    print(f"[*] Starting fuzz smoke on '{name}' (seed={seed}) for {max_seconds:.1f}s...")
    corpus_dir = CORPUS_DIR / corpus_subpath
    seeds = [p.read_bytes() for p in sorted(corpus_dir.glob("*")) if p.is_file()]
    if not seeds:
        print(f"[!] Warning: no seed files found in {corpus_dir}")
        seeds = [b""]

    # 1. Test baseline corpus seeds and any regression inputs
    for idx, corpus_payload in enumerate(seeds):
        try:
            target_fn(corpus_payload)
        except Exception as exc:
            meta_file = persist_failing_seed(
                name, seed, -idx - 1, corpus_payload, exc, crash_dir=crash_dir
            )
            print(f"[!] Persisted crashing corpus seed metadata to {meta_file}", file=sys.stderr)
            raise

    start_time = time.time()
    iterations = 0
    rng = random.Random(seed)

    while time.time() - start_time < max_seconds:
        base = rng.choice(seeds)
        mutated = mutate(base, rng)
        for _ in range(rng.randint(0, 4)):
            mutated = mutate(mutated, rng)
        try:
            target_fn(mutated)
        except Exception as exc:
            meta_file = persist_failing_seed(
                name, seed, iterations, mutated, exc, crash_dir=crash_dir
            )
            print(f"[!] Persisted crashing fuzz input metadata to {meta_file}", file=sys.stderr)
            raise
        iterations += 1

    elapsed = max(1e-6, time.time() - start_time)
    print(
        f"[+] '{name}' passed cleanly (seed={seed}): {iterations} mutations executed in {elapsed:.2f}s ({iterations / elapsed:.0f} exec/s)."
    )
    return iterations


def main() -> int:
    parser = argparse.ArgumentParser(description="EWM Engine Bounded Fuzz Smoke Runner")
    parser.add_argument(
        "--max-seconds", type=float, default=3.0, help="Max execution time per target in seconds"
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Deterministic PRNG seed for fuzz mutations"
    )
    parser.add_argument(
        "--crash-dir",
        type=Path,
        default=DEFAULT_CRASH_DIR,
        help="Directory where crashing inputs and reproduction metadata are persisted",
    )
    parser.add_argument(
        "--replay-crash",
        type=Path,
        default=None,
        help="Optional path to a persisted crash .bin file to replay across all targets",
    )
    args = parser.parse_args()

    if args.replay_crash is not None:
        payload = args.replay_crash.read_bytes()
        for t_name, t_fn in [
            ("YAML Loader", fuzz_yaml),
            ("JSON Loader", fuzz_json),
            ("WSL Compiler", fuzz_wsl),
        ]:
            print(f"[*] Replaying {args.replay_crash} against {t_name}...")
            t_fn(payload)
        print("[OK] Replay completed without crash.")
        return 0

    print(
        f"=== EWM Engine Fuzz Smoke Suite (Target duration: {args.max_seconds}s/target, seed={args.seed}) ==="
    )
    total_execs = 0
    try:
        total_execs += run_target(
            "YAML Loader",
            fuzz_yaml,
            "yaml",
            args.max_seconds,
            seed=args.seed,
            crash_dir=args.crash_dir,
        )
        total_execs += run_target(
            "JSON Loader",
            fuzz_json,
            "json",
            args.max_seconds,
            seed=args.seed + 1,
            crash_dir=args.crash_dir,
        )
        total_execs += run_target(
            "WSL Compiler",
            fuzz_wsl,
            "wsl",
            args.max_seconds,
            seed=args.seed + 2,
            crash_dir=args.crash_dir,
        )
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
