"""PM4Py AGPL-3.0 isolation boundary for EWM Engine.

ARCHITECTURE INVARIANT (ADR-047):
    PM4Py is licensed under AGPL-3.0. To protect the permissive Apache-2.0 / MIT
    licensing of the EWM Engine core, PM4Py is strictly quarantined behind an
    out-of-process subprocess boundary.

    EWM Engine core NEVER imports PM4Py directly in-process. All interactions
    are mediated via subprocess execution with standard JSON stdin/stdout serialization.
    When PM4Py is not installed, EWM uses its native pure-Python Inductive Miner.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError


def is_pm4py_available() -> bool:
    """Check whether PM4Py is installed in the current Python environment without importing it."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import pm4py; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def run_pm4py_inductive_miner_out_of_process(
    traces: list[list[str]],
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Execute Inductive Miner via an isolated subprocess worker.

    Args:
        traces: List of event name sequences (one sequence per object/case).
        timeout_sec: Maximum execution time before subprocess termination.

    Returns:
        Structured dictionary containing transitions, DFG edges, fitness, and precision.
    """
    if not is_pm4py_available():
        raise SimulationConfigurationError(
            "PM4Py is not installed. To use the PM4Py engine, install `pip install pm4py` "
            "or use EWM's native pure-Python inductive miner (engine='native')."
        )

    worker_script = Path(__file__).parent / "pm4py_worker.py"
    input_payload = json.dumps({"traces": traces})

    try:
        proc = subprocess.run(
            [sys.executable, str(worker_script)],
            input=input_payload,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise SimulationConfigurationError(
            f"PM4Py subprocess timed out after {timeout_sec}s."
        ) from exc

    if proc.returncode != 0:
        raise SimulationConfigurationError(
            f"PM4Py worker subprocess failed (code {proc.returncode}): {proc.stderr.strip()}"
        )

    try:
        output_data: dict[str, Any] = json.loads(proc.stdout)
        return output_data
    except Exception as exc:
        raise SimulationConfigurationError(
            f"Failed to parse PM4Py worker output: {proc.stdout[:200]}"
        ) from exc
