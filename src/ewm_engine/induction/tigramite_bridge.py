"""Tigramite GPL-3.0 isolation boundary for EWM Engine.

ARCHITECTURE INVARIANT (ADR-048):
    Tigramite is licensed under GPL-3.0. To protect the permissive Apache-2.0 / MIT
    licensing of the EWM Engine core, Tigramite is strictly quarantined behind an
    out-of-process subprocess boundary.

    EWM Engine core NEVER imports Tigramite directly in-process. All interactions
    are mediated via subprocess execution with standard JSON stdin/stdout serialization.
    When Tigramite is not installed or when running in the permissive default profile,
    EWM uses its native pure-Python / NumPy temporal causal discovery engine (DYNOTEARS / lag-MCI).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError


def is_tigramite_available() -> bool:
    """Check whether Tigramite is installed in the current Python environment without importing it."""
    try:
        res = subprocess.run(
            [sys.executable, "-c", "import tigramite; print('OK')"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return res.returncode == 0 and "OK" in res.stdout
    except Exception:
        return False


def run_tigramite_out_of_process(
    series_data: list[list[float]],
    var_names: list[str],
    max_lag: int = 1,
    alpha: float = 0.05,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Execute Tigramite PCMCI+ via an isolated subprocess worker.

    Args:
        series_data: Time-series observations matrix (rows = time steps, cols = variables).
        var_names: Variable names corresponding to columns.
        max_lag: Maximum temporal lag tau_max.
        alpha: Significance threshold for conditional independence tests.
        timeout_sec: Maximum execution time before subprocess termination.

    Returns:
        Structured dictionary containing discovered temporal causal links.
    """
    if not is_tigramite_available():
        raise SimulationConfigurationError(
            "Tigramite is not installed. To use the Tigramite engine, install `pip install tigramite` "
            "or use EWM's native temporal causal discovery engine (engine='native')."
        )

    worker_script = Path(__file__).parent / "tigramite_worker.py"
    input_payload = json.dumps(
        {
            "series_data": series_data,
            "var_names": var_names,
            "max_lag": max_lag,
            "alpha": alpha,
        }
    )

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
            f"Tigramite subprocess timed out after {timeout_sec}s."
        ) from exc

    if proc.returncode != 0:
        raise SimulationConfigurationError(
            f"Tigramite worker subprocess failed (code {proc.returncode}): {proc.stderr.strip()}"
        )

    try:
        output_data: dict[str, Any] = json.loads(proc.stdout)
        return output_data
    except json.JSONDecodeError as exc:
        raise SimulationConfigurationError(
            f"Failed to parse JSON output from Tigramite worker: {proc.stdout}"
        ) from exc
