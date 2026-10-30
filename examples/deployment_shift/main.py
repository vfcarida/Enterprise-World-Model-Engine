"""Executable entrypoint for the ADR-053 Deployment-Shift Walkthrough (`examples/deployment_shift/main.py`)."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, _REPO_ROOT)

from examples.deployment_shift_walkthrough import (  # noqa: E402
    DeploymentShiftWalkthroughSummary,
    run_deployment_shift_walkthrough,
)

__all__ = ["DeploymentShiftWalkthroughSummary", "run_deployment_shift_walkthrough"]

if __name__ == "__main__":  # pragma: no cover
    summary = run_deployment_shift_walkthrough()
    print(f"Pre-shift MAE:              {summary.pre_shift_mae:.4f}")
    print(f"Peak post-shift MAE:        {summary.peak_post_shift_mae:.4f}")
    print(f"Recovered post-shift MAE:   {summary.recovered_post_shift_mae:.4f}")
    print(f"Static unrecovered MAE:     {summary.static_unrecovered_mae:.4f}")
    print(f"Steps until recovery:       {summary.steps_until_recovery}")
    print(f"Hidden resource masked:     {summary.hidden_resource_masked_in_observation}")
    print(f"Ontic fingerprint invariant:{summary.ontic_fingerprint_invariant_to_epistemic}")
