"""Main entrypoint for the ADR-053 Fraud Cascade Flagship Showcase (`examples/fraud_cascade/main.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:  # pragma: no cover
    sys.path.insert(0, _REPO_ROOT)

from examples.fraud_cascade.adr053_full_cascade import (
    ADR053FraudCascadeShowcaseResult,
    run_adr053_fraud_cascade_showcase,
)

__all__ = ["ADR053FraudCascadeShowcaseResult", "run_adr053_fraud_cascade_showcase"]

if __name__ == "__main__":  # pragma: no cover
    summary = run_adr053_fraud_cascade_showcase()
    print(f"Steps simulated (6-part ADR-053): {summary.steps_simulated}")
    print(f"Myopic do(tau=0.40) regret:       ${summary.myopic_regret_usd:,.2f}")
    print(f"Balanced EWM do(tau=0.55) regret: ${summary.balanced_regret_usd:,.2f}")
    print(f"Max cohort conservation drift:    {summary.cohort_conservation_max_drift:.3e}")
    print(f"Latent fraud masked in O_t:       {summary.latent_fraud_masked_in_o_t}")
