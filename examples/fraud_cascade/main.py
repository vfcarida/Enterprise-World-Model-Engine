"""Main entrypoint for the ADR-053 Fraud Cascade Flagship Showcase."""

from __future__ import annotations

from examples.fraud_cascade.adr053_full_cascade import (
    ADR053FraudCascadeSummary,
    run_adr053_fraud_cascade,
)

__all__ = ["ADR053FraudCascadeSummary", "run_adr053_fraud_cascade"]

if __name__ == "__main__":  # pragma: no cover
    summary = run_adr053_fraud_cascade()
    print(f"Steps simulated (6-part ADR-053): {summary.steps_simulated}")
    print(f"Myopic do(tau=0.40) regret:       ${summary.myopic_regret_usd:,.2f}")
    print(f"Balanced EWM do(tau=0.55) regret: ${summary.balanced_regret_usd:,.2f}")
    print(f"Max cohort conservation drift:    {summary.max_conservation_drift:.3e}")
    print(f"Latent fraud masked in O_t:       {summary.latent_fraud_masked_in_observation}")
