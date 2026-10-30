"""Unit tests for the Deployment-Shift Walkthrough and Fraud Cascade ADR-053 Integration."""

from __future__ import annotations

from examples.deployment_shift_walkthrough import run_deployment_shift_walkthrough
from examples.fraud_cascade.adr053_full_cascade import run_adr053_fraud_cascade_showcase


def test_deployment_shift_walkthrough_adr053_invariants() -> None:
    summary = run_deployment_shift_walkthrough(horizon=24, shift_step=10, audit_delay_steps=2)
    assert summary.pre_shift_mae == 0.0
    assert summary.peak_post_shift_mae > 20.0
    assert summary.recovered_post_shift_mae == 0.0
    assert summary.static_unrecovered_mae > 20.0
    assert summary.steps_until_recovery == 2
    assert summary.hidden_resource_masked_in_observation is True
    assert summary.ontic_fingerprint_invariant_to_epistemic is True


def test_fraud_cascade_adr053_six_part_state_integration() -> None:
    res = run_adr053_fraud_cascade_showcase(horizon=12)
    assert res.steps_simulated == 12
    assert res.myopic_regret_usd > 100000.0
    assert res.balanced_regret_usd == 0.0
    assert res.cohort_conservation_max_drift <= 1e-12
    assert res.latent_fraud_masked_in_o_t is True

    step0 = res.trajectory_balanced[0]
    assert "latent_true_fraud_attempts" in step0.ontic_state_s_t.resources
    assert "latent_true_fraud_attempts" not in step0.observation_o_t.resources
    assert "latent_true_fraud_attempts" in step0.belief_state_b_t.epistemic_uncertainty
    assert step0.context_k_t.version == "1.0.0"
    assert step0.action_a_t.parameters["tau"] == 0.55
    assert len(step0.epistemic_state_p_t.epistemic_fingerprint) == 64
