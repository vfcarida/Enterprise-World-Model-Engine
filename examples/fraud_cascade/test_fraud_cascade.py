"""Tests for the ADR-053 Fraud Cascade Showcase (`examples/fraud_cascade/`)."""

from __future__ import annotations

from tests.unit.test_fraud_cascade_adr053 import (
    test_adr053_fraud_cascade_abstains_under_extreme_out_of_support_drift,
    test_adr053_fraud_cascade_six_part_state_and_regret,
)

__all__ = [
    "test_adr053_fraud_cascade_abstains_under_extreme_out_of_support_drift",
    "test_adr053_fraud_cascade_six_part_state_and_regret",
]
