"""EU AI Act Annex IV Technical Documentation readiness renderer.

Regulation (EU) 2024/1689 (EU AI Act) mandates detailed technical documentation
for high-risk AI systems under Annex IV (applicable from August 2026).

DISCLAIMER:
    This module produces technical documentation readiness inputs reprojecting
    simulation provenance, model cards, and scenario evaluations into Annex IV
    sections. It is provided for engineering transparency and does NOT constitute
    legal advice or a formal declaration of regulatory conformity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ewm_engine.cards.models import ModelCard, ScenarioCard


DISCLAIMER_TEXT = (
    "This document provides technical documentation readiness inputs per EU AI Act Annex IV "
    "(Regulation (EU) 2024/1689). High-risk system obligations become enforceable from August 2026. "
    "This artifact reflects simulator configuration and execution data; it does not constitute a "
    "formal declaration of conformity or legal certification."
)


def render_model_annex_iv(card: ModelCard) -> dict[str, Any]:
    """Reproject a ModelCard into EU AI Act Annex IV technical documentation structure."""
    return {
        "regulation": "Regulation (EU) 2024/1689 (EU AI Act)",
        "annex": "Annex IV - Technical Documentation Referred to in Article 11(1)",
        "status": "Readiness Input (Non-Legal Representation)",
        "disclaimer": DISCLAIMER_TEXT,
        "section_1_general_description": {
            "title": "General description of the AI system",
            "system_identifier": card.model_id,
            "system_name": card.model_name,
            "version": card.version,
            "intended_purpose": card.intended_use,
            "out_of_scope_use_cases": card.out_of_scope,
            "developer_details": card.model_details.get(
                "developer", "Enterprise World Model Engine Maintainers"
            ),
            "interaction_with_hardware_or_software": card.model_details.get(
                "interactions", ["Python 3.11/3.12 runtime", "EWM Simulation Kernel"]
            ),
        },
        "section_2_system_elements_and_development": {
            "title": "Detailed description of system elements and development process",
            "design_specifications": card.description,
            "theoretical_assumptions": card.assumptions,
            "constraints_exercised": card.constraints_exercised,
            "training_data_specifications": card.training_data or card.provenance_sources,
            "validation_and_testing_data": card.eval_data,
            "evaluation_metrics": card.metrics,
            "quantitative_analyses": card.quantitative_analyses,
            "population_calibration_score": card.calibration_score,
        },
        "section_3_monitoring_functioning_and_control": {
            "title": "Detailed information about monitoring, functioning, and control",
            "capabilities_and_limitations": card.limitations,
            "operational_factors_and_subgroups": card.factors,
            "extrapolation_safeguards": (
                "Monitored via EWM Engine support boundary detection and Mahalanobis covariance gating."
            ),
            "human_oversight_measures": card.model_details.get(
                "human_oversight",
                "Simulation outputs require human decision-maker review; bare point estimates are blocked.",
            ),
        },
        "section_4_risk_management": {
            "title": "Risk management system & foreseeable risks",
            "ethical_considerations": card.ethical_considerations,
            "caveats_and_recommendations": card.caveats_and_recommendations,
            "safety_invariants": [
                "Strict state immutability (AC-005)",
                "Physical & accounting conservation law gates (AC-006, AC-008)",
                "Anti-auto-promotion of causal evidence levels (SPEC 9 / R12)",
            ],
        },
        "section_5_lifecycle_and_provenance": {
            "title": "Changes throughout the lifecycle & cryptographic provenance",
            "artifact_fingerprint": card.artifact_fingerprint,
            "schema_version": card.schema_version,
            "maturity": card.maturity,
        },
    }


def render_scenario_annex_iv(card: ScenarioCard) -> dict[str, Any]:
    """Reproject a ScenarioCard into EU AI Act Annex IV evaluation and test evidence."""
    return {
        "regulation": "Regulation (EU) 2024/1689 (EU AI Act)",
        "annex": "Annex IV - Verification and Testing Documentation",
        "status": "Readiness Input (Non-Legal Representation)",
        "disclaimer": DISCLAIMER_TEXT,
        "section_1_test_scope": {
            "scenario_identifier": card.scenario_id,
            "description": card.description,
            "horizon_steps": card.horizon,
            "monte_carlo_samples": card.samples,
            "master_seed": card.seed,
            "applied_interventions": card.interventions,
        },
        "section_2_assumptions_and_boundaries": {
            "assumptions": card.assumptions,
            "out_of_scope_conditions": card.out_of_scope,
            "environmental_context": card.environmental_context,
            "limitations": card.limitations,
        },
        "section_3_test_results_and_uncertainty": {
            "summary_metrics": card.metrics,
            "quantitative_analyses": card.quantitative_analyses,
        },
        "section_4_provenance": {
            "artifact_fingerprint": card.artifact_fingerprint,
            "schema_version": card.schema_version,
            "maturity": card.maturity,
        },
    }
