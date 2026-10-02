"""Causal and epistemic foundations for Enterprise World Model simulations."""

from __future__ import annotations

from enum import StrEnum


class EvidenceLevel(StrEnum):
    """Epistemic classification of dynamics, relationships, and simulation traces.

    EWM Engine distinguishes observational correlations from interventional mechanics:
    - STRUCTURAL: Grounded in deterministic physical laws, accounting conservation, or hard rules.
    - INTERVENTIONAL: Empirically validated under controlled intervention / do-calculus.
    - QUASI_CAUSAL: Derived from econometric/quasi-experimental designs (e.g., diff-in-diff, IV).
    - PREDICTIVE: Observational statistical or machine-learned association P(Y | X).
    - ASSUMED: Working hypothesis or heuristic assumption not yet validated.
    """

    STRUCTURAL = "structural"
    INTERVENTIONAL = "interventional"
    QUASI_CAUSAL = "quasi_causal"
    PREDICTIVE = "predictive"
    ASSUMED = "assumed"
