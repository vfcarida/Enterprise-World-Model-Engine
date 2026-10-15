"""Decision-focused scenario discovery and multi-robustness evaluation package.

Anchored in:
    - Elmachtoub & Grigas (2022), Smart Predict-then-Optimize (DFL).
    - Lempert et al. / RAND, Robust Decision Making (RDM).
    - McPhail et al. (2018), Robustness metric taxonomy & disagreement analysis.
    - Friston (2010), Da Costa et al. (2024), Active Inference Expected Free Energy (EFE).
    - Kwakkel (2017), EMA Workbench scenario discovery (PRIM/CART).
"""

from __future__ import annotations

from ewm_engine.decision.discovery import (
    PeelingStep,
    ScenarioDiscoveryResult,
    VulnerabilityBox,
    is_ema_workbench_available,
    run_cart,
    run_prim,
)
from ewm_engine.decision.efe import ActiveInferenceScorer, ExpectedFreeEnergyScore
from ewm_engine.decision.engine import DecisionEngine
from ewm_engine.decision.report import DecisionReport
from ewm_engine.decision.robustness import (
    ActionRobustnessScore,
    DecisionComparisonResult,
    evaluate_decision_robustness,
)

__all__ = [
    "ActionRobustnessScore",
    "ActiveInferenceScorer",
    "DecisionComparisonResult",
    "DecisionEngine",
    "DecisionReport",
    "ExpectedFreeEnergyScore",
    "PeelingStep",
    "ScenarioDiscoveryResult",
    "VulnerabilityBox",
    "evaluate_decision_robustness",
    "is_ema_workbench_available",
    "run_cart",
    "run_prim",
]
