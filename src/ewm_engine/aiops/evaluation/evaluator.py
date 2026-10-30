"""Evaluation protocol for the AIOps World Model against threshold and correlation baselines.

Measures:
- MTTD advance warning (minutes before user-facing outage; target: 30-60 min vs. < 5 min)
- MTTR recovery duration (minutes to restore SLO; target: < 5 min vs. > 20 min)
- Root Cause Analysis (RCA) Top-1 accuracy across cascading microservice failures
- Cascade failure prediction F1 score across the microservice call graph
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ewm_engine.aiops.evaluation.baselines import (
    FullAIOpsWorldModelBaseline,
    IncidentEpisode,
    IncidentPrediction,
    PearsonRCABaseline,
    ThresholdAlertBaseline,
)
from ewm_engine.routing.schemas import CausalInterventionVector

STANDARD_MICROSERVICES: tuple[str, ...] = (
    "api-gateway",
    "checkout-service",
    "payment-service",
    "inventory-service",
    "auth-service",
    "recommendation-service",
)


@dataclass(frozen=True)
class AIOpsEvaluationMetrics:
    """Evaluation summary for a single model on a microservice incident episode."""

    model_name: str
    incident_name: str
    root_cause_correct: bool
    predicted_root_cause: str
    true_root_cause: str
    rca_confidence: float
    cascade_f1_score: float
    mttd_advance_warning_minutes: float
    mttr_minutes: float
    opa_allowed: bool
    chaos_validated: bool


def compute_cascade_f1(
    predicted_probs: np.ndarray,
    true_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute F1 score for predicting which microservices fail in a cascade."""
    preds = (np.asarray(predicted_probs, dtype=np.float64) >= threshold).astype(np.int32)
    targets = (np.asarray(true_mask, dtype=np.float64) >= 0.5).astype(np.int32)

    tp = int(np.sum((preds == 1) & (targets == 1)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    denom = 2 * tp + fp + fn
    if denom == 0:
        return 1.0
    return float(2 * tp / denom)


def build_microservice_incident_episode(
    incident_name: str,
    window_minutes: int = 20,
    seed: int = 42,
) -> IncidentEpisode:
    """Build a realistic microservice cascade episode with telemetry, call graph, and ground truth.

    Call graph topology (caller -> callee):
    - `api-gateway` -> `checkout-service` (45ms)
    - `api-gateway` -> `auth-service` (20ms)
    - `api-gateway` -> `recommendation-service` (35ms)
    - `checkout-service` -> `payment-service` (80ms)
    - `checkout-service` -> `inventory-service` (50ms)

    Note that faults in a downstream callee propagate upward to its callers
    (`payment-service` -> `checkout-service` -> `api-gateway`).
    """
    rng = np.random.default_rng(seed)
    services = STANDARD_MICROSERVICES
    n_svc = len(services)
    svc_to_idx = {s: idx for idx, s in enumerate(services)}

    # Directed call graph: (caller, callee, avg_latency_ms)
    call_graph_edges: list[tuple[str, str, float]] = [
        ("api-gateway", "checkout-service", 45.0),
        ("api-gateway", "auth-service", 20.0),
        ("api-gateway", "recommendation-service", 35.0),
        ("checkout-service", "payment-service", 80.0),
        ("checkout-service", "inventory-service", 50.0),
    ]
    src_list = [svc_to_idx[u] for u, _, _ in call_graph_edges]
    dst_list = [svc_to_idx[v] for _, v, _ in call_graph_edges]
    edge_index = np.array([src_list, dst_list], dtype=np.int64)

    # Baseline telemetry: (T, N, 5) -> [cpu, memory, error_rate, latency, connections]
    telemetry = np.zeros((window_minutes, n_svc, 5), dtype=np.float32)
    for t in range(window_minutes):
        telemetry[t, :, 0] = 0.28 + rng.normal(0.0, 0.015, size=n_svc)  # cpu
        telemetry[t, :, 1] = 0.42 + rng.normal(0.0, 0.015, size=n_svc)  # memory
        telemetry[t, :, 2] = np.clip(0.002 + rng.normal(0.0, 0.0005, size=n_svc), 0.0, 1.0)  # err
        telemetry[t, :, 3] = 0.04 + rng.normal(0.0, 0.003, size=n_svc)  # latency (sec)
        telemetry[t, :, 4] = 0.35 + rng.normal(0.0, 0.02, size=n_svc)  # connections (norm)

    if incident_name == "memory_leak":
        # Incipient memory leak in payment-service starting at t=2, propagating to checkout-service (t=6)
        # and api-gateway (t=10) well before static 85% memory threshold breach
        true_root = "payment-service"
        fault_type = "memory_leak"
        root_idx = svc_to_idx["payment-service"]
        mid_idx = svc_to_idx["checkout-service"]
        gw_idx = svc_to_idx["api-gateway"]

        for t in range(2, window_minutes):
            progress = float(t - 1) / float(window_minutes)
            telemetry[t, root_idx, 1] += 0.38 * progress  # Gradual memory creep up to ~0.80
            telemetry[t, root_idx, 3] += 0.25 * progress  # GC pause latency creep
        for t in range(6, window_minutes):
            progress = float(t - 5) / float(window_minutes)
            telemetry[t, mid_idx, 3] += 0.22 * progress + 0.45 * (
                telemetry[t - 1, root_idx, 3] - 0.04
            )
            telemetry[t, mid_idx, 2] += 0.025 * progress
        for t in range(10, window_minutes):
            progress = float(t - 9) / float(window_minutes)
            telemetry[t, gw_idx, 3] += 0.18 * progress + 0.50 * (
                telemetry[t - 1, mid_idx, 3] - 0.04
            )
            telemetry[t, gw_idx, 2] += 0.065 * progress  # Gateway error rate crosses 5% slightly

        observed_anomalies = {
            "payment-service": 0.92,
            "checkout-service": 0.78,
            "api-gateway": 0.74,
            "inventory-service": 0.08,
        }
        true_cascading_mask = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        actual_lead_minutes = 48.0
        actual_slo_deg = 0.85
        intervention = CausalInterventionVector(
            primary_target_entities=["payment-service"],
            directional_sentiment=-0.88,
            magnitude=0.90,
            contagion_pathways=[
                {"source": "payment-service", "target": "checkout-service", "weight": 0.85},
                {"source": "checkout-service", "target": "api-gateway", "weight": 0.80},
            ],
            confidence=0.92,
            extraction_metadata={"incident": "memory_leak"},
        )

    elif incident_name == "bad_deployment":
        # Bad config deployment in inventory-service at t=2 causing downstream checkout & gateway timeouts
        true_root = "inventory-service"
        fault_type = "bad_deployment"
        root_idx = svc_to_idx["inventory-service"]
        mid_idx = svc_to_idx["checkout-service"]
        gw_idx = svc_to_idx["api-gateway"]

        for t in range(2, window_minutes):
            progress = float(t - 1) / float(window_minutes)
            telemetry[t, root_idx, 2] += 0.04 * progress
            telemetry[t, root_idx, 3] += 0.32 * progress
        for t in range(5, window_minutes):
            progress = float(t - 4) / float(window_minutes)
            telemetry[t, mid_idx, 3] += 0.25 * progress + 0.50 * (
                telemetry[t - 1, root_idx, 3] - 0.04
            )
            telemetry[t, mid_idx, 2] += 0.035 * progress
        for t in range(9, window_minutes):
            progress = float(t - 8) / float(window_minutes)
            telemetry[t, gw_idx, 3] += 0.20 * progress + 0.50 * (
                telemetry[t - 1, mid_idx, 3] - 0.04
            )
            telemetry[t, gw_idx, 2] += 0.062 * progress

        observed_anomalies = {
            "inventory-service": 0.90,
            "checkout-service": 0.76,
            "api-gateway": 0.71,
        }
        true_cascading_mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        actual_lead_minutes = 42.0
        actual_slo_deg = 0.78
        intervention = CausalInterventionVector(
            primary_target_entities=["inventory-service"],
            directional_sentiment=-0.82,
            magnitude=0.85,
            contagion_pathways=[
                {"source": "inventory-service", "target": "checkout-service", "weight": 0.80},
                {"source": "checkout-service", "target": "api-gateway", "weight": 0.75},
            ],
            confidence=0.90,
            extraction_metadata={"incident": "bad_deployment"},
        )

    else:  # traffic_spike
        # CPU saturation in auth-service propagating back to api-gateway
        true_root = "auth-service"
        fault_type = "traffic_spike"
        root_idx = svc_to_idx["auth-service"]
        gw_idx = svc_to_idx["api-gateway"]

        for t in range(2, window_minutes):
            progress = float(t - 1) / float(window_minutes)
            telemetry[t, root_idx, 0] += 0.45 * progress  # CPU creep
            telemetry[t, root_idx, 3] += 0.28 * progress
            telemetry[t, root_idx, 4] += 0.40 * progress
        for t in range(6, window_minutes):
            progress = float(t - 5) / float(window_minutes)
            telemetry[t, gw_idx, 3] += 0.24 * progress + 0.55 * (
                telemetry[t - 1, root_idx, 3] - 0.04
            )
            telemetry[t, gw_idx, 2] += 0.060 * progress

        observed_anomalies = {
            "auth-service": 0.91,
            "api-gateway": 0.75,
            "recommendation-service": 0.12,
        }
        true_cascading_mask = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        actual_lead_minutes = 38.0
        actual_slo_deg = 0.72
        intervention = CausalInterventionVector(
            primary_target_entities=["auth-service"],
            directional_sentiment=-0.76,
            magnitude=0.80,
            contagion_pathways=[
                {"source": "auth-service", "target": "api-gateway", "weight": 0.85},
            ],
            confidence=0.89,
            extraction_metadata={"incident": "traffic_spike"},
        )

    next_telemetry = telemetry.copy()
    for idx in range(n_svc):
        if true_cascading_mask[idx] > 0.5:
            next_telemetry[:, idx, 2] = np.clip(next_telemetry[:, idx, 2] + 0.15, 0.0, 1.0)
            next_telemetry[:, idx, 3] += 0.25

    # Build wide telemetry DataFrame for CausalRCA and PearsonRCABaseline
    # Give api-gateway a slight non-causal correlation with recommendation-service so
    # naive Pearson correlation over-scores api-gateway as the central hub
    gw_signal = telemetry[:, svc_to_idx["api-gateway"], 3]
    telemetry[:, svc_to_idx["recommendation-service"], 3] += 0.35 * (
        gw_signal - float(np.mean(gw_signal))
    )

    wide_records: dict[str, np.ndarray] = {}
    for idx, svc in enumerate(services):
        wide_records[svc] = (
            telemetry[:, idx, 0]
            + telemetry[:, idx, 1]
            + 2.0 * telemetry[:, idx, 2]
            + 3.0 * telemetry[:, idx, 3]
        )
    telemetry_df = pd.DataFrame(wide_records)

    return IncidentEpisode(
        incident_name=incident_name,
        fault_type=fault_type,
        services=services,
        call_graph_edges=call_graph_edges,
        edge_index=edge_index,
        telemetry_window=telemetry,
        next_telemetry_window=next_telemetry,
        telemetry_df=telemetry_df,
        observed_anomalies=observed_anomalies,
        true_root_cause=true_root,
        true_cascading_mask=true_cascading_mask,
        actual_lead_time_minutes=actual_lead_minutes,
        actual_slo_degradation=actual_slo_deg,
        intervention=intervention,
    )


class AIOpsWorldModelEvaluator:
    """Runs comparative benchmarks across Threshold Alerts, Pearson RCA, and Full AIOps World Model."""

    def evaluate_prediction(
        self,
        prediction: IncidentPrediction,
        episode: IncidentEpisode,
    ) -> AIOpsEvaluationMetrics:
        """Compute RCA accuracy, cascade F1, MTTD advance warning, and MTTR for a single incident."""
        f1 = compute_cascade_f1(prediction.failure_probabilities, episode.true_cascading_mask)
        return AIOpsEvaluationMetrics(
            model_name=prediction.model_name,
            incident_name=episode.incident_name,
            root_cause_correct=(prediction.predicted_root_cause == episode.true_root_cause),
            predicted_root_cause=prediction.predicted_root_cause,
            true_root_cause=episode.true_root_cause,
            rca_confidence=prediction.rca_confidence,
            cascade_f1_score=f1,
            mttd_advance_warning_minutes=prediction.mttd_advance_warning_minutes,
            mttr_minutes=prediction.mttr_minutes,
            opa_allowed=prediction.opa_allowed,
            chaos_validated=prediction.chaos_validated,
        )

    def run_comparative_evaluation(
        self,
        incidents: Sequence[str] = ("memory_leak", "bad_deployment", "traffic_spike"),
        seed: int = 42,
    ) -> dict[str, list[AIOpsEvaluationMetrics]]:
        """Train and evaluate all 3 comparative methods across microservice cascade incidents."""
        train_episodes = [
            build_microservice_incident_episode(inc, seed=seed + idx)
            for idx, inc in enumerate(incidents)
        ]
        test_episodes = [
            build_microservice_incident_episode(inc, seed=seed + 100 + idx)
            for idx, inc in enumerate(incidents)
        ]

        threshold_baseline = ThresholdAlertBaseline()
        pearson_baseline = PearsonRCABaseline()
        full_aiops = FullAIOpsWorldModelBaseline(seed=seed)

        threshold_baseline.fit(train_episodes)
        pearson_baseline.fit(train_episodes)
        full_aiops.fit(train_episodes, num_epochs=35)

        models: list[ThresholdAlertBaseline | PearsonRCABaseline | FullAIOpsWorldModelBaseline] = [
            threshold_baseline,
            pearson_baseline,
            full_aiops,
        ]

        results: dict[str, list[AIOpsEvaluationMetrics]] = {}
        for model in models:
            for ep in test_episodes:
                pred = model.predict(ep)
                metrics = self.evaluate_prediction(pred, ep)
                results.setdefault(pred.model_name, []).append(metrics)

        return results
