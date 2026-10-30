"""AIOps World Model vertical for zero-downtime resilience engineering.

Provides:
- Telemetry ingestion (`PrometheusCollector`, `JaegerCollector`, `LogParser`)
- Spatial-Temporal Graph Convolutional Network (`STGCNEncoder`, `CascadeFailurePredictor`, `AIOpsWorldModel`)
- Root Cause Analysis via PC algorithm & Pearl's do-calculus (`CausalRCA`)
- Kubernetes remediation synthesis & OPA policy validation (`RemediationGenerator`, `OPAPolicyValidator`)
- Physical fault injection & recovery validation (`ChaosMeshHarness`)
- Comparative benchmarks against static threshold alerts and Pearson correlation (`AIOpsWorldModelEvaluator`)
"""

from ewm_engine.aiops.chaos import ChaosMeshHarness
from ewm_engine.aiops.data import (
    STANDARD_SRE_METRICS,
    JaegerCollector,
    LogParser,
    PrometheusCollector,
)
from ewm_engine.aiops.evaluation import (
    STANDARD_MICROSERVICES,
    AIOpsEvaluationMetrics,
    AIOpsWorldModelEvaluator,
    FullAIOpsWorldModelBaseline,
    IncidentEpisode,
    IncidentPrediction,
    PearsonRCABaseline,
    ThresholdAlertBaseline,
    build_microservice_incident_episode,
    compute_cascade_f1,
)
from ewm_engine.aiops.models import (
    AIOpsWorldModel,
    CascadeFailurePredictor,
    CascadePredictionOutput,
    STGCNEncoder,
)
from ewm_engine.aiops.rca import CausalRCA, CausalRCAResult
from ewm_engine.aiops.remediation import (
    OPAPolicyEvaluation,
    OPAPolicyValidator,
    RemediationGenerator,
)

__all__ = [
    "STANDARD_MICROSERVICES",
    "STANDARD_SRE_METRICS",
    "AIOpsEvaluationMetrics",
    "AIOpsWorldModel",
    "AIOpsWorldModelEvaluator",
    "CascadeFailurePredictor",
    "CascadePredictionOutput",
    "CausalRCA",
    "CausalRCAResult",
    "ChaosMeshHarness",
    "FullAIOpsWorldModelBaseline",
    "IncidentEpisode",
    "IncidentPrediction",
    "JaegerCollector",
    "LogParser",
    "OPAPolicyEvaluation",
    "OPAPolicyValidator",
    "PearsonRCABaseline",
    "PrometheusCollector",
    "RemediationGenerator",
    "STGCNEncoder",
    "ThresholdAlertBaseline",
    "build_microservice_incident_episode",
    "compute_cascade_f1",
]
