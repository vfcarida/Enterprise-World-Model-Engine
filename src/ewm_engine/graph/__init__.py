"""Continuous-Time Temporal Graph Neural Network (CT-TGNN) package.

Exposes the in-memory ContinuousGraphRegistry, domain node feature schemas,
EvolveGCN recurrent graph convolution modules, and temporal training pipeline.
"""

from __future__ import annotations

from ewm_engine.graph.aiops_features import AIOpsNodeFeatures
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node
from ewm_engine.graph.financial_features import FinancialNodeFeatures

try:
    from ewm_engine.graph.evolvegcn import EvolveGCN, EvolveGCNLayer, GCNConv
    from ewm_engine.graph.training_pipeline import EvolveGCNTrainer
except ImportError:  # pragma: no cover - optional [ml] extra (torch)
    EvolveGCN = None  # type: ignore[assignment,misc]
    EvolveGCNLayer = None  # type: ignore[assignment,misc]
    GCNConv = None  # type: ignore[assignment,misc]
    EvolveGCNTrainer = None  # type: ignore[assignment,misc]

__all__ = [
    "AIOpsNodeFeatures",
    "ContinuousGraphRegistry",
    "Edge",
    "EvolveGCN",
    "EvolveGCNLayer",
    "EvolveGCNTrainer",
    "FinancialNodeFeatures",
    "GCNConv",
    "Node",
]
