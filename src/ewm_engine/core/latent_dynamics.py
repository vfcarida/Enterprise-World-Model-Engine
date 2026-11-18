"""Deprecated latent state rollout predictors isolated for Continual Discovery Agent (CDA) migration.

Architectural Shift: From Latent Rollout Prediction to Explicit Continual Discovery (E3)
----------------------------------------------------------------------------------------
Historically, world-model simulators relied on static neural latent-space predictors of the form:
    z_{t+1} = f_\\theta(z_t, a_t)
where $z_t \\in \\mathbb{R}^d$ is a compressed latent embedding and $\\theta$ represents frozen,
offline-trained neural network weights.

In enterprise socio-technical systems (as demonstrated by CascadeBench), static latent predictors
suffer catastrophic distribution shift whenever operational business rules, approval thresholds,
routing topologies, or regulatory policies ($c_t$) are modified at runtime. Because $\\theta$
encodes historical correlations from a stale policy regime $c_0$, latent rollouts silently
hallucinate invalid state trajectories under $c_t \\neq c_0$.

To guarantee zero-shot adaptation to runtime configuration shifts without retraining, EWM Engine
isolates and deprecates all weight-based latent rollout predictors in this module and replaces
forward prediction with the Continual Discovery Agent (CDA) retrieve-then-reason pipeline:
    s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)
implemented by ``RetrieveAndReasonEngine`` in ``ewm_engine.core.dynamics`` / ``ewm_engine.dynamics.cda``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from ewm_engine.core.deprecation import deprecated
from ewm_engine.core.types import DynamicsInterface, RandomGenerator

_CDA_REDIRECT_MESSAGE = (
    "Static neural weight-based latent state predictors (z_{t+1} = f_theta(z_t, a_t)) are "
    "deprecated and disabled because offline latent priors collapse under runtime configuration "
    "shifts. Redirecting to the Continual Discovery Agent (CDA) pipeline: use "
    "`RetrieveAndReasonEngine` (s_{t+1} ~ P(s_{t+1} | s_t, a_t, c)) in `ewm_engine.dynamics.cda` "
    "or `ewm_engine.core.dynamics`."
)


class LatentStatePredictor(DynamicsInterface):
    """Legacy neural weight-based latent state predictor ($z_{t+1} = f_\\theta(z_t, a_t)$).

    .. deprecated:: 1.5.0
       Static latent rollout predictors are isolated and disabled in favor of explicit
       Continual Discovery Agent (CDA) retrieve-then-reason dynamics
       (``RetrieveAndReasonEngine``), which condition transitions directly on retrieved
       runtime business rules $c$ via $s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)$.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        action_dim: int = 64,
        model_weights_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.latent_dim: int = latent_dim
        self.action_dim: int = action_dim
        self.model_weights_path: Optional[str] = model_weights_path
        self.metadata: Dict[str, Any] = dict(metadata or {})

    @deprecated(
        since="1.5.0",
        removed_in="2.0.0",
        alternative="Use RetrieveAndReasonEngine.predict_diff(state, action) in the CDA pipeline.",
    )
    def predict_latent(self, z_t: Any, a_t: Any) -> Any:
        """Legacy $z_{t+1}$ forward step; raises NotImplementedError redirecting to CDA."""
        raise NotImplementedError(_CDA_REDIRECT_MESSAGE)

    @deprecated(
        since="1.5.0",
        removed_in="2.0.0",
        alternative="Use RetrieveAndReasonEngine in the CDA pipeline.",
    )
    def forward(self, z_t: Any, a_t: Any = None) -> Any:
        """Legacy neural forward pass; raises NotImplementedError redirecting to CDA."""
        raise NotImplementedError(_CDA_REDIRECT_MESSAGE)

    @deprecated(
        since="1.5.0",
        removed_in="2.0.0",
        alternative="Use RetrieveAndReasonEngine in the CDA pipeline.",
    )
    def rollout(self, z_0: Any, actions: Sequence[Any]) -> List[Any]:
        """Legacy multi-step latent rollout; raises NotImplementedError redirecting to CDA."""
        raise NotImplementedError(_CDA_REDIRECT_MESSAGE)

    @deprecated(
        since="1.5.0",
        removed_in="2.0.0",
        alternative="Use RetrieveAndReasonEngine.transition(...) in the CDA pipeline.",
    )
    def transition(
        self,
        state: Any,
        actions: Sequence[Any],
        exogenous_events: Sequence[Any],
        rng: RandomGenerator,
    ) -> Any:
        """Legacy DynamicsInterface entrypoint; raises NotImplementedError redirecting to CDA."""
        raise NotImplementedError(_CDA_REDIRECT_MESSAGE)


class LatentRolloutPredictor(LatentStatePredictor):
    """Deprecated alias for multi-step latent embedding rollout predictors ($z_{t+1}$ loops).

    Isolated to preserve backwards-compatible imports while enforcing migration to
    ``RetrieveAndReasonEngine`` ($s_{t+1} \\sim P(s_{t+1} \\mid s_t, a_t, c)$).
    """


class NeuralLatentDynamics(LatentStatePredictor):
    """Deprecated neural latent dynamics wrapper redirecting callers to the CDA pipeline."""


__all__ = [
    "LatentRolloutPredictor",
    "LatentStatePredictor",
    "NeuralLatentDynamics",
]
