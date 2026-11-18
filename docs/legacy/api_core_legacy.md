# Core Primitives & World-Model Context API Reference

This module defines the fundamental data structures, entities, ontic/epistemic state models, runtime context snapshots ($K_t$), partial observability filters ($O_t, B_t$), and Responsible AI governance gates of the Enterprise World Model Engine.

::: ewm_engine.core.state
    options:
      show_root_heading: true
      show_source: false
      members:
        - WorldState

::: ewm_engine.core.world
    options:
      show_root_heading: true
      show_source: false
      members:
        - World

::: ewm_engine.core.context
    options:
      show_root_heading: true
      show_source: false
      members:
        - ContextSnapshot
        - ObservationSpec
        - ObservationFilter
        - ObservationPacket
        - BeliefState
        - BeliefUpdater
        - ExponentialSmoothingBeliefUpdater
        - EpistemicState
        - MechanismKind
        - MechanismProvenance
        - InfeasibleResidualPolicy
        - DriftKind
        - DriftDiagnostic
        - ContextDiscoveryProvider
        - StaticContextDiscovery
        - AuditMetadataContextDiscovery

::: ewm_engine.core.trust
    options:
      show_root_heading: true
      show_source: false
      members:
        - ResponsibleAIGate
        - ScenarioForecastResult
        - validate_causal_claim

::: ewm_engine.core.entities
    options:
      show_root_heading: true
      show_source: false
      members:
        - Entity
        - Relationship

::: ewm_engine.core.resources
    options:
      show_root_heading: true
      show_source: false
      members:
        - Resource
