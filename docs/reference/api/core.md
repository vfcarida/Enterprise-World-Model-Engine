# Core Primitives, OCEL 2.0 State, CDA Dynamics & DoWhy-GCM Causal API Reference

This module defines the fundamental data structures, OCEL 2.0 relational state encoders (`StateEncoder`), Continual Discovery Agent retrieve-then-reason dynamics (`RetrieveAndReasonEngine`), DoWhy-GCM structural causal models (`EngineCausalInterface`), ontic/epistemic state models, runtime context snapshots ($K_t$), and Responsible AI governance gates of the Enterprise World Model Engine.

For full executable examples of `StateEncoder`, `RetrieveAndReasonEngine`, and `EngineCausalInterface`, see **[OCEL 2.0, CDA & Causal API Reference](ocel_cda_causal.md)**.

::: ewm_engine.core.ocel
    options:
      show_root_heading: true
      show_source: false
      members:
        - StateEncoder
        - OcelEventRecord
        - OcelObjectTarget
        - OcelObjectRelation
        - read_ocel2_sqlite
        - validate_ocel2_sqlite
        - export_to_ocel_json
        - validate_ocel2_json
        - ingest_csv_event_stream
        - ingest_xes_event_log

::: ewm_engine.core.dynamics
    options:
      show_root_heading: true
      show_source: false
      members:
        - RetrieveAndReasonEngine
        - ConfigurationDatastore
        - BusinessRule
        - CDAPromptHarness
        - StateDiff
        - RuleAttribution
        - parse_cda_configuration_json
        - parse_sandboxed_state_diff

::: ewm_engine.core.causal
    options:
      show_root_heading: true
      show_source: false
      members:
        - EngineCausalInterface
        - validate_strict_dag
        - load_ocel_historical_dataframe
        - CausalDAGValidationError
        - CausalMechanismError

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
