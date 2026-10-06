# Durability & Event Sourcing API Reference

This module implements event-sourced durability (Track T1) and cryptographic result store memoization (Track T2).

---

## Event Stores & Transition Logs

::: ewm_engine.durability.protocol
    options:
      show_root_heading: true
      show_source: false
      members:
        - EventStore
        - ResultStore

::: ewm_engine.durability.events
    options:
      show_root_heading: true
      show_source: false
      members:
        - TraceLog
        - TransitionEvent

---

## Storage Backends

::: ewm_engine.durability.backends
    options:
      show_root_heading: true
      show_source: false
      members:
        - SqliteEventStore
        - FilesystemResultStore
        - InMemoryEventStore
        - InMemoryResultStore

---

## Replay & Memoization

::: ewm_engine.durability.replay
    options:
      show_root_heading: true
      show_source: false
      members:
        - replay_trajectory
        - fold_events
        - verify_trajectory_replay

::: ewm_engine.durability.memoize
    options:
      show_root_heading: true
      show_source: false
      members:
        - compute_simulation_fingerprint
        - run_memoized
