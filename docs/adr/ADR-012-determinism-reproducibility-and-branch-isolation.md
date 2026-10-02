# ADR-012: Determinism Invariant, Monte Carlo RNG Derivation & Branch Isolation

## Status
Accepted

## Context
A defining requirement for an Enterprise World Model Engine is rigorous, auditable reproducibility.
Unlike typical agent frameworks or ad-hoc simulation scripts, enterprise risk evaluation and counterfactual comparisons must guarantee that running an experiment with the same inputs, seed, and component versions produces the exact same logical trajectory across runs and machines.
Furthermore:
1. Monte Carlo rollouts must sample independent, statistically sound pseudo-random streams without overlapping RNG sequences.
2. Global RNG state (`random.seed`, `numpy.random.seed`) must never introduce hidden coupling or accidental side effects across threads or libraries.
3. Counterfactual branching (`Scenario.branch`, `branch_world`) must enforce strict data isolation so that alternative branches never contaminate parent snapshots or peer executions.

## Decision
1. **The Reproducibility Invariant (AC-004)**:
   We codify the formal contract:
   $$\text{WorldState}_0 + \text{Scenario} + \text{ComponentVersions} + \text{Seed} \implies \text{Trajectory}_{1:H}$$
   The engine guarantees bitwise identical logical trajectories for all built-in models and standard constraints.
   *Caveat*: External ML/GPU adapters (e.g. PyTorch, ONNX, external solvers) carry an explicit caveat that floating-point non-determinism across GPU kernels and hardware architectures is outside this bitwise guarantee.
2. **Explicit Per-Rollout RNG Streams (AC-009)**:
   - The engine spawns independent child seeds using `numpy.random.SeedSequence(scenario.seed).spawn(scenario.samples)`.
   - Each child seed yields an isolated `np.random.default_rng(child_seed)` instance threaded explicitly to dynamics, exogenous event sources, and stochastic actors.
   - Per-rollout seeds are derived via `int(child_seed.generate_state(1, dtype=np.uint32)[0])` and recorded on each `Trajectory`.
   - Core modules are prohibited by automated AST tests from calling global `random` or `numpy.random` methods.
3. **Scenario Branching & Scheduled Actions (AC-005)**:
   - Added `ScheduledAction(step, action)` to specify deterministic actions scheduled for future time steps.
   - Added `Scenario.branch(scenario_id=..., scheduled_actions=..., seed=..., intervention=...)` creating an independent counterfactual branch that preserves `initial_state` (by fingerprint identity) while leaving the parent scenario completely unmodified.
   - `branch_world(world)` provides deep snapshot isolation so simulations run on branches cannot mutate the origin world.
4. **Self-Check Determinism Helper**:
   - Implemented `SimulationEngine.verify_determinism(world, scenario)` executing two independent runs and verifying logical bitwise identity across states, steps, metrics, and fingerprints.

## Consequences
- **Positive**:
  - Full auditability and reproducible counterfactual comparisons (A/B testing, policy stress testing).
  - Eliminates flaky tests caused by hidden global RNG mutations.
  - Zero state leakage across branches in exploratory tree search, Monte Carlo planning, or MPC.
- **Negative**:
  - All stochastic components must explicitly accept `rng: np.random.Generator` in their interface rather than calling convenience global functions.

## Acceptance Criteria Satisfied
- **AC-004**: Identical input + versions + seed produces identical built-in logical trajectory.
- **AC-005**: Branches derived from the same snapshot never contaminate each other.
- **AC-009**: Monte Carlo returns exactly `samples` rollouts with deterministically-derived RNG streams.
