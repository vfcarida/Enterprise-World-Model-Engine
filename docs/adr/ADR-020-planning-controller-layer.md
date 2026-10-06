# ADR-020: Planning and Controller Layer, Pluggable Rollout Scoring, and Continuous Re-Grounding (MPC)

## Status
Accepted (v1.3.0-experimental)

## Context
In version 1.0, EWM Engine provided an initial prototype of Model Predictive Control (`RecedingHorizonSimulator` in `ewm_engine.simulation.mpc`). While it demonstrated the basic re-grounding loop, it exhibited several architectural limitations:
1. **Naive Objective Argmax**: Decisions were selected strictly by comparing the mean of a single scalar metric, without support for multi-criteria trade-offs, constraint violation penalties, or risk preferences.
2. **Ignorance of Uncertainty and Tail Risk**: Real-world decision makers face asymmetric risks. In high-variance environments, an action with a slightly higher expected return may carry unacceptable catastrophe or default risks in its tail distribution (e.g., negative cash, stockouts, or capacity breaches).
3. **No Pluggable Controller / Scorer Protocol**: The lookahead simulation loop was tightly coupled to inline simulation code rather than a pluggable `Planner` or `RolloutScorer` abstraction.
4. **Lack of Integration with the Actor Architecture**: There was no standard mechanism to use a model-predictive planner as an `Actor` within multi-agent worlds.

### The Epistemic Stance: Long Open-Loop Rollouts Are Not Forecasts
A foundational epistemic principle of the EWM Engine is that **long open-loop simulations are not prophetic forecasts**. In complex organizational and socio-technical dynamics, unmodeled exogenous factors, structural shifts (Lucas Critique), and autoregressive model drift inevitably compound over time (empirically measured in the `LongHorizon` benchmark family).

Therefore, **continuous state re-grounding** is a first-class control paradigm:
$$\text{Observe Current State } s_t \to \text{Simulate Short Lookahead } H \to \text{Score Candidates } \to \text{Execute Step 1 } \to \text{Re-ground from Reality } s_{t+1} \to \text{Repeat}$$

The world model serves as an *internal sandbox* for short-horizon hypothesis testing and risk assessment, while actual progression is constrained and corrected by continuous ground-truth observations.

## Decision

1. **Formalize Controller & Planning Protocols (`ewm_engine.experimental.planning`)**:
   - `RolloutScorer(Protocol)`: Evaluates a candidate's Monte Carlo rollouts and produces a scalar score and breakdown metadata.
   - `Planner(Protocol)`: Given an active world state, a set of candidate options (actions, action sequences, or interventions), lookahead horizon, and sample budget, evaluates candidates using a `RolloutScorer` and returns a `PlanningDecision`.
2. **Provide Four Standard Pluggable Scorers**:
   - **`ExpectedObjectiveScorer`**: Evaluates the mathematical expectation $\mathbb{E}[M]$ of a designated metric across stochastic rollout trajectories.
   - **`CVaRScorer` (Conditional Value at Risk / Risk-Averse)**: Evaluates tail risk at quantile level $\alpha \in (0, 1)$, optimizing worst-case outcomes rather than average-case expectations.
   - **`ConstraintPenalizedScorer`**: Incorporates hard and soft constraint violations, penalizing candidates that stress or violate world boundaries: $\text{BaseScore} - \lambda \cdot (\text{violations})$.
   - **`UncertaintyPenalizedScorer`**: Penalizes variance or spread across rollouts ($\mu \pm \lambda \cdot \sigma$) to reward decisions that are robust to stochastic shocks.
3. **Sealed Decision Provenance (`PlanningDecision`)**:
   - Every planning step records immutable audit metadata: candidate IDs, per-candidate scores and breakdowns, chosen candidate, simulation seed, lookahead horizon, sample budget, pre-decision state hash, and execution timestamps.
4. **`PlanningActor` Adapter**:
   - Implements the engine's `Actor` protocol, allowing any `Planner` + `RolloutScorer` to act as an autonomous decision agent within standard simulation environments.
5. **Inviolability of the Constraint Pipeline**:
   - Planners and controllers *propose* decisions. During execution in the real or host world, every action must pass through the world's constraint pipeline (`ConstraintRegistry`). Planning never bypasses invariants or permission gates.
6. **Refactor `RecedingHorizonSimulator`**:
   - Refactor `RecedingHorizonSimulator` to delegate lookahead evaluation to `RolloutPlanner` and `RolloutScorer`, preserving 100% backward compatibility for existing scripts and tests.

## Consequences

### Positive
- Enables decision makers and researchers to evaluate risk-sensitive policies (e.g. CVaR vs expected value) and observe how risk aversion shifts chosen actions.
- Establishes a clean, extensible protocol for world-model-based planning (compatible with RL/Dreamer-style latent rollouts and heuristic planners).
- Guarantees full auditability and decision provenance for organizational governance.
- Planners can be dropped directly into multi-agent simulations as `PlanningActor` instances.

### Negative / Trade-offs
- Monte Carlo lookahead planning incurs a computational cost proportional to $\mathcal{O}(|\text{candidates}| \times \text{samples} \times H)$.
- Requires appropriate tuning of penalty hyperparameters ($\lambda$) and risk quantiles ($\alpha$).
