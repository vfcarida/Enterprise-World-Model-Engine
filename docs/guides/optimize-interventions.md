# Optimizing Interventions: Multi-Objective Pareto Frontiers & Evolutionary Search

This guide demonstrates how to search and optimize policy interventions across single- and multi-objective landscapes using built-in optimizers, evolutionary strategies (`pycma`, `nevergrad`), and genetic Pareto search (`pymoo`) in `ewm_engine.experimentation`.

---

## 1. Defining Multi-Objective Vectors & Pareto Dominance

When evaluating interventions, define metrics with explicit optimization directions (`"minimize"` or `"maximize"`):

```python
from ewm_engine.experimentation import ObjectiveVector

# Policy A: Cost = 80, SLA = 0.95
obj_a = ObjectiveVector(
    values={"cost": 80.0, "sla": 0.95},
    directions={"cost": "minimize", "sla": "maximize"},
)

# Policy B: Cost = 120, SLA = 0.90 (Worse in both)
obj_b = ObjectiveVector(
    values={"cost": 120.0, "sla": 0.90},
    directions={"cost": "minimize", "sla": "maximize"},
)

# Policy A Pareto-dominates Policy B
print(f"Policy A dominates B: {obj_a.dominates(obj_b)}")  # True
```

---

## 2. Zero-Dependency Optimizers (Pure NumPy)

The core engine provides `RandomSearchOptimizer` and `HillClimbingOptimizer` without requiring any heavy solver packages:

```python
from ewm_engine.experimentation import (
    ParameterDef,
    ParameterSpace,
    HillClimbingOptimizer,
)

space = ParameterSpace(
    name="PolicyInterventionSpace",
    parameters=(
        ParameterDef(name="staffing_buffer", type="continuous", bounds=(5.0, 30.0), default=10.0),
        ParameterDef(name="safety_stock", type="continuous", bounds=(50.0, 200.0), default=100.0),
    ),
)


def evaluate_policy(point: dict[str, float]) -> ObjectiveVector:
    cost = point["staffing_buffer"] * 100.0 + point["safety_stock"] * 2.0
    sla = 1.0 - (10.0 / (point["staffing_buffer"] + 1.0)) * 0.05
    return ObjectiveVector(
        values={"cost": cost, "sla": sla},
        directions={"cost": "minimize", "sla": "maximize"},
    )


optimizer = HillClimbingOptimizer(step_fraction=0.15)
result = optimizer.optimize(evaluate_policy, space, n_evaluations=40, seed=42)

print(f"Best Found Cost: {result.best_objective.values['cost']:.2f}")
print(f"Best Found SLA: {result.best_objective.values['sla']:.3f}")
print(f"Evaluations: {result.evaluations_count}")
```

---

## 3. Evolutionary Search via pycma & Nevergrad (`[opt-evolutionary]`)

Install the evolutionary optimization extra:
```bash
pip install 'ewm-engine[opt-evolutionary]'
```

### CMA-ES Search via PyCMAOptimizer
```python
from ewm_engine.experimentation import PyCMAOptimizer

cma_opt = PyCMAOptimizer(sigma0=0.25)
cma_result = cma_opt.optimize(evaluate_policy, space, n_evaluations=50, seed=42)
print("CMA-ES Optimal Parameters:", cma_result.best_parameters)
```

### Nevergrad Gradient-Free Search
```python
from ewm_engine.experimentation import NevergradOptimizer

ng_opt = NevergradOptimizer(algorithm="NGOpt")
ng_result = ng_opt.optimize(evaluate_policy, space, n_evaluations=50, seed=42)
print("Nevergrad Optimal Parameters:", ng_result.best_parameters)
```

---

## 4. Multi-Objective Pareto Frontiers via pymoo (`[opt-pareto]`)

Install the Pareto optimization extra:
```bash
pip install 'ewm-engine[opt-pareto]'
```

Search multi-objective trade-offs and return a non-dominated `ParetoFront`:

```python
from ewm_engine.experimentation import PymooParetoOptimizer

# Run NSGA-II multi-objective genetic algorithm
pareto_opt = PymooParetoOptimizer(population_size=20, n_generations=15)
result = pareto_opt.optimize(evaluate_policy, space, seed=42)

front = result.pareto_front
print(f"Discovered {front.size} non-dominated Pareto-optimal policies:")

for i, policy in enumerate(front.policies[:5]):
    cost = policy.objectives.values["cost"]
    sla = policy.objectives.values["sla"]
    print(
        f"  Policy #{i + 1}: Cost=${cost:,.2f}, SLA={sla:.2%}, "
        f"Parameters={policy.parameters}, Fingerprint={policy.policy_fingerprint[:12]}..."
    )
```

> [!TIP]
> **No Arbitrary Argmax:** Instead of picking an arbitrary single policy, present the Pareto front to enterprise decision-makers with quantified trade-offs between expenditure and operational service levels.
