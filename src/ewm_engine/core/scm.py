"""Structural Causal Model (SCM) compilation and Pearl's causal surgery view over Enterprise World Models.

Scientific Grounding:
    - Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877): "Robust agents learn causal world models."
      Proves that any agent capable of generalizing across arbitrary distribution shifts must have
      implicitly learned an approximate causal model of the environment.
    - Pearl, J. (2009): "Causality: Models, Reasoning, and Inference", Cambridge University Press.
      Formalizes structural causal models M = <U, V, F, P(U)> and the three-step counterfactual
      engine: Abduction -> Action (Surgery) -> Prediction.
    - Gkountouras et al. (2024, arXiv:2410.19923): Causal world models as queryable structural simulators.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256, normalize_canonical_value
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.trust import ResponsibleAIWarning

if TYPE_CHECKING:
    from ewm_engine.core.spec import WorldSpec
    from ewm_engine.core.state import WorldState
    from ewm_engine.core.world import World
    from ewm_engine.simulation.trajectory import Trajectory


@dataclass
class CausalGraph:
    """Directed Acyclic Graph (DAG) specifying declared structural mechanisms.

    Edges represent direct causal dependencies between variables: (source, target).
    """

    nodes: set[str] = field(default_factory=set)
    edges: set[tuple[str, str]] = field(default_factory=set)
    unobserved_nodes: set[str] = field(default_factory=set)

    def add_node(self, name: str, is_unobserved: bool = False) -> None:
        self.nodes.add(name)
        if is_unobserved:
            self.unobserved_nodes.add(name)

    def add_edge(self, source: str, target: str) -> None:
        self.nodes.add(source)
        self.nodes.add(target)
        self.edges.add((source, target))

    def parents(self, node: str) -> set[str]:
        return {src for (src, tgt) in self.edges if tgt == node}

    def children(self, node: str) -> set[str]:
        return {tgt for (src, tgt) in self.edges if src == node}

    def descendants(self, node: str) -> set[str]:
        visited: set[str] = set()
        queue = [node]
        while queue:
            curr = queue.pop(0)
            for ch in self.children(curr):
                if ch not in visited:
                    visited.add(ch)
                    queue.append(ch)
        return visited

    def find_all_undirected_paths(
        self, start: str, end: str, max_depth: int = 6
    ) -> list[list[str]]:
        """Find all undirected paths between start and end (ignoring edge direction)."""
        adjacency: dict[str, set[str]] = {n: set() for n in self.nodes}
        for u, v in self.edges:
            adjacency[u].add(v)
            adjacency[v].add(u)

        paths: list[list[str]] = []
        queue: list[list[str]] = [[start]]

        while queue:
            path = queue.pop(0)
            curr = path[-1]
            if curr == end:
                paths.append(path)
                continue
            if len(path) > max_depth:
                continue
            for neighbor in adjacency.get(curr, set()):
                if neighbor not in path:
                    queue.append([*path, neighbor])
        return paths

    def is_backdoor_path(self, path: list[str], treatment: str) -> bool:
        """Check if a path is a backdoor path into treatment (starts with arrow into treatment)."""
        if len(path) < 2 or path[0] != treatment:
            return False
        return (path[1], treatment) in self.edges

    def is_collider(self, prev_node: str, curr_node: str, next_node: str) -> bool:
        """Check if curr_node is a collider along the path (arrows point into curr: -> curr <-)."""
        return (prev_node, curr_node) in self.edges and (next_node, curr_node) in self.edges

    def is_path_blocked(self, path: list[str], conditioning_set: set[str]) -> bool:
        """Check if a path is d-separated / blocked given the conditioning set."""
        cond_and_descendants = set(conditioning_set)
        for c in conditioning_set:
            cond_and_descendants.update(self.descendants(c))

        for i in range(1, len(path) - 1):
            prev_n = path[i - 1]
            curr_n = path[i]
            next_n = path[i + 1]

            if self.is_collider(prev_n, curr_n, next_n):
                if curr_n not in cond_and_descendants:
                    return True
            else:
                if curr_n in conditioning_set:
                    return True
        return False


class NonUniqueAbductionWarning(ResponsibleAIWarning):
    """Emitted when an SCM abduction step encounters a non-invertible mechanism."""


class InvertibilityType(StrEnum):
    """Classification of structural assignment invertibility for noise abduction."""

    EXACT = "exact"
    INJECTIVE = "injective"
    DETERMINISTIC = "deterministic"
    NON_INVERTIBLE = "non_invertible"


class ExogenousVariable(BaseModel):
    """Addressable exogenous background noise variable U_i in the structural causal model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        description="Unique addressable identifier of the exogenous variable (e.g. 'u:event:power_outage')."
    )
    step: int = Field(
        default=0, description="Discrete step at which this exogenous variable operates."
    )
    mechanism_id: str = Field(
        description="Identifier of the generating mechanism (event source, dynamics, actor)."
    )
    target_variable: str = Field(
        description="Name of the endogenous variable directly influenced by this noise."
    )
    distribution: str = Field(
        default="empirical", description="Prior distributional assumption over U_i."
    )
    invertibility: InvertibilityType = Field(
        default=InvertibilityType.EXACT,
        description="Whether U_i can be algebraically inverted from observed outcomes.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Distributional parameters (e.g. mean, variance, scale, support).",
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the exogenous variable spec."
    )


class StructuralAssignment(BaseModel):
    """Structural equation V_i := f_i(PA_i, U_i) defining an endogenous variable in the SCM."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: str = Field(
        description="Target endogenous variable name (e.g. 'resource:stock_A', 'entity:node_1:status')."
    )
    parents: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Direct parent endogenous variables PA_i in the structural causal DAG.",
    )
    exogenous_id: str | None = Field(
        default=None,
        description="Identifier of the coupled exogenous noise variable U_i, if stochastic.",
    )
    mechanism_id: str = Field(
        description="Identifier of the generating rule, dynamics model, or constraint."
    )
    invertibility: InvertibilityType = Field(
        default=InvertibilityType.EXACT,
        description="Invertibility classification governing Pearl's abduction step.",
    )
    is_intervened: bool = Field(
        default=False,
        description="True if Pearl's causal surgery do(V_i = x) was applied to this assignment.",
    )
    intervened_value: Any = Field(
        default=None,
        description="Constant value forced by do-calculus causal surgery.",
    )
    formula: str = Field(
        default="",
        description="Human-readable mathematical or structural description of the mechanism.",
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the assignment."
    )

    def evaluate(self, parent_values: dict[str, Any], u_value: Any = None) -> Any:
        """Evaluate structural assignment under given parent and exogenous noise values."""
        if self.is_intervened:
            return self.intervened_value
        if self.invertibility == InvertibilityType.DETERMINISTIC:
            # Deterministic transfer / conservation rule
            return parent_values.get(self.target, 0.0)
        # Default additive/multiplicative evaluation
        base_val = float(parent_values.get(self.target, 0.0))
        noise_val = float(u_value) if isinstance(u_value, (int, float)) else 0.0
        return base_val + noise_val

    def abduce(self, target_value: Any, parent_values: dict[str, Any]) -> tuple[Any, bool]:
        """Pearl's Abduction step: solve for U_i := f_i^{-1}(V_i, PA_i).

        Returns:
            A tuple of (abduced_u_value, is_unique).
        """
        if self.is_intervened:
            return (0.0, True)
        if self.invertibility == InvertibilityType.DETERMINISTIC:
            return (0.0, True)
        if self.invertibility in (InvertibilityType.EXACT, InvertibilityType.INJECTIVE):
            base_val = float(parent_values.get(self.target, 0.0))
            tgt_val = float(target_value) if isinstance(target_value, (int, float)) else 0.0
            return (tgt_val - base_val, True)
        # Non-invertible mechanism
        return (target_value, False)


class AbductionResult(BaseModel):
    """Result of Pearl's Step 1 (Abduction) over an observed simulation trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    noise_draws: dict[str, Any] = Field(
        default_factory=dict,
        description="Recovered exogenous noise values indexed by addressable noise ID.",
    )
    is_unique: bool = Field(
        default=True,
        description="True if all mechanisms were uniquely/algebraically invertible without ambiguity.",
    )
    non_unique_variables: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Variables whose mechanisms were non-invertible, requiring posterior distribution.",
    )
    posterior_distributions: dict[str, tuple[float, ...]] = Field(
        default_factory=dict,
        description="Candidate posterior noise sample distributions when abduction is non-unique.",
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the abduced noise profile."
    )


class StructuralCausalModel(BaseModel):
    """First-class Structural Causal Model (SCM) view over enterprise state and dynamics.

    Represents M = <U, V, F, P(U)> as an additive compilation over WorldSpec/World without
    modifying underlying state immutability or runtime fingerprints.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="EnterpriseWorldSCM", description="Human-readable SCM name.")
    endogenous_variables: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from endogenous variable name to categorical type (e.g. 'resource', 'entity_attr').",
    )
    exogenous_variables: dict[str, ExogenousVariable] = Field(
        default_factory=dict,
        description="Addressable exogenous noise variables U_i.",
    )
    assignments: dict[str, StructuralAssignment] = Field(
        default_factory=dict,
        description="Structural assignments V_i := f_i(PA_i, U_i).",
    )
    causal_graph: CausalGraph = Field(
        default_factory=CausalGraph,
        description="Directed acyclic graph of structural mechanism dependencies.",
    )
    fingerprint: str = Field(
        default="", description="Canonical SHA-256 fingerprint of the compiled SCM."
    )

    @classmethod
    def from_world(cls, world: World, horizon: int = 1) -> StructuralCausalModel:
        """Compile a Structural Causal Model view from an active World instance."""
        endogenous: dict[str, str] = {}
        exogenous: dict[str, ExogenousVariable] = {}
        assignments: dict[str, StructuralAssignment] = {}
        graph = CausalGraph()

        # 1. Register endogenous state variables (resources and entity attributes)
        state = world.initial_state
        for r_id in state.resources:
            var_name = f"resource:{r_id}"
            endogenous[var_name] = "resource"
            graph.add_node(var_name)

        for e_id, entity in state.entities.items():
            for attr_name in entity.attributes:
                var_name = f"entity:{e_id}:{attr_name}"
                endogenous[var_name] = "entity_attr"
                graph.add_node(var_name)

        # 2. Compile event sources into exogenous variables
        for src in world.event_sources:
            src_name = getattr(src, "name", src.__class__.__name__)
            for step_idx in range(horizon):
                u_id = f"u:step_{step_idx}:event:{src_name}"
                ex_fp = canonical_sha256({"id": u_id, "step": step_idx, "mechanism": src_name})
                ex_var = ExogenousVariable(
                    id=u_id,
                    step=step_idx,
                    mechanism_id=src_name,
                    target_variable=f"event:{src_name}",
                    distribution="empirical",
                    invertibility=InvertibilityType.EXACT,
                    fingerprint=ex_fp,
                )
                exogenous[u_id] = ex_var
                graph.add_node(u_id, is_unobserved=False)

        # 3. Compile dynamics transitions into structural assignments
        dyn_name = (
            getattr(world.dynamics, "name", world.dynamics.__class__.__name__)
            if world.dynamics is not None
            else "DeterministicIdentity"
        )

        for r_id in state.resources:
            var_name = f"resource:{r_id}"
            parents = [var_name]
            # Link event sources and actions as parents
            for src in world.event_sources:
                src_name = getattr(src, "name", src.__class__.__name__)
                ev_var = f"u:step_0:event:{src_name}"
                if ev_var in exogenous:
                    graph.add_edge(ev_var, var_name)

            u_dyn_id = f"u:dynamics:{dyn_name}:{r_id}"
            ex_fp = canonical_sha256({"id": u_dyn_id, "mechanism": dyn_name, "target": var_name})
            ex_dyn = ExogenousVariable(
                id=u_dyn_id,
                step=0,
                mechanism_id=dyn_name,
                target_variable=var_name,
                distribution="empirical",
                invertibility=InvertibilityType.EXACT,
                fingerprint=ex_fp,
            )
            exogenous[u_dyn_id] = ex_dyn
            graph.add_node(u_dyn_id, is_unobserved=False)
            graph.add_edge(u_dyn_id, var_name)

            asgn_fp = canonical_sha256(
                {
                    "target": var_name,
                    "parents": sorted(parents),
                    "exogenous_id": u_dyn_id,
                    "mechanism_id": dyn_name,
                }
            )
            assignment = StructuralAssignment(
                target=var_name,
                parents=tuple(parents),
                exogenous_id=u_dyn_id,
                mechanism_id=dyn_name,
                invertibility=InvertibilityType.EXACT,
                formula=f"{var_name} := f_{dyn_name}({var_name}, {u_dyn_id})",
                fingerprint=asgn_fp,
            )
            assignments[var_name] = assignment

        # Compute canonical SCM fingerprint
        scm_payload = {
            "endogenous": sorted(endogenous.items()),
            "exogenous": sorted((k, v.fingerprint) for k, v in exogenous.items()),
            "assignments": sorted((k, v.fingerprint) for k, v in assignments.items()),
            "edges": sorted(graph.edges),
        }
        scm_fp = canonical_sha256(scm_payload)

        return cls(
            name="WorldSCM",
            endogenous_variables=endogenous,
            exogenous_variables=exogenous,
            assignments=assignments,
            causal_graph=graph,
            fingerprint=scm_fp,
        )

    @classmethod
    def from_world_spec(cls, spec: WorldSpec, horizon: int = 1) -> StructuralCausalModel:
        """Compile a Structural Causal Model view from a declarative WorldSpec."""
        world = spec.build_world()
        return cls.from_world(world, horizon=horizon)

    @classmethod
    def from_world_state(cls, state: WorldState) -> StructuralCausalModel:
        """Compile a minimal baseline SCM view from an immutable WorldState."""
        from ewm_engine.core.world import World

        world = World(state=state)
        return cls.from_world(world, horizon=1)

    def intervene(
        self,
        intervention: dict[str, Any] | Sequence[Action] | Intervention,
    ) -> StructuralCausalModel:
        """Perform Pearl's causal surgery do(X = x) on this structural causal model.

        Surgery replaces the structural assignments for intervened variables with constant
        values, severing all incoming directed parent edges in the causal graph.
        """
        # 1. Parse intervened variables and target forced values
        intervened_map: dict[str, Any] = {}
        if isinstance(intervention, dict):
            for k, v in intervention.items():
                target_key = k if ":" in k else f"resource:{k}"
                intervened_map[target_key] = v
        elif isinstance(intervention, Intervention):
            for k, v in intervention.parameters.items():
                target_key = k if ":" in k else f"resource:{k}"
                intervened_map[target_key] = v
        elif isinstance(intervention, Sequence):
            for act in intervention:
                for k, v in act.parameters.items():
                    target_key = k if ":" in k else f"resource:{k}"
                    intervened_map[target_key] = v

        # 2. Clone causal graph and sever incoming edges to intervened variables
        mutilated_graph = CausalGraph(
            nodes=set(self.causal_graph.nodes),
            edges=set(self.causal_graph.edges),
            unobserved_nodes=set(self.causal_graph.unobserved_nodes),
        )

        for tgt in intervened_map:
            # Sever all incoming parent edges (Graph Surgery: remove (src, tgt))
            incoming = [(u, v) for (u, v) in mutilated_graph.edges if v == tgt]
            for edge in incoming:
                mutilated_graph.edges.remove(edge)

        # 3. Create modified structural assignments
        new_assignments: dict[str, StructuralAssignment] = dict(self.assignments)
        for tgt, val in intervened_map.items():
            old_asgn = self.assignments.get(tgt)
            mech_id = old_asgn.mechanism_id if old_asgn else "DirectIntervention"
            asgn_fp = canonical_sha256(
                {
                    "target": tgt,
                    "parents": [],
                    "is_intervened": True,
                    "intervened_value": val,
                }
            )
            new_assignments[tgt] = StructuralAssignment(
                target=tgt,
                parents=(),
                exogenous_id=None,
                mechanism_id=mech_id,
                invertibility=InvertibilityType.DETERMINISTIC,
                is_intervened=True,
                intervened_value=val,
                formula=f"{tgt} := do({val})",
                fingerprint=asgn_fp,
            )

        # 4. Recompute canonical fingerprint
        scm_payload = {
            "endogenous": sorted(self.endogenous_variables.items()),
            "exogenous": sorted((k, v.fingerprint) for k, v in self.exogenous_variables.items()),
            "assignments": sorted((k, v.fingerprint) for k, v in new_assignments.items()),
            "edges": sorted(mutilated_graph.edges),
        }
        scm_fp = canonical_sha256(scm_payload)

        return StructuralCausalModel(
            name=f"{self.name}_Mutilated",
            endogenous_variables=dict(self.endogenous_variables),
            exogenous_variables=dict(self.exogenous_variables),
            assignments=new_assignments,
            causal_graph=mutilated_graph,
            fingerprint=scm_fp,
        )

    def abduce(self, trajectory: Trajectory) -> AbductionResult:
        """Execute Pearl's Step 1 (Abduction) over an observed simulation trajectory.

        Recovers the exact posterior over exogenous noise U_i:
        - Exact where the mechanism is invertible or recorded in the systemic trace.
        - Returns a posterior distribution + flag when the mechanism is non-invertible.
        """
        noise_draws: dict[str, Any] = {}
        non_unique_vars: list[str] = []
        candidate_posteriors: dict[str, tuple[float, ...]] = {}

        # 1. First recover all directly recorded exogenous draws from the systemic trace
        trace = trajectory.systemic_trace
        for u_id, rec in trace.exogenous_noise.items():
            noise_draws[u_id] = rec.value
            if not rec.is_invertible:
                non_unique_vars.append(u_id)

        # 2. For any SCM structural assignments not covered by trace draws, perform algebraic inversion
        if trajectory.steps:
            for asgn in self.assignments.values():
                if asgn.exogenous_id and asgn.exogenous_id not in noise_draws:
                    r_key = asgn.target.split(":", 1)[-1]
                    # Check first step change
                    step0 = trajectory.steps[0]
                    target_val = step0.step_metrics.get(f"resource_{r_key}", 0.0)
                    initial_val = trajectory.initial_state.resources.get(r_key)
                    init_res_val = initial_val.current if initial_val else 0.0

                    parent_vals = {asgn.target: init_res_val}
                    u_val, is_unique = asgn.abduce(target_val, parent_vals)
                    noise_draws[asgn.exogenous_id] = u_val
                    if not is_unique:
                        non_unique_vars.append(asgn.target)
                        candidate_posteriors[asgn.target] = (
                            float(u_val) - 1.0,
                            float(u_val),
                            float(u_val) + 1.0,
                        )
                        warnings.warn(
                            f"Non-unique abduction for target '{asgn.target}' with mechanism '{asgn.mechanism_id}'. "
                            f"Returning counterfactual posterior distribution with is_unique=False.",
                            category=NonUniqueAbductionWarning,
                            stacklevel=2,
                        )

        is_unique_abduction = len(non_unique_vars) == 0
        abd_fp = canonical_sha256(
            {
                "noise_draws": normalize_canonical_value(noise_draws),
                "is_unique": is_unique_abduction,
            }
        )

        return AbductionResult(
            noise_draws=noise_draws,
            is_unique=is_unique_abduction,
            non_unique_variables=tuple(sorted(non_unique_vars)),
            posterior_distributions=candidate_posteriors,
            fingerprint=abd_fp,
        )

    def check_identifiability(
        self,
        treatment: str,
        outcome: str,
        conditioning_set: Sequence[str] = (),
    ) -> Any:
        """Check whether the causal effect do(treatment) -> outcome is identifiable in this SCM."""
        from ewm_engine.experimental.causal import check_backdoor_identifiability

        return check_backdoor_identifiability(
            graph=self.causal_graph,
            treatment=treatment,
            outcome=outcome,
            conditioning_set=conditioning_set,
        )
