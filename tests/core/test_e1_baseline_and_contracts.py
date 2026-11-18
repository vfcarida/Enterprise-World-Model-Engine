"""Verification suite for Prompt E1: Baseline Reproduction, Regression Protection, and Shared Contracts.

Verifies:
1. Pinned dependency versions (`==`) in root `requirements.txt` and `pytest.ini` configuration.
2. Central `types.py` (`ewm_engine.core.types` and `ewm_engine.types`) exposing core protocol
   interfaces (`StateInterface`, `DynamicsInterface`, `ConstraintInterface`, `ActionInterface`,
   `EventInterface`, `EventSourceInterface`, `ActorInterface`, `TransitionResultInterface`)
   with complete docstrings on all base classes.
3. Strict Python typing (`typing.Dict`, `typing.List`, `typing.Optional`) across `/core/` modules.
4. Explicit subclassing of all custom engine exceptions from `EngineError`.
5. Seeded randomness and isolated temporary directory file I/O (`tmp_path`).
6. Dummy metamorphic test confirming that structurally identical inputs yield identical outputs.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import ewm_engine.core.types as core_types
import ewm_engine.types as pkg_types
from ewm_engine.actors.base import ActorContext
from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.results import ConstraintPhase
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.bitemporal import BitemporalLeakageError
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.time import TemporalMonotonicityError
from ewm_engine.core.trust import (
    DecisionBlockedError,
    InductionRefusalError,
    UncertaintyRequiredError,
    UnqualifiedCausalClaimError,
)
from ewm_engine.core.types import (
    ActionInterface,
    ActorInterface,
    ConstraintInterface,
    DynamicsInterface,
    EngineError,
    EventInterface,
    StateInterface,
    TransitionResultInterface,
)
from ewm_engine.core.world import World
from ewm_engine.domains.swe.constraints import StructuralViolationError
from ewm_engine.domains.swe.state import (
    CircularDependencyError,
    StaleObservationError,
    SWEGraphCapacityError,
)
from ewm_engine.dynamics.calibration import AbstainException
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.router import NumericalInstabilityError
from ewm_engine.evaluation.evaluator import BenchmarkTimeoutError
from ewm_engine.exceptions import (
    BranchingError,
    ConstraintViolationError,
    ConstraintViolationTeachingError,
    DynamicsError,
    EWMError,
    InvalidActionError,
    InvalidWorldStateError,
    NonIdentifiableCausalQueryError,
    ProvenanceError,
    RefusalToRankError,
    ResourceBoundsError,
    ScenarioError,
    SerializationError,
    SerializationSecurityError,
    SimulationConfigurationError,
    TeachingError,
    UncertifiedCalibrationError,
)
from ewm_engine.experimental.domain_composition import CompositionAssumptionError
from ewm_engine.provenance.explainability import NarrativeFabricationError
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_DIR = REPO_ROOT / "src" / "ewm_engine" / "core"


def test_e1_pinned_requirements_and_pytest_ini_exist() -> None:
    """Verify root requirements.txt pins exact versions with '==' and pytest.ini is configured."""
    req_path = REPO_ROOT / "requirements.txt"
    assert req_path.exists(), "Root requirements.txt must exist"

    lines: List[str] = [
        line.strip()
        for line in req_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) >= 15, "requirements.txt must pin core and testing dependencies"
    for req_line in lines:
        assert "==" in req_line, (
            f"Dependency in requirements.txt must be pinned with '==': {req_line}"
        )

    pinned_map: Dict[str, str] = {}
    for req_line in lines:
        pkg, ver = req_line.split("==", 1)
        pinned_map[pkg.strip().lower()] = ver.strip()

    for mandatory_pkg in ("pydantic", "numpy", "pyyaml", "pytest", "mypy", "hypothesis", "ruff"):
        assert mandatory_pkg in pinned_map, f"Missing mandatory pinned package '{mandatory_pkg}'"

    pytest_ini_path = REPO_ROOT / "pytest.ini"
    assert pytest_ini_path.exists(), "Root pytest.ini must exist"
    ini_text = pytest_ini_path.read_text(encoding="utf-8")
    assert "[pytest]" in ini_text
    assert "filterwarnings" in ini_text


def test_e1_types_protocols_and_docstrings(rng: np.random.Generator) -> None:
    """Verify types.py exposes core protocols with docstrings and runtime conformance."""
    assert core_types.StateInterface is pkg_types.StateInterface
    assert core_types.DynamicsInterface is pkg_types.DynamicsInterface
    assert core_types.EngineError is pkg_types.EngineError

    # Every class defined in ewm_engine.core.types must have a non-empty docstring
    for name, obj in inspect.getmembers(core_types, inspect.isclass):
        if obj.__module__ == core_types.__name__:
            doc: Optional[str] = inspect.getdoc(obj)
            assert doc and len(doc.strip()) > 20, (
                f"Class '{name}' in types.py is missing a docstring"
            )

    # Verify runtime protocol checks
    state = WorldState(
        entities=[Entity(id="wh_1", type="warehouse")],
        resources=[Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0)],
    )
    assert isinstance(state, StateInterface)

    action = Action(
        id="act_1",
        type="transfer_resource",
        parameters={"source_resource": "stock_a", "target_resource": "stock_a", "quantity": 0.0},
    )
    assert isinstance(action, ActionInterface)

    event = ExogenousEvent(id="ev_1", type="demand_shock", severity=1.5)
    assert isinstance(event, EventInterface)

    dynamics = DeterministicTransferDynamics()
    assert isinstance(dynamics, DynamicsInterface)

    tr_res = dynamics.transition(state, [action], [event], rng)
    assert isinstance(tr_res, TransitionResultInterface)

    constraint = ResourceCapacityConstraint(resource_id="stock_a")
    assert isinstance(constraint, ConstraintInterface)
    c_res = constraint.evaluate(state, [action], phase=ConstraintPhase.POST_TRANSITION)
    assert c_res.satisfied is True

    actor = ThresholdReplenishmentActor(
        actor_id="rb_actor",
        source_resource="stock_a",
        target_resource="stock_a",
        reorder_point=10.0,
        order_quantity=25.0,
    )
    assert isinstance(actor, ActorInterface)
    ctx = ActorContext(step=0, timestamp=0.0, rng=rng)
    assert list(actor.act(state, ctx)) == []


def test_e1_core_modules_strict_typing_and_exception_hierarchy() -> None:
    """Verify /core/ modules use typing.Dict/List/Optional and all exceptions subclass EngineError."""
    core_files = sorted(CORE_DIR.glob("*.py"))
    assert len(core_files) >= 20

    typing_target_names = {"Dict", "List", "Optional"}
    for py_file in core_files:
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        imported_typing_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                for alias in node.names:
                    imported_typing_names.add(alias.name)
        assert imported_typing_names.intersection(typing_target_names), (
            f"Module {py_file.name} in /core/ must utilize strict typing imports "
            f"(Dict, List, Optional from typing)"
        )

    all_exception_classes = (
        EWMError,
        InvalidWorldStateError,
        ConstraintViolationError,
        InvalidActionError,
        ResourceBoundsError,
        SimulationConfigurationError,
        DynamicsError,
        ScenarioError,
        BranchingError,
        ProvenanceError,
        SerializationError,
        SerializationSecurityError,
        TeachingError,
        NonIdentifiableCausalQueryError,
        UncertifiedCalibrationError,
        ConstraintViolationTeachingError,
        RefusalToRankError,
        BitemporalLeakageError,
        TemporalMonotonicityError,
        UnqualifiedCausalClaimError,
        UncertaintyRequiredError,
        DecisionBlockedError,
        InductionRefusalError,
        StructuralViolationError,
        SWEGraphCapacityError,
        CircularDependencyError,
        StaleObservationError,
        NumericalInstabilityError,
        AbstainException,
        BenchmarkTimeoutError,
        CompositionAssumptionError,
        NarrativeFabricationError,
    )
    for exc_cls in all_exception_classes:
        assert issubclass(exc_cls, EngineError), (
            f"Exception '{exc_cls.__name__}' must explicitly subclass EngineError"
        )

    err = EngineError("test engine error", details={"code": "E1_TEST"})
    assert str(err) == "test engine error"
    assert err.details == {"code": "E1_TEST"}


def test_e1_dummy_metamorphic_structural_identity_and_isolated_io(
    isolated_temp_dir: Path,
) -> None:
    """Confirm structurally identical inputs yield identical outputs with seeded RNG and isolated I/O."""
    state_1 = WorldState(
        entities=[
            Entity(id="dc_east", type="distribution_center", attributes={"region": "east"}),
            Entity(id="dc_west", type="distribution_center", attributes={"region": "west"}),
        ],
        relationships=[
            Relationship(source="dc_east", target="dc_west", type="corridor"),
        ],
        resources=[
            Resource(
                id="inv_east", entity_id="dc_east", current=200.0, min_value=0.0, max_value=500.0
            ),
            Resource(
                id="inv_west", entity_id="dc_west", current=80.0, min_value=0.0, max_value=500.0
            ),
        ],
        memory={"cycle": 1},
        active_rules={"min_reserve": 20.0},
        context={"demand_regime": "stable"},
        step=0,
        timestamp=0.0,
    )

    # Construct structurally identical state with reversed entity/resource declaration order
    state_2_permuted = WorldState(
        entities=[
            Entity(id="dc_west", type="distribution_center", attributes={"region": "west"}),
            Entity(id="dc_east", type="distribution_center", attributes={"region": "east"}),
        ],
        relationships=[
            Relationship(source="dc_east", target="dc_west", type="corridor"),
        ],
        resources=[
            Resource(
                id="inv_west", entity_id="dc_west", current=80.0, min_value=0.0, max_value=500.0
            ),
            Resource(
                id="inv_east", entity_id="dc_east", current=200.0, min_value=0.0, max_value=500.0
            ),
        ],
        memory={"cycle": 1},
        active_rules={"min_reserve": 20.0},
        context={"demand_regime": "stable"},
        step=0,
        timestamp=0.0,
    )

    # Persist and reload via isolated_temp_dir
    io_path = isolated_temp_dir / "state_2.json"
    io_path.write_text(json.dumps(state_2_permuted.to_dict(), sort_keys=True), encoding="utf-8")
    state_2 = WorldState.from_dict(json.loads(io_path.read_text(encoding="utf-8")))

    assert state_1.state_hash == state_2.state_hash
    assert state_1.fingerprint == state_2.fingerprint

    action = Action(
        id="rebalance_50",
        type="transfer_resource",
        parameters={"source_resource": "inv_east", "target_resource": "inv_west", "quantity": 50.0},
    )
    scenario = Scenario(
        scenario_id="e1_metamorphic",
        horizon=4,
        samples=2,
        seed=42,
        scheduled_actions=(ScheduledAction(step=0, action=action),),
    )

    world_1 = World(state=state_1, dynamics=DeterministicTransferDynamics())
    world_2 = World(state=state_2, dynamics=DeterministicTransferDynamics())

    engine = SimulationEngine()
    res_1 = engine.run(world_1, scenario)
    res_2 = engine.run(world_2, scenario)

    assert len(res_1.trajectories) == len(res_2.trajectories) == 2
    for t1, t2 in zip(res_1.trajectories, res_2.trajectories, strict=True):
        assert t1.seed == t2.seed
        assert t1.final_state.fingerprint == t2.final_state.fingerprint
        assert [s.state_hash for s in t1.steps] == [s.state_hash for s in t2.steps]
