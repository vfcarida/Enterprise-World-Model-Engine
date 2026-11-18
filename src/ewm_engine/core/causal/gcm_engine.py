"""DoWhy-GCM Structural Causal Model (SCM) Interface for Enterprise World Models (E4).

Integrates graphical Structural Causal Models (`dowhy.gcm.StructuralCausalModel`) to
mathematically distinguish between Rung-1 observational association ($P(Y \\mid X=x)$ via
`.predict()`) and Rung-2 causal intervention ($P(Y \\mid do(X=x))$ via
`.simulate_intervention()`), climbing Pearl's Causal Hierarchy.

Scientific & Mathematical Grounding
-----------------------------------
A Structural Causal Model $\\mathcal{M} = \\langle U, V, F, P(U) \\rangle$ over a Directed
Acyclic Graph $\\mathcal{G} = (V, E)$ factorizes the joint observational distribution into
independent, modular structural mechanisms:
    $$P(V_1, \\dots, V_d) = \\prod_{i=1}^d P(V_i \\mid \\text{PA}_i)$$
Under the **Modularity (Independent Mechanisms) Assumption**, an atomic intervention
$do(V_s = v_s)$ performs graph surgery ($\\mathcal{G}_{\\overline{V_s}}$), replacing only the
target structural assignment $V_s := f_s(\\text{PA}_s, U_s)$ with the constant $V_s := v_s$
while leaving all other mechanisms $\\{f_j\\}_{j \\neq s}$ and the topological sort intact:
    $$P(V_1, \\dots, V_d \\mid do(V_s = v_s)) = \\mathbb{I}(V_s = v_s) \\prod_{j \\neq s} P(V_j \\mid \\text{PA}_j)$$
Consequently:
- Upstream ancestors $X \\in \\text{Anc}(V_s)$ remain strictly invariant ($P(X \\mid do(V_s = v_s)) = P(X)$).
- The intervened node $V_s$ exhibits exact value $v_s$ with zero variance ($\\text{Var}(V_s) = 0$).
- Downstream descendants $Z \\in \\text{Desc}(V_s)$ update according to their fitted causal mechanisms.
"""

from __future__ import annotations

import sqlite3
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ewm_engine.core.types import EngineError

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd


class CausalDAGValidationError(EngineError):
    """Raised when a provided causal graph fails strict DAG acyclicity or structural checks."""


class CausalMechanismError(EngineError):
    """Raised when causal mechanism assignment, fitting, or interventional sampling fails."""


def _import_networkx() -> Any:
    """Lazily import networkx to preserve zero-heavy-import startup contracts."""
    import networkx as nx

    return nx


def _import_pandas() -> Any:
    """Lazily import pandas to preserve minimal wheel startup contracts."""
    import pandas as pd

    return pd


def _import_dowhy_gcm() -> Any:
    """Lazily import dowhy.gcm with Python 3.12 SyntaxWarning suppression."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"^dowhy(\.|$)")
        from dowhy import gcm

    gcm.config.disable_progress_bars()
    return gcm


def validate_strict_dag(dag: Any) -> Tuple[str, ...]:
    """Validate that `dag` is a non-empty `networkx.DiGraph` satisfying strict acyclicity.

    Operational Safeguard (E4 Requirement 11):
    Enforces `networkx.is_directed_acyclic_graph(dag)` and checks for self-loops before
    any causal mechanisms are assigned or fitted.

    Args:
        dag: Candidate `networkx.DiGraph` instance.

    Returns:
        Tuple of node names in topological sort order.

    Raises:
        CausalDAGValidationError: If `dag` is not a `networkx.DiGraph`, is empty, contains
            self-loops, or contains directed cycles.
    """
    nx = _import_networkx()
    if not isinstance(dag, nx.DiGraph):
        raise CausalDAGValidationError(
            f"Expected a networkx.DiGraph for SCM initialization, got {type(dag).__name__}."
        )
    if dag.number_of_nodes() == 0:
        raise CausalDAGValidationError("Causal DAG must contain at least one node.")

    self_loops = list(nx.selfloop_edges(dag))
    if self_loops:
        raise CausalDAGValidationError(
            f"Causal Graph violates DAG acyclicity due to self-loops: {self_loops}",
            details={"self_loops": self_loops},
        )

    if not nx.is_directed_acyclic_graph(dag):
        try:
            cycle = list(nx.find_cycle(dag, orientation="original"))
        except Exception:
            cycle = []
        raise CausalDAGValidationError(
            f"Causal Graph violates strict DAG acyclicity check (cycle detected: {cycle}).",
            details={"cycle": cycle},
        )

    return tuple(str(n) for n in nx.topological_sort(dag))


def load_ocel_historical_dataframe(
    sqlite_path: Union[str, Path],
    *,
    object_type_table: Optional[str] = None,
    event_type_table: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Extract historical numerical/tabular observations from an E2 OCEL 2.0 SQLite store.

    Supports reading either:
    - An OCEL 2.0 `object_<map_type>` or `event_<map_type>` table containing attribute columns, or
    - Pivoting E2O / O2O attribute records from the normalized OCEL 2.0 relational tables.

    Args:
        sqlite_path: Path to the E2 OCEL 2.0 SQLite database file.
        object_type_table: Optional specific `object_<map_type>` table name to query.
        event_type_table: Optional specific `event_<map_type>` table name to query.
        columns: Optional list of column names (matching DAG nodes) to select.

    Returns:
        A `pandas.DataFrame` containing historical observations for SCM mechanism fitting.
    """
    db_path = Path(sqlite_path)
    if not db_path.exists():
        raise CausalMechanismError(f"OCEL 2.0 SQLite database not found at '{db_path}'.")

    pd_mod = _import_pandas()
    conn = sqlite3.connect(str(db_path))
    try:
        target_table = event_type_table or object_type_table
        if target_table is None:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND (name LIKE 'event_%' OR name LIKE 'object_%') "
                "AND name NOT IN ('event', 'object', 'event_map_type', 'object_map_type', "
                "'event_object', 'object_object') ORDER BY name ASC"
            )
            candidates = [row[0] for row in cursor.fetchall()]
            if not candidates:
                raise CausalMechanismError(
                    f"No typed attribute tables found in OCEL 2.0 database '{db_path}'."
                )
            target_table = candidates[0]

        safe_tbl = str(target_table).replace('"', '""')
        df: pd.DataFrame = pd_mod.read_sql_query(
            f'SELECT * FROM "{safe_tbl}"',  # nosec B608
            conn,
        )
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise CausalMechanismError(
                    f"Columns {missing} not found in OCEL table '{target_table}' "
                    f"(available: {list(df.columns)})."
                )
            df = df[columns].copy()
        return df
    finally:
        conn.close()


class EngineCausalInterface:
    """DoWhy-GCM Structural Causal Model (SCM) Engine Interface (E4).

    Wraps `dowhy.gcm.StructuralCausalModel` over an application-provided `networkx.DiGraph`
    and historical observational data (from a `pandas.DataFrame` or E2 OCEL 2.0 SQLite store)
    to support:
    1. Automatic causal mechanism assignment and fitting (`assign_mechanisms`),
    2. Observational Rung-1 prediction/sampling (`predict`), and
    3. Interventional Rung-2 do-calculus simulation (`simulate_intervention`).
    """

    def __init__(
        self,
        dag: nx.DiGraph,
        *,
        seed: int = 42,
    ) -> None:
        """Initialize the DoWhy-GCM `StructuralCausalModel` after strict DAG validation.

        Args:
            dag: Directed Acyclic Graph (`networkx.DiGraph`) defining causal topology.
            seed: Deterministic random seed for mechanism fitting and sampling.

        Raises:
            CausalDAGValidationError: If `dag` is not a valid acyclic `networkx.DiGraph`.
        """
        self._topological_nodes: Tuple[str, ...] = validate_strict_dag(dag)
        self._dag: nx.DiGraph = dag.copy()
        self._seed: int = int(seed)
        self._is_fitted: bool = False
        self._assigned_mechanisms: Dict[str, str] = {}
        self._training_data: Optional[pd.DataFrame] = None
        self._last_intervention_time_ms: float = 0.0

        gcm = _import_dowhy_gcm()
        self._scm: Any = gcm.StructuralCausalModel(self._dag)

    @property
    def dag(self) -> nx.DiGraph:
        """Return a defensive copy of the underlying `networkx.DiGraph`."""
        return self._dag.copy()

    @property
    def scm(self) -> Any:
        """Return the underlying `dowhy.gcm.StructuralCausalModel` instance."""
        return self._scm

    @property
    def topological_nodes(self) -> Tuple[str, ...]:
        """Return the DAG nodes in topological sort order."""
        return self._topological_nodes

    @property
    def assigned_mechanisms(self) -> Dict[str, str]:
        """Return a mapping of DAG node names to their assigned DoWhy-GCM mechanism descriptions."""
        return dict(self._assigned_mechanisms)

    @property
    def is_fitted(self) -> bool:
        """Return True if causal mechanisms have been assigned and fitted to historical data."""
        return self._is_fitted

    @property
    def last_intervention_time_ms(self) -> float:
        """Return the wall-clock execution time (in ms) of the most recent `simulate_intervention` call."""
        return self._last_intervention_time_ms

    def assign_mechanisms(
        self,
        data: Union[pd.DataFrame, str, Path],
        *,
        override_models: bool = True,
        fit: bool = True,
        ocel_table: Optional[str] = None,
    ) -> Dict[str, str]:
        """Assign and optionally fit causal mechanisms to the SCM using `dowhy.gcm.auto.assign_causal_mechanisms`.

        Enforces strict DAG acyclicity prior to mechanism assignment and maps each node in
        the DAG to an appropriate root distribution (e.g., `EmpiricalDistribution`) or
        conditional mechanism (e.g., `AdditiveNoiseModel` with `LinearRegression` / regressor).

        Args:
            data: Historical observational dataset as a `pandas.DataFrame` or path to an
                E2 OCEL 2.0 SQLite database file.
            override_models: Whether to overwrite any previously assigned mechanisms on the SCM.
            fit: If True (default), fits the assigned causal mechanisms to `data` via `dowhy.gcm.fit`.
            ocel_table: Optional table name when loading `data` from an OCEL 2.0 SQLite file.

        Returns:
            Dictionary mapping each node in topological order to its assigned mechanism string.

        Raises:
            CausalDAGValidationError: If the graph violates strict DAG acyclicity.
            CausalMechanismError: If required DAG node columns are missing from `data`.
        """
        # Operational safeguard: re-verify strict DAG acyclicity before fitting
        self._topological_nodes = validate_strict_dag(self._dag)
        pd_mod = _import_pandas()

        if isinstance(data, (str, Path)):
            df = load_ocel_historical_dataframe(
                data,
                event_type_table=ocel_table,
                columns=list(self._topological_nodes),
            )
        elif isinstance(data, pd_mod.DataFrame):
            df = data.copy()
        else:
            raise CausalMechanismError(
                f"Expected pandas.DataFrame or OCEL SQLite path, got {type(data).__name__}."
            )

        missing_cols = [node for node in self._topological_nodes if node not in df.columns]
        if missing_cols:
            raise CausalMechanismError(
                f"Historical data is missing columns for DAG nodes: {missing_cols}."
            )
        if len(df) < 2:
            raise CausalMechanismError(
                "Historical dataset must contain at least 2 rows to assign and fit causal mechanisms."
            )

        gcm = _import_dowhy_gcm()
        np.random.seed(self._seed)

        gcm.auto.assign_causal_mechanisms(
            self._scm,
            df,
            override_models=override_models,
        )

        if fit:
            gcm.fit(self._scm, df)
            self._is_fitted = True

        self._training_data = df
        self._assigned_mechanisms = {
            node: str(self._scm.causal_mechanism(node)) for node in self._topological_nodes
        }
        return dict(self._assigned_mechanisms)

    def predict(
        self,
        observed_data: Optional[pd.DataFrame] = None,
        *,
        n_samples: int = 100,
    ) -> pd.DataFrame:
        """Generate observational Rung-1 samples or forward predictions without graph surgery.

        Distinction from `.simulate_intervention()` (E4 Requirement 12):
        ----------------------------------------------------------------
        - `.predict()` operates on **Rung 1 (Association / Observation)** of Pearl's Causal
          Hierarchy ($P(V)$ or $P(Y \\mid X = x)$). It evaluates the fitted structural mechanisms
          under the natural observational graph $\\mathcal{G}$ without severing incoming parent
          edges. Conditioning on an observed variable $X = x$ in observational data reflects both
          downstream causal paths and upstream/backdoor confounding associations.
        - `.simulate_intervention()` operates on **Rung 2 (Intervention / Action)** of Pearl's
          Causal Hierarchy ($P(Y \\mid do(X = x))$). It performs graph surgery ($\\mathcal{G}_{\\overline{X}}$)
          via the $do$-operator, severing all incoming edges into the target node $X$, fixing
          $X := x$ with zero variance, leaving upstream ancestors invariant, and propagating
          only true causal effects to downstream descendants.

        Args:
            observed_data: Optional DataFrame of observed root/parent values. If None, draws
                `n_samples` joint observational samples from `dowhy.gcm.draw_samples`.
            n_samples: Number of observational samples to draw when `observed_data` is None.

        Returns:
            `pandas.DataFrame` containing observational samples across all DAG nodes.
        """
        if not self._is_fitted:
            raise CausalMechanismError(
                "Causal mechanisms must be assigned and fitted via assign_mechanisms() before calling predict()."
            )

        gcm = _import_dowhy_gcm()
        np.random.seed(self._seed)

        if observed_data is None:
            sampled: pd.DataFrame = gcm.draw_samples(self._scm, num_samples=int(n_samples))
            return sampled[list(self._topological_nodes)]

        # Evaluate mechanisms in topological order given observed root/parent columns
        nx = _import_networkx()
        result_df = observed_data.copy()
        for node in self._topological_nodes:
            if node in result_df.columns:
                continue
            parents = list(self._dag.predecessors(node))
            if not parents:
                drawn = gcm.draw_samples(self._scm, num_samples=len(result_df))
                result_df[node] = drawn[node].to_numpy()
            else:
                parent_matrix = result_df[parents].to_numpy()
                mech = self._scm.causal_mechanism(node)
                result_df[node] = np.asarray(mech.draw_samples(parent_matrix)).reshape(-1)
        _ = nx
        return result_df[list(self._topological_nodes)]

    def simulate_intervention(
        self,
        target_node: Union[str, Mapping[str, Any]],
        value: Optional[Any] = None,
        n_samples: int = 1000,
    ) -> pd.DataFrame:
        """Simulate an atomic or multi-node causal intervention $do(V_s = v_s)$ via DoWhy-GCM.

        Distinction from `.predict()` (E4 Requirement 12):
        --------------------------------------------------
        Unlike `.predict()` (which samples from the passive observational distribution $P(V)$),
        `.simulate_intervention()` executes Pearl's $do$-operator ($do(V_s = v_s)$) by wrapping
        `dowhy.gcm.interventional_samples`. Adhering strictly to the **Modularity Assumption**
        of Structural Causal Models:
        1. Only the structural equation of `target_node` ($V_s$) is replaced by the constant
           assignment $V_s := v_s$ (severing incoming edges $\\text{PA}_s \\to V_s$).
        2. Upstream and non-descendant variables ($X \\notin \\text{Desc}(V_s)$) retain their
           original independent mechanisms and marginal distributions without upstream leakage.
        3. The intervened variable in the returned DataFrame matches `value` with exact zero
           variance ($\\text{Var}(V_s) = 0$).
        4. Downstream variables ($Z \\in \\text{Desc}(V_s)$) update deterministically/causally
           through their structural equations $f_Z(\\text{PA}_Z, U_Z)$.

        Args:
            target_node: Name of the DAG node to intervene on (e.g., `'Y'`), or a mapping of
                `{node_name: forced_value}` for simultaneous interventions.
            value: Constant value $v_s$ assigned by $do(V_s = v_s)$ when `target_node` is a string.
            n_samples: Number of interventional samples to draw (default: 1,000).

        Returns:
            `pandas.DataFrame` of `n_samples` rows across all DAG nodes under $do(V_s = v_s)$.

        Raises:
            CausalMechanismError: If the SCM has not been fitted, `target_node` is not in the DAG,
                or `n_samples < 1`.
        """
        if not self._is_fitted:
            raise CausalMechanismError(
                "Causal mechanisms must be assigned and fitted via assign_mechanisms() "
                "before calling simulate_intervention()."
            )
        if int(n_samples) < 1:
            raise CausalMechanismError(f"n_samples must be >= 1, got {n_samples}.")

        if isinstance(target_node, str):
            if value is None:
                raise CausalMechanismError(
                    f"Intervention value must be provided when target_node='{target_node}' is a string."
                )
            raw_interventions: Dict[str, Any] = {target_node: value}
        elif isinstance(target_node, Mapping):
            raw_interventions = dict(target_node)
        else:
            raise CausalMechanismError(
                f"target_node must be a node name (str) or mapping, got {type(target_node).__name__}."
            )

        if not raw_interventions:
            raise CausalMechanismError("At least one target_node intervention must be specified.")

        unknown_nodes = [n for n in raw_interventions if n not in self._dag.nodes]
        if unknown_nodes:
            raise CausalMechanismError(
                f"Cannot intervene on unknown node(s) {unknown_nodes} not present in DAG {list(self._topological_nodes)}."
            )

        # Build modular intervention callables for dowhy.gcm.interventional_samples
        gcm_interventions: Dict[str, Any] = {}
        for node_name, forced_val in raw_interventions.items():
            if callable(forced_val):
                gcm_interventions[node_name] = forced_val
            else:
                scalar_val = (
                    float(forced_val) if isinstance(forced_val, (int, float)) else forced_val
                )
                gcm_interventions[node_name] = lambda _x, _v=scalar_val: _v

        gcm = _import_dowhy_gcm()
        np.random.seed(self._seed)

        t0 = time.perf_counter()
        samples_df: pd.DataFrame = gcm.interventional_samples(
            self._scm,
            gcm_interventions,
            num_samples_to_draw=int(n_samples),
        )
        self._last_intervention_time_ms = (time.perf_counter() - t0) * 1000.0

        # Ensure intervened numeric columns have exact float value with zero floating-point drift
        for node_name, forced_val in raw_interventions.items():
            if not callable(forced_val) and isinstance(forced_val, (int, float)):
                samples_df[node_name] = float(forced_val)

        return samples_df[list(self._topological_nodes)]


__all__ = [
    "CausalDAGValidationError",
    "CausalMechanismError",
    "EngineCausalInterface",
    "load_ocel_historical_dataframe",
    "validate_strict_dag",
]
