"""Verification suite for Prompt E4: Causal Mechanisms via DoWhy-GCM.

Verifies:
1. Initialization of `dowhy.gcm.StructuralCausalModel` via `EngineCausalInterface`.
2. `EngineCausalInterface.assign_mechanisms()` using `dowhy.gcm.auto.assign_causal_mechanisms`
   mapped to the application-provided `networkx.DiGraph` (both from a `pandas.DataFrame`
   and directly from an E2 OCEL 2.0 SQLite event store).
3. `EngineCausalInterface.simulate_intervention(target_node, value, n_samples)` wrapping
   `dowhy.gcm.interventional_samples` on a known linear SCM ($X \\rightarrow Y \\rightarrow Z$).
4. Modularity and causal invariance: intervening $do(Y = y)$ yields a `pandas.DataFrame`
   where $Y = y$ with exact zero variance, downstream $Z$ updates deterministically, and
   upstream $X$ remains invariant without upstream leakage.
5. Operational safeguards: strict DAG acyclicity checks (`networkx.is_directed_acyclic_graph`)
   rejecting cycles, self-loops, and non-DiGraph inputs with `CausalDAGValidationError`.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ewm_engine.core.causal import (
    CausalDAGValidationError,
    CausalMechanismError,
    EngineCausalInterface,
    load_ocel_historical_dataframe,
    validate_strict_dag,
)
from ewm_engine.core.ocel import OcelEventRecord, OcelObjectTarget, StateEncoder
from ewm_engine.core.types import EngineError


def _generate_linear_chain_data(
    n_rows: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic linear chain data for X -> Y -> Z.

    Structural equations:
        X ~ Normal(10.0, 1.5^2)
        Y := 2.0 * X + 5.0
        Z := 3.0 * Y - 4.0
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=10.0, scale=1.5, size=n_rows)
    y = 2.0 * x + 5.0
    z = 3.0 * y - 4.0
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def test_linear_scm_intervention_modularity_and_zero_variance() -> None:
    """Test X -> Y -> Z linear SCM: do(Y=y) has zero variance on Y, updates Z, leaves X invariant."""
    dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    df = _generate_linear_chain_data(n_rows=500, seed=42)

    causal_engine = EngineCausalInterface(dag=dag, seed=42)
    assert causal_engine.topological_nodes == ("X", "Y", "Z")

    mechanisms = causal_engine.assign_mechanisms(df)
    assert causal_engine.is_fitted is True
    assert set(mechanisms.keys()) == {"X", "Y", "Z"}
    assert "Empirical" in mechanisms["X"]
    assert "AdditiveNoiseModel" in mechanisms["Y"]
    assert "AdditiveNoiseModel" in mechanisms["Z"]

    # Intervene on Y: do(Y = 20.0) -> expected Z = 3.0 * 20.0 - 4.0 = 56.0
    samples_y20 = causal_engine.simulate_intervention(
        target_node="Y",
        value=20.0,
        n_samples=1000,
    )
    assert isinstance(samples_y20, pd.DataFrame)
    assert len(samples_y20) == 1000
    assert list(samples_y20.columns) == ["X", "Y", "Z"]
    assert causal_engine.last_intervention_time_ms > 0.0

    # Acceptance Criterion 1: Intervened variable Y matches assigned value with zero variance
    assert (samples_y20["Y"] == 20.0).all()
    assert float(samples_y20["Y"].var(ddof=0)) == 0.0
    assert float(samples_y20["Y"].mean()) == pytest.approx(20.0, abs=1e-12)

    # Acceptance Criterion 2: Downstream Z updates deterministically to 3 * 20 - 4 = 56.0
    assert np.allclose(samples_y20["Z"].to_numpy(), 56.0, atol=1e-5)
    assert float(samples_y20["Z"].mean()) == pytest.approx(56.0, abs=1e-5)

    # Acceptance Criterion 3: Upstream X remains invariant without upstream leakage
    # Compare X under do(Y = 20.0) vs do(Y = 100.0) and observational X
    samples_y100 = causal_engine.simulate_intervention(
        target_node="Y",
        value=100.0,
        n_samples=1000,
    )
    assert (samples_y100["Y"] == 100.0).all()
    assert float(samples_y100["Y"].var(ddof=0)) == 0.0
    assert np.allclose(samples_y100["Z"].to_numpy(), 296.0, atol=1e-5)

    # Because the RNG seed is reset deterministically and X's mechanism is untouched by do(Y=y),
    # the upstream samples of X under do(Y=20) and do(Y=100) are identical and match P(X)!
    assert np.allclose(samples_y20["X"].to_numpy(), samples_y100["X"].to_numpy(), atol=1e-12)
    assert float(samples_y20["X"].mean()) == pytest.approx(float(df["X"].mean()), abs=0.15)
    assert float(samples_y20["X"].std(ddof=0)) == pytest.approx(
        float(df["X"].std(ddof=0)), abs=0.15
    )


def test_scm_integration_with_e2_ocel2_sqlite_store(tmp_path: Path) -> None:
    """Verify EngineCausalInterface ingests historical data directly from an E2 OCEL 2.0 SQLite store."""
    sqlite_file = tmp_path / "causal_ocel2.sqlite"
    df_hist = _generate_linear_chain_data(n_rows=250, seed=101)

    with StateEncoder(sqlite_file) as encoder:
        for idx, row in df_hist.iterrows():
            encoder.encode_event(
                OcelEventRecord(
                    event_id=f"ev_{idx}",
                    event_type="process_batch",
                    timestamp=1700000000.0 + float(idx),
                    attributes={
                        "X": float(row["X"]),
                        "Y": float(row["Y"]),
                        "Z": float(row["Z"]),
                    },
                    objects=(
                        OcelObjectTarget(
                            object_id="line_1",
                            object_type="ProductionLine",
                            qualifier="executes",
                        ),
                    ),
                )
            )

    loaded_df = load_ocel_historical_dataframe(sqlite_file, columns=["X", "Y", "Z"])
    assert len(loaded_df) == 250
    assert list(loaded_df.columns) == ["X", "Y", "Z"]

    dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    causal_engine = EngineCausalInterface(dag=dag, seed=101)
    mechanisms = causal_engine.assign_mechanisms(sqlite_file)
    assert set(mechanisms.keys()) == {"X", "Y", "Z"}

    interventional_df = causal_engine.simulate_intervention("Y", 15.0, n_samples=500)
    assert (interventional_df["Y"] == 15.0).all()
    assert float(interventional_df["Y"].var(ddof=0)) == 0.0
    # Z = 3 * 15 - 4 = 41.0
    assert np.allclose(interventional_df["Z"].to_numpy(), 41.0, atol=1e-5)


def test_predict_vs_simulate_intervention_and_docstrings() -> None:
    """Verify observational .predict() vs interventional .simulate_intervention() behavior and docstrings."""
    predict_doc = inspect.getdoc(EngineCausalInterface.predict) or ""
    intervene_doc = inspect.getdoc(EngineCausalInterface.simulate_intervention) or ""
    assert "simulate_intervention" in predict_doc
    assert "Rung 1" in predict_doc
    assert "predict" in intervene_doc
    assert "Modularity" in intervene_doc

    dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    df = _generate_linear_chain_data(n_rows=300, seed=7)
    engine = EngineCausalInterface(dag=dag, seed=7)
    engine.assign_mechanisms(df)

    obs_samples = engine.predict(n_samples=300)
    assert len(obs_samples) == 300
    # Under observational sampling, Y varies with X (non-zero variance)
    assert float(obs_samples["Y"].var(ddof=0)) > 5.0
    assert np.allclose(
        obs_samples["Z"].to_numpy(),
        3.0 * obs_samples["Y"].to_numpy() - 4.0,
        atol=1e-5,
    )

    # Forward prediction from provided root X observations
    custom_x = pd.DataFrame({"X": [0.0, 10.0, 20.0]})
    pred_from_x = engine.predict(observed_data=custom_x)
    assert np.allclose(pred_from_x["Y"].to_numpy(), [5.0, 25.0, 45.0], atol=1e-5)
    assert np.allclose(pred_from_x["Z"].to_numpy(), [11.0, 71.0, 131.0], atol=1e-5)


def test_strict_dag_acyclicity_safeguards() -> None:
    """Operational safeguard: cyclic graphs, self-loops, and non-DiGraphs are rejected."""
    assert issubclass(CausalDAGValidationError, EngineError)
    assert issubclass(CausalMechanismError, EngineError)

    # 1. Directed 3-node cycle: X -> Y -> Z -> X
    cyclic_graph = nx.DiGraph([("X", "Y"), ("Y", "Z"), ("Z", "X")])
    with pytest.raises(CausalDAGValidationError, match="strict DAG acyclicity"):
        EngineCausalInterface(cyclic_graph)

    # 2. Self-loop: X -> X
    self_loop_graph = nx.DiGraph([("X", "X"), ("X", "Y")])
    with pytest.raises(CausalDAGValidationError, match="self-loops"):
        validate_strict_dag(self_loop_graph)

    # 3. Undirected graph
    undirected = nx.Graph([("X", "Y")])
    with pytest.raises(CausalDAGValidationError, match="DiGraph"):
        validate_strict_dag(undirected)

    # 4. Empty DiGraph
    empty_dag = nx.DiGraph()
    with pytest.raises(CausalDAGValidationError, match="at least one node"):
        EngineCausalInterface(empty_dag)

    # 5. Calling simulate_intervention before assign_mechanisms raises CausalMechanismError
    valid_dag = nx.DiGraph([("X", "Y")])
    unfitted = EngineCausalInterface(valid_dag)
    with pytest.raises(CausalMechanismError, match="assign_mechanisms"):
        unfitted.simulate_intervention("X", 1.0, n_samples=10)
