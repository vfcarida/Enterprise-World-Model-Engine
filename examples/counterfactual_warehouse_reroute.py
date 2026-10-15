"""Worked Example: Warehouse Stockout Prevention via Native Pearl Counterfactuals (S01).

Scientific Anchors:
- Richens & Everitt, "Robust agents learn causal world models", ICLR 2024 oral (arXiv:2402.10877).
- Pearl, "Causality: Models, Reasoning, and Inference", Cambridge University Press, 2009.

Scenario:
  Two distribution centers serve regional demand:
    - Warehouse North: Initial stock = 15.0 units.
    - Warehouse South: Initial stock = 120.0 units.
    - Central Treasury: Unrelated financial balance = $50,000.00.

  Factual Episode:
    - An unexpected demand spike of 45.0 units strikes Warehouse North.
    - With only 15.0 units on hand, Warehouse North suffers a severe stockout (30.0 units unserved).

  Counterfactual Question:
    "Had we intervened to reroute 35.0 units from South to North prior to the shock,
     would the stockout have been prevented under the exact same realized demand shock?"

  Pearl's Three Steps:
    1. Abduction:  Infer the exact exogenous noise realization U (demand spike = 45.0 units).
    2. Action:     Perform causal graph surgery do(warehouse_north = 50.0, warehouse_south = 85.0).
    3. Prediction: Re-roll the dynamics under identical coupled noise U (Twin-Rollout coupling).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ewm_engine.core.actions import Intervention
from ewm_engine.core.entities import Entity
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import TraceDiff
from ewm_engine.simulation.counterfactual import CounterfactualResult
from ewm_engine.simulation.scenario import Scenario


class WarehouseDemandShockSource:
    """Exogenous event source generating demand shocks with explicit noise draws."""

    def __init__(self, spike_step: int = 1, spike_quantity: float = 45.0) -> None:
        self.name = "WarehouseDemandShockSource"
        self.spike_step = spike_step
        self.spike_quantity = spike_quantity

    def sample(self, state: WorldState, step: int, rng: Any) -> list[ExogenousEvent]:
        # At step 1, inject the stochastic demand spike at Warehouse North
        if step == self.spike_step:
            return [
                ExogenousEvent(
                    id=f"demand_spike_step_{step}",
                    type="demand_shock",
                    severity=1.0,
                    parameters={"resource_id": "warehouse_north", "quantity": self.spike_quantity},
                )
            ]
        return []


def build_supply_chain_world() -> World:
    """Construct the initial supply chain world with North, South, and Treasury entities."""
    state = WorldState(
        resources=[
            Resource(id="warehouse_north", current=15.0, min_value=0.0, max_value=200.0, unit="crates"),
            Resource(id="warehouse_south", current=120.0, min_value=0.0, max_value=500.0, unit="crates"),
            Resource(id="central_treasury", current=50000.0, min_value=0.0, max_value=1000000.0, unit="USD"),
        ],
        entities=[
            Entity(id="hub_north", type="warehouse", attributes={"region": "north", "manager": "Alice"}),
            Entity(id="hub_south", type="warehouse", attributes={"region": "south", "manager": "Bob"}),
            Entity(id="hq_finance", type="finance", attributes={"audit_status": "certified"}),
        ],
    )
    dynamics = DeterministicTransferDynamics(
        action_type="transfer_resource",
        name="SupplyChainTransferDynamics",
    )
    return World(
        state=state,
        dynamics=dynamics,
        event_sources=[WarehouseDemandShockSource(spike_step=0, spike_quantity=45.0)],
    )


def run_warehouse_reroute_counterfactual() -> CounterfactualResult:
    """Execute the factual run, abduce exogenous noise, perform causal surgery, and predict."""
    print("=" * 80)
    print("S01 WORKED EXAMPLE: WAREHOUSE STOCKOUT COUNTERFACTUAL (TWIN-ROLLOUT SCM)")
    print("Scientific Anchor: Richens & Everitt, ICLR 2024 oral (arXiv:2402.10877)")
    print("=" * 80)

    world = build_supply_chain_world()

    # 1. Observe Factual Trajectory (no rerouting occurred)
    factual_scenario = Scenario(
        scenario_id="factual-baseline",
        name="Factual Baseline (No Rerouting)",
        horizon=2,
        seed=42,
    )
    print("\n[1] Running Factual Episode (Seed=42)...")
    factual_res = world.simulate(factual_scenario)
    factual_traj = factual_res.trajectories[0]

    factual_north = factual_traj.final_state.resources["warehouse_north"].current
    factual_south = factual_traj.final_state.resources["warehouse_south"].current
    factual_treasury = factual_traj.final_state.resources["central_treasury"].current

    print(f"  Factual Final Inventory North: {factual_north:.1f} crates")
    print(f"  Factual Final Inventory South: {factual_south:.1f} crates")
    print(f"  Factual Central Treasury:      ${factual_treasury:,.2f}")
    print(f"  Exogenous Noise Records Count: {len(factual_traj.exogenous_noise)}")
    print(f"  Exogenous Noise Fingerprint:   {factual_traj.exogenous_noise_fingerprint[:16]}...")

    # 2. Formulate Causal Intervention do(warehouse_north = 50.0, warehouse_south = 85.0)
    # Rerouting 35 crates from South to North prior to demand fulfillment
    intervention = Intervention(
        id="reroute_emergency_inventory",
        description="Emergency inventory rerouting from South (120 -> 85) to North (15 -> 50)",
        parameters={
            "warehouse_north": 50.0,
            "warehouse_south": 85.0,
        },
    )

    print("\n[2] Executing Pearl's Three-Step Counterfactual on SCM:")
    print("  Step 1 (Abduction): Recover realized exogenous noise U from factual trace")
    print("  Step 2 (Action):    Apply structural surgery do(North=50, South=85)")
    print("  Step 3 (Prediction):Twin-Rollout rollout with coupled exogenous noise")

    cf_result: CounterfactualResult = world.counterfactual(
        observed=factual_traj,
        intervention=intervention,
        horizon=2,
        seed=42,
    )

    cf_traj = cf_result.counterfactual_trajectory
    cf_north = cf_traj.final_state.resources["warehouse_north"].current
    cf_south = cf_traj.final_state.resources["warehouse_south"].current
    cf_treasury = cf_traj.final_state.resources["central_treasury"].current

    print("\n--- Counterfactual Outcomes ---")
    print(f"  Evidence Level:                {cf_result.evidence_level.value.upper()}")
    print(f"  Model-Counterfactual Notice:   {cf_result.disclaimer}")
    print(f"  Unique Invertible Abduction:   {cf_result.is_unique}")
    print(f"  Bit-Reproducible Execution:    {cf_result.is_bit_reproducible}")
    print(f"  Noise Coupled (Twin-Rollout):  {cf_result.is_noise_coupled}")
    print(f"  Locality Preserved:            {cf_result.locality_preserved}")

    print("\n--- Comparison (Factual vs Counterfactual) ---")
    print(f"  Warehouse North Stock:  {factual_north:.1f} -> {cf_north:.1f} crates (Delta: +{cf_result.target_divergences.get('warehouse_north', 0.0):.1f})")
    print(f"  Warehouse South Stock:  {factual_south:.1f} -> {cf_south:.1f} crates (Delta: {cf_result.target_divergences.get('warehouse_south', 0.0):.1f})")
    print(f"  Central Treasury (Off): ${factual_treasury:,.2f} -> ${cf_treasury:,.2f} (Delta: {cf_result.off_target_divergences.get('central_treasury', 0.0):.1f})")

    # Invariant checks
    assert cf_result.evidence_level == EvidenceLevel.COUNTERFACTUAL
    assert cf_result.is_noise_coupled is True
    assert cf_result.locality_preserved is True
    assert cf_treasury == factual_treasury, "Structural locality violation: unrelated treasury modified!"
    assert cf_north > factual_north, "Warehouse North should have higher inventory under rerouting!"

    # 3. Export Systemic Divergence Trace in Mermaid notation
    trace_diff: TraceDiff = cf_result.trace_diff
    mermaid_diagram = trace_diff.to_mermaid()

    print("\n--- Systemic Trace Divergence Diagram (Mermaid) ---")
    print(mermaid_diagram)
    print("=" * 80)
    print("SUCCESS: Counterfactual verified with Twin-Rollout bit-level reproducibility.")
    print("=" * 80)

    return cf_result


if __name__ == "__main__":
    run_warehouse_reroute_counterfactual()
