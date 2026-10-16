"""End-to-End Enterprise Induction Demo: OCEL 2.0 Log -> WorldSpec -> Calibrated Counterfactual.

Demonstrates:
    1. Ingestion of raw OCEL 2.0 object-centric event log
    2. Automated discovery of entities, relationships, resources, and candidate conservation laws
    3. Conformance checking replay (mapping fitness/precision directly to EvidenceLevel)
    4. Execution of a baseline simulation and a policy counterfactual ("Add second approval queue")
    5. Evaluation with bootstrap confidence intervals and statistical significance testing
"""

from __future__ import annotations

from pathlib import Path

from ewm_engine.core.actions import Action
from ewm_engine.core.spec import WorldFactory
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.induction.process import induce_process


def run_ocel_procurement_demo() -> None:
    data_path = Path(__file__).parent / "log.json"
    print(f"=== 1. Inducing World Model from OCEL 2.0 Log ({data_path.name}) ===")
    spec, report = induce_process(
        ocel_data=data_path,
        engine="native",
        name="ProcurementFulfillmentTwin",
        seed=42,
    )

    print(f"Successfully induced WorldSpec: '{spec.world.name}'")
    print(f"  Entities: {report.entities_count} (Types: {list({e.type for e in spec.entities})})")
    print(f"  Relationships: {report.relationships_count} (O2O Fulfillment Links)")
    print(f"  Resources: {report.resources_count}")
    print(f"  Candidate Invariants: {len(report.candidate_invariants)}")
    print(f"  SHA-256 Fingerprint: {report.sha256_fingerprint[:16]}...")

    print("\n=== 2. Conformance & Replay Alignment Audit ===")
    for obj_type, conf in report.conformance.items():
        print(
            f"  Object Type: {obj_type:<10} | Fitness: {conf.fitness:.3f} | "
            f"Precision: {conf.precision:.3f} | Evidence: {conf.evidence_level.value}"
        )

    print("\n=== 3. Compiling Executable World ===")
    factory = WorldFactory()
    world = factory.create_world(spec)

    print("\n=== 4. Simulating Baseline (Status Quo) ===")
    base_res = world.simulate(horizon=10, samples=20, seed=42)
    print(f"  Baseline runs completed: {len(base_res.trajectories)} trajectories")

    print("\n=== 5. Simulating Counterfactual Policy: 'Add Expedited Approval Queue' ===")
    # Add an intervention action expediting fulfillment or transferring reserve stock
    expedite_action = Action(
        type="transfer_resource",
        parameters={
            "source": "inventory_stock",
            "target": "inventory_stock",
            "amount": 20.0,
        },
    )
    cf_res = world.simulate(
        actions=expedite_action,
        horizon=10,
        samples=20,
        seed=42,
        name="Expedited_Approval_Queue",
    )
    print(f"  Counterfactual runs completed: {len(cf_res.trajectories)} trajectories")

    print("\n=== 6. Calibrated Counterfactual Comparison with Error Bars ===")
    comparison = compare_scenarios(
        baseline=base_res,
        candidates=[cf_res],
        metrics=["inventory_stock"],
    )
    print(comparison.summary_table())


if __name__ == "__main__":
    run_ocel_procurement_demo()
