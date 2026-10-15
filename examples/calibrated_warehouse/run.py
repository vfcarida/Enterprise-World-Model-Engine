"""Worked Example: Amortized Bayesian Calibration with Certified Coverage (`ewm calibrate`).

Disciplines & Scientific Anchors:
- Simulation Intelligence Motif 3 (Lavin et al., arXiv:2112.03235).
- Amortized Simulation-Based Inference (Boelts et al., arXiv:2411.17337; Gloeckler et al., Simformer, ICML 2024).
- Mandatory Coverage Certification: TARP (arXiv:2302.03026) and Posterior-SBC (arXiv:2502.03279).
- Misspecification Robustness & Decision Refusal (Huang et al., NeurIPS 2023, arXiv:2305.15871).

Scenario:
  We have a regional distribution warehouse model with unknown initial inventory
  and depletion rate. We observe 6 weeks of empirical pallet counts and backlog orders.
  We calibrate the model parameters using amortized SBI and certify coverage before
  permitting downstream decision use.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import yaml

from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.workflow import CalibrationWorkflow
from ewm_engine.core.spec import WorldSpec
from ewm_engine.simulation.scenario import Scenario


def run_calibrated_warehouse_demo() -> None:
    current_dir = Path(__file__).resolve().parent
    spec_path = current_dir / "spec.yaml"
    data_path = current_dir / "empirical_data.yaml"
    priors_path = current_dir / "priors.yaml"

    print("======================================================================")
    print("  EWM Engine S04: Amortized Calibration with Certified Coverage")
    print("======================================================================\n")

    # 1. Load declarative WorldSpec
    spec = WorldSpec.from_yaml(spec_path)
    print(f"[1] Loaded WorldSpec: '{spec.world.name}' with {len(spec.resources)} tracked resources.")

    # 2. Load Bayesian Priors
    priors_raw = yaml.safe_load(priors_path.read_text(encoding="utf-8"))
    priors = [CalibrationPrior(**p) for p in priors_raw["priors"]]
    prior_set = PriorSet(priors=tuple(priors))
    print(f"[2] Declared {prior_set.dim} parameter priors:")
    for p in prior_set.priors:
        print(f"    - {p.name} ({p.target}): bounds={p.bounds}, distribution={p.distribution}")

    # 3. Load Empirical Observations
    data_raw = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    obs = data_raw["observations"]
    print("\n[3] Empirical Target Data (6 steps):")
    for r, vals in obs.items():
        print(f"    - {r}: {vals}")

    # 4. Execute Calibration Workflow with Coverage Certification
    print("\n[4] Running amortized SBI simulation batch & TARP + SBC certification...")
    scenario = Scenario(scenario_id="demo_calib", horizon=5, samples=1, seed=42)
    workflow = CalibrationWorkflow(
        spec=spec,
        prior_set=prior_set,
        tracked_resources=["inventory"],
        scenario=scenario,
    )

    report = workflow.run_calibration(
        target_data=obs,
        n_simulations=100,
        n_posterior_samples=300,
        seed=42,
    )

    # 5. Display Certified Results
    print("\n" + report.summary_table())

    print("\n[5] Posterior Predictive Check Overlay:")
    for rid in ["inventory"]:
        pred = report.posterior_predictive[rid]
        print(f"    Resource '{rid}':")
        print(f"      Observed:        {pred['observed']}")
        print(f"      Predictive Mean: {[round(v, 1) for v in pred['predictive_mean']]}")
        print(f"      90% CI Lower:    {[round(v, 1) for v in pred['ci_5']]}")
        print(f"      90% CI Upper:    {[round(v, 1) for v in pred['ci_95']]}")

    # 6. Epistemic Guardrail Demonstration: Refuse Uncertified / Misspecified Models
    print("\n[6] Demonstrating Mandatory Refusal Gate (Huang et al., 2023):")
    print("    Attempting calibration against impossible observations (inventory = 9999.0)...")
    impossible_obs = {"inventory": [9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0]}

    refusal_report = workflow.run_calibration(
        target_data=impossible_obs,
        n_simulations=25,
        n_posterior_samples=100,
        seed=42,
    )

    print(f"    Status:          {refusal_report.certificate.status}")
    print(f"    Is Certified:    {refusal_report.certificate.is_certified}")
    print(f"    Decision Usable: {refusal_report.certificate.decision_usable}")
    print(f"    Misspecification Score: {refusal_report.certificate.misspecification_score:.1f} std")
    print(f"    Refusal Reason:  {refusal_report.certificate.refusal_reason}")

    print("\n======================================================================")
    print("  Reproduction Command via CLI:")
    print(f"  ewm calibrate {spec_path} --data {data_path} --priors {priors_path}")
    print("======================================================================")


if __name__ == "__main__":
    run_calibrated_warehouse_demo()
