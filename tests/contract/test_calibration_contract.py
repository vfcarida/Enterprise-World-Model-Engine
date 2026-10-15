"""Contract tests for Amortized Calibration with Certified Coverage (S04).

Enforces:
1. Non-point distribution contract: PosteriorDistribution always reports full distributions,
   credible intervals, and covariances — never a single point estimate.
2. Mandatory coverage certification: under-covered or misspecified models are flagged and
   refused for downstream decision use (decision_usable=False).
3. Invariant 5 (reproducibility): identical seeds produce bitwise-identical calibration artifacts.
4. Epistemic honesty: evidence_level is strictly PREDICTIVE, never auto-promoted to CAUSAL/STRUCTURAL.
5. CLI integration: `ewm calibrate` executes end-to-end, writing JSON/Markdown artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.workflow import CalibrationWorkflow
from ewm_engine.cli.main import main
from ewm_engine.core.spec import WorldSpec
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.scenario import Scenario


@pytest.fixture
def sample_spec() -> WorldSpec:
    spec_yaml = """
schema_version: "1.0.0"
world:
  name: "CalibrationWarehouseWorld"
  description: "Test world for SBI calibration contract"
entities:
  - id: "wh_facility"
    type: "warehouse"
resources:
  - id: "stock"
    current: 50.0
    min_value: 0.0
    max_value: 200.0
    unit: "crates"
  - id: "capacity"
    current: 100.0
    min_value: 0.0
    max_value: 500.0
    unit: "crates"
"""
    return WorldSpec.from_yaml(spec_yaml)


def test_contract_non_point_distribution_output(sample_spec: WorldSpec) -> None:
    """Contract: calibration must return a full posterior distribution with credible intervals, never a bare point."""
    priors = PriorSet(
        priors=(
            CalibrationPrior(
                name="initial_stock",
                target="resource",
                target_id="stock",
                distribution="uniform",
                bounds=(10.0, 90.0),
            ),
        )
    )

    workflow = CalibrationWorkflow(
        spec=sample_spec,
        prior_set=priors,
        tracked_resources=["stock"],
        scenario=Scenario(scenario_id="sbi_scen", horizon=5, samples=1, seed=42),
    )

    target_data = {"stock": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]}
    report = workflow.run_calibration(
        target_data=target_data,
        n_simulations=20,
        n_posterior_samples=50,
        seed=42,
    )

    # Invariant: Posterior has full distribution fields
    assert isinstance(report.posterior.mean, dict)
    assert isinstance(report.posterior.std, dict)
    assert isinstance(report.posterior.median, dict)
    assert isinstance(report.posterior.credible_intervals_90, dict)
    assert isinstance(report.posterior.covariance, list)
    assert len(report.posterior.samples) > 0

    # Never a single scalar
    ci = report.posterior.credible_intervals_90["initial_stock"]
    assert ci[0] < ci[1]
    assert report.posterior.std["initial_stock"] > 0.0

    # Posterior predictive overlay is present
    assert "stock" in report.posterior_predictive
    assert "observed" in report.posterior_predictive["stock"]
    assert "predictive_mean" in report.posterior_predictive["stock"]
    assert "ci_5" in report.posterior_predictive["stock"]
    assert "ci_95" in report.posterior_predictive["stock"]


def test_contract_refusal_on_misspecified_or_undercovered(sample_spec: WorldSpec) -> None:
    """Contract: severely misspecified data must be flagged and refused for decision use."""
    priors = PriorSet(
        priors=(
            CalibrationPrior(
                name="initial_stock",
                target="resource",
                target_id="stock",
                bounds=(10.0, 90.0),
            ),
        )
    )

    workflow = CalibrationWorkflow(
        spec=sample_spec,
        prior_set=priors,
        tracked_resources=["stock"],
        scenario=Scenario(scenario_id="sbi_scen", horizon=5, samples=1, seed=42),
    )

    # Target data 10,000 crates is impossible under the prior bounds [10, 90]
    target_data = {"stock": [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0]}
    report = workflow.run_calibration(
        target_data=target_data,
        n_simulations=20,
        n_posterior_samples=50,
        seed=42,
    )

    # Refusal contract
    assert not report.certificate.is_certified
    assert not report.certificate.decision_usable
    assert report.certificate.status == "UNCERTIFIED_MISSPECIFIED"
    assert report.certificate.refusal_reason is not None
    assert report.certificate.is_misspecified


def test_contract_epistemic_honesty(sample_spec: WorldSpec) -> None:
    """Contract: EvidenceLevel of calibration is PREDICTIVE (calibrated != causally identified)."""
    priors = PriorSet(
        priors=(
            CalibrationPrior(
                name="stock_init",
                target="resource",
                target_id="stock",
                bounds=(20.0, 80.0),
            ),
        )
    )

    workflow = CalibrationWorkflow(
        spec=sample_spec,
        prior_set=priors,
        tracked_resources=["stock"],
        scenario=Scenario(scenario_id="scen", horizon=3, samples=1, seed=42),
    )

    target_data = {"stock": [50.0, 50.0, 50.0, 50.0]}
    report = workflow.run_calibration(target_data=target_data, n_simulations=15, seed=42)

    assert report.evidence_level == EvidenceLevel.PREDICTIVE


def test_contract_invariant5_reproducibility(sample_spec: WorldSpec) -> None:
    """Contract: Calibration run with identical seed produces bitwise identical report and samples."""
    priors = PriorSet(
        priors=(
            CalibrationPrior(
                name="stock_init",
                target="resource",
                target_id="stock",
                bounds=(30.0, 70.0),
            ),
        )
    )

    workflow = CalibrationWorkflow(
        spec=sample_spec,
        prior_set=priors,
        tracked_resources=["stock"],
        scenario=Scenario(scenario_id="scen", horizon=4, samples=1, seed=123),
    )

    target_data = {"stock": [50.0, 50.0, 50.0, 50.0, 50.0]}

    report1 = workflow.run_calibration(target_data=target_data, n_simulations=20, seed=777)
    report2 = workflow.run_calibration(target_data=target_data, n_simulations=20, seed=777)

    assert report1.target_data_hash == report2.target_data_hash
    assert report1.posterior.mean == report2.posterior.mean
    assert report1.posterior.std == report2.posterior.std
    assert report1.posterior.samples == report2.posterior.samples
    assert report1.certificate.status == report2.certificate.status


def test_contract_cli_calibrate_command(tmp_path: Path, sample_spec: WorldSpec) -> None:
    """Contract: `ewm calibrate` executes from CLI and generates valid artifacts."""
    spec_path = tmp_path / "world.yaml"
    spec_path.write_text(
        """
schema_version: "1.0.0"
world:
  name: "CLIWarehouse"
resources:
  - id: "inventory"
    current: 40.0
    min_value: 0.0
    max_value: 200.0
""",
        encoding="utf-8",
    )

    data_path = tmp_path / "target.json"
    data_path.write_text(
        json.dumps({"inventory": [40.0, 40.0, 40.0, 40.0, 40.0]}),
        encoding="utf-8",
    )

    out_json = tmp_path / "calib_out.json"
    out_md = tmp_path / "calib_out.md"

    exit_code = main(
        [
            "calibrate",
            str(spec_path),
            "--data",
            str(data_path),
            "--samples",
            "15",
            "--posterior-samples",
            "30",
            "--seed",
            "42",
            "--out",
            str(out_json),
            "--markdown-out",
            str(out_md),
            "--allow-uncertified",
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "calibration_id" in data
    assert "posterior" in data
    assert "mean" in data["posterior"]
    assert "certificate" in data
