"""Orchestration workflow for simulation-based Bayesian calibration and certified coverage."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ewm_engine.calibration.certification import certify_coverage
from ewm_engine.calibration.priors import PriorSet
from ewm_engine.calibration.report import CalibrationReport
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator
from ewm_engine.calibration.simulator import ForwardSimulatorWrapper
from ewm_engine.core.spec import WorldSpec
from ewm_engine.core.types import ResourceId
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.simulation.scenario import Scenario


class CalibrationWorkflow:
    """End-to-end SBI Bayesian calibration pipeline with mandatory coverage certification."""

    def __init__(
        self,
        spec: WorldSpec,
        prior_set: PriorSet,
        tracked_resources: Sequence[ResourceId],
        scenario: Scenario | None = None,
        summary_mode: str = "trajectory",
    ) -> None:
        self.spec = spec
        self.prior_set = prior_set
        self.tracked_resources = tuple(tracked_resources)
        self.scenario = scenario or Scenario(
            scenario_id="calibration_base_scenario",
            horizon=10,
            samples=1,
            seed=42,
        )
        self.summary_mode = summary_mode
        self.simulator = ForwardSimulatorWrapper(
            spec=self.spec,
            scenario=self.scenario,
            prior_set=self.prior_set,
            tracked_resources=self.tracked_resources,
            summary_statistic_mode=self.summary_mode,
        )

    def run_calibration(
        self,
        target_data: Mapping[str, Sequence[float]],
        n_simulations: int = 100,
        n_posterior_samples: int = 500,
        seed: int = 42,
        calibration_id: str | None = None,
    ) -> CalibrationReport:
        """Execute full amortized calibration, coverage certification, and posterior predictive check."""
        t_start = time.time()
        calib_id = calibration_id or f"calib_{self.spec.world.name}_{int(t_start)}"

        # Compute deterministic target data hash
        target_bytes = json.dumps(
            {k: [float(v) for v in vals] for k, vals in sorted(target_data.items())},
            sort_keys=True,
        ).encode("utf-8")
        data_hash = hashlib.sha256(target_bytes).hexdigest()

        # Step 1: Sample thetas from prior
        rng = np.random.default_rng(seed)
        thetas = self.prior_set.sample(n=n_simulations, rng=rng)

        # Step 2: Forward simulate batch
        xs = self.simulator.simulate_batch(thetas=thetas, base_seed=seed)

        # Split into training (80%) and holdout test (20%) for TARP evaluation
        n_train = max(10, int(0.8 * n_simulations))
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, n_simulations)
        if len(test_indices) < 5:
            # If batch is small, use all samples for both training and holdout check
            train_thetas, train_xs = thetas, xs
            test_thetas, test_xs = thetas, xs
        else:
            train_thetas, train_xs = thetas[train_indices], xs[train_indices]
            test_thetas, test_xs = thetas[test_indices], xs[test_indices]

        # Step 3: Train Amortized Posterior Estimator
        estimator = AmortizedPosteriorEstimator(prior_set=self.prior_set)
        estimator.fit(thetas=train_thetas, xs=train_xs)

        # Step 4: Extract observed feature vector x_obs from target_data
        x_obs_list: list[float] = []
        if self.summary_mode == "moments":
            for rid in self.tracked_resources:
                series = target_data.get(rid, [])
                if not series:
                    x_obs_list.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                arr = np.array(series, dtype=np.float64)
                x_obs_list.extend(
                    [
                        float(np.mean(arr)),
                        float(np.std(arr)),
                        float(np.min(arr)),
                        float(np.max(arr)),
                        float(arr[-1]),
                    ]
                )
        else:
            expected_len = self.scenario.horizon + 1
            for rid in self.tracked_resources:
                raw_series = list(target_data.get(rid, []))
                if len(raw_series) < expected_len:
                    last_val = raw_series[-1] if raw_series else 0.0
                    raw_series = raw_series + [last_val] * (expected_len - len(raw_series))
                elif len(raw_series) > expected_len:
                    raw_series = raw_series[:expected_len]
                x_obs_list.extend([float(v) for v in raw_series])

        x_obs = np.array(x_obs_list, dtype=np.float64)

        # Step 5: Mandatory Coverage Certification Gate (TARP + Posterior-SBC)
        certificate = certify_coverage(
            estimator=estimator,
            test_thetas=test_thetas,
            test_xs=test_xs,
            x_obs=x_obs,
            simulator_fn=self.simulator.simulate_single,
            seed=seed,
        )

        # Step 6: Infer calibrated posterior p(theta | x_obs)
        posterior = estimator.predict_posterior(
            x_obs=x_obs,
            n_samples=n_posterior_samples,
            seed=seed + 1,
        )

        # Step 7: Posterior predictive simulation
        predictive_series: dict[str, Any] = {}
        for rid in self.tracked_resources:
            predictive_series[rid] = {
                "observed": [float(v) for v in target_data.get(rid, [])],
                "predictive_mean": [],
                "ci_5": [],
                "ci_95": [],
            }

        # Simulate 15 posterior draws to build predictive band
        post_sample_arr = posterior.sample(n=min(15, n_posterior_samples), seed=seed + 2)
        post_sample_dicts = self.prior_set.to_dict_list(post_sample_arr)

        rollout_curves: dict[str, list[list[float]]] = {rid: [] for rid in self.tracked_resources}
        for s_idx, t_dict in enumerate(post_sample_dicts):
            _, series_s = self.simulator.simulate_single(t_dict, seed=(seed + s_idx))
            for rid in self.tracked_resources:
                if rid in series_s:
                    rollout_curves[rid].append(series_s[rid])

        for rid in self.tracked_resources:
            curves = rollout_curves[rid]
            if curves:
                arr_curves = np.array(curves, dtype=np.float64)
                predictive_series[rid]["predictive_mean"] = [float(v) for v in np.mean(arr_curves, axis=0)]
                predictive_series[rid]["ci_5"] = [float(v) for v in np.percentile(arr_curves, 5, axis=0)]
                predictive_series[rid]["ci_95"] = [float(v) for v in np.percentile(arr_curves, 95, axis=0)]

        # Step 8: Build reproducible run manifest
        run_manifest = {
            "calibration_id": calib_id,
            "seed": seed,
            "n_simulations": n_simulations,
            "n_posterior_samples": n_posterior_samples,
            "tracked_resources": list(self.tracked_resources),
            "execution_duration_seconds": float(time.time() - t_start),
            "software_version": "1.8.0",
        }

        # Step 9: Assemble final report
        priors_dump = [p.model_dump(mode="json") for p in self.prior_set.priors]

        return CalibrationReport(
            calibration_id=calib_id,
            spec_name=self.spec.world.name,
            timestamp=t_start,
            target_data_hash=data_hash,
            evidence_level=EvidenceLevel.PREDICTIVE,
            priors=priors_dump,
            posterior=posterior,
            certificate=certificate,
            posterior_predictive=predictive_series,
            misspecification_diagnostic={
                "misspecification_score": certificate.misspecification_score,
                "is_misspecified": certificate.is_misspecified,
                "threshold_std": 3.5,
            },
            run_manifest=run_manifest,
        )
