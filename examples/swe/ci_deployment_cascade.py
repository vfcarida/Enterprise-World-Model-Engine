"""Software Engineering Progressive-Delivery & Shared-DB-Pool Cascade Application.

Demonstrates how `ewm_engine.experimental.swe_ecosystem` models hidden cross-service
contention (`checkout_service` retry storm exhausting shared `orders_db` pool and starving
`payment_worker`) that static dependency graphs miss, while respecting SWE negative findings
(`SWE-N1`, `SWE-N2`, `SWE-N3`).
"""

from __future__ import annotations

from dataclasses import dataclass

from ewm_engine.experimental.swe_ecosystem import (
    compute_static_transitive_dependents,
    run_swe_ecosystem_experiment,
)


@dataclass(frozen=True)
class SWEDeploymentApplicationSummary:
    """Summary of the SWE progressive-delivery counterfactual evaluation."""

    static_dependents_of_auth_lib: tuple[str, ...]
    dynamic_impacted_under_burst: tuple[str, ...]
    payment_worker_missed_by_static_graph: bool
    static_graph_deployment_regret_usd: float
    ewm_deployment_regret_usd: float
    bitemporal_withheld_records_at_t5: int
    negative_findings_confirmed: int
    output_fingerprint: str


def run_swe_deployment_application(
    seed: int = 42,
    n_seeds: int = 3,
) -> SWEDeploymentApplicationSummary:
    """Execute the SWE ecosystem deployment planning evaluation and extract key metrics."""
    static_deps = compute_static_transitive_dependents("auth_lib")
    report = run_swe_ecosystem_experiment(seed=seed, n_seeds=n_seeds)

    rung_by_id = {r.rung_id: r for r in report.ladder_results}
    static_regret = rung_by_id["rung1_static_dependency_graph"].deployment_decision_regret_usd
    ewm_regret = rung_by_id["rung5_experimental_swe_world_model"].deployment_decision_regret_usd
    neg_confirmed = sum(
        1 for n in report.negative_findings_and_transfer_limits if n.get("confirmed", False)
    )

    return SWEDeploymentApplicationSummary(
        static_dependents_of_auth_lib=tuple(sorted(static_deps)),
        dynamic_impacted_under_burst=tuple(report.true_dynamic_impacted_under_burst_rollout),
        payment_worker_missed_by_static_graph=(
            "payment_worker" not in static_deps
            and "payment_worker" in report.true_dynamic_impacted_under_burst_rollout
        ),
        static_graph_deployment_regret_usd=static_regret,
        ewm_deployment_regret_usd=ewm_regret,
        bitemporal_withheld_records_at_t5=report.bitemporal_withheld_records_at_t5,
        negative_findings_confirmed=neg_confirmed,
        output_fingerprint=report.output_fingerprint,
    )


if __name__ == "__main__":  # pragma: no cover
    summary = run_swe_deployment_application()
    print(f"Static dependents of auth_lib:    {summary.static_dependents_of_auth_lib}")
    print(f"Dynamic impacted under burst:     {summary.dynamic_impacted_under_burst}")
    print(f"payment_worker missed statically: {summary.payment_worker_missed_by_static_graph}")
    print(f"Static graph deployment regret:   ${summary.static_graph_deployment_regret_usd:,.2f}")
    print(f"Dynamic EWM deployment regret:    ${summary.ewm_deployment_regret_usd:,.2f}")
    print(f"Negative findings confirmed:      {summary.negative_findings_confirmed}/3")
