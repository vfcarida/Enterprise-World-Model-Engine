"""Unit tests for STGCN, CascadeFailurePredictor, CausalRCA, Remediation, OPA, and Chaos Mesh."""
# ruff: noqa: E402

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")

from ewm_engine.aiops.chaos.harness import ChaosMeshHarness
from ewm_engine.aiops.evaluation.evaluator import build_microservice_incident_episode
from ewm_engine.aiops.models.cascade_predictor import AIOpsWorldModel
from ewm_engine.aiops.models.stgcn_encoder import STGCNEncoder
from ewm_engine.aiops.rca.causal_rca import CausalRCA
from ewm_engine.aiops.remediation.generator import RemediationGenerator
from ewm_engine.aiops.remediation.opa_validator import OPAPolicyValidator


@pytest.mark.unit
def test_stgcn_encoder_and_cascade_predictor_forward_and_gradients() -> None:
    """Verify STGCNEncoder and AIOpsWorldModel produce bounded predictions and clean gradients."""
    encoder = STGCNEncoder(
        in_channels=5,
        spatial_hidden=32,
        temporal_hidden=32,
        out_channels=16,
        num_gcn_layers=2,
    )
    x = torch.rand(2, 10, 6, 5)
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 4, 2, 3]], dtype=torch.long)

    s_t = encoder(x, edge_index)
    assert s_t.shape == (2, 16)
    assert torch.isfinite(s_t).all()

    with pytest.raises(ValueError, match="Expected 4D tensor"):
        encoder(torch.rand(6, 5), edge_index)

    with pytest.raises(ValueError, match="Expected in_channels=5"):
        encoder(torch.rand(1, 5, 6, 4), edge_index)

    model = AIOpsWorldModel(num_services=6, in_channels=5, latent_dim=16)
    x_next = torch.rand(2, 10, 6, 5)
    target_fail = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    target_lead = torch.full((2, 6), 45.0)
    target_slo = torch.tensor([0.85, 0.70])

    loss = model.compute_training_loss(
        x_t=x,
        x_t1=x_next,
        edge_index=edge_index,
        target_failures=target_fail,
        target_lead_minutes=target_lead,
        target_slo_degradation=target_slo,
    )
    loss.backward()
    assert float(loss.item()) > 0.0

    _, out = model(x, edge_index)
    assert out.failure_probabilities.shape == (2, 6)
    assert (out.failure_probabilities >= 0.0).all() and (out.failure_probabilities <= 1.0).all()
    assert (out.predicted_lead_time_minutes >= 30.0).all()
    assert (out.predicted_lead_time_minutes <= 60.0).all()


@pytest.mark.unit
def test_causal_rca_pc_and_do_calculus_isolates_root_cause() -> None:
    """Verify CausalRCA learns causal DAG and isolates upstream root cause from downstream symptoms."""
    rca = CausalRCA()

    # Calling identify_root_cause before learn_causal_graph raises ValueError
    with pytest.raises(ValueError, match="Causal graph not learned yet"):
        rca.identify_root_cause({"payment-service": 0.9}, slo_impact=0.8)

    episode = build_microservice_incident_episode("memory_leak", seed=42)
    dag = rca.learn_causal_graph(
        episode.telemetry_df,
        call_graph_edges=episode.call_graph_edges,
    )
    assert len(dag.nodes) == len(episode.services)

    root_cause, confidence = rca.identify_root_cause(
        observed_anomalies=episode.observed_anomalies,
        slo_impact=episode.actual_slo_degradation,
    )
    assert root_cause == "payment-service"
    assert 0.35 < confidence <= 1.0

    # Also test long-format DataFrame support in learn_causal_graph
    long_df = pd.DataFrame(
        [
            {"timestamp": i, "service_name": "svc-a", "cpu": 0.2 + 0.05 * i, "latency": 0.1 * i}
            for i in range(15)
        ]
        + [
            {
                "timestamp": i,
                "service_name": "svc-b",
                "cpu": 0.2 + 0.05 * max(0, i - 2),
                "latency": 0.1 * max(0, i - 2),
            }
            for i in range(15)
        ]
    )
    long_dag = rca.learn_causal_graph(long_df)
    assert "svc-a" in long_dag.nodes and "svc-b" in long_dag.nodes


@pytest.mark.unit
def test_remediation_generator_and_opa_policy_validator() -> None:
    """Verify RemediationGenerator manifests and OPAPolicyValidator admission rules."""
    gen = RemediationGenerator()
    validator = OPAPolicyValidator()

    rollback_m = gen.generate_manifest("payment-service", "rollback", namespace="production")
    assert rollback_m["kind"] == "Deployment"
    assert (
        rollback_m["spec"]["template"]["spec"]["containers"][0]["image"]
        == "payment-service:previous-stable"
    )
    assert "apiVersion: apps/v1" in gen.to_yaml(rollback_m)

    scale_m = gen.generate_manifest("auth-service", "scale_up", replicas=5)
    assert scale_m["spec"]["replicas"] == 5

    res_m = gen.generate_manifest("payment-service", "adjust_resources", memory_limit="2Gi")
    assert (
        res_m["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] == "2Gi"
    )

    with pytest.raises(ValueError, match="Unknown remediation_type"):
        gen.generate_manifest("payment-service", "invalid_action")

    # Valid remediation passes OPA Gatekeeper
    eval_ok = validator.validate_remediation(
        scale_m, pdb_min_available=3, predicted_blast_radius=2, max_allowed_blast_radius=3
    )
    assert eval_ok.allowed is True
    assert eval_ok.violations == []

    # Violating PodDisruptionBudget (replicas=1 < min_available=3) is denied
    bad_scale = gen.generate_manifest("auth-service", "scale_up", replicas=1)
    eval_pdb = validator.validate_remediation(bad_scale, pdb_min_available=3)
    assert eval_pdb.allowed is False
    assert any("PodDisruptionBudget" in v for v in eval_pdb.violations)

    # Violating blast radius is denied
    eval_blast = validator.validate_remediation(
        rollback_m, predicted_blast_radius=5, max_allowed_blast_radius=3
    )
    assert eval_blast.allowed is False
    assert any("blast radius" in v for v in eval_blast.violations)

    # Unpinned ':latest' tag is denied
    latest_m = gen.generate_manifest("payment-service", "rollback")
    latest_m["spec"]["template"]["spec"]["containers"][0]["image"] = "payment-service:latest"
    eval_latest = validator.validate_remediation(latest_m)
    assert eval_latest.allowed is False
    assert any(":latest" in v for v in eval_latest.violations)


@pytest.mark.unit
def test_chaos_mesh_harness_fault_injection_and_shadow_recovery() -> None:
    """Verify ChaosMeshHarness injects StressChaos/NetworkChaos/PodChaos into shadow only."""
    harness = ChaosMeshHarness()

    # Refuses to inject faults into production namespace
    with pytest.raises(ValueError, match="Refusing to inject Chaos Mesh fault into production"):
        harness.inject_predicted_fault("production", "payment-service", "memory_leak")

    stress_crd = harness.inject_predicted_fault(
        "shadow", "payment-service", "memory_leak", duration_seconds=180
    )
    assert stress_crd["kind"] == "StressChaos"
    assert stress_crd["spec"]["stressors"]["memory"]["size"] == "256MB"

    net_crd = harness.inject_predicted_fault("canary", "checkout-service", "network_delay")
    assert net_crd["kind"] == "NetworkChaos"

    pod_crd = harness.inject_predicted_fault("shadow", "inventory-service", "pod_kill")
    assert pod_crd["kind"] == "PodChaos"

    with pytest.raises(ValueError, match="Unknown fault_type"):
        harness.inject_predicted_fault("shadow", "payment-service", "disk_burn")

    gen = RemediationGenerator()
    manifest = gen.generate_manifest("payment-service", "adjust_resources", namespace="shadow")
    patched = harness.apply_remediation_to_shadow(manifest, namespace="shadow")
    assert patched["metadata"]["name"] == "payment-service"

    metrics = harness.measure_stabilization_metrics(
        "shadow",
        "payment-service",
        post_remediation_error_rates=[0.05, 0.02, 0.005, 0.001],
        step_seconds=15.0,
    )
    assert metrics["slo_restored"] is True
    assert metrics["ttr_seconds"] == pytest.approx(45.0)
    assert metrics["blast_radius"] == 2
