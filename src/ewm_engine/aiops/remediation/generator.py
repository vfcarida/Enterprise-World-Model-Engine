"""AI-driven Kubernetes remediation manifest generator for the AIOps World Model.

Synthesizes declarative Kubernetes `apps/v1` `Deployment` manifests for automated
incident remediation:
- `rollback`: reverts container image to `:previous-stable`
- `scale_up`: increases replica count to absorb traffic/CPU spikes
- `adjust_resources`: updates container CPU and memory requests/limits for memory leaks
"""

from __future__ import annotations

from typing import Any

import yaml


class RemediationGenerator:
    """Generates Kubernetes remediation manifests for predicted microservice failures."""

    def generate_manifest(
        self,
        root_cause_service: str,
        remediation_type: str,  # 'rollback', 'scale_up', 'adjust_resources'
        namespace: str = "production",
        replicas: int = 5,
        memory_limit: str = "1Gi",
        cpu_limit: str = "1000m",
    ) -> dict[str, Any]:
        """Generate a Kubernetes `Deployment` manifest for remediation.

        Args:
            root_cause_service: Name of the root-cause microservice Deployment.
            remediation_type: One of `'rollback'`, `'scale_up'`, or `'adjust_resources'`.
            namespace: Target Kubernetes namespace (default `'production'`).
            replicas: Target replica count for `'scale_up'` (default `5`).
            memory_limit: Target memory limit for `'adjust_resources'` (default `'1Gi'`).
            cpu_limit: Target CPU limit for `'adjust_resources'` (default `'1000m'`).

        Returns:
            Dictionary representing a valid Kubernetes `apps/v1` `Deployment` manifest.
        """
        if not root_cause_service or not root_cause_service.strip():
            raise ValueError("root_cause_service must be a non-empty string")

        if remediation_type == "rollback":
            manifest: dict[str, Any] = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": root_cause_service,
                    "namespace": namespace,
                    "annotations": {
                        "ewm.aiops/remediation-type": "rollback",
                    },
                },
                "spec": {
                    "replicas": 3,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": root_cause_service,
                                    "image": f"{root_cause_service}:previous-stable",
                                }
                            ]
                        }
                    },
                },
            }
        elif remediation_type == "scale_up":
            manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": root_cause_service,
                    "namespace": namespace,
                    "annotations": {
                        "ewm.aiops/remediation-type": "scale_up",
                    },
                },
                "spec": {
                    "replicas": replicas,
                },
            }
        elif remediation_type == "adjust_resources":
            manifest = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": root_cause_service,
                    "namespace": namespace,
                    "annotations": {
                        "ewm.aiops/remediation-type": "adjust_resources",
                    },
                },
                "spec": {
                    "replicas": 3,
                    "template": {
                        "spec": {
                            "containers": [
                                {
                                    "name": root_cause_service,
                                    "image": f"{root_cause_service}:stable",
                                    "resources": {
                                        "requests": {
                                            "memory": "512Mi",
                                            "cpu": "500m",
                                        },
                                        "limits": {
                                            "memory": memory_limit,
                                            "cpu": cpu_limit,
                                        },
                                    },
                                }
                            ]
                        }
                    },
                },
            }
        else:
            raise ValueError(f"Unknown remediation_type: {remediation_type}")

        return manifest

    def recommend_remediation(
        self,
        root_cause_service: str,
        anomaly_type: str,
        namespace: str = "production",
    ) -> tuple[str, dict[str, Any]]:
        """Select remediation strategy from incident anomaly classification and build manifest."""
        mapping = {
            "bad_deployment": "rollback",
            "config_change": "rollback",
            "traffic_spike": "scale_up",
            "cpu_saturation": "scale_up",
            "memory_leak": "adjust_resources",
        }
        remediation_type = mapping.get(anomaly_type, "rollback")
        manifest = self.generate_manifest(
            root_cause_service=root_cause_service,
            remediation_type=remediation_type,
            namespace=namespace,
        )
        return remediation_type, manifest

    @staticmethod
    def to_yaml(manifest: dict[str, Any]) -> str:
        """Serialize a Kubernetes remediation manifest dictionary to YAML."""
        dumped: str = yaml.safe_dump(manifest, sort_keys=False)
        return dumped
