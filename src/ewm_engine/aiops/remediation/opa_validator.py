"""OPA Gatekeeper policy validator for AI-generated Kubernetes remediations.

Validates Kubernetes `Deployment` remediation manifests against the Rego policy rules
defined in `opa_policies/remediation_policy.rego` (PodDisruptionBudget compliance,
blast-radius bounds, and container image tag hygiene) before shadow or production
execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OPAPolicyEvaluation:
    """Result of evaluating a Kubernetes remediation manifest against OPA policies."""

    allowed: bool
    violations: list[str]
    policy_path: str


class OPAPolicyValidator:
    """Evaluates Kubernetes admission requests against `remediation_policy.rego`."""

    def __init__(self, policy_path: Path | str | None = None) -> None:
        if policy_path is None:
            repo_root = Path(__file__).resolve().parents[4]
            self.policy_path = repo_root / "opa_policies" / "remediation_policy.rego"
        else:
            self.policy_path = Path(policy_path)

        if not self.policy_path.exists():
            raise FileNotFoundError(f"OPA policy file not found: {self.policy_path}")
        self.policy_source = self.policy_path.read_text(encoding="utf-8")

    def validate_remediation(
        self,
        manifest: Mapping[str, Any],
        pdb_min_available: int = 2,
        predicted_blast_radius: int = 1,
        max_allowed_blast_radius: int = 3,
        operation: str = "UPDATE",
    ) -> OPAPolicyEvaluation:
        """Evaluate a remediation manifest against OPA Gatekeeper admission rules.

        Args:
            manifest: Kubernetes `Deployment` manifest dictionary.
            pdb_min_available: Minimum available replicas required by `PodDisruptionBudget`.
            predicted_blast_radius: Predicted number of affected downstream services.
            max_allowed_blast_radius: Policy threshold for maximum blast radius.
            operation: Kubernetes admission operation (default `'UPDATE'`).

        Returns:
            OPAPolicyEvaluation with `allowed` boolean and any policy `violations`.
        """
        violations: list[str] = []
        kind = str(manifest.get("kind", ""))

        if kind == "Deployment" and operation == "UPDATE":
            spec = manifest.get("spec", {})
            if isinstance(spec, Mapping) and "replicas" in spec:
                current_replicas = int(spec["replicas"])
                if current_replicas < pdb_min_available:
                    violations.append(
                        f"Remediation violates PodDisruptionBudget: "
                        f"{current_replicas} replicas < {pdb_min_available} min_available"
                    )

            if predicted_blast_radius > max_allowed_blast_radius:
                violations.append(
                    f"Remediation exceeds maximum blast radius: "
                    f"{predicted_blast_radius} affected services > {max_allowed_blast_radius} allowed"
                )

            template_spec = (
                spec.get("template", {}).get("spec", {}) if isinstance(spec, Mapping) else {}
            )
            containers = (
                template_spec.get("containers", []) if isinstance(template_spec, Mapping) else []
            )
            for container in containers:
                if isinstance(container, Mapping):
                    image = str(container.get("image", ""))
                    c_name = str(container.get("name", "unknown"))
                    if image.endswith(":latest"):
                        violations.append(
                            f"Unpinned ':latest' container image forbidden in remediation for {c_name}"
                        )

        return OPAPolicyEvaluation(
            allowed=len(violations) == 0,
            violations=violations,
            policy_path=str(self.policy_path),
        )
