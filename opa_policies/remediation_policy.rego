package kubernetes.admission

import data.kubernetes.podsdisruptionbudgets

deny[msg] {
    input.request.kind.kind == "Deployment"
    input.request.operation == "UPDATE"

    deployment := input.request.object
    pdb := podsdisruptionbudgets[deployment.metadata.namespace][deployment.metadata.name]

    # Check if scaling down violates PodDisruptionBudget
    current_replicas := deployment.spec.replicas
    pdb_min_available := pdb.spec.minAvailable

    current_replicas < pdb_min_available

    msg := sprintf("Remediation violates PodDisruptionBudget: %v replicas < %v min_available", [current_replicas, pdb_min_available])
}

deny[msg] {
    input.request.kind.kind == "Deployment"
    input.request.operation == "UPDATE"

    predicted_blast := input.context.predicted_blast_radius
    max_blast := input.context.max_allowed_blast_radius

    predicted_blast > max_blast

    msg := sprintf("Remediation exceeds maximum blast radius: %v affected services > %v allowed", [predicted_blast, max_blast])
}

deny[msg] {
    input.request.kind.kind == "Deployment"
    input.request.operation == "UPDATE"

    deployment := input.request.object
    container := deployment.spec.template.spec.containers[_]
    endswith(container.image, ":latest")

    msg := sprintf("Unpinned ':latest' container image forbidden in remediation for %v", [container.name])
}
