"""Croissant JSON-LD metadata generator for EWM simulation outputs.

Conforms to the MLCommons Croissant 1.0 specification (arXiv:2403.19546, NeurIPS 2024 D&B).
Provides structured, machine-actionable dataset manifests describing Monte Carlo
simulation rollouts, trajectories, state variables, and deterministic SHA-256 fingerprints
for seamless discovery on HuggingFace, Kaggle, and OpenML.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ewm_engine.simulation.trajectory import SimulationResult

CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def generate_croissant_manifest(
    result: SimulationResult,
    dataset_name: str | None = None,
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
) -> dict[str, Any]:
    """Generate a Croissant 1.0 JSON-LD dataset manifest for a SimulationResult.

    Parameters:
        result: SimulationResult containing trajectories, provenance, and run metrics.
        dataset_name: Optional explicit name for the dataset.
        license_url: Machine-readable URL of the dataset license.

    Returns:
        Dictionary conforming to MLCommons Croissant 1.0 specification.
    """
    name = dataset_name or f"ewm_sim_{result.fingerprint[:12]}"
    scenario = result.scenario

    # Inspect first trajectory step to extract available metrics
    metric_keys: list[str] = []
    if result.trajectories and result.trajectories[0].steps:
        sample_step = result.trajectories[0].steps[0]
        metric_keys = sorted(sample_step.step_metrics.keys())

    # Build field definitions for the trajectory_steps RecordSet
    fields: list[dict[str, Any]] = [
        {
            "@type": "cr:Field",
            "name": "sample_id",
            "description": "Monte Carlo sample index (0-indexed).",
            "dataType": "sc:Integer",
        },
        {
            "@type": "cr:Field",
            "name": "step",
            "description": "Discrete temporal rollout step index.",
            "dataType": "sc:Integer",
        },
        {
            "@type": "cr:Field",
            "name": "timestamp",
            "description": "Logical simulation timestamp.",
            "dataType": "sc:Float",
        },
        {
            "@type": "cr:Field",
            "name": "state_hash",
            "description": "Bit-exact canonical SHA-256 fingerprint of the world state at this step.",
            "dataType": "sc:Text",
        },
    ]

    for m in metric_keys:
        fields.append(
            {
                "@type": "cr:Field",
                "name": f"metric_{m}",
                "description": f"Simulation step metric: {m}",
                "dataType": "sc:Float",
            }
        )

    record_sets = [
        {
            "@type": "cr:RecordSet",
            "name": "trajectories",
            "description": "Ensemble of simulated Monte Carlo trajectory steps.",
            "field": fields,
        }
    ]

    manifest: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "Dataset",
        "name": name,
        "description": (
            f"Enterprise World Model simulation rollouts for scenario '{scenario.name}' "
            f"({len(result.trajectories)} samples, horizon {scenario.horizon}, seed {scenario.seed}). "
            f"Provenance Fingerprint: {result.fingerprint}"
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": license_url,
        "datePublished": "2026-10-16",
        "creator": {
            "@type": "Organization",
            "name": "Enterprise World Model Engine",
            "url": "https://github.com/vfcarida/Enterprise-World-Model-Engine",
        },
        "keywords": [
            "world-model",
            "simulation",
            "digital-twin",
            "monte-carlo",
            "reproducible",
            "deterministic",
        ],
        "distribution": [
            {
                "@type": "cr:FileObject",
                "name": "simulation_result_json",
                "description": "Full serialized simulation result file.",
                "encodingFormat": "application/json",
                "contentUrl": f"file://simulation_result_{result.fingerprint[:12]}.json",
            }
        ],
        "recordSet": record_sets,
        "extra": {
            "fingerprint": result.fingerprint,
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "samples": len(result.trajectories),
            "horizon": scenario.horizon,
            "duration_seconds": result.run_metrics.simulation_duration_seconds,
        },
    }

    return manifest
