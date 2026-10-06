"""Croissant 1.1 JSON-LD generator for simulation datasets.

Conforms to Track T9: Hand-emitted Croissant 1.1 JSON-LD (stdlib only).
Standard specification: MLCommons Croissant 1.1 (http://mlcommons.org/croissant/).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ewm_engine.cards.models import DatasetCard

CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "cr": "http://mlcommons.org/croissant/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
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
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


def to_croissant_dataset(card: DatasetCard) -> dict[str, Any]:
    """Emit standard MLCommons Croissant 1.1 JSON-LD metadata for a DatasetCard.

    Args:
        card: Source DatasetCard describing simulation trajectories.

    Returns:
        A dictionary conforming to Croissant 1.1 JSON-LD specification.
    """
    fields: list[dict[str, Any]] = [
        {
            "@type": "cr:Field",
            "@id": "steps/sample_id",
            "name": "sample_id",
            "description": "Monte Carlo rollout sample identifier",
            "dataType": "sc:Integer",
        },
        {
            "@type": "cr:Field",
            "@id": "steps/step",
            "name": "step",
            "description": "Discrete simulation time step index",
            "dataType": "sc:Integer",
        },
        {
            "@type": "cr:Field",
            "@id": "steps/timestamp",
            "name": "timestamp",
            "description": "Continuous simulation time",
            "dataType": "sc:Float",
        },
        {
            "@type": "cr:Field",
            "@id": "steps/state_hash",
            "name": "state_hash",
            "description": "Cryptographic SHA-256 fingerprint of state before transition",
            "dataType": "sc:Text",
        },
    ]

    for feat in card.features:
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"steps/{feat}",
                "name": feat,
                "description": f"Simulation feature or resource metric: {feat}",
                "dataType": "sc:Float",
            }
        )

    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": card.name,
        "description": card.description or f"Simulation trace dataset {card.name}",
        "version": card.version,
        "license": card.license,
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "trajectories-json",
                "name": "trajectories-json",
                "description": "Complete JSON-serialized simulation rollout trajectories",
                "contentUrl": "trajectories.json",
                "encodingFormat": "application/json",
                "sha256": card.artifact_fingerprint,
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "steps",
                "name": "steps",
                "description": "Discrete time step records across rollout trajectories",
                "field": fields,
            }
        ],
    }


def to_croissant_json(card: DatasetCard, indent: int = 2) -> str:
    """Serialize a DatasetCard as Croissant 1.1 JSON-LD string."""
    return json.dumps(to_croissant_dataset(card), indent=indent)
