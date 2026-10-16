"""World Model Interchange Format (WMIF) self-describing container standard (ADR-051).

Architectural Anchors:
- FMI 3.0 / Reference-FMUs model: Self-describing PKZIP container (.wmif) encapsulating
  modelDescription.json, the deterministic model payload (WSL / WorldSpec), and
  reference simulation results keyed by seed and canonical SHA-256 fingerprint.
- SBML Levels and Packages: Modular, independently versioned growth (Level 1 core
  deterministic kernel, optional packages: PkgCausal, PkgCosim, PkgDecisions, PkgStochastic)
  supporting fail-closed requirement gates.
- JSON Schema Draft 2020-12 / OpenAPI 3.1: Manifest schema for universal validation.
- Conformance Oracle: Exact SHA-256 fingerprint match against reference results.

Status:
  Proposed / Experimental (v1.0.0-proposed).
"""

from __future__ import annotations

import json
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.world import World
from ewm_engine.exceptions import SerializationError, SimulationConfigurationError
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.serialization.wsl import compile_wsl, export_wsl, parse_wsl_yaml

if TYPE_CHECKING:
    from ewm_engine.simulation.scenario import Scenario
    from ewm_engine.simulation.trajectory import SimulationResult


class WMIFPackage(BaseModel):
    """Specification of an optional or required SBML-style extension package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Package name (e.g. 'PkgCausal', 'PkgCosim').")
    version: str = Field(default="1.0.0", description="Semantic version of the package.")
    required: bool = Field(
        default=False,
        description="Whether supporting this package is mandatory to run the world.",
    )
    description: str = Field(
        default="", description="Narrative description of the package capability."
    )


# Known standard packages
PKG_CAUSAL = WMIFPackage(
    name="PkgCausal", version="1.0.0", required=False, description="Causal DAGs and SCM equations."
)
PKG_COSIM = WMIFPackage(
    name="PkgCosim", version="1.0.0", required=False, description="Co-simulation interface ports."
)
PKG_DECISIONS = WMIFPackage(
    name="PkgDecisions",
    version="1.0.0",
    required=False,
    description="Decision scenario discovery spaces.",
)
PKG_STOCHASTIC = WMIFPackage(
    name="PkgStochastic", version="1.0.0", required=False, description="Counter-based PRNG streams."
)

SUPPORTED_PACKAGES: set[str] = {
    PKG_CAUSAL.name,
    PKG_COSIM.name,
    PKG_DECISIONS.name,
    PKG_STOCHASTIC.name,
}


class WMIFReferenceResult(BaseModel):
    """Reference execution golden record keyed by seed and fingerprint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scenario_id: str
    seed: int
    samples: int
    horizon: int
    fingerprint: str
    per_branch_fingerprints: list[str] = Field(default_factory=list)
    summary_metrics: dict[str, float] = Field(default_factory=dict)
    result_path: str = ""


class WMIFManifest(BaseModel):
    """Root metadata manifest (modelDescription.json) for a WMIF container."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Draft 2020-12 schema version.")
    format: str = Field(default="WMIF/1.0", description="Container format identifier.")
    level: int = Field(default=1, description="SBML-style specification Level.")
    status: str = Field(
        default="Proposed/Experimental",
        description="Standardization status (e.g., 'Proposed/Experimental').",
    )
    model_id: str = Field(description="Unique world model identifier.")
    model_name: str = Field(description="Human-readable model name.")
    description: str = Field(
        default="", description="Narrative description of the enterprise model."
    )
    author: str = Field(default="Enterprise World Model Engine", description="Model creator.")
    created_at: str = Field(default="", description="ISO-8601 timestamp of package creation.")
    model_file: str = Field(
        default="model.wsl", description="Relative path to model definition payload."
    )
    model_fingerprint: str = Field(
        default="", description="SHA-256 fingerprint of the model definition."
    )
    packages: list[WMIFPackage] = Field(
        default_factory=list, description="Active SBML-style extension packages."
    )
    reference_results: list[WMIFReferenceResult] = Field(
        default_factory=list,
        description="Golden reference simulation runs for conformance verification.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Creation provenance metadata."
    )


def export_wmif(
    world: World,
    output_path: str | Path,
    *,
    scenarios: Sequence[Scenario] | None = None,
    packages: Sequence[WMIFPackage] | None = None,
    model_name: str | None = None,
    description: str = "",
    author: str = "EWM Engine",
) -> Path:
    """Export an Enterprise World Model into a self-describing WMIF (.wmif) container.

    Parameters:
        world: The World model instance to package.
        output_path: Destination path for the .wmif file.
        scenarios: Optional reference scenarios to execute and include as golden conformance results.
        packages: Optional list of SBML-style extension packages.
        model_name: Optional human-readable model name.
        description: High-level narrative description.
        author: Author or organization name.

    Returns:
        Path to the created .wmif container.
    """
    out_path = Path(output_path)
    if not out_path.name.endswith(".wmif") and not out_path.name.endswith(".zip"):
        out_path = out_path.with_suffix(".wmif")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Export model definition to WSL document string
    wsl_doc = export_wsl(world)
    if model_name:
        wsl_doc = wsl_doc.model_copy(
            update={
                "metadata": wsl_doc.metadata.model_copy(
                    update={"name": model_name, "description": description}
                )
            }
        )
    wsl_yaml_str = wsl_doc.to_yaml()
    model_hash = canonical_sha256(wsl_doc.model_dump())

    # 2. Run reference scenarios if provided to capture golden fingerprints
    ref_results: list[WMIFReferenceResult] = []
    result_files: dict[str, str] = {}

    if scenarios:
        for scn in scenarios:
            res: SimulationResult = world.simulate(scenario=scn)
            branch_fps = [t.final_state.fingerprint for t in res.trajectories]
            rel_res_path = f"reference_results/{scn.scenario_id}_{res.fingerprint[:12]}.json"

            # Compute summary stats
            metrics_summary: dict[str, float] = {}
            if res.trajectories and res.trajectories[0].steps:
                last_step = res.trajectories[0].steps[-1]
                for k in list(last_step.step_metrics.keys())[:5]:
                    dist = res.get_metric_distribution(k)
                    metrics_summary[f"{k}_mean"] = round(dist.mean, 4)

            ref_rec = WMIFReferenceResult(
                scenario_id=scn.scenario_id,
                seed=scn.seed,
                samples=len(res.trajectories),
                horizon=scn.horizon,
                fingerprint=res.fingerprint,
                per_branch_fingerprints=branch_fps,
                summary_metrics=metrics_summary,
                result_path=rel_res_path,
            )
            ref_results.append(ref_rec)

            # Store result payload
            result_payload = {
                "fingerprint": res.fingerprint,
                "scenario": scn.model_dump(),
                "trajectories_count": len(res.trajectories),
                "per_branch_fingerprints": branch_fps,
                "summary_metrics": metrics_summary,
            }
            result_files[rel_res_path] = json.dumps(result_payload, indent=2, sort_keys=True)

    # 3. Build modelDescription manifest
    manifest = WMIFManifest(
        schema_version="1.0.0",
        format="WMIF/1.0",
        level=1,
        status="Proposed/Experimental",
        model_id=wsl_doc.metadata.id,
        model_name=model_name or wsl_doc.metadata.name,
        description=description or wsl_doc.metadata.description,
        author=author,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model_file="model.wsl",
        model_fingerprint=model_hash,
        packages=list(packages) if packages else [],
        reference_results=ref_results,
        provenance=Provenance().model_dump(),
    )

    # 4. Pack into ZIP archive
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr("modelDescription.json", manifest.model_dump_json(indent=2))
        # Write model payload
        zf.writestr("model.wsl", wsl_yaml_str)
        # Write reference results
        for r_path, r_content in result_files.items():
            zf.writestr(r_path, r_content)

    return out_path


def import_wmif(wmif_path: str | Path) -> tuple[World, WMIFManifest]:
    """Import and validate an Enterprise World Model from a WMIF (.wmif) container.

    Security & Validation:
    - Strictly checks against ZIP path traversal vulnerabilities (Zip Slip).
    - Validates modelDescription.json against WMIFManifest schema.
    - Evaluates required packages and fails closed if any required package is unsupported.
    - Reconstitutes the executable World model without code execution.

    Parameters:
        wmif_path: Path to the .wmif container file.

    Returns:
        Tuple of (reconstituted World model, WMIFManifest).
    """
    path = Path(wmif_path)
    if not path.exists():
        raise SerializationError(f"WMIF container file not found: {path}")

    with zipfile.ZipFile(path, mode="r") as zf:
        # Security: verify no path traversal
        for member in zf.namelist():
            if member.startswith("/") or ".." in member:
                raise SerializationError(f"Insecure archive entry rejected: {member}")

        # 1. Read and parse manifest
        if "modelDescription.json" not in zf.namelist():
            raise SerializationError("Invalid WMIF container: missing 'modelDescription.json'")

        manifest_data = json.loads(zf.read("modelDescription.json").decode("utf-8"))
        manifest = WMIFManifest.model_validate(manifest_data)

        # 2. Check required packages (SBML Level/Package fail-closed rule)
        for pkg in manifest.packages:
            if pkg.required and pkg.name not in SUPPORTED_PACKAGES:
                raise SimulationConfigurationError(
                    f"WMIF model requires unsupported package '{pkg.name}' (v{pkg.version}). "
                    f"Supported packages: {sorted(SUPPORTED_PACKAGES)}"
                )

        # 3. Read model definition
        model_file = manifest.model_file
        if model_file not in zf.namelist():
            raise SerializationError(
                f"Model payload '{model_file}' declared in manifest but missing from container"
            )

        model_yaml_str = zf.read(model_file).decode("utf-8")
        wsl_doc = parse_wsl_yaml(model_yaml_str)

        # 4. Compile into World
        world = compile_wsl(wsl_doc)

    return world, manifest


def verify_wmif_conformance(wmif_path: str | Path) -> dict[str, Any]:
    """Verify that a WMIF container reproduces its embedded reference results with bit-exact fidelity.

    Executes each scenario in reference_results and asserts that:
    1. The scenario reproduces the exact golden per-branch SHA-256 fingerprints.
    2. The aggregate simulation fingerprint matches bit-for-bit.

    Returns:
        Dictionary detailing conformance verification status per reference scenario.
    """
    world, manifest = import_wmif(wmif_path)
    if not manifest.reference_results:
        return {"conformance": "no_reference_results", "passed": True, "scenarios": []}

    path = Path(wmif_path)
    all_passed = True
    details: list[dict[str, Any]] = []

    with zipfile.ZipFile(path, mode="r") as zf:
        for ref in manifest.reference_results:
            from ewm_engine.simulation.scenario import Scenario

            # Load scenario from result file if available, or construct from ref metadata
            scn: Scenario | None = None
            if ref.result_path and ref.result_path in zf.namelist():
                try:
                    raw_res = json.loads(zf.read(ref.result_path).decode("utf-8"))
                    if isinstance(raw_res, dict) and "scenario" in raw_res:
                        scn = Scenario.model_validate(raw_res["scenario"])
                except Exception:
                    scn = None

            if scn is None:
                scn = Scenario(
                    scenario_id=ref.scenario_id,
                    horizon=ref.horizon,
                    samples=ref.samples,
                    seed=ref.seed,
                )
            sim_res = world.simulate(scenario=scn)
            actual_branch_fps = [t.final_state.fingerprint for t in sim_res.trajectories]

            match_aggregate = sim_res.fingerprint == ref.fingerprint
            match_branches = (
                actual_branch_fps == ref.per_branch_fingerprints
                if ref.per_branch_fingerprints
                else True
            )

            passed = match_aggregate and match_branches
            if not passed:
                all_passed = False

            details.append(
                {
                    "scenario_id": ref.scenario_id,
                    "expected_fingerprint": ref.fingerprint,
                    "actual_fingerprint": sim_res.fingerprint,
                    "passed": passed,
                    "branch_count": len(actual_branch_fps),
                }
            )

    return {
        "conformance": "pass" if all_passed else "fail",
        "passed": all_passed,
        "scenarios": details,
    }
