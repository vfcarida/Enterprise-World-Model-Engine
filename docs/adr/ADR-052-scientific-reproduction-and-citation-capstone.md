# ADR-052: Scientific Reproduction, Persistent Identifiers, and Independent Adoption Capstone

## Context and Problem Statement

Scientific credibility compounds in sequence:
$$\text{Citable Mathematical Rigor} \longrightarrow \text{Reproduce in One Command} \longrightarrow \text{Independent Third-Party Replication} \longrightarrow \text{Widespread Adoption}$$

While Enterprise World Model Engine (EWM) established strong software quality mechanisms (JOSS draft, SLSA L2+ provenance, strict typing, and comprehensive test suites), research credibility required addressing three critical challenges:
1. **The "Rotting Documentation" Pitfall**: Simulation benchmark results frequently degrade into static screenshots or hard-coded figures that drift from underlying code revisions over time.
2. **Local Setup Friction**: Strangers and peer reviewers evaluating complex simulation software frequently encounter environment setup barriers (Python version mismatches, BLAS library divergence, missing system headers).
3. **Absence of Independent Validation**: Internal test passes, while essential, do not provide external credibility. Rigorous scientific validation requires independent third-party replication evaluated through peer-reviewed open science venues.

EWM's deterministic and cryptographically fingerprinted substrate enables a unique claim: *"Anyone, on any machine, reproduces our exact SHA-256 state fingerprints with one command."* To realize this potential, we must provide a hermetic reproduction layer, persistent archival identifiers, and formal independent replication packages.

## Considered Options

1. **Status Quo (Static Documentation & Manual CLI Instructions)**:
   - Provide manual `pip install` commands and static screenshots in `docs/results/`.
   - *Rejected*: Prone to documentation rot, dependency breakage across operating systems, and lacks cryptographic provenance.

2. **Local Script Only (`scripts/reproduce.py`)**:
   - Provide an automated Python script, but require reviewers to configure local Python virtual environments.
   - *Rejected*: Fails for reviewers on locked-down corporate machines, tablets, or environments without local Python 3.10–3.12 installations.

3. **Multi-Modal Hermetic Reproduction Layer + Persistent Identifiers + Independent Validation (Adopted)**:
   - **Single-Command Orchestrator (`make reproduce` / `scripts/reproduce.py`)**: Re-runs all seven benchmark cards, verifies WMIF reference conformance, checks Structure-of-Arrays (SoA) state tensor bit-identity, and emits a cryptographic run-manifest (`reproduction_manifest.json`).
   - **Rot-Free Vector Artifacts**: Re-generates dynamic SVG graphics (`ground_truth_overlay.svg`, `calibration_pit.svg`, `multiseed_error_bars.svg`) directly from seeded simulation traces.
   - **Zero-Setup Cloud & In-Browser Execution**:
     - **Binder**: Interactive JupyterLab notebook (`notebooks/reproduce.ipynb`) with `.binder/` environment specification.
     - **GitHub Codespaces**: Cloud development container (`.devcontainer/devcontainer.json`, `.devcontainer/Dockerfile`) with automated post-creation reproduction.
     - **Pinned Docker Container**: Hermetic OCI image (`Dockerfile`) verifying lockfile hashes.
   - **Persistent Archival Identifiers (Metadata & Readiness)**:
     - **Zenodo**: Metadata configured in `.zenodo.json` (*Concept and Version DOIs to be assigned upon release deposition*).
     - **Software Heritage**: SWHID documentation in `docs/scientific/swhid.md` (ISO/IEC 18670:2025; *public SWHID to be assigned upon snapshot ingestion*).
     - **Cross-Linking**: Synchronized metadata across `CITATION.cff`, `.zenodo.json`, and `reproduction_manifest.json` (with explicit `CERTIFIED_RELEASE` vs. `REPRODUCIBLE_DIRTY_WORKTREE` worktree certification).
   - **Peer-Reviewed Manuscripts**:
     - **JOSS Software Paper**: Focuses on software design, modularity, and statement of need (`paper/paper.md`, `paper/paper.bib`).
     - **arXiv cs.MS & cs.DC Companion**: Methods and benchmark paper (*Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty*) evaluated under the NeurIPS Datasets & Benchmarks checklist.
   - **External Replication Packages**:
     - **ML Reproducibility Challenge (MLRC 2026)**: Replication package for NeurIPS 2026 track.
     - **ReScience C**: Open-review replication submission dossier (ISSN 2430-3658).
     - **ACM Artifact Review Badges**: Systematic evidence pursuit across all three badges: *Artifacts Available*, *Artifacts Evaluated - Functional & Reusable*, and *Results Reproduced*.
     - **Pre-Registered Study Protocol**: Stage 1 Registered Report protocol defining hypotheses and stopping rules.
     - **Hugging Face Papers**: Discoverability and community replication feed configuration.

## Decision Outcome

Chosen option: **Option 3 (Multi-Modal Reproduction Layer + Persistent Identifiers + Independent Validation)**.

### Architectural Rules & Invariants

1. **Evidence Over Assertion (No Static Screenshots)**:
   - Every figure and metric in `docs/results/` must be generated by code from an explicit seed.
   - Vector SVG artifacts are committed and re-verified during `make reproduce`.

2. **Cryptographic Run-Manifest Provenance**:
   - `scripts/reproduce.py` collects host operating system, Python build, NumPy version, BLAS provider (`np.show_config()`), `uv.lock` SHA-256 hash, and exact state fingerprints.
   - Output manifest `reproduction_manifest.json` is synced to `docs/results/reproduction_manifest.json`.

3. **Zero-Setup Accessibility**:
   - A stranger must be able to click a Binder badge or Codespaces link and verify our exact published fingerprints in under 60 seconds with zero local setup.

4. **Honest Epistemics & Anti-Overclaiming**:
   - Papers claim only what is implemented, verified, and reproducible.
   - All citations are verified against primary literature databases; suspect or unverified identifiers are strictly excluded.
   - Forbidden superlative marketing claims (*"state-of-the-art"*, *"world's first"*) are prohibited.

5. **Persistent Identifier Integrity**:
   - Releases must preserve synchronization across `CITATION.cff`, `.zenodo.json`, Software Heritage SWHID, and the arXiv companion paper.

## Consequences

### Positive
- Independent reviewers can audit and reproduce every result with zero friction across local, cloud, and containerized platforms.
- Eliminates documentation rot by anchoring all gallery cards and visuals to automated run-manifests.
- Positions EWM Engine for premier scientific adoption through MLRC 2026, ReScience C, and ACM Artifact badges.

### Negative / Maintenance Overhead
- Updating benchmark figures requires re-running `scripts/reproduce.py` and syncing `reproduction_manifest.json`.
- Cloud environments (Binder, Codespaces) must be kept aligned with `uv.lock` and container specifications.
