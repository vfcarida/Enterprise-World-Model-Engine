# Software Heritage Persistent Identifiers (SWHID)

> **Standard**: ISO/IEC 18670:2025 (Information technology â€” Software Heritage persistent identifiers)
> **Repository Origin**: `https://github.com/vfcarida/Enterprise-World-Model-Engine`
> **Public Archival Status**: *Not yet published in the Software Heritage public vault (SWHID to be assigned upon release ingestion; intrinsic Git-tree SWHIDs can be computed locally via `swh identify`)*

---

## 1. Overview of SWHID & ISO/IEC 18670:2025

Persistent, immutable identification of software artifacts is a fundamental prerequisite for computational reproducibility. While URLs and software forge repositories can be deleted, moved, or altered, **SoftWare Heritage Identifiers (SWHIDs)** provide content-addressed, cryptographic references to source code entities stored in the [Software Heritage Archive](https://www.softwareheritage.org/).

In 2025, the SWHID specification was formally adopted as **ISO/IEC 18670:2025**, recognizing SWHIDs as the global international standard for persistent software identification.

### Anatomy of an SWHID

An SWHID follows a standard cryptographic syntax:

```text
swh:<version>:<type>:<hash>[;<qualifiers>]
```

Where:
- `swh`: The canonical URI scheme prefix.
- `<version>`: Identifier schema version (`1` for Git-compatible SHA-1/Git Object hashes).
- `<type>`: The object type in the Merkle DAG:
  - `cnt` (content / blob): Individual source code file.
  - `dir` (directory): Directory tree mapping file names to blobs or subdirectories.
  - `rev` (revision / commit): Snapshot of a repository commit, including parent pointers and commit metadata.
  - `rel` (release / tag): Annotated release tag pointing to a specific revision.
  - `snp` (snapshot): Complete state of all references (branches, tags) of a repository visit.
- `<hash>`: The hexadecimal cryptographic hash of the object.
- `<qualifiers>`: Contextual metadata (e.g., `;origin=<repo_url>`, `;visit=<snapshot>`, `;lines=<start>-<end>`).

---

## 2. EWM Engine SWHID Structure (Template)

Once a release tag is ingested by Software Heritage, intrinsic identifiers follow these schema levels:

| Level | Object Type | Identifier Template | Purpose |
| :--- | :--- | :--- | :--- |
| **Release v1.5.0** | `rel` | `swh:1:rel:<git_tag_sha1>;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Tagged release snapshot for v1.5.0 (*to be assigned*) |
| **Commit Revision** | `rev` | `swh:1:rev:<git_commit_sha1>;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Exact Git commit revision of reproduction baseline |
| **Root Tree Directory** | `dir` | `swh:1:dir:<git_tree_sha1>;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Root file tree containing `src`, `tests`, and `conformance` |
| **WMIF Conformance** | `dir` | `swh:1:dir:<subdir_tree_sha1>;lines=1-250` | Standalone WMIF specification and conformance oracle |

---

## 3. Resolving and Archiving EWM Engine via SWH

### Resolving via the Web Resolver
Once ingested, any SWHID can be resolved directly in a browser via the Software Heritage public resolver:
```text
https://archive.softwareheritage.org/swh:1:rev:<git_commit_sha1>/
```

### Triggering Automated Ingestion (Save Code Now)
To archive new releases or commits, the repository is submitted to Software Heritage's "Save Code Now" webhook:
```bash
curl -X POST https://archive.softwareheritage.org/api/1/origin/save/git/url/https://github.com/vfcarida/Enterprise-World-Model-Engine/
```

### Checking Archival Status
To check ingestion status:
```bash
curl https://archive.softwareheritage.org/api/1/origin/save/git/url/https://github.com/vfcarida/Enterprise-World-Model-Engine/
```

---

## 4. Local Cryptographic Verification

You can compute and verify the exact SWHID of your local checkout using the official `swh-model` Python library:

```bash
# Install the official Software Heritage CLI
pip install swh.model

# Compute directory SWHID for current workspace
swh identify .
```

Output format:
```text
swh:1:dir:<computed_40_hex_sha1>
```

Additionally, `scripts/reproduce.py` computes a deterministic SHA-256 Merkle digest over all tracked repository files (`source_tree_sha256` in `reproduction_manifest.json`) to verify worktree integrity independently of external archival services.

---

## 5. Cross-Linking Architecture

Upon external publication and archival, EWM Engine links its artifacts across four metadata surfaces:

```
                  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
                  â”‚       Companion Manuscript Draft       â”‚
                  â”‚         (paper/arxiv/paper.md)         â”‚
                  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                                      â”‚
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
            â–¼                         â–¼                         â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚     CITATION.cff      â”‚ â”‚     .zenodo.json      â”‚ â”‚    ISO/IEC 18670      â”‚
â”‚  (Machine-Readable    â”‚ â”‚  (Zenodo Metadata     â”‚ â”‚   Software Heritage   â”‚
â”‚   Repo Citation)      â”‚ â”‚   Configuration)      â”‚ â”‚      (SWHID)          â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```
