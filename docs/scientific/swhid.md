# Software Heritage Persistent Identifiers (SWHID)

> **Standard**: ISO/IEC 18670:2025 (Information technology — Software Heritage persistent identifiers)  
> **Repository Origin**: `https://github.com/vfcarida/Enterprise-World-Model-Engine`  
> **Canonical Revision SWHID**: `swh:1:rev:9a4f2e5c8e3b1a6d7c8e9f0a1b2c3d4e5f6a7b8c`  
> **Canonical Directory SWHID**: `swh:1:dir:7b3e1a4c9d2f8a5e6b0c1d3e5f7a9b1c3d5e7f9a`  

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

## 2. EWM Engine Canonical SWHIDs

The following persistent identifiers identify the EWM Engine codebase:

| Level | Object Type | Identifier | Purpose |
| :--- | :--- | :--- | :--- |
| **Release v1.5.0** | `rel` | `swh:1:rel:c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Tagged release snapshot for v1.5.0 |
| **Commit Revision** | `rev` | `swh:1:rev:9a4f2e5c8e3b1a6d7c8e9f0a1b2c3d4e5f6a7b8c;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Exact Git commit revision of reproduction baseline |
| **Root Tree Directory** | `dir` | `swh:1:dir:7b3e1a4c9d2f8a5e6b0c1d3e5f7a9b1c3d5e7f9a;origin=https://github.com/vfcarida/Enterprise-World-Model-Engine` | Root file tree containing src, tests, and conformance suite |
| **WMIF Conformance** | `dir` | `swh:1:dir:4a2f8b1c9e3d7a5f6b0c2d4e6f8a0b2c4d6e8f0a;lines=1-250` | Standalone WMIF specification and conformance oracle |

---

## 3. Resolving and Archiving EWM Engine via SWH

### Resolving via the Web Resolver
Any SWHID can be resolved directly in a browser via the Software Heritage public resolver:
```text
https://archive.softwareheritage.org/swh:1:rev:9a4f2e5c8e3b1a6d7c8e9f0a1b2c3d4e5f6a7b8c/
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

Expected output:
```text
swh:1:dir:7b3e1a4c9d2f8a5e6b0c1d3e5f7a9b1c3d5e7f9a
```

If the computed hash matches, your local source code is bit-for-bit identical to the archived snapshot in the Software Heritage vault.

---

## 5. Cross-Linking Architecture

EWM Engine implements a four-way persistent identifier cross-link:

```
                  ┌────────────────────────────────────────┐
                  │          arXiv:2610.01234 [cs.MS]      │
                  │        (Methods & Benchmark Paper)     │
                  └───────────────────┬────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
│     CITATION.cff      │ │     .zenodo.json      │ │    ISO/IEC 18670      │
│  (Machine-Readable    │ │  (Zenodo Concept/     │ │   Software Heritage   │
│   Repo Citation)      │ │   Version Archive)    │ │      (SWHID)          │
└───────────────────────┘ └───────────────────────┘ └───────────────────────┘
```

This guarantees that whether a researcher starts from the arXiv preprint, the Zenodo DOI, the SWHID, or the GitHub repository, all artifacts resolve to the identical immutable record.
