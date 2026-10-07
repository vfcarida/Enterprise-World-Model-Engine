# Security Policy

## Supported Versions

Only the latest minor release branch is supported with security updates.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the Enterprise World Model Engine (EWM Engine), please report it responsibly by emailing **maintainers@ewm-engine.org** rather than using public GitHub issues.

Please include:
1. A description of the vulnerability.
2. Steps or minimal code to reproduce the issue.
3. Potential impact or attack vectors.

We will acknowledge receipt within 48 hours and provide a remediation timeline.

## Security Principles & Invariants

EWM Engine adheres to strict software safety standards:
1. **No Arbitrary Code Execution**: World specifications, states, and configurations must never dynamically execute arbitrary code or evaluate untrusted Python expressions (`eval`, `exec`).
2. **Safe Serialization**: The engine uses strict, typed schema serialization (`Pydantic` and JSON). Python `pickle` or arbitrary unsafe YAML object deserialization is strictly prohibited.
3. **No Unsolicited Network Calls**: The core engine operates completely offline. It never transmits telemetry, logs, or state snapshots over the network without explicit user instrumentation.
4. **Zero Embedded Secrets**: Credentials, API tokens, or secrets must never be embedded in the codebase or logged by the simulation engine.

## Supply-Chain Security Posture

EWM Engine implements modern software supply-chain hardening in accordance with [ADR-034](docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md):

1. **Zero Static Publishing Secrets**: All package distributions to PyPI are published exclusively via OpenID Connect (OIDC) Trusted Publishing (`id-token: write`) without long-lived static API tokens.
2. **Cryptographic Build Provenance (SLSA Level 2+)**: Releases automatically generate Sigstore-signed in-toto build provenance (`actions/attest-build-provenance`). Artifacts can be independently verified via:
   ```bash
   gh attestation verify dist/ewm_engine-*.whl --repo vfcarida/Enterprise-World-Model-Engine
   ```
3. **Dual Software Bill of Materials (SBOM)**: Every release attaches both a logical CycloneDX JSON manifest (`cyclonedx-bom.json`) and a post-build Syft SPDX manifest (`syft-bom.spdx.json`) generated directly from compiled wheels to detect phantom binary dependencies.
4. **Vulnerability Gating (`pip-audit`)**: Pull requests are blocked on dependencies containing known CVEs. Nightly advisory scans run OSV-Scanner v2 and Anchore Grype.
5. **OpenSSF Scorecard & Best Practices**: The repository is audited weekly via the OpenSSF Scorecard targeting an aggregate score $\ge 7.5/10$ across Critical/High vectors (Token Permissions, Pinned Dependencies, Branch Protection).

