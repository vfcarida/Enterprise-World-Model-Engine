# Security Policy

## Supported Versions

Only the latest minor release series is actively maintained with security patches.

| Version | Supported          | Security Maintenance Status |
| :------ | :----------------: | :-------------------------- |
| `1.5.x` | :white_check_mark: | Active support & patches    |
| `< 1.5` | :x:                | End of Life (upgrade to `1.5.x`) |

## Reporting a Vulnerability

We treat all vulnerabilities in the Enterprise World Model Engine (EWM Engine) with the highest severity. **Please do not report security vulnerabilities via public GitHub issues, discussions, or pull requests.**

### Preferred Channel: GitHub Private Vulnerability Reporting
Please submit reports directly via **[GitHub Private Vulnerability Reporting](https://github.com/vfcarida/Enterprise-World-Model-Engine/security/advisories/new)**. This creates an encrypted, private advisory workspace directly with the maintainers and enables coordinated CVE assignment.

### Alternative Channel: Encrypted Email
If you cannot use GitHub's advisory system, submit your report via email directly to the project lead maintainer:
- **Contact:** `vfcarida@gmail.com`
- **Subject:** `[SECURITY] EWM Engine Vulnerability Report`

Please include in your report:
1. **Description**: Summary of the vulnerability, potential attack vector, and affected components (`ewm_engine.*`).
2. **Reproduction**: Minimal, reproducible proof-of-concept (Python script, serialized YAML/JSON world, or CLI command).
3. **Impact**: Assessment of severity, confidentiality/integrity/availability impact, and whether exploitation requires local access or untrusted input.
4. **Suggested Fix**: If you have a remediation patch or mitigation proposal.

## Response SLA & Disclosure Timeline

We adhere to a structured, transparent response SLA for all reported security vulnerabilities:

| Phase | Target SLA | Description |
| :--- | :---: | :--- |
| **Initial Acknowledgment** | $\le 48$ hours | Maintainer confirms receipt and begins initial impact assessment. |
| **Triage & Reproducibility** | $\le 7$ calendar days | Maintainer validates findings, sets CVSS severity, and confirms feasibility. |
| **Fix Development & Testing** | $\le 30$ calendar days | Secure patch prepared in private fork; verified against regression gates. |
| **Coordinated Disclosure** | Scheduled with reporter | Patch released in a point release (e.g., `1.5.1`); CVE published via GitHub Advisories. |

## RFC 9116 Security Metadata
In compliance with [RFC 9116](https://www.rfc-editor.org/rfc/rfc9116), security metadata is published at [`/.well-known/security.txt`](.well-known/security.txt).

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

