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
