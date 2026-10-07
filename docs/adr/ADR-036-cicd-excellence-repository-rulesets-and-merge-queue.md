# ADR-036: CI/CD Excellence, Repository Rulesets & Merge Queue Governance

- **Status**: Accepted
- **Date**: 2026-10-07
- **Authors**: Vinicius Caridá <vfcarida@gmail.com>
- **Deciders**: Enterprise World Model Engine Maintainers
- **Consulted**: Security Engineers, CI/CD Maintainers, OpenSSF Working Group
- **Informed**: All Contributors and Ecosystem Consumers

---

## 1. Context & Problem Statement

The Enterprise World Model Engine (`ewm-engine`) maintains a comprehensive continuous integration and delivery architecture comprising 19 required automated quality gates spanning linting, formatting, strict static typing, packaging integrity, supply-chain provenance, numerical reproducibility, differential fuzzing, mutation testing, and deterministic memory ceilings.

However, as CI pipelines expand in scope and concurrency, modern GitHub Actions security research (Zizmor, StepSecurity, OpenSSF Scorecard, and Google SLSA) identifies critical supply-chain attack surfaces and operational bottlenecks:

1. **Mutable Action References**: Referencing external actions by mutable git tags (e.g., `@v4`, `@v5`, `@main`) introduces arbitrary code execution vulnerabilities if upstream repository tags are hijacked, repointed, or deleted.
2. **Workflow Template Injections**: Interpolating untrusted expressions (e.g., `${{ github.base_ref }}`, `${{ github.head_ref }}`) directly inside shell `run:` blocks exposes runners to arbitrary shell injection.
3. **Overly Broad Runner Permissions**: Default runner tokens with write access to repository contents, issues, or pull requests violate the principle of least privilege.
4. **Cache Poisoning Attacks**: Allowing untrusted pull requests to overwrite GitHub Actions dependency caches can compromise subsequent CI runs and build outputs.
5. **Branch Protection Bottlenecks & Race Conditions**: Traditional GitHub branch protections do not scale to high-volume pull request throughput, resulting in merge races, broken trunk commits, and manual rebase churn.
6. **Ecosystem Horizon Divergence**: Without explicit Scientific Python Ecosystem Coordination (SPEC 0) gating and proactive free-threaded CPython evaluation (PEP 703, Python 3.13t/3.14t), dependency drift and GIL assumptions silently accumulate.

---

## 2. Decision Drivers

- **Zero-Trust CI Security**: Enforce 100% immutable action pinning, least-privilege token permissions, template injection elimination, and automated static security auditing via `zizmor`.
- **Reproducible & Poison-Resistant Caching**: Decouple cache restoration from cache mutation so pull requests operate strictly in restore-only mode, with cache writes restricted to verified pushes on `main`.
- **DRY & Maintainable Pipeline Structure**: Unify repetitive Python and `uv` installation steps into a single reusable composite action (`.github/actions/setup-python-uv`).
- **Atomic & Linear Merge Automation**: Replace legacy branch protection rules with GitHub Repository Rulesets paired with GitHub Merge Queue (`merge_group`), eliminating trunk breaks.
- **Future-Proof Multi-OS & Free-Threaded Matrix**: Validate cross-platform stability across Linux, macOS, and Windows, while introducing advisory free-threaded (`3.13t`) and prerelease (`3.14-dev`) canaries alongside SPEC 0 enforcement.

---

## 3. Considered Options

1. **Option 1: Status Quo (Legacy Branch Protection, Mutable Tags, Independent Actions)**
   - *Pros*: Zero migration effort.
   - *Cons*: Fails OpenSSF Scorecard requirements; vulnerable to tag mutability and cache poisoning; duplicate boilerplate across 19 jobs; frequent merge conflicts on `main`.
2. **Option 2: Partial Pinning with External SaaS Platforms**
   - *Pros*: Offloads merge queue and security scanning to third-party proprietary vendors.
   - *Cons*: Introduces vendor lock-in, recurring operational cost, and third-party webhook permissions.
3. **Option 3: Pure-Engine Architecture (Harden-Runner, Zizmor, Composite Action, Repository Rulesets, Native Merge Queue) [Chosen]**
   - *Pros*: 100% native GitHub platform features, zero external subscription cost, deterministic supply-chain immutability, automated linting via `zizmor` and `scripts/check_action_pins.py`, full SPEC 0 compliance.
   - *Cons*: Requires strict SHA update cadence (managed via automated Dependabot/Renovate PRs with metadata comments).

---

## 4. Decision Outcome

We decided to implement **Option 3: Modern CI/CD Excellence, Repository Rulesets & Merge Queue Governance**.

### 4.1 Immutable Commit SHA Pinning & Verification
Every `uses:` reference across `.github/workflows/` and `.github/actions/` is pinned to a full 40-character hexadecimal commit SHA followed by a human-readable tag comment (e.g., `uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2`).

Enforcement is validated by two automated gates:
1. `scripts/check_action_pins.py`: Scans all workflow files and asserts 100% SHA pinning and tag comment presence.
2. `zizmor`: Audits workflows against the `unpinned-uses` blanket policy.

### 4.2 Security Auditing & Runtime Network Boundary
- **Zizmor Security Gate**: Added as a required blocking gate in `.github/workflows/ci.yml`. Enforces zero findings at `medium` and `high` severity across `template-injection`, `dangerous-triggers`, `excessive-permissions`, `unpinned-uses`, and `cache-poisoning`.
- **StepSecurity Harden-Runner**: Integrated into all Linux runner jobs (`step-security/harden-runner@351661ca32ac09a36dc5ee2d536e3128f2a3c8ed # v2.22.0`) with `egress-policy: audit`, capturing network baseline calls for future blocklist enforcement.
- **Template Injection Sanitization**: All runtime shell scripts expand context variables (such as `github.base_ref`) strictly through intermediate environment variables (`env: BASE_REF: ${{ github.base_ref }}`).
- **Least-Privilege Scoping**: Workflows declare `permissions: contents: read` globally at the top level, escalating permissions (`id-token: write`, `attestations: write`, `security-events: write`, `pages: write`) solely on specific publishing or security-reporting jobs.

### 4.3 Composite Setup Action & Cache Hardening
- **Composite Action (`.github/actions/setup-python-uv`)**: Centralizes `setup-uv` and `setup-python` installation.
- **Cache Isolation**: `astral-sh/setup-uv` is configured with `cache-dependency-glob: "uv.lock"`.
- **Restore-Only Pull Requests**: Pull request jobs set `save-cache: false`, restoring existing warm caches without write privileges. Only verified pushes to `refs/heads/main` execute cache writes.
- **Cache Pruning**: `uv cache prune --ci` is invoked prior to cache finalization to eliminate intermediate build artifacts.
- **Concurrency Governance**: PR runs cancel obsolete superseded commits (`cancel-in-progress: true`), whereas merge queue runs and releases retain execution integrity (`cancel-in-progress: false`).

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event_name == 'merge_group' && github.run_id || github.ref }}
  cancel-in-progress: ${{ github.event_name != 'merge_group' }}
```

### 4.4 Repository Ruleset & Merge Queue Governance
Branch protection on `main` is codified and managed via `.github/rulesets/main-ruleset.json`:
- **Branch Protection**: Prohibits deletion, enforces `non_fast_forward` (no force-pushes), and mandates linear history (`required_linear_history`).
- **PR Approvals**: Requires at least 1 approving review from verified code owners (`.github/CODEOWNERS` mapping to `@vfcarida`), with dismissal of stale approvals on new pushes.
- **Status Checks**: Requires all 19 CI quality gates plus CodeQL and OpenSSF Scorecard.
- **Merge Queue Integration**: Configures atomic merge batches (`grouping_strategy: ALLGREEN`, `max_entries_to_merge: 5`, `timeout: 60m`). Every gating workflow explicitly handles the `merge_group` trigger.

### 4.5 Matrix Expansion & SPEC 0 Compliance
- **Core OS & Python Matrix**: Tests Ubuntu, macOS, and Windows on Python 3.11 and 3.12.
- **Future-Proof Canaries**: Adds advisory legs (`continue-on-error: true`) for Python 3.13, free-threaded CPython `3.13t` (PEP 703), and prerelease `3.14-dev`.
- **SPEC 0 Enforcement**: Created `tests/packaging/test_spec0_compliance.py` asserting that `requires-python` (`>=3.11`) and `numpy` (`>=1.26.0`) strictly honor the Scientific Python Ecosystem Coordination support windows (drop Python >36 months, drop NumPy >24 months).

---

## 5. Consequences

### Positive
- **Supply-Chain Integrity**: Eliminates risk of compromised upstream action tags.
- **Elimination of Merge Race Conditions**: Merge Queue serializes and validates PR combinations against the latest trunk tip before merging.
- **Protection Against Cache Poisoning**: Untrusted PRs cannot inject compromised artifacts into the team's shared CI cache.
- **Immediate Feedback on Security Flaws**: Zizmor and SHA pinning checks catch workflow configuration vulnerabilities in seconds.
- **Zero Technical Debt on Python Runtime**: Early signal on free-threaded Python and prereleases prevents runtime friction upon ecosystem transitions.

### Negative / Trade-offs
- **SHA Maintenance Overhead**: Action updates require updating both the commit SHA and comment tag. (Mitigated: Dependabot/Renovate supports automatic SHA updates when comments follow `# vX.Y.Z`).
- **Merge Queue Queuing Latency**: When multiple PRs merge concurrently, merge queue runs an integrated verification check before applying the merge.

---

## 6. References

- [OpenSSF Scorecard: Pinned Dependencies](https://scorecard.dev/docs/checks/#pinned-dependencies)
- [Zizmor: GitHub Actions Static Analysis](https://woodruffw.github.io/zizmor/)
- [StepSecurity: Harden-Runner Architecture](https://www.stepsecurity.io/)
- [GitHub Actions Documentation: Managing Repository Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- [GitHub Actions Documentation: Merge Queue](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
- [SPEC 0: Minimum Supported Versions](https://scientific-python.org/specs/spec-0000/)
- [PEP 703: Making the Global Interpreter Lock Optional in CPython](https://peps.python.org/pep-0703/)
