# Community Health & Triage Guidelines

## 1. Overview

The Enterprise World Model Engine (`ewm-engine`) structures its contributor workflow and triage automation using open-source community health practices inspired by the **[CHAOSS Project](https://chaoss.community/)** and **[Scientific Python SPEC 9](https://scientific-python.org/specs/spec-0009/)**.

- **Governance Framework**: [GOVERNANCE.md](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/GOVERNANCE.md)
- **Code Ownership**: [`.github/CODEOWNERS`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/CODEOWNERS)
- **Starter Tasks**: [`docs/good-first-issues.md`](good-first-issues.md)

---

## 2. Community Health Objectives

### 2.1. Knowledge Distribution (Bus Factor)
- **CHAOSS Reference**: [Risk: Bus Factor](https://chaoss.community/kb/metric-bus-factor/)
- **Implementation**:
  - Subsystem ownership documented in [`CODEOWNERS`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/CODEOWNERS).
  - Five-tier contributor advancement ladder defined in [GOVERNANCE.md](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/GOVERNANCE.md).
  - Architectural decisions recorded in [`docs/adr/`](adr/ADR-038-open-source-governance-community-health-and-rfc-process.md) and [`rfcs/`](https://github.com/vfcarida/Enterprise-World-Model-Engine/tree/main/rfcs).

### 2.2. Issue & Pull Request Triage
- **CHAOSS Reference**: [Evolution: Time to First Response](https://chaoss.community/kb/metric-time-to-first-response/)
- **Implemented Automation**:
  - Automated welcome workflow for first-time contributors ([`greetings.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/greetings.yml)).
  - Automated component labeling via `actions/labeler` ([`labeler.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/labeler.yml) and [`.github/labeler.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/labeler.yml)).
  - Automated dormant issue management ([`stale.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/stale.yml)).

### 2.3. Contributor Onboarding
- Curated starter tasks with acceptance criteria in [`docs/good-first-issues.md`](good-first-issues.md).
- Structured issue templates in [`.github/ISSUE_TEMPLATE/`](https://github.com/vfcarida/Enterprise-World-Model-Engine/tree/main/.github/ISSUE_TEMPLATE).

---

## 3. Repository Triage Automation

| Workflow | Purpose | Configuration |
| :--- | :--- | :--- |
| **Labeler Action** | Automated PR routing and component tagging | [`.github/workflows/labeler.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/labeler.yml) |
| **First-Interaction Bot** | Welcoming new contributors with repository guidelines | [`.github/workflows/greetings.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/greetings.yml) |
| **Stale Management** | Notification of dormant issues with exempt labels | [`.github/workflows/stale.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/stale.yml) |

