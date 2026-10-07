# ADR-038: Open-Source Governance, Community Health, and RFC Process

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-07
* Technical Story: [RFC-0001](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/rfcs/RFC-0001-rfc-process.md)
* Supersedes: None

## Context and Problem Statement

As the Enterprise World Model Engine evolves from an initial foundational implementation into a multi-contributor, reference-grade open-source platform, the project requires an institutional governance framework. The solo-maintainer model lacks an explicit advancement pathway for outside contributors, has no formal mechanism for debating large architectural proposals before code implementation, lacks community health observability, and has no established vehicle for non-profit fiscal sponsorship.

How should EWM Engine organize its community governance, contributor recognition, health analytics, proposal lifecycle, and decision-making authority?

## Decision Drivers

* **Scientific Credibility & Ecosystem Standards**: Adhere to established standards from the Scientific Python ecosystem (SPEC 9 for governance, SPEC 8 for secure releases).
* **Contributor Empowerment & Meritocracy**: Provide a clear, transparent maintainer ladder with distinct rights and responsibilities.
* **Community Safety & Inclusion**: Adopt the industry-standard Contributor Covenant 2.1 with actionable enforcement procedures.
* **Risk Mitigation**: Continuously monitor community health (Bus Factor, response times, organizational concentration) via CHAOSS standards.
* **Long-Term Sustainability**: Pursue 501(c)(3) fiscal sponsorship under NumFOCUS.
* **Proposal Governance**: Separate broad, strategic proposals (RFCs) from granular internal engineering decisions (ADRs).

## Considered Options

* **Option 1: Informal Solo Maintainer (Status Quo)** — Fast decision-making, but creates a Bus Factor of 1, discourages external investment, and limits academic/industrial adoption.
* **Option 2: Pure Corporate Consortium** — Backed by specific enterprises, but risks vendor capture and violates NumFOCUS open science principles.
* **Option 3: Meritocratic Steering Council with SPEC 9 Ladder, CHAOSS Health, and RFC Process (Chosen)** — Aligns with the wider Scientific Python ecosystem, establishes transparent advancement, and enables non-profit sponsorship.

## Decision Outcome

Chosen option: **Option 3: Meritocratic Steering Council with SPEC 9 Ladder, CHAOSS Health, and RFC Process**.

### Consequences

* **Good, because**:
  * Establishes a five-tier maintainer ladder (Contributor → Triager → Committer → Maintainer → Steering Council) per Scientific Python SPEC 9.
  * Formulates a 2-tier decision model: lazy consensus for routine maintenance and 2/3 Council voting for major architectural changes.
  * Adopts Contributor Covenant 2.1 with a four-tier enforcement ladder (Correction, Warning, Temporary Ban, Permanent Ban) and direct contact (`vfcarida@gmail.com`).
  * Establishes a formal RFC process in `rfcs/` for user-facing API changes and paradigm shifts, distinct from internal ADRs.
  * Implements automated triage and contributor recognition via `all-contributors`, `actions/labeler`, `actions/stale`, and first-time greeting automation.
  * Tracks project health via CHAOSS metrics (Bus Factor $\ge 3$, TTFR $< 48$h, Elephant Factor) with a live CollectOSS dashboard link.
  * Targets NumFOCUS fiscal sponsorship and configures `.github/FUNDING.yml`.
  * Adopts MADR 4.x for subsequent Architecture Decision Records and enforces an automated ADR link and status linter in CI.
* **Bad, because**:
  * Introduces governance overhead and deliberation time for cross-cutting proposals (14-day RFC review window).
  * Requires active triage monitoring to maintain the $< 48$h TTFR SLA.

### Confirmation

* Governance policies ratified in `GOVERNANCE.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `rfcs/RFC-0001-rfc-process.md`.
* Triage automation verified via `.github/workflows/labeler.yml`, `.github/workflows/stale.yml`, and `.github/workflows/greetings.yml` passing `check_action_pins.py` and `zizmor`.
* ADR integrity and cross-references verified by `scripts/check_adr_links.py` in CI.

## Pros and Cons of the Options

### Option 1: Informal Solo Maintainer

* Good, because zero governance overhead or delay in approving pull requests.
* Bad, because project Bus Factor remains 1, blocking institutional grant funding and NumFOCUS eligibility.

### Option 2: Pure Corporate Consortium

* Good, because guaranteed corporate developer allocations.
* Bad, because compromises scientific vendor neutrality and risks alienating independent academic researchers.

### Option 3: Meritocratic Steering Council (SPEC 9) + RFC Process

* Good, because standard across respected scientific projects (NumPy, SciPy, NetworkX, Matplotlib).
* Good, because clear separation of concerns between user-facing RFCs and engineering ADRs.
* Bad, because requires ongoing maintainer coordination.

## More Information

* Related RFC: [`RFC-0001: The EWM Engine RFC Process`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/rfcs/RFC-0001-rfc-process.md)
* Related Policy: [`GOVERNANCE.md`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/GOVERNANCE.md), [`CODE_OF_CONDUCT.md`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/CODE_OF_CONDUCT.md)
* Community Health: [`docs/community-health.md`](../community-health.md)
* NumFOCUS Prospectus: [`docs/numfocus-sponsorship.md`](../numfocus-sponsorship.md)
