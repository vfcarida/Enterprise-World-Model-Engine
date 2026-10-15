# EWM Engine — Round 5 Expanded Roadmap (v1.6 → v2.0)

> Prepared: 2026-10-14. Companion to `00_OVERVIEW_AND_STATE.md` and the `prompts/S01`–`S09` package. Maps the Round-5 transformative capabilities (and the still-open Round-4 `Q`-package) onto the repository's **own declared milestone frontier**, respecting its maturity framework and SemVer discipline.
>
> **Anchoring to the repo's declared plan (verified in `ROADMAP.md` / README @ `main`):** `v1.5.0` **Shipped** · `v1.6.0` **"Learned Dynamics & Causal Epistemics" — Active** · `v2.0-alpha` **"Relational GNNs & WSL" — Horizon**. Round 5 does not invent a parallel roadmap; it *fills in* what v1.6.0 should contain to be world-class, adds the two intermediate minors the frontier research justifies (`v1.7`, `v1.8`), and states what graduates the transformative kernel to **Stable** at `v2.0.0`.

---

## 1. Maturity framework (unchanged, from the repo's `ROADMAP.md`)
- **Stable** — frozen public contract (`1.x`), back-compat guarantees, contract gates, deprecation lifecycle.
- **Beta Adapter / Beta** — production-grade, tested, time/resource-bounded; may evolve with deprecation.
- **Experimental** — `ewm_engine.experimental.*`; API may change across minors without deprecation.
- **Research** — scientific instruments / benchmarks without production claims.

Every Round-5 capability enters at **Experimental** (behind an extra), earns **Beta** once it has tests + an ADR + reproducible evidence, and only the subset that proves bit-reproducible, honest-by-construction, and spec-stable reaches **Stable** at `v2.0.0`. Nothing is marketed above its earned tier (invariant #7).

## 2. Design principles that drive the sequencing
1. **Positioning is version-independent and ships first** (`S06`/`S07`) — it attacks the 0-stars problem now and costs no core risk; treat as `v1.5.x` docs/metadata patches, not a feature release.
2. **The counterfactual SCM kernel (`S01`) is the keystone** — it is the moat *and* the dependency for the honest-evaluation suite (`S05`) and the credibility of calibration (`S04`). It leads v1.6.0.
3. **Each capability lands with its honesty guardrail in the same minor** (Round-5 meta-rule) — never a power without its refusal/abstention path and correctness test.
4. **Evidence is a release gate** — a capability is not "done" for a milestone until a committed `make reproduce` artifact regenerates its headline figure/number in CI.
5. **Round-4 `Q`-items that are prerequisites are pulled in** where a Round-5 capability needs them (e.g. `Q01` metrics underpin `S05`; `Q03` validation packs feed `S04`/`S09`).

---

## 3. Release sequencing (at a glance)

| Milestone | Theme | Round-5 (and pulled-in Round-4) content | Entry maturity |
|---|---|---|---|
| **v1.5.x** (patch, now) | **Positioning & narrative** | `S06` README/identity + topics; `S07` `docs/executive/` + governance mapping; `S08` handbook *scaffold* | Docs only |
| **v1.6.0** (active) | **Causal epistemics & learned dynamics** | `S01` counterfactual SCM kernel · `S03` grey-box conservation + residual learning · `S05` causal-correctness + metamorphic suite · (pull-in `Q01` numpy metrics, `Q02` identifiability gate) | Experimental → Beta |
| **v1.7.0** | **Decision intelligence** | `S02` `ewm decide` (scenario discovery + robustness metrics + EFE) · (pull-in `Q05` EWM-Bench *lite* if desired) | Experimental → Beta |
| **v1.8.0** | **Calibration to reality** | `S04` `ewm calibrate` (SBI + certified coverage) · (pull-in `Q03` real-data validation packs, `Q04` batched engine for study scale) | Experimental → Beta |
| **v1.9.0** | **Flywheel & neuro-symbolic maturation** | `S03` symbolic-regression term-promotion workflow matures · `S05` abstention wired across decision+calibration · handbook (`S08`) completes | Beta |
| **v2.0.0** | **Reference release** | Transformative kernel (`S01`) → **Stable**; `S09` results gallery + arXiv companion + coordinated launch; (repo's own `v2.0` GNN/WSL + Round-4 `Q07`/`Q08` WSL/WMIF) | Stable + Experimental GNN/WSL |

> `S08` (handbook/explorable docs) and the positioning track are **continuous/cross-cutting**, not a single milestone — they accrete across every release and are called out per-milestone below.

---

## 4. Milestone detail

### v1.5.x — Positioning & narrative (ship immediately; docs/metadata only)
**Why first:** zero core risk, highest adoption ROI, and it fixes the active liability (EWM mis-filed as a neural/video world model; 0 stars).
- `S06`: mechanism-first README, neural-world-model disambiguation line, CI-generated "proof hero" (ground-truth overlay), 18→~8 badges, uncrowded GitHub topics, `docs/positioning/landscape.md` (2×2 + nearest-neighbor table + non-goals).
- `S07`: `docs/executive/` (lineage, decision-grade guarantees, C-suite use-case gallery, maturity model, reference architectures) + `docs/governance/compliance-mapping.md` (NIST AI RMF, EU AI Act).
- `S08` (scaffold): stand up the `docs/handbook/` skeleton (question-organized ToC + the "When to refuse a claim" chapter stub).
- **Exit:** README leads with mechanism + the honest identity claim; executive docs link only to shipped features; no forbidden claim; proof-hero regenerates in CI.

### v1.6.0 — Causal Epistemics & Learned Dynamics (the repo's declared active milestone)
**Theme match:** this is *exactly* the repo's declared v1.6.0; Round 5 supplies the world-class content.
- **`S01` Twin-Rollout counterfactual SCM kernel** *(keystone)* — typed graph → native SCM with addressable seeded exogenous noise; `world.counterfactual()` via abduction–action–prediction with coupled noise; bit-reproducible; rung-3 scoping + non-unique-abduction distribution+flag; ID-gate integration. **Richens & Everitt (2402.10877)** becomes the headline citation.
- **`S03` grey-box (first half)** — residual-only learned-dynamics interface (zero residual ≡ symbolic kernel, fingerprint-identical) + architectural conservation-projection layer (enterprise invariants impossible to violate by construction) + structural-zero-violation honesty metric.
- **`S05` causal-correctness + metamorphic suite** — `do(x)`-descendants-only selectivity gates (uses `S01`), label-free MR-cards against the learned layer + mutation-adequacy score, first-class epistemic abstention; the Physics-IQ "no plausibility-as-correctness" rule.
- **Pull-in from Round 4:** `Q01` numpy metrics (rollout error/compounding, CRPS, PIT/ECE, violation-rate) and `Q02` identifiability gate + E-values — these are prerequisites `S05` and `S01` lean on.
- **Maturity:** `S01` enters Experimental, targets Beta by milestone close (tests + ADR + reproducible counterfactual example); `S03`/`S05` Experimental→Beta.
- **Exit:** a bit-reproducible entity-level counterfactual ships with its honesty guardrails; the causal-correctness gate fails a wrong-sign/non-descendant `do(x)`; MR suite runs label-free with proven mutation adequacy; all behind extras with core green without them.

### v1.7.0 — Decision Intelligence (`ewm decide`)
**Why here:** reframes EWM from "what could happen?" to "what should we do, and when does it break?" — the single highest-value conceptual upgrade, and it consumes the v1.6 counterfactual/uncertainty substrate.
- **`S02`** — scenario discovery (PRIM/CART vulnerability boxes) over the ensemble; multi-robustness-metric decision comparison (min-max regret, percentile/CVaR regret, satisficing) with **explicit refusal to over-rank when metrics disagree**; optional EFE action scoring; `ewm decide` CLI + decision report.
- **Optional pull-in:** `Q05` EWM-Bench *lite* (shift families as scored sub-leaderboards) if the benchmark track is prioritized here rather than at v2.0.
- **Maturity:** Experimental → Beta (deterministic/seeded discovery, reproducible report).
- **Exit:** `ewm decide` returns vulnerability scenarios + a ≥3-metric decision comparison and declines a single winner when metrics disagree (tested).

### v1.8.0 — Calibration to Reality (`ewm calibrate`)
**Why here:** bridges synthetic worlds to real published ground truth — the move that makes the empirical story real, and it benefits from v1.7's decision layer (uncertified posteriors are refused for decision use).
- **`S04`** — SBI module wrapping the engine as forward model; amortized posterior over WorldSpec params; **mandatory TARP + Posterior-SBC coverage certification** + misspecification-robust stats + refusal path; reproducible calibration artifact; worked example on a real series.
- **Pull-in from Round 4:** `Q03` real-data validation packs (SEIR/COVID, Beer-Game, WORLD3, OpenEI) as calibration targets; `Q04` batched/vectorized Monte Carlo to make the SBI simulation budget tractable at scale (fingerprint-identical to pure-Python).
- **Linkage:** a *certified* calibration promotes `S01`'s counterfactuals from "model-only" toward real-relevance — documented, with `EvidenceLevel` kept honest (calibrated ≠ identified).
- **Maturity:** Experimental → Beta.
- **Exit:** `ewm calibrate` returns a certified posterior (or refuses, under engineered under-coverage); a calibrated SEIR/Beer-Game example reproduces published ground truth from one command.

### v1.9.0 — Flywheel & neuro-symbolic maturation
**Why here:** the pieces now compose; this milestone hardens them into a coherent, teachable whole before the v2.0 reference release.
- **`S03` (second half)** — symbolic-regression residual extraction (SINDy/PySR) + the human-in-the-loop, spec-driven **promotion workflow** (LEARNED → KNOWN); LLM-SR advisory-only.
- **`S05` abstention wired end-to-end** — out-of-competence signal down-weights/excludes low-competence rollouts in `ewm decide` and gates `ewm calibrate`.
- **`S08`** — the World Modeling Handbook completes (method-quartet chapters for `S01`–`S05`, Distill-style explorable explainers on the live Pyodide kernel, whole-game-first tutorial ladder).
- **Maturity:** the full learned/decision/calibration stack consolidates at **Beta**.
- **Exit:** a discovered residual term is promoted into the WSL ruleset via an ADR/spec step (tested); the handbook teaches each capability with all four Diátaxis types; explorable explainer runs client-side.

### v2.0.0 — Reference Release
**Theme match:** aligns with the repo's declared `v2.0` (Relational GNNs & WSL) and makes it the *reference* moment.
- **Graduations to Stable:** `S01` counterfactual SCM kernel (bit-reproducible, honest-by-construction, spec-stable) and the grey-box residual/conservation contract — each via API-Change-Proposal + contract gates.
- **`S09` capstone** — CI-reproducible `docs/results/` gallery (result cards: ground-truth overlay + calibration + multi-seed error bars + `make reproduce`), the arXiv methods/benchmark companion paper (NeurIPS-D&B-checklist QA, re-verified citations, cross-linked to CITATION.cff/Zenodo), the reproducible comparison harness, and the coordinated launch led by proof.
- **Repo's own v2.0 + Round-4 ecosystem:** heterogeneous relational graph state & GNN dynamics (Experimental), WSL maturation (`Q07`), and the WMIF interchange format + conformance suite (`Q08`) — the "own the standard" layer.
- **Maturity:** Stable transformative kernel + Experimental GNN/WSL/WMIF.
- **Exit:** the identity claim — *the only world-model engine scored on being right about its own uncertainty* — is demonstrated by the gallery and the paper, and the Stable surface is contract-gated.

---

## 5. Dependency graph (what blocks what)

```
Positioning (S06,S07)  ──► independent, ship at v1.5.x
                                     │ (gives the narrative every later artifact reuses)
Q01 metrics ─┐                       ▼
Q02 ID gate ─┴─► S01 SCM kernel ──► S05 causal-correctness + metamorphic
                      │                     ▲
                      ├──────────► S02 ewm decide ◄── (abstention from S05)
                      │                     ▲
                      └──────────► S04 ewm calibrate ◄── Q03 packs, Q04 batch
                                          │
S03 grey-box (residual+conservation) ─────┼──► S03 flywheel (SINDy/PySR promotion)
                                          ▼
                              S08 handbook (teaches S01–S05)
                                          ▼
                              S09 gallery + arXiv + launch  ──► v2.0 reference
```

- **Critical path:** `S01` → (`S05`, `S04`) → `S09`. Everything else parallelizes around it.
- **Fast, independent wins:** `S06`/`S07` (now), `S03`-conservation-layer (self-contained).
- **Round-4 pull-ins are prerequisites, not duplicates:** `Q01`/`Q02` under `S05`; `Q03`/`Q04` under `S04`; `Q05` optional under `S02`; `Q07`/`Q08` land with the v2.0 ecosystem layer.

---

## 6. Honest risks & how the roadmap mitigates them

| Risk | Mitigation in this plan |
|---|---|
| **Scope explosion** (5 big capabilities + 2 doc tracks) | Each is an extra behind `experimental`; core stays `pydantic`/`numpy`/`pyyaml`; small PRs; evidence-gated milestones. Positioning (cheap) ships first to buy momentum. |
| **Capability without honesty** (overclaiming creep) | Round-5 meta-rule enforced per-milestone: no power ships without its refusal/abstention path + correctness test. `S05` is a milestone peer of `S01`, not an afterthought. |
| **Fingerprint drift** from learned/batched/SBI paths | Invariant #5: bit-identical per-branch fingerprints vs pure-Python, CI-enforced; GPU/float-nondeterministic paths explicitly throughput-only and labeled. |
| **Citation hygiene** (future-dated/unverified IDs, analyst figures) | `REFERENCES.md` tiers; load-bearing anchors venue-validated only; `S09` paper gate re-verifies every ID/DOI; market figures dated+sourced or omitted. |
| **Adoption still lags** despite capability | Positioning (`S06`/`S07`) + handbook (`S08`) + launch (`S09`) are first-class milestones, not side effects; the bit-exact reproducibility story is the press-worthy differentiator. |
| **"World model" naming confusion persists** | `S06` disambiguation + uncrowded topics + landscape doc land at v1.5.x, before any capability PR, so new visitors file EWM correctly from the start. |

---

## 7. One-paragraph framing for the repo's `ROADMAP.md`
> Round 5 fills the repo's declared **v1.6.0 "Learned Dynamics & Causal Epistemics"** with world-class content — a bit-reproducible counterfactual SCM kernel (grounded in Richens & Everitt's theorem that robust agents must learn causal world models), a grey-box that enforces enterprise conservation laws by construction and whitens itself over time, and a causal-correctness + metamorphic test suite that scores the engine on *interventional* correctness and *epistemic abstention*, not just rollout error. It then adds decision intelligence (`v1.7`, `ewm decide`: scenario discovery + robustness metrics that refuse to over-rank under deep uncertainty) and calibration to reality (`v1.8`, `ewm calibrate`: simulation-based inference with mandatory coverage certification), consolidates the neuro-symbolic flywheel and the teaching handbook (`v1.9`), and culminates in a **v2.0 reference release** — the counterfactual kernel graduated to Stable, a reproducible results gallery, and an arXiv companion — alongside the repo's own relational-GNN/WSL horizon and the WMIF interchange standard. Positioning and executive narrative ship immediately as `v1.5.x` docs, because the binding constraint today is not quality but a crowded name and zero adoption. Every capability lands honest-by-construction: no new power without its refusal path and its correctness test, so the engine's identity — *the only world-model engine scored on being right about its own uncertainty* — strengthens with each release.
