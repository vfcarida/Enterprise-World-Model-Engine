# EWM Engine Launch Announcement Narrative

> Headline Asset: **Proof, Not Assertion.**
> Through-line Identity: *"A hybrid causal simulation engine scored on structural validity, conservation bounds, and uncertainty calibration."*

This document coordinates the public announcement across Hacker News, Reddit, academic communities, and social media. In accordance with our anti-overclaiming policy, all communications lead with reproducible evidence, transparent benchmarks, and explicit failure/refusal modes rather than buzzwords.

---

## 1. Hacker News (Show HN)

### Title
```text
Show HN: Enterprise World Model – A causal simulation engine scored on uncertainty honesty
```

### Submission Text
```markdown
Hi HN,

Most recent work in "world models" focuses on video generation or autoregressive LLM rollouts (e.g., predicting pixels or next tokens). While impressive, these systems struggle in socio-technical and operational settings: they accumulate error over long horizons, create or destroy mass, and hallucinate unwarranted certainty on counterfactual queries.

We built the Enterprise World Model (EWM) Engine to approach this from a different angle:
1. **Explicit Separation of Known vs. Learned**: Hard physical and structural invariants (conservation of mass, capacity limits, double-entry balance) are enforced via nullspace projection within IEEE-754 machine precision ($\le 10^{-12}$).
2. **Native Structural Causal Models**: Pearlian counterfactuals via three-step Abduction-Action-Prediction with coupled noise ("Twin-Rollout"), with automated backdoor/frontdoor/ID checks that fail closed (`UnqualifiedCausalClaimError`) on unidentifiable claims.
3. **Scored on Honesty, Not Just Rollout Error**: Beyond MSE, the engine evaluates coverage calibration (TARP / split-conformal prediction), metamorphic physical tests (Physics-IQ), and sensitivity bounds (VanderWeele & Ding E-values).
4. **Deterministic & Auditable**: Every run generates a cryptographic SHA-256 fingerprint under the `BITWISE_REPRODUCIBLE` pure-Python/NumPy tier. The entire test suite and results gallery run on standard single-core CPUs without requiring GPUs or API keys.

You can reproduce our full results gallery (7 domain cards, including SEIR mass conservation, Beer Game bullwhip amplification, and TARP posterior calibration) and 12-family benchmark suite with a single command:

    git clone https://github.com/vfcarida/Enterprise-World-Model-Engine
    cd Enterprise-World-Model-Engine
    pip install -e .
    python scripts/reproduce_results.py --all

- **Docs & Results Gallery**: https://vfcarida.github.io/Enterprise-World-Model-Engine/results/
- **Reproducible Benchmarks**: https://vfcarida.github.io/Enterprise-World-Model-Engine/results/benchmarks/
- **Companion Paper Preprint**: https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/paper/arxiv/paper.md
- **GitHub**: https://github.com/vfcarida/Enterprise-World-Model-Engine

We'd love your feedback on the architecture, the calibration bounds, and how you approach counterfactual decision-making in complex systems.
```

---

## 2. Reddit (`r/MachineLearning`, `r/causalinference`, `r/datascience`)

### Title
```text
[Project] Enterprise World Model Engine: Causal, Calibrated, Conservation-Preserving System Simulation (Paper + Reproducible Benchmarks)
```

### Post Body
```markdown
Hello everyone,

We are sharing our open-source framework and companion methods paper: **Enterprise World Model (EWM) Engine**.

- **Companion Paper (Preprint Draft)**: https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/paper/arxiv/paper.md
- **GitHub Repository**: https://github.com/vfcarida/Enterprise-World-Model-Engine
- **Interactive Results Gallery**: https://vfcarida.github.io/Enterprise-World-Model-Engine/results/

### Why Another World Model Engine?
Recent theoretical work by Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877) established that agents which optimize reward across arbitrary distribution shifts must internally represent the correct structural causal model (SCM). Standard generative simulators approximate dynamics purely as predictive Markov transitions, leading to two severe vulnerabilities:
1. Confounded correlation is mistreated as causal intervention capability.
2. Stochastic rollout errors break fundamental conservation invariants.

### Key Contributions:
1. **Grey-Box SCM Kernel**: Separates known constraints (enforced via algebraic projection) from learned probabilistic dynamics.
2. **Coupled Twin-Rollout**: Implements Pearl's counterfactual definition $Y_{X \leftarrow x}(u)$ by conditioning exogenous noise on historical observed states before intervening.
3. **Empirical Calibration Diagnostics**: Validates posterior distributions using Tests of Accuracy with Random Points (TARP, Lemos et al., ICML 2023) and split-conformal coverage intervals.
4. **Proof-Carrying Integrity**: Automatically refuses to assert causal effects when unobserved confounding violates the backdoor/ID criteria.

### Quick Verification:
The repository includes a Polars/Ruff-style reproducible comparison harness:
```bash
python scripts/reproduce_comparison.py
```
Outputs empirical comparisons against unconstrained baselines across mass conservation, PRNG branch isolation, and conformal calibration.

We welcome critiques from the causal inference and simulation-based inference communities!
```

---

## 3. Social Media & Academic Community (X / Mastodon / LinkedIn)

### Thread Plan (8 Tweets / Posts)
1. **The Core Thesis**: Most world models predict the future by autoregressively hallucinating the next token or frame. In complex operational systems, you need physical conservation, counterfactual validity, and honest uncertainty bounds. Introducing the Enterprise World Model Engine 🧵👇
2. **The Results Gallery**: We are launching with a fully reproducible, CI-verified results gallery. Every card (SEIR, Beer Game, WORLD3, TARP calibration, SCM counterfactuals) runs from a committed seed in <2 seconds on a laptop CPU.
3. **Hard Conservation**: In our comparative benchmark, unconstrained neural dynamics violate mass conservation in 99.9% of steps. EWM projects dynamics onto the invariant nullspace: 0.0% violations, machine precision $< 10^{-12}$.
4. **Epistemic Honesty**: When a user queries a confounded relationship without valid instruments or adjustment sets, EWM refuses with `UnqualifiedCausalClaimError`. We score the engine on being right about its own uncertainty.
5. **Certified Coverage**: Deep models frequently produce overconfident confidence intervals (64% coverage for a 90% target). EWM incorporates TARP and conformalized prediction to guarantee nominal coverage bounds (90.2% on test).
6. **Zero Mandatory GPU**: Pure Python + NumPy + Pydantic. Runs client-side, in Pyodide/WebAssembly, and in standard CI pipelines. No vendor lock-in, no GPU cluster required.
7. **Paper & Artifact**: Check out our companion arXiv paper ("Beyond Rollout Error") alongside the NeurIPS Datasets & Benchmarks self-audit checklist in `paper/arxiv/`.
8. **Get Started**: Run `pip install -e .` and `python scripts/reproduce_results.py --all`. Full docs, explorable guides, and code at github.com/vfcarida/Enterprise-World-Model-Engine!
