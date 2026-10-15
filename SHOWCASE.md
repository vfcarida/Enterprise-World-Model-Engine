# EWM Engine Showcase & Reference Benchmarks

> *"A world-model engine scored on being right about its own uncertainty."*

Welcome to the EWM Engine Showcase. We adhere to a strict scientific honesty policy: **we do not display unverified logos, fabricated corporate endorsements, or synthetic testimonials**. 

Instead, this page showcases the official, reproducible **reference validation packs** included with the framework, alongside instructions for community researchers and organizations to submit their real-world implementations.

---

## 1. Reference Validation Packs

All showcase benchmarks are fully reproducible directly from the repository using seeded run manifests:

| Validation Pack | Domain | Primary Dynamic / Law Tested | Key Credibility Metric | Reproduction Command |
| :--- | :--- | :--- | :--- | :--- |
| [SEIR Epidemic Outbreak](docs/results/seir-outbreak.md) | Epidemiology / Public Health | Closed mass conservation ($S+E+I+R=N$) | Drift: $4.54 \times 10^{-13}$, Physics-IQ: $1.00$ | `python scripts/reproduce_results.py --card seir` |
| [Beer Game Bullwhip](docs/results/beer-game-bullwhip.md) | Supply Chain Operations | Information distortion & multi-echelon delay | Bullwhip ratio: $3.80\times$, Inventory cost balance | `python scripts/reproduce_results.py --card beer_game` |
| [WORLD3 Limits to Growth](docs/results/world3-resource-depletion.md) | System Dynamics / Sustainability | Multi-loop non-linear carrying capacity | 100-step numerical feedback stability | `python scripts/reproduce_results.py --card world3` |
| [Amortized Posterior Calibration](docs/results/calibrated-posterior-s04.md) | Bayesian Inference (SBI) | Tests of Accuracy of Recovery of Posterior (TARP) | Coverage error: $0.0283$, TARP p-val: $0.78$ | `python scripts/reproduce_results.py --card calibrated_posterior` |
| [Twin-Rollout Counterfactual SCM](docs/results/counterfactual-scm-s01.md) | Structural Causal Models | Abduction-Action-Prediction twin noise coupling | Identifiable ATE: $+35.0$, Epistemic gate refusal | `python scripts/reproduce_results.py --card counterfactual_scm` |
| [Decision Scenario Discovery](docs/results/decision-discovery-s02.md) | Robust Decision Making | PRIM bump-hunting + Minimax Regret | Box density: $91.2\%$, Coverage: $80.0\%$ | `python scripts/reproduce_results.py --card decision_discovery` |

To verify all cards cryptographically in one pass:
```bash
python scripts/reproduce_results.py --all
```

---

## 2. Who is Using EWM Engine?

EWM Engine is designed for:
- **Academic Researchers**: In causal inference, simulation-based inference (SBI), system dynamics, and robust decision making (RDM).
- **Operations & Supply Chain Engineers**: Evaluating multi-echelon risk, capacity allocation, and buffer inventory under deep epistemic uncertainty.
- **Policy & Strategic Decision Teams**: Modeling coupled socioeconomic dynamics where black-box generative models hallucinate unconstrained trajectories.

### Featured Community Implementations
*(Submissions are open! Follow the guidelines below to submit your project or paper).*

---

## 3. How to Submit Your Project

We welcome submissions from open-source contributors, academic labs, and enterprise data teams.

### Inclusion Criteria:
1. **Public Preprint or Open Code**: Must link to a public repository, whitepaper, arXiv preprint, or peer-reviewed publication demonstrating EWM Engine usage.
2. **Reproducibility**: Must include configuration files, parameters, and random seeds enabling independent verification.
3. **Transparent Uncertainty**: Must report confidence intervals, posterior coverage, or sensitivity metrics.

To submit, open a Pull Request modifying this file (`SHOWCASE.md`) and add your card to `docs/results/community/` using the template in `docs/results/index.md`.

---

## 4. How to Cite

If you use EWM Engine in academic research, benchmark evaluations, or technical publications, please cite both the software artifact and the companion methods paper.

### Software Citation (JOSS / Zenodo)
```bibtex
@software{ewm_engine_software,
  author       = {Enterprise World Model Engine Contributors},
  title        = {Enterprise World Model Engine: A Principled Architecture for Deterministic, Causal, and Calibrated System Simulation},
  year         = {2026},
  publisher    = {GitHub / Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.1234567},
  url          = {https://github.com/vfcarida/Enterprise-World-Model-Engine}
}
```

### Methods & Benchmark Companion Paper
```bibtex
@article{ewm_engine_arxiv_2026,
  author       = {Carida, Vinicius and contributors},
  title        = {Beyond Rollout Error: A Causal, Calibrated, and Conservation-Preserving Architecture for Enterprise World Models},
  journal      = {arXiv preprint arXiv:2603.xxxxx},
  year         = {2026},
  eprint       = {2603.xxxxx},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG}
}
```

For complete citation metadata, refer to [CITATION.cff](CITATION.cff).
