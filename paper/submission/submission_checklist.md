# Academic Peer-Review & Preprint Submission Checklist

**Manuscript**: [`paper/preprint/ewm_engine_preprint.md`](../preprint/ewm_engine_preprint.md)  
**Target Venues**: *INFORMS Journal on Computing* (Software Tools Track) / *NeurIPS Datasets & Benchmarks*  
**Date**: 2026-10-30  

---

## 1. Pre-Submission Readiness Gates

- [x] **G-01 (Benchmark Measurement Audit)**: All 45 quantitative claims classified in [`benchmarks/measurement_classification.md`](../../benchmarks/measurement_classification.md) (`100%` of headline metrics runtime-computed `MEASURED` / `DERIVED_MEASURED`).
- [x] **G-02 (Clean Checkout Certification)**: `python scripts/reproduce.py --require-clean` verified on clean tree (`CERTIFIED_RELEASE`, SHA-256 `618142f9eab6b9e04bad9f3903313fd12525672f9844ea82f9d05d9bdcc160c6`).
- [x] **G-03 (Pre-Registration & Negative Results)**: Pre-registered under `EWME-PREREG-20261023-LADDER` with 4 first-class negative results (`N1`–`N4`) and Cautionary Case A construct-validity failure reported prominently.
- [x] **G-04 (Economic TCO Justification)**: TCO protocol and Fraud Cascade ROI/sensitivity analysis documented in [`docs/results/fraud_cascade_tco.md`](../../docs/results/fraud_cascade_tco.md).
- [x] **G-05 (Hermetic Reproduction Container)**: [`Dockerfile.reproduction`](../../Dockerfile.reproduction) and [`reproduction_package/zenodo.json`](../../reproduction_package/zenodo.json) prepared for archival deposit.
