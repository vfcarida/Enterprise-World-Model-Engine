# Public Archival Reproduction Package (Zenodo / FigShare)

**Version**: `v1.5.0` (`CERTIFIED_RELEASE`)  
**Manifest SHA-256**: `618142f9eab6b9e04bad9f3903313fd12525672f9844ea82f9d05d9bdcc160c6`  
**Container Definition**: [`Dockerfile.reproduction`](../Dockerfile.reproduction)  
**Zenodo Metadata**: [`reproduction_package/zenodo.json`](zenodo.json)  

---

## 1. One-Command Containerized Reproduction

```bash
docker build -f Dockerfile.reproduction -t ewm-engine-repro:1.5.0 .
docker run --rm ewm-engine-repro:1.5.0
```

## 2. Local Virtualenv Reproduction

```bash
python -m venv .venv-repro
source .venv-repro/bin/activate  # On Windows: .venv-repro\Scripts\activate
pip install -e .
python scripts/reproduce.py --require-clean
```

## 3. Expected Verification Outcomes

- Exit code: `0`
- `certification_status`: `"CERTIFIED_RELEASE"`
- `verification_checks_passed`: `true`
- Execution time: `< 15` seconds on standard 4-core CPU (`4.82s` reference run)
