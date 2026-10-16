# Enterprise World Model (EWM) Conformance Suite

> **"Own the Suite = Own the Standard."**
> A reference implementation becomes *the* standard by publishing an interchange format and owning the executable test suite everyone validates against.
> Conformance Criterion: **Bit-exact SHA-256 state and rollout fingerprints for pinned seeds.**

Modeled on the [JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite) and SBML Test Suite discipline, this directory contains language-agnostic, executable conformance cases.

---

## 1. Directory Structure

```text
conformance/
├── LICENSE                    # Independently licensed (MIT) for third-party adoption
├── README.md                  # This documentation
├── runner.py                  # Standalone CI-automatable conformance test runner
├── reference_worlds/          # Reference-FMUs model implementations
│   ├── ref_seed_trivial.py
│   ├── ref_closed_form_stochastic.py
│   └── ref_constraint_stress.py
└── tests/
    ├── latest/                # Mandatory current Level 1 conformance cases
    │   ├── deterministic_rollout.json
    │   ├── stochastic_ensemble.json
    │   └── constraint_invalidation.json
    ├── optional/              # Optional SBML-style extension packages (PkgCausal, etc.)
    │   └── pkg_causal_counterfactual.json
    └── proposals/             # Draft / experimental future proposals
        └── proposal_wsl_v2.json
```

---

## 2. Conformance Case Schema

Each test fixture defines:
```json
{
  "name": "Deterministic Trivial Rollout",
  "category": "deterministic",
  "level": 1,
  "description": "5-step deterministic transfer between two depots with seed 42",
  "world_manifest": { ... },
  "scenario": {
    "scenario_id": "ref_seed_trivial_scen",
    "horizon": 5,
    "samples": 1,
    "seed": 42
  },
  "expected_fingerprint": "5dbb3db473211cc725a549e714449a2ac43c7d563e301470a70447f323e72dda",
  "expected_branch_fingerprints": [
    "5f230e86d8715edfb8939ff8ec4ab28c2769ca60ea456ec44e8db9178a7845ab"
  ],
  "expected_status": "completed",
  "valid": true
}
```

---

## 3. Running the Conformance Suite

### Command-Line Execution
```bash
# Run all mandatory latest cases
python conformance/runner.py

# Filter by category (deterministic, stochastic, constraints, wmif)
python conformance/runner.py --category stochastic

# Generate JSON test report
python conformance/runner.py --report-json conformance_report.json
```

### Negative Mutation Testing
The runner includes `--test-negative` mode to deliberately mutate golden fingerprints and verify that non-conformant implementations fail loudly.
