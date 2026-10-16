# World Specification Language (WSL) Version 2.0 Specification

> **Discipline**: Modelica-level formal declarative DSL specification for Enterprise World Models.
> **Safety Guarantee**: Strictly non-executable declarative grammar. No `eval`, no `exec`, no dynamic code execution.

---

## 1. Scope & Design Principles

The World Specification Language (WSL) is a human-authored and machine-verifiable domain-specific language for declaring discrete-time enterprise world models.

### Core Principles
1. **Separation of Definition and Execution**: A WSL document defines states, topologies, constraints, and initial resources. It never executes code or defines arbitrary algorithms.
2. **Deterministic Parity**: An authored `.wsl` document compiles into an immutable `World` that yields **100% bit-identical SHA-256 fingerprints** across execution environments.
3. **Draft 2020-12 Schema Alignment**: Every construct maps 1-to-1 to the versioned `world.schema.json` and OpenAPI 3.1 representations.

---

## 2. Lexical Grammar & Syntax (EBNF)

WSL uses a block-structured declarative syntax with C-style curly braces `{}` and Python-style comments (`#`).

```ebnf
WorldDocument      ::= "world" [ StringLiteral ] "{" WorldStatement* "}"
WorldStatement     ::= MetadataStmt | TemporalBlock | EntityBlock | ResourceBlock 
                     | RelationshipBlock | ConstraintBlock | ScenarioBlock

TemporalBlock      ::= "temporal" "{" TemporalStmt* "}"
TemporalStmt       ::= "time_unit" ":" StringLiteral
                     | "step_duration" ":" NumberLiteral
                     | "default_horizon" ":" IntegerLiteral

EntityBlock        ::= "entity" Identifier "{" EntityStmt* "}"
EntityStmt         ::= "type" ":" StringLiteral
                     | "tags" ":" "[" [ StringLiteral ("," StringLiteral)* ] "]"
                     | Identifier ":" Value

ResourceBlock      ::= "resource" Identifier "{" ResourceStmt* "}"
ResourceStmt       ::= "entity" ":" StringLiteral
                     | "current" ":" NumberLiteral
                     | "min" ":" NumberLiteral
                     | "max" ":" NumberLiteral
                     | "unit" ":" StringLiteral

RelationshipBlock  ::= "relationship" [ Identifier ] "{" RelationshipStmt* "}"
RelationshipStmt   ::= "source" ":" StringLiteral
                     | "target" ":" StringLiteral
                     | "type" ":" StringLiteral
                     | "weight" ":" NumberLiteral

ConstraintBlock    ::= "constraint" [ Identifier ] "{" ConstraintStmt* "}"
ConstraintStmt     ::= "type" ":" StringLiteral
                     | "severity" ":" ("HARD" | "SOFT" | "hard" | "soft")
                     | "parameters" "{" (Identifier ":" Value)* "}"

ScenarioBlock      ::= "scenario" Identifier "{" ScenarioStmt* "}"
ScenarioStmt       ::= "horizon" ":" IntegerLiteral
                     | "samples" ":" IntegerLiteral
                     | "seed" ":" IntegerLiteral
```

---

## 3. Tooling & Language Server Protocol (LSP)

Authored `.wsl` files are supported by:
1. **Lark Parser (`ewm_engine.serialization.wsl_parser`)**: Bounded-time LALR(1) compiler.
2. **Language Server (`ewm_engine.serialization.lsp_server`)**: Real-time diagnostic validation, hover tooltips, and keyword completions.
3. **VS Code Extension (`editors/vscode`)**: Syntax highlighting and language client integration.

---

## 4. Cross-Domain Example Gallery

### Example A: Multi-Echelon Supply Chain Logistics
```wsl
world "SupplyChainDistribution" {
    id: "sc_global_network"
    version: "2.0.0"
    description: "Two-echelon supply network linking central warehouse to regional hubs"

    temporal {
        time_unit: "day"
        step_duration: 1.0
        default_horizon: 30
    }

    entity central_depot {
        type: "distribution_center"
        tier: 1
    }

    entity regional_hub {
        type: "retail_depot"
        tier: 2
    }

    resource central_stock {
        entity: "central_depot"
        current: 1000.0
        min: 0.0
        max: 5000.0
        unit: "pallets"
    }

    resource hub_stock {
        entity: "regional_hub"
        current: 250.0
        min: 0.0
        max: 1000.0
        unit: "pallets"
    }

    relationship freight_lane {
        source: "central_depot"
        target: "regional_hub"
        type: "highway_route"
        weight: 1.0
    }

    constraint hub_capacity {
        type: "ResourceCapacityConstraint"
        severity: "HARD"
        parameters {
            resource_id: "hub_stock"
        }
    }

    scenario stress_test {
        horizon: 30
        samples: 50
        seed: 20261016
    }
}
```

### Example B: Epidemic SEIR Disease Dynamics
```wsl
world "EpidemicSEIRModel" {
    id: "seir_metropolitan"
    version: "2.0.0"
    description: "Metropolitan compartmental epidemic simulation"

    temporal {
        time_unit: "day"
        step_duration: 1.0
        default_horizon: 60
    }

    entity metro_pop {
        type: "population"
        census: 1000000
    }

    resource susceptible {
        entity: "metro_pop"
        current: 999900.0
        min: 0.0
        max: 1000000.0
        unit: "individuals"
    }

    resource infectious {
        entity: "metro_pop"
        current: 100.0
        min: 0.0
        max: 1000000.0
        unit: "individuals"
    }

    scenario baseline_spread {
        horizon: 60
        samples: 25
        seed: 42
    }
}
```

### Example C: Financial Bank Liquidity & Tail Run
```wsl
world "BankLiquidityRun" {
    id: "bank_liquidity_v1"
    version: "2.0.0"
    description: "Wholesale funding run on fractional reserve deposits"

    temporal {
        time_unit: "hour"
        step_duration: 1.0
        default_horizon: 72
    }

    entity commercial_bank {
        type: "financial_institution"
    }

    resource liquid_reserves {
        entity: "commercial_bank"
        current: 500.0
        min: 50.0
        max: 2000.0
        unit: "million_usd"
    }

    constraint minimum_liquidity_buffer {
        type: "ResourceCapacityConstraint"
        severity: "HARD"
        parameters {
            resource_id: "liquid_reserves"
        }
    }

    scenario panic_run {
        horizon: 72
        samples: 100
        seed: 777
    }
}
```
