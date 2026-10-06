# Enterprise World Model Engine — Especificação de Spec-Driven Development para a v1

> **Canonical project name:** Enterprise World Model Engine  
> **Short name:** EWM Engine  
> **Target file:** `docs/specs/spec-driven-development.md`  
> **Specification status:** Authoritative  
> **Target contract:** v1  
> **Primary implementation language:** Python  
> **Repository language:** English  
> **This specification language:** Portuguese (pt-BR)  
> **Last research review:** 2026-10-02  
> **Owner:** Repository Owner / Maintainer

---

## Resumo executivo e autoridade da especificação

Este documento define o contrato de desenvolvimento da versão 1 do Enterprise World Model Engine (EWM Engine). Seu propósito é transformar a tese conceitual e o Master Implementation Prompt em uma especificação suficientemente precisa para que humanos e diferentes coding agents implementem o repositório de forma incremental sem reinterpretar, a cada interação, o que o projeto é, quais invariantes devem ser preservados e quais funcionalidades pertencem ou não ao núcleo.

A escolha por Spec-Driven Development é deliberada. A documentação atual do GitHub Spec Kit organiza o processo em Specify → Plan → Tasks → Implement → Converge, mantendo intenção e evidência como artefatos persistentes antes e depois da geração de código; o mecanismo de convergência compara implementação, especificação, plano e tarefas para identificar lacunas restantes. Essa filosofia é particularmente adequada ao EWM Engine porque o principal risco do projeto não é “não gerar código”, mas deixar agentes sucessivos criarem abstrações incompatíveis, expandirem escopo, introduzirem dependências ou confundirem hipóteses científicas com capacidades implementadas.

A especificação adota como tese arquitetural permanente que um Enterprise World Model deve representar um ambiente evolutivo sobre o qual ações possam ser experimentadas antes de serem executadas no mundo real. Essa direção é consistente com a literatura de world models: o trabalho clássico de Ha e Schmidhuber aprende uma representação espaço-temporal comprimida do ambiente na qual um agente pode treinar, e DreamerV3 melhora comportamento imaginando futuros por meio de um modelo aprendido do ambiente. O diferencial proposto pelo EWM Engine é transpor essa abstração para sistemas organizacionais e socio-técnicos, combinando estado, ações, eventos exógenos, regras explícitas, dinâmica e incerteza, conforme a hipótese de pesquisa original do projeto.

O projeto, entretanto, não adotará como fato científico que JEPA, SMT ou uma arquitetura neuro-simbólica específica seja a implementação correta de um Enterprise World Model. JEPA é uma direção legítima de pesquisa: I-JEPA demonstrou previsão de representações em espaço latente sem reconstrução generativa, e V-JEPA 2 estendeu a abordagem a um world model action-conditioned usado para planejamento robótico. Isso justifica JEPA como futura implementação de `DynamicsModel`, não como dependência estrutural do engine. A proposta externa recebida anteriormente, que tratava JEPA + SMT + projeções diferenciáveis como solução praticamente mandatória, será preservada apenas como hipótese de pesquisa e direção de adapters, e não como verdade arquitetural da v1.

A v1, portanto, será bem-sucedida quando provar uma proposição menor, porém forte:
> **Um desenvolvedor consegue representar um mundo, explicitar suas restrições, plugar uma dinâmica, partir do mesmo estado para intervenções diferentes, simular futuros estocásticos reproduzíveis, identificar violações e rastrear como os resultados emergiram, comparando alternativas sem confundir a simulação com uma alegação causal.**

O engine deverá permanecer industry-neutral. Nenhum exemplo público será específico de dados bancários proprietários ou de regras financeiras internas. A aplicação financeira pertence a uma instanciação futura da tese, não ao contrato público do framework. A generalidade do repositório preserva o refinamento feito ao longo da pesquisa: os artefatos públicos devem servir a logística, saúde operacional, energia, manufatura, varejo, mobilidade, serviços públicos ou finanças sem que o núcleo conheça nenhum desses domínios.

As palavras MUST/DEVE, MUST NOT/NÃO DEVE, SHOULD/DEVERIA e MAY/PODE neste documento são normativas. A distinção entre API pública e interna é especialmente importante porque SemVer exige uma API pública declarada; em SemVer, `1.0.0` passa a definir esse contrato e mudanças incompatíveis posteriores exigem incremento da versão major.

### Ordem de autoridade dentro do repositório

| Prioridade | Artefato | Autoridade |
| :--- | :--- | :--- |
| **Mais alta** | `docs/specs/spec-driven-development.md` | Contrato de produto, arquitetura e engenharia da v1 |
| **Alta** | ADRs aceitos em `docs/adr/` | Decisões arquiteturais que refinam, mas não contradizem esta spec |
| **Alta** | Feature specs em `docs/specs/features/` | Contrato da mudança específica |
| **Média** | Testes de contrato e aceitação | Evidência executável de conformidade |
| **Média** | Código-fonte | Implementação do contrato |
| **Informativa** | `README.md`, guides e examples | Explicação e experiência de uso |
| **Histórica** | Master Implementation Prompt | Origem da intenção; esta spec prevalece em caso de ambiguidade |
| **Exploratória** | Research notes e roadmap | Hipóteses não necessariamente implementadas |

Uma alteração que contradiga este documento não pode ser legitimada apenas por um ADR. A alteração deve modificar a própria spec, incluir justificativa, impacto, testes e, quando pertinente, proposta de mudança de API.

---

## Visão do produto, escopo e contrato da v1

**Propósito.** O EWM Engine é um framework open-source para representar e executar modelos de ambientes complexos, permitindo perguntar “what could happen if we do this?” em vez de limitar a computação a “what is likely to happen next?”. Essa distinção acompanha a própria motivação dos world models: modelar dinâmica para permitir que agentes ou políticas considerem trajetórias futuras, em vez de apenas produzir uma inferência pontual.

O projeto não assume que uma simulação condicional é automaticamente causal. Há literatura mostrando explicitamente que aprender regularidades espaço-temporais observacionais pode não ser suficiente para responder perguntas contrafactuais quando existem confundidores. Portanto, scenario branching é a terminologia normativa da v1. A palavra counterfactual deve aparecer apenas de forma qualificada na documentação ou em futuras extensões que explicitem suas hipóteses de identificação causal.

**Público-alvo.** A v1 deve ser utilizável por:
- ML/research engineers que desejem experimentar diferentes modelos de dinâmica;
- software engineers construindo simulações e sistemas de decisão;
- pesquisadores de world models, sistemas complexos, multiagentes ou neuro-simbólica;
- desenvolvedores de agentes que precisem de um ambiente independente do agente;
- equipes de operações, ciência de dados ou decisão que necessitem modelar regras explícitas e evolução estocástica.

**Fora do escopo da v1.** O EWM Engine não é um chatbot, framework de RAG, orquestrador de agentes, clone de Gym, biblioteca de causal inference, biblioteca de forecasting, digital twin visual, solver matemático ou implementação JEPA. Essas tecnologias podem interagir com o engine por meio de adapters.

### Preservação da tese e das hipóteses

| Elemento da pesquisa | Status na v1 | Decisão normativa |
| :--- | :--- | :--- |
| **EWM como “physics engine” do ambiente** | Preservar | Traduzir para estado + dinâmica + ações + eventos + constraints |
| **Modelos/agentes especialistas orbitam o EWM** | Preservar | `Actor` e adapters ficam ao redor do engine; não são o núcleo |
| **Regras explícitas + dinâmica aprendida** | Preservar | Constraints e dynamics são abstrações separadas |
| **Simular alternativas antes da ação real** | Preservar | Scenario branching é first-class |
| **Incerteza** | Preservar | Monte Carlo e distribuição de trajetórias fazem parte da v1 |
| **Rastreabilidade sistêmica** | Preservar | Provenance e systemic trace são first-class |
| **Causalidade automática da simulação** | Rejeitar | Simulação não é evidência causal por si só |
| **JEPA como arquitetura obrigatória** | Rebaixar a hipótese | Adapter/future learned dynamics |
| **SMT/Z3 obrigatório no núcleo** | Rebaixar a adapter | Constraints básicas no core; Z3 opcional posteriormente |
| **Constraint differentiable projection** | Experimental futuro | Não corrigir estados automaticamente na v1 |
| **LLM como requisito** | Rejeitar | Core deve funcionar sem LLM |
| **RLlib/AutoGen/LangGraph etc. no core** | Rejeitar | Adapters opcionais |
| **Aplicação financeira pública** | Fora do escopo | Repositório generalista |
| **CivicFlow** | Preservar | Demo social, sintética e educativa |
| **Benchmark como produto principal** | Não na v1 | Infraestrutura para benchmark, não benchmark como finalidade |

---

### Tabela A — Matriz MUST / SHOULD / FUTURE / PROMOÇÕES

| Capacidade | Prioridade Inicial | Maturidade Atual | Governança / ADR | Critério Normativo |
| :--- | :--- | :--- | :--- | :--- |
| `World` + `WorldState` | **MUST** | **Stable (v1.0.0)** | ADR-001, ADR-009 | Estado profundamente imutável, versionável e branchable |
| `Entity`, `Relationship`, `Resource` | **MUST** | **Stable (v1.0.0)** | ADR-001, ADR-007 | Representação tipada, serializável e defensiva |
| `Action` e `ExogenousEvent` | **MUST** | **Stable (v1.0.0)** | ADR-001, ADR-007 | Eventos controláveis e não controláveis separados |
| `DynamicsModel` protocol | **MUST** | **Stable (v1.0.0)** | ADR-002, ADR-007 | Dinâmica intercambiável sem alterar o simulation engine |
| Deterministic dynamics | **MUST** | **Stable (v1.0.0)** | ADR-002, ADR-005 | Implementação de referência para transferências de recursos |
| Stochastic dynamics | **MUST** | **Stable (v1.0.0)** | ADR-005, ADR-012 | Usa RNG explícito e `SeedSequence` derivado |
| Composite dynamics | **MUST** | **Stable (v1.0.0)** | ADR-002 | Composição determinística de componentes |
| Hard/soft constraints | **MUST** | **Stable (v1.0.0)** | ADR-003, ADR-008 | Semântica explícita `PRE_ACTION` e `POST_TRANSITION` |
| Scenario branching | **MUST** | **Stable (v1.0.0)** | ADR-005, ADR-012 | Mesmo snapshot, políticas isoladas sem contaminação |
| Monte Carlo rollouts | **MUST** | **Stable (v1.0.0)** | ADR-005, ADR-012 | `samples >= 1`, seed reproduzível e reordenamento determinístico |
| `Trajectory` e `SimulationResult` | **MUST** | **Stable (v1.0.0)** | ADR-001, ADR-007 | Histórico completo e machine-readable |
| Provenance + fingerprint | **MUST** | **Stable (v1.0.0)** | ADR-009, ADR-010 | SHA-256 estável e metadados de execução para auditoria |
| `EvidenceLevel` | **MUST** | **Stable (v1.0.0)** | ADR-010 | Qualificar epistemicamente relações e traces sem sobre-alegação |
| Systemic trace | **MUST** | **Stable (v1.0.0)** | ADR-010 | Grafo de dependência direcionado; proíbe arestas `"causes"` |
| Scenario comparison | **MUST** | **Stable (v1.0.0)** | ADR-001, ADR-016 | Comparação de métricas entre resultados com CIs bootstrap |
| JSON serialization | **MUST** | **Stable (v1.0.0)** | ADR-011 | Formato canônico canônico ordenado |
| Safe YAML input | **MUST** | **Stable (v1.0.0)** | ADR-011, ADR-018 | Alternativa humana segura; rejeita tags customizadas |
| JSON Schema artifacts | **MUST** | **Stable (v1.0.0)** | ADR-011 | Versionados em `schemas/` sob Draft 2020-12 |
| Minimal Warehouse | **MUST** | **Stable (v1.0.0)** | AC-012 | Quickstart e acceptance fixture canônica |
| CivicFlow | **MUST** | **Stable (v1.0.0)** | AC-013 | Flagship cross-domain/social demo |
| CI, typing, lint, tests, docs | **MUST** | **Stable (v1.0.0)** | ADR-014 | Quality gates bloqueantes de release |
| Hook/event protocol | **SHOULD** | **Stable (v1.0.0)** | ADR-013 | Observabilidade extensível e `RunMetrics` |
| Minimal CLI | **SHOULD** | **Stable (v1.0.0)** | ADR-007 | Camada de conveniência; API Python prevalece |
| Trace graph export | **SHOULD** | **Stable (v1.0.0)** | ADR-010 | Mermaid e NetworkX sem dependência mandatória |
| Scenario evaluation & Bootstrap | **FUTURE** | **Stable (v1.1.0)** | ADR-016, ACP-001 | CIs percentis bootstrap em deltas e fronteira de Pareto |
| Distributed Monte Carlo | **FUTURE** | **Beta Adapter (v1.1.0)** | ADR-017, ACP-003 | Execução paralela preservando `SeedSequence` (extra `parallel`) |
| OpenTelemetry adapter | **SHOULD** | **Beta Adapter (v1.1.0)** | ADR-024 | Observabilidade API-only sem acoplamento ao core (extra `otel`) |
| OR-Tools adapter | **FUTURE** | **Beta Adapter (v1.1.0)** | ADR-018 | CP-SAT discrete optimization com limites de tempo (extra `or`) |
| SciPy Continuous Planner | **FUTURE** | **Beta Adapter (v1.1.0)** | ADR-018 | Continuous allocation planner via HiGHS com limites de tempo |
| Z3 SMT solver adapter | **FUTURE** | **Beta Adapter (v1.1.0)** | ADR-018 | Verificação formal de invariantes simbólicos com timeout |
| Gymnasium RL adapter | **FUTURE** | **Beta Adapter (v1.1.0)** | ADR-018 | Wrapper `gymnasium.Env` para treinamento de políticas de agentes |
| Learned dynamics eval harness | **FUTURE** | **Experimental (v1.2.0)** | ADR-019 | Medição de erro multi-step, calibração e invariantes |
| Torch neural residual dynamics | **FUTURE** | **Experimental (v1.2.0)** | ADR-019 | Baseline neural MLP com symlog scaling (extra `ml`) |
| Scientific benchmark families | **FUTURE** | **Research (v1.2.0)** | ADR-025 | 5 famílias sintéticas de shift estrutural em `benchmarks/` |
| Planning & controller layer | **FUTURE** | **Experimental (v1.3.0)** | ADR-020 | Scorers de rollout (CVaR, constraints) e controle em horizonte móvel |
| OOD & regime-shift detection | **FUTURE** | **Experimental (v1.4.0)** | ADR-021 | Detecção de suporte e covariância de Mahalanobis sem auto-downgrade |
| Honest causal diagnostics | **FUTURE** | **Experimental (v1.4.0)** | ADR-021 | Backdoor, sobreposição de positividade e Twin Rollouts acoplados |
| Heterogeneous graph world state | **FUTURE** | **Experimental (v2.0-alpha)**| ADR-022, ACP-004 | Projeção em grafo heterogêneo temporal e migração v1 <-> v2 |
| Relational GNN dynamics | **FUTURE** | **Experimental (v2.0-alpha)**| ADR-022 | Message passing relacional sobre topologia temporal (extra `ml`) |
| World Specification Language (WSL)| **FUTURE**| **Experimental (v2.0-alpha)**| ADR-023, ACP-004 | Gramática declarativa segura, validação, compilação e exportação |
| MLflow adapter | **SHOULD** | **FUTURE (v1.2.0 T9)** | — | Optional tracking extra |
| RSSM/Dreamer-like dynamics | **FUTURE** | **Research** | — | Research latent adapter |
| JEPA dynamics | **FUTURE** | **Research** | — | Research latent adapter |
| LLM actor adapter | **FUTURE** | **Beta Adapter** | ADR-018 | `CallableActorAdapter` para agentes externos |
| Real-time data ingestion | **FUTURE** | **FUTURE (v1.5.0 T5)** | — | External adapter de telemetria |
| Web dashboard | **FUTURE** | **FUTURE** | — | Fora do engine |

---

### Objetivos normativos da v1

A versão `1.0.0` DEVE:
1. fornecer um kernel de simulação independente de domínio;
2. ter API pública documentada e semanticamente estável;
3. permitir bifurcação de cenários sem compartilhar estado mutável;
4. preservar determinismo sob configuração, versões e seed equivalentes dentro dos limites documentados;
5. tratar restrições explícitas como componentes de primeira classe;
6. registrar incerteza, provenance e systemic trace;
7. não introduzir implicitamente causalidade;
8. funcionar sem GPU, LLM, cloud, banco de dados ou solver externo;
9. instalar como pacote Python padrão;
10. possuir exemplos que executem como parte da suíte de testes.

---

### Tabela B — Critérios de aceitação mapeados a artefatos e testes

| ID | Critério testável | Artifact principal | Teste obrigatório |
| :--- | :--- | :--- | :--- |
| **AC-001** | Todos os símbolos Stable são importáveis da raiz pública | `src/ewm_engine/__init__.py` | `tests/contract/test_public_api.py` |
| **AC-002** | Modelos públicos serializáveis fazem JSON round-trip sem perda semântica | `serialization/` | `tests/contract/test_serialization_roundtrip.py` |
| **AC-003** | JSON Schemas versionados correspondem aos modelos atuais | `schemas/*.schema.json` | `tests/contract/test_schema_snapshots.py` |
| **AC-004** | Mesma entrada + versões + seed produz mesma trajetória built-in | `simulation/engine.py` | `tests/property/test_reproducibility.py` |
| **AC-005** | Branches derivados do mesmo snapshot não se contaminam | `simulation/branching.py` | `tests/property/test_branch_isolation.py` |
| **AC-006** | Hard constraint pre-action rejeita ação inválida | `constraints/` | `tests/integration/test_hard_constraints.py` |
| **AC-007** | Soft constraint registra violação e permite continuidade | `constraints/` | `tests/integration/test_soft_constraints.py` |
| **AC-008** | Hard post-transition violation invalida trajectory; não corrige estado silenciosamente | `simulation/engine.py` | `tests/integration/test_post_transition_violation.py` |
| **AC-009** | Monte Carlo retorna exatamente `samples` rollouts com RNG derivado | `simulation/engine.py` | `tests/property/test_monte_carlo.py` |
| **AC-010** | Provenance contém configuração mínima exigida e fingerprint estável | `provenance/models.py` | `tests/contract/test_provenance.py` |
| **AC-011** | Systemic trace contém IDs, timestamps/steps, edge type e EvidenceLevel | `provenance/trace.py` | `tests/contract/test_trace_contract.py` |
| **AC-012** | Warehouse fixture produz resultados normativos | `examples/minimal_warehouse/` | `tests/examples/test_minimal_warehouse.py` |
| **AC-013** | CivicFlow compara duas políticas e respeita hard constraints | `examples/civicflow/` | `tests/examples/test_civicflow.py` |
| **AC-014** | Todos os exemplos documentados executam | `README.md`, `docs/` | `tests/examples/test_documented_examples.py` |
| **AC-015** | Lint, format, type-check e testes passam | `pyproject.toml`, workflows | CI required checks |
| **AC-016** | Wheel/sdist constroem e wheel instala em ambiente limpo | packaging | `package-build` workflow |
| **AC-017** | Pacote distribui typing metadata | `src/ewm_engine/py.typed` | `tests/contract/test_packaging.py` |
| **AC-018** | YAML inseguro/custom tags são rejeitados; pickle não é aceito | `serialization/yaml.py` | `tests/security/test_deserialization.py` |
| **AC-019** | Core instala sem PyTorch, LLM SDK, solver ou database | `pyproject.toml` | `tests/contract/test_core_dependencies.py` |
| **AC-020** | Boundary rules entre módulos não são violadas | package tree | `tests/architecture/test_import_boundaries.py` |
| **AC-021** | README Quickstart executa do início ao fim | `README.md` | docs/example test |
| **AC-022** | Documentação compila em strict mode | `mkdocs.yml`, `docs/` | `docs-build` workflow |
| **AC-023** | Release não contém secrets ou credenciais | repo/workflows | secret/security scanning |
| **AC-024** | Public API de 1.x não quebra fixtures de contrato sem API Change Proposal | `tests/contract/` | compatibility gate |
| **AC-025** | Execução distribuída de Monte Carlo preserva resultados idênticos à serial | `simulation/executors.py` | `tests/property/test_distributed_determinism.py` |
| **AC-026** | Observabilidade OpenTelemetry é API-only e no-op quando dependência ausente | `integrations/otel.py` | `tests/unit/test_otel_adapter.py` |
| **AC-027** | Adapters de solvers e planners impõem limites de tempo e recursos com fallback | `integrations/` | `tests/integration/test_solvers_and_planners.py` |
| **AC-028** | Harness de dinâmica aprendida avalia divergência multi-step e invariantes | `experimental/dynamics_eval.py` | `tests/unit/test_dynamics_eval.py` |
| **AC-029** | Famílias de benchmark científico são reproduzíveis a partir de seed + versões | `benchmarks/protocol.py` | `tests/benchmark/test_scientific_benchmarks.py` |
| **AC-030** | Decisões de planejamento (MPC) são determinísticas, filtradas e auditadas | `experimental/planning.py` | `tests/unit/test_planning_scorers.py` |
| **AC-031** | Detector OOD reporta regime empírico; diagnóstico causal não auto-rotula EvidenceLevel | `experimental/ood.py`, `causal.py` | `tests/unit/test_ood_detection.py`, `test_causal_diagnostics.py` |
| **AC-032** | WSL rejeita código arbitrário, usa carregador seguro e compila para World container | `serialization/wsl.py` | `tests/unit/test_wsl.py`, `tests/unit/test_migration.py` |


---

## API pública, modelos e serialização

Superfície pública Stable para `1.0.0`:
```python
from ewm_engine import (
    Action,
    Constraint,
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
    DynamicsModel,
    Entity,
    EvidenceLevel,
    ExogenousEvent,
    Provenance,
    Relationship,
    Resource,
    Scenario,
    SimulationEngine,
    SimulationResult,
    TraceEdge,
    Trajectory,
    TrajectoryStatus,
    TransitionResult,
    World,
    WorldState,
    compare_scenarios,
)
```

Somente símbolos deliberadamente reexportados por `ewm_engine.__all__` pertencem à API Stable. Funcionalidades ainda instáveis devem residir sob:
```text
ewm_engine.experimental
```
e carregar explicitamente o rótulo Experimental.

### Modelo conceitual da API

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class EvidenceLevel(StrEnum):
    STRUCTURAL = "structural"
    INTERVENTIONAL = "interventional"
    QUASI_CAUSAL = "quasi_causal"
    PREDICTIVE = "predictive"
    ASSUMED = "assumed"


class Entity(FrozenModel):
    id: str
    kind: str
    attributes: Mapping[str, JsonValue] = Field(default_factory=dict)


class Relationship(FrozenModel):
    source_id: str
    target_id: str
    kind: str
    attributes: Mapping[str, JsonValue] = Field(default_factory=dict)


class Resource(FrozenModel):
    id: str
    owner_id: str | None = None
    quantity: float
    unit: str
    attributes: Mapping[str, JsonValue] = Field(default_factory=dict)


class Action(FrozenModel):
    id: str
    kind: str
    actor_id: str | None = None
    target_ids: tuple[str, ...] = ()
    parameters: Mapping[str, JsonValue] = Field(default_factory=dict)


class ExogenousEvent(FrozenModel):
    id: str
    kind: str
    target_ids: tuple[str, ...] = ()
    payload: Mapping[str, JsonValue] = Field(default_factory=dict)


class WorldState(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    state_id: str
    step: int
    timestamp: datetime
    entities: Mapping[str, Entity] = Field(default_factory=dict)
    relationships: tuple[Relationship, ...] = ()
    resources: Mapping[str, Resource] = Field(default_factory=dict)
    memory: Mapping[str, JsonValue] = Field(default_factory=dict)
    context: Mapping[str, JsonValue] = Field(default_factory=dict)
    active_rule_ids: tuple[str, ...] = ()


class TransitionResult(FrozenModel):
    next_state: WorldState
    evidence_level: EvidenceLevel = EvidenceLevel.ASSUMED
    component_id: str
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class DynamicsModel(Protocol):
    @property
    def component_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        *,
        rng: np.random.Generator,
    ) -> TransitionResult: ...
```

### Constraints

```python
class ConstraintPhase(StrEnum):
    PRE_ACTION = "pre_action"
    POST_TRANSITION = "post_transition"


class ConstraintSeverity(StrEnum):
    HARD = "hard"
    SOFT = "soft"


class ConstraintResult(FrozenModel):
    constraint_id: str
    constraint_version: str
    phase: ConstraintPhase
    severity: ConstraintSeverity
    satisfied: bool
    message: str | None = None
    entity_ids: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)


class Constraint(Protocol):
    @property
    def constraint_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def severity(self) -> ConstraintSeverity: ...

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult: ...
```

**Semântica normativa:**
- `HARD + PRE_ACTION + violated` → a ação deve ser rejeitada antes da transição;
- `SOFT + PRE_ACTION + violated` → registrar e continuar;
- `HARD + POST_TRANSITION + violated` → marcar aquele rollout como `INVALID` e interrompê-lo;
- `SOFT + POST_TRANSITION + violated` → registrar e continuar;
- o core NÃO DEVE projetar silenciosamente um estado inválido para um estado válido.

### Scenario, branching e execução

```python
class ScheduledAction(FrozenModel):
    step: int
    action: Action


class Scenario(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    scenario_id: str
    name: str
    initial_state: WorldState
    horizon: int = Field(gt=0)
    samples: int = Field(default=1, gt=0)
    seed: int
    scheduled_actions: tuple[ScheduledAction, ...] = ()
    metadata: Mapping[str, JsonValue] = Field(default_factory=dict)

    def branch(
        self,
        *,
        scenario_id: str,
        name: str | None = None,
        scheduled_actions: Sequence[ScheduledAction] | None = None,
        seed: int | None = None,
    ) -> "Scenario": ...


class SimulationEngine:
    def run(
        self,
        *,
        world: World,
        scenario: Scenario,
    ) -> SimulationResult: ...
```

---

## Templates Normativos

### ADR Template (`.github/ADR_TEMPLATE.md`)
```markdown
# ADR-XXX: <Decision Title>

Status: Proposed
Date: YYYY-MM-DD
Owners: <names>
Related Spec: <path>
Related Issues: <ids>

## Context

Describe the problem, constraints, and why a decision is needed.

## Decision

State the decision precisely.

## Decision Drivers

- <driver>
- <driver>

## Considered Options

### Option A

Description.

Pros:
- ...

Cons:
- ...

### Option B

Description.

Pros:
- ...

Cons:
- ...

## Consequences

### Positive

- ...

### Negative

- ...

## Compatibility Impact

Public API:
Serialization:
Determinism:
Security:
Performance:
Dependencies:

## Validation

List tests, benchmarks, or experiments that validate the decision.

## Rollback / Supersession

Describe how the decision could be reverted or superseded.
```

### API Change Proposal Template (`.github/API_CHANGE_PROPOSAL.md`)
```markdown
# API Change Proposal: <Title>

Status: Proposed
Target Release: <version>
Owner: <name>

## Motivation

Why is the API change needed?

## Current Contract

```python
# Current API
```

## Proposed Contract

```python
# Proposed API
```

## Compatibility Classification
[ ] Backward-compatible addition
[ ] Deprecation
[ ] Backward-incompatible change
[ ] Serialization change
[ ] Behavioral change

## Impacted Artifacts
Public imports:
Type signatures:
JSON schemas:
Documentation:
Examples:
Tests:

## Migration
Describe the consumer migration path.

## Deprecation Plan
First deprecated version:
Earliest removal version:
Earliest removal date:

## Acceptance Criteria
- [ ] ...
- [ ] ...

## Alternatives Considered
Describe alternatives and why they were rejected.
```

### Release Checklist Template (`.github/RELEASE_CHECKLIST.md`)
```markdown
# Release Checklist

Release: X.Y.Z

## Specification

- [ ] All target acceptance criteria are satisfied.
- [ ] No unresolved specification conflicts remain.
- [ ] Required ADRs are accepted.
- [ ] API changes are approved.

## Quality

- [ ] Ruff lint passes.
- [ ] Ruff format check passes.
- [ ] Static type checking passes.
- [ ] Unit tests pass.
- [ ] Property tests pass.
- [ ] Integration tests pass.
- [ ] Contract tests pass.
- [ ] Security tests pass.
- [ ] Architecture tests pass.
- [ ] Example tests pass.
- [ ] Coverage gates pass.

## Documentation

- [ ] Documentation builds in strict mode.
- [ ] README Quickstart is executable.
- [ ] Public API reference is current.
- [ ] CHANGELOG is updated.
- [ ] Migration notes are present when required.
- [ ] Version references are current.

## Packaging

- [ ] Source distribution builds.
- [ ] Wheel builds.
- [ ] Wheel installs in a clean environment.
- [ ] `py.typed` is present in the distribution.
- [ ] Core install does not pull optional ML/solver/LLM dependencies.

## Security

- [ ] Security scanning passes.
- [ ] No secrets are present.
- [ ] Release workflows use approved pinned actions.
- [ ] Publishing uses trusted identity/OIDC where available.

## Release

- [ ] Version is SemVer-compliant.
- [ ] Git tag is created.
- [ ] GitHub Release is created.
- [ ] Package is published.
- [ ] Documentation is published.
- [ ] Release notes link to relevant ADRs/API changes.
```

### Test-Case Template (`.github/TEST_CASE_TEMPLATE.md`)
```markdown
# Test Case: <Title>

ID: TC-XXX
Related Acceptance Criteria: AC-XXX
Type: Unit | Integration | Property | Contract | Security | Regression

## Purpose

Describe the contract being validated.

## Preconditions

- ...

## Input

Describe or link to the fixture.

## Action

Describe the operation under test.

## Expected Result

Describe exact observable behavior.

## Invariants

- ...

## Failure Meaning

Explain what a failure indicates about the product contract.
```

### Pull Request Checklist Template (`.github/PULL_REQUEST_TEMPLATE.md`)
```markdown
## Purpose

Describe what this PR changes and why.

## Related Specification

- Issue:
- Feature spec:
- Acceptance criteria:
- ADR:
- API change proposal:

## Change Type

- [ ] Feature
- [ ] Bug fix
- [ ] Documentation
- [ ] Refactor
- [ ] Performance
- [ ] Security
- [ ] API change

## Verification

- [ ] Tests added or updated.
- [ ] Regression test added for a bug fix.
- [ ] Property tests added where an invariant is involved.
- [ ] Documentation updated.
- [ ] Examples updated when public behavior changed.
- [ ] JSON schemas updated when serialized models changed.
- [ ] Ruff passes.
- [ ] Type checking passes.
- [ ] Full relevant test suite passes.
- [ ] Documentation builds.
- [ ] Package builds.

## Architecture and Safety

- [ ] No unrelated dependency was added.
- [ ] No unsafe deserialization was introduced.
- [ ] No hidden network call was introduced.
- [ ] No global random state was introduced.
- [ ] No domain-specific logic leaked into the core.
- [ ] No causal claim was introduced without explicit evidence.
- [ ] No public API changed without an approved API proposal.

## Reviewer Notes

Describe trade-offs, known limitations, or follow-up work.
```

---

## Acceptance Fixtures

### Warehouse Acceptance Scenario
A fixture determinística mínima:
```yaml
scenario:
  id: "warehouse-transfer-v1"
  horizon: 1
  seed: 42

warehouses:
  warehouse_a:
    capacity: 150
    inventory: 100
  warehouse_b:
    capacity: 150
    inventory: 20

demand:
  warehouse_b: 50
```

- Baseline (No transfer):
  - `served_demand`: 20.0
  - `unserved_demand`: 30.0
  - `final_inventory`: `{"warehouse_a": 100.0, "warehouse_b": 0.0}`
  - `hard_constraint_violations`: 0

- Intervention (Transfer 30 units from A to B before demand):
  - `served_demand`: 50.0
  - `unserved_demand`: 0.0
  - `final_inventory`: `{"warehouse_a": 70.0, "warehouse_b": 0.0}`
  - `hard_constraint_violations`: 0

### CivicFlow Acceptance Scenario
Simulação educativa/pesquisa sobre inundação e logística humanitária:
- Compara `nearest_shelter_first` vs `capacity_aware_allocation`.
- `capacity_aware_allocation` reduz `unserved_demand` sem violação de hard constraints.
- Systemic trace documenta dependências com `EvidenceLevel` apropriado sem alegação causal direta.

---

## Definition of Done da v1

```text
A v1.0.0 exists only when:

- the public contract is explicit;
- the core is domain-independent;
- states are safely branchable;
- dynamics are pluggable;
- constraints are first-class;
- stochastic rollouts are reproducible;
- provenance is complete;
- traces are inspectable without being mislabeled as causal;
- schemas are safe and versioned;
- Warehouse proves the minimal concept;
- CivicFlow proves a richer systemic scenario;
- tests enforce the invariants;
- documentation explains the boundaries;
- CI enforces engineering quality;
- optional technologies remain optional;
- and a new coding agent can continue development from the repository
  without needing the original conversation that created it.
```

---

## Evolução Pós-v1: Extensão Normativa e Critérios de Aceitação (v1.1.0 — v2.0-alpha)

Para garantir que coding agents futuros e contribuidores mantenham a integridade arquitetural sem reinterpretar intenções, esta seção estende formalmente o contrato normativo além da versão `1.0.0`:

### 1. Critérios de Aceitação Pós-v1 (AC-025 a AC-032)

#### AC-025: Determinismo Rigoroso em Monte Carlo Distribuído (v1.1.0)
- **Declaração Normativa:** A execução paralela/distribuída de rollouts de Monte Carlo DEVE produzir trajetórias e métricas logicamente e numericamente idênticas à execução serial sob os mesmos seeds, cenários e configurações de mundo.
- **Invariante:** Cada worker consome a mesma sub-semente gerada deterministicamente por `np.random.SeedSequence(scenario.seed).spawn(samples)[rollout_idx]`. Os resultados DEVEM ser reagrupados na ordem ordinal estrita dos rollouts (índices `0` a `samples - 1`), independentemente da ordem assíncrona de conclusão dos processos.
- **Artefato:** `src/ewm_engine/simulation/executors.py`
- **Governança:** ADR-017, ACP-003. Teste obrigatório: `tests/property/test_distributed_determinism.py`.

#### AC-026: Observabilidade OpenTelemetry API-Only e Desacoplada (v1.1.0)
- **Declaração Normativa:** O adapter de telemetria DEVE registrar-se como listener no `HookRegistry` existente consumindo exclusivamente a API pública de instrumentação, sem impor o SDK do OpenTelemetry como dependência do core.
- **Invariante:** Quando o extra opcional `[otel]` não estiver instalado, a importação ou presença do adapter NÃO DEVE lançar exceções e DEVE degradar graciosamente para no-op sem qualquer degradação de performance no loop de simulação.
- **Artefato:** `src/ewm_engine/integrations/otel.py`
- **Governança:** ADR-013, ADR-024. Teste obrigatório: `tests/unit/test_otel_adapter.py`.


#### AC-027: Limites Estritos de Recursos e Timeouts em Planners e Solvers (v1.1.0)
- **Declaração Normativa:** Todo adapter de Operations Research (Google OR-Tools, SciPy) e de verificação formal (Z3 SMT) DEVE expor parâmetros obrigatórios de limite de tempo (`time_limit_seconds` ou `timeout_ms`).
- **Invariante:** Quando a busca exceder o tempo limite alocado, o solver/planner DEVE abortar graciosamente retornando um resultado explícito de timeout (`satisfied=False`, `timed_out=True`), sem travar o kernel de simulação, sem vazamento de memória e sem lançar exceções não tratadas. Planners propõem ações; constraints dispõem.
- **Artefato:** `src/ewm_engine/integrations/` (`ortools.py`, `scipy_planner.py`, `solvers.py`)
- **Governança:** ADR-018. Teste obrigatório: `tests/integration/test_solvers_and_planners.py`.


#### AC-028: Harness de Avaliação e Invariantes para Dinâmica Aprendida (v1.2.0)
- **Declaração Normativa:** O framework DEVE disponibilizar harness científico para medir qualquer implementação de `DynamicsModel` ou `LearnedDynamics` contra métricas canônicas de erro multi-step, calibração estocástica e verificação de invariantes.
- **Invariante:** O harness DEVE medir: (1) erro de previsão pontual em 1-passo ($MAE$, $RMSE$); (2) divergência autorregressiva em rollouts multi-step; (3) calibração estocástica via CRPS e coverage; (4) taxa de violação de invariantes declarados de recursos ($[min, max]$); (5) hiato de generalização intervencional ($MAE_{\text{interventional}} - MAE_{\text{in-distribution}}$).
- **Artefato:** `src/ewm_engine/experimental/dynamics_eval.py`
- **Governança:** ADR-019. Teste obrigatório: `tests/unit/test_dynamics_eval.py`.

#### AC-029: Reprodutibilidade de Famílias de Benchmark Científico (v1.2.0)
- **Declaração Normativa:** O repositório DEVE disponibilizar cinco famílias canônicas de benchmark sintético (`InterventionShift`, `RuleShift`, `ConstraintStress`, `LongHorizon`, `MultiAgentCascade`) como instrumentos de medição para pesquisa em world models.
- **Invariante:** Qualquer execução de benchmark com os mesmos seeds canônicos e versões de dependências DEVE produzir relatórios estruturados idênticos (`BenchmarkReport`) contendo proveniência criptográfica completa. O benchmark é um instrumento de medição, não o produto do repositório.
- **Artefato:** `benchmarks/protocol.py`, `benchmarks/families/`
- **Governança:** ADR-025. Teste obrigatório: `tests/benchmark/test_scientific_benchmarks.py`.

#### AC-030: Determinismo e Auditabilidade na Camada de Planejamento e Controle (v1.3.0)
- **Declaração Normativa:** O controlador de horizonte móvel (MPC) e scorers de rollout DEVEM tomar decisões determinísticas e auditáveis, operando sob re-ancoragem contínua (re-grounding).
- **Invariante:** Rollouts abertos longos NÃO SÃO previsões. Cada decisão de planejamento DEVE emitir `PlanningDecision` registrando candidatos avaliados, sementes de lookahead, amostras, pontuações de utilidade e hashes de estado antes e depois da execução. Ações propostas passam obrigatoriamente pelo portão de restrições do mundo hospedeiro.
- **Artefato:** `src/ewm_engine/experimental/planning.py`, `src/ewm_engine/simulation/mpc.py`
- **Governança:** ADR-020. Teste obrigatório: `tests/unit/test_planning_scorers.py`, `tests/integration/test_planning_controller.py`.

#### AC-031: Detecção de Regime OOD e Honestidade Causal sem Auto-Rotulação (v1.4.0)
- **Declaração Normativa:** O engine DEVE diagnosticar quando uma trajetória deixa o suporte empírico calibrado e avaliar hipóteses causais explicitamente declaradas, sem jamais emitir alegações causais automáticas a partir de dados observacionais.
- **Invariante:** O detector de OOD computa `grounded_fraction` e anota o `SystemicTrace` sem alterar o status da simulação nem elevar arbitrariamente o `EvidenceLevel`. Verificações de Backdoor e diagnósticos de sensibilidade de Rosenbaum operam como ferramentas epistêmicas: o engine JAMAIS descobre DAGs causais automaticamente a partir de correlações brutas.
- **Artefato:** `src/ewm_engine/experimental/ood.py`, `src/ewm_engine/experimental/causal.py`
- **Governança:** ADR-021. Teste obrigatório: `tests/unit/test_ood_detection.py`, `tests/unit/test_causal_diagnostics.py`.

#### AC-032: Execução Declarativa Segura na World Specification Language (WSL) (v2.0-alpha)
- **Declaração Normativa:** A especificação declarativa de mundos em YAML/JSON (WSL) DEVE ser 100% livre de execução de código arbitrário e validar contra schema JSON formal.
- **Invariante:** `parse_wsl_file` e `parse_wsl_yaml` utilizam carregador seguro estrito (`StrictSafeLoader`), rejeitando tags customizadas (ex: `!python/object`), chamadas inline (`eval`, `exec`) e importações arbitrárias de strings. Componentes dinâmicos são instanciados exclusivamente a partir de um `ComponentRegistry` programático pré-aprovado. A migração entre `schema_version = "1.0.0"` e `"2.0.0"` DEVE ser bidirecional e sem perda de dados.
- **Artefato:** `src/ewm_engine/serialization/wsl.py`, `src/ewm_engine/core/graph.py`, `src/ewm_engine/core/migration.py`
- **Governança:** ADR-022, ADR-023, ACP-004. Teste obrigatório: `tests/unit/test_wsl.py`, `tests/unit/test_migration.py`.

---

### 2. Regra de Promoção de Maturidade e Governança SemVer

1. **Promoção FUTURE $\to$ BETA / EXPERIMENTAL:**
   Nenhum item pode ser promovido de `FUTURE` para `BETA` ou `EXPERIMENTAL` sem um Architecture Decision Record (ADR) aceito e testes de unidade/integração correspondentes.
2. **Promoção BETA / EXPERIMENTAL $\to$ STABLE:**
   Exige um API Change Proposal (ACP) formal aprovado, snapshot de schema validado, freeze de assinaturas públicas em `tests/contract/test_api_compatibility.py` e período mínimo de estabilidade.
3. **Imutabilidade do Core v1:**
   A inclusão de novos módulos experimentais (`ewm_engine.experimental.*`) ou adapters (`ewm_engine.integrations.*`) NÃO PODE introduzir dependências obrigatórias no core nem quebrar a compatibilidade retroativa dos 22 símbolos Stable declarados em `ewm_engine.__all__`.

