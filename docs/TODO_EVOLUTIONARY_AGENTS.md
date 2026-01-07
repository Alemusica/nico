# 🧬 TODO: Evolutionary Agents & AI Supervision

> **Status**: Future Implementation  
> **Priority**: HIGH  
> **Dependencies**: API Registry, Data Aggregation Pipeline, SurrealDB Knowledge Graph

---

## 🎯 Objective

Implement an **evolutionary multi-agent system** for:
1. **Data Quality Supervision** - Agent monitors and validates incoming data
2. **Pattern Discovery** - Agents explore causal relationships
3. **Knowledge Distillation** - Transfer learning from historical events
4. **Human-in-the-Loop** - Expert validation of discovered patterns

---

## 📚 Reference Frameworks

### DeepMind Approaches
- [ ] **AlphaFold-style iterative refinement** - Multiple rounds of prediction/correction
- [ ] **MuZero planning** - Model-based RL for prediction
- [ ] **RLHF (Reinforcement Learning from Human Feedback)** - Human validation loop

### Evolutionary Algorithms
- [ ] **NSGA-II** - Multi-objective optimization (from Remembrance)
- [ ] **CMA-ES** - Covariance Matrix Adaptation
- [ ] **Novelty Search** - Explore for novel precursors

### Multi-Agent Systems
- [ ] **AutoGen** - Microsoft's multi-agent framework
- [ ] **CrewAI** - Role-based agent collaboration
- [ ] **LangGraph** - Graph-based agent workflows

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┐ │
│  │  Data QA     │  Pattern     │  Knowledge   │  Validation  │ │
│  │  Agent       │  Discovery   │  Distiller   │  Agent       │ │
│  │              │  Agent       │  Agent       │              │ │
│  └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘ │
│         │              │              │              │         │
│         ▼              ▼              ▼              ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SHARED MEMORY / SURREALDB                  │   │
│  │  - Observations, Patterns, Validations, Fitness scores │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                     ┌─────────────────┐                        │
│                     │ HUMAN IN LOOP   │                        │
│                     │ Review Interface│                        │
│                     └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Implementation Tasks

### Phase 1: Foundation (Current Sprint)
- [x] API Registry with status tracking
- [x] Health checker for all data sources
- [x] SurrealDB ingestion scripts
- [ ] Base Agent class with common interface
- [ ] Shared memory/state in SurrealDB

### Phase 2: Data Quality Agent
- [ ] Schema validation for incoming data
- [ ] Anomaly detection (statistical + ML)
- [ ] Missing data handling strategies
- [ ] Data freshness monitoring
- [ ] Cross-source consistency checks

### Phase 3: Pattern Discovery Agent
- [ ] Integrate PCMCI for causal discovery
- [ ] Novelty search for unexplored correlations
- [ ] Multi-scale pattern detection (hourly → monthly)
- [ ] Cross-region teleconnection discovery
- [ ] Confidence scoring for patterns

### Phase 4: Knowledge Distiller Agent
- [ ] Extract patterns from historical events
- [ ] Paper/literature semantic search
- [ ] News article entity extraction
- [ ] Witness testimony parsing
- [ ] Pattern library building

### Phase 5: Validation Agent (Human-in-Loop)
- [ ] Gray zone pattern queue
- [ ] Expert review interface
- [ ] Feedback collection → agent learning
- [ ] Pattern promotion/demotion workflow
- [ ] Explanation generation for experts

### Phase 6: Evolutionary Optimization
- [ ] Population of competing hypotheses
- [ ] Fitness function: physics + experience + novelty
- [ ] Selection/crossover/mutation operators
- [ ] Transfer learning from Remembrance RDNN
- [ ] Curriculum learning for complex patterns

---

## 🔧 Technical Components

### Agent Base Class
```python
class BaseAgent:
    """Base class for all CTW agents."""
    
    def __init__(self, name: str, memory: SurrealDBMemory):
        self.name = name
        self.memory = memory
        self.state = AgentState()
    
    async def observe(self) -> Observation:
        """Gather data from environment."""
        pass
    
    async def think(self, observation: Observation) -> Action:
        """Decide next action."""
        pass
    
    async def act(self, action: Action) -> Result:
        """Execute action."""
        pass
    
    async def learn(self, result: Result):
        """Update internal state from feedback."""
        pass
    
    async def report(self) -> Dict:
        """Report status to supervisor."""
        pass
```

### Shared Memory Schema (SurrealDB)
```sql
-- Agent observations
DEFINE TABLE agent_observation SCHEMAFULL;
DEFINE FIELD agent_id ON agent_observation TYPE string;
DEFINE FIELD timestamp ON agent_observation TYPE datetime;
DEFINE FIELD observation_type ON agent_observation TYPE string;
DEFINE FIELD data ON agent_observation TYPE object;
DEFINE FIELD confidence ON agent_observation TYPE float;

-- Discovered patterns
DEFINE TABLE discovered_pattern SCHEMAFULL;
DEFINE FIELD pattern_id ON discovered_pattern TYPE string;
DEFINE FIELD discovered_by ON discovered_pattern TYPE string;
DEFINE FIELD variables ON discovered_pattern TYPE array;
DEFINE FIELD lag_hours ON discovered_pattern TYPE int;
DEFINE FIELD strength ON discovered_pattern TYPE float;
DEFINE FIELD physics_score ON discovered_pattern TYPE float;
DEFINE FIELD experience_score ON discovered_pattern TYPE float;
DEFINE FIELD status ON discovered_pattern TYPE string; -- pending, validated, rejected, gray_zone

-- Human validations
DEFINE TABLE human_validation SCHEMAFULL;
DEFINE FIELD pattern_id ON human_validation TYPE string;
DEFINE FIELD expert_id ON human_validation TYPE string;
DEFINE FIELD decision ON human_validation TYPE string;
DEFINE FIELD confidence ON human_validation TYPE float;
DEFINE FIELD notes ON human_validation TYPE string;
DEFINE FIELD timestamp ON human_validation TYPE datetime;

-- Agent fitness/performance
DEFINE TABLE agent_fitness SCHEMAFULL;
DEFINE FIELD agent_id ON agent_fitness TYPE string;
DEFINE FIELD generation ON agent_fitness TYPE int;
DEFINE FIELD accuracy ON agent_fitness TYPE float;
DEFINE FIELD novelty ON agent_fitness TYPE float;
DEFINE FIELD physics_compliance ON agent_fitness TYPE float;
DEFINE FIELD human_approval_rate ON agent_fitness TYPE float;
```

---

## 📖 References

### Papers
1. **AlphaFold2** - Jumper et al. 2021 - Iterative prediction
2. **MuZero** - Schrittwieser et al. 2020 - Model-based planning
3. **RLHF** - Ouyang et al. 2022 - Human feedback integration
4. **PCMCI** - Runge et al. 2019 - Causal discovery
5. **Novelty Search** - Lehman & Stanley 2011 - Open-ended evolution

### Repositories
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/joaomdmoura/crewAI)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Remembrance RDNN](../binaural_golden/src/core/rdnn_memory.py)

### Related CTW Components
- `src/surge_shazam/causal/pcmci_runner.py` - Causal discovery
- `src/pattern_engine/early_warning.py` - Alert system
- `src/data_manager/causal_graph.py` - SurrealDB graph
- `audit_agents/` - Existing agent framework

---

## 📅 Timeline

| Phase | Description | Est. Time |
|-------|-------------|-----------|
| 1 | Foundation | ✅ Done |
| 2 | Data QA Agent | 1 week |
| 3 | Pattern Discovery | 2 weeks |
| 4 | Knowledge Distiller | 2 weeks |
| 5 | Human-in-Loop | 1 week |
| 6 | Evolutionary | 3 weeks |

**Total estimated**: ~10 weeks for full implementation

---

## 🔗 Integration Points

### With Remembrance
```python
# Transfer RDNN memory patterns
from remembrance.binaural_golden.src.core.rdnn_memory import RDNNMemory

class PatternDiscoveryAgent(BaseAgent):
    def __init__(self, ...):
        # Reuse RDNN for experience transfer
        self.rdnn = RDNNMemory(
            input_dim=...,  # Observation features
            hidden_dim=128,
            output_dim=...  # Action suggestions
        )
```

### With Existing Audit Agents
```python
# audit_agents/ already has agent structure
# Extend for new capabilities
```

---

*Last updated: 2026-01-07*
