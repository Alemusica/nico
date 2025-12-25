# 🔍 Project Audit & Multi-Agent Architecture

**Data**: 25 Dicembre 2025  
**Branch**: refactor/router-split  
**Status**: Sprint 1.2 Complete, Knowledge System Operational

---

## 📊 ANALISI ARCHITETTURA COMPLETA

### 🏗️ Componenti Sistema

#### 1. **API Layer** (FastAPI)
```
api/
├── main.py                     # App FastAPI, router registration
├── config.py                   # Settings (Pydantic)
├── exceptions.py               # Custom exceptions
├── logging_config.py           # StructLog + rotation
├── middleware.py               # Request ID tracking
├── rate_limit.py               # Limitatore chiamate
├── security.py                 # Headers, input validation
├── connection_pool.py          # Circuit breaker pattern
└── routers/
    ├── analysis_router.py      # Causal discovery (PCMCI)
    ├── chat_router.py          # LLM conversation
    ├── data_router.py          # Upload/cache dataset
    ├── health_router.py        # Health checks
    ├── investigation_router.py # WebSocket streaming
    ├── knowledge_router.py     # Papers/Events/Patterns
    └── pipeline_router.py      # Full research pipeline
```

**Status**:
- ✅ 7 routers modulari (da monolitico 960 righe)
- ✅ Production middleware (logging, rate limit, security)
- ✅ Connection pool con circuit breaker
- ⚠️ Test coverage 40% (target: 80%)

#### 2. **Services Layer**
```
api/services/
├── llm_service.py              # Ollama integration
├── causal_service.py           # PCMCI + Tigramite
├── data_service.py             # Dataset operations
└── knowledge_service.py        # Knowledge base interface
    ├── knowledge_service.py    # Abstract base (20+ methods)
    ├── minimal_knowledge.py    # In-memory impl (CURRENT)
    ├── surrealdb_knowledge.py  # Full impl (incomplete)
    └── neo4j_knowledge.py      # Legacy fallback
```

**Status**:
- ✅ MinimalKnowledgeService operativo (in-memory)
- ✅ Factory con fallback graceful
- ⚠️ SurrealDB incomplete (18 abstract methods mancanti)
- ❌ Neo4j missing `get_stats()` 

#### 3. **Investigation Agent** (Core Logic)
```
src/agent/
├── investigation_agent.py      # Orchestrator principale
├── investigation_logger.py     # Structured logging
├── investigation_validators.py # Data quality checks
└── tools/
    ├── geo_resolver.py         # Location → BBox
    ├── literature_scraper.py   # arXiv/Copernicus
    └── (altri tools)
```

**Flusso Investigation**:
1. Parse query → `EventContext` (location, dates, event_type)
2. **Parallel data collection** (asyncio.gather):
   - CMEMS: SSH, currents
   - ERA5: wind, pressure, temperature, humidity
   - Climate indices: NAO, AO, ENSO
   - Papers: arXiv search
3. Validation → structured logging
4. **BUG FIX**: Papers ora salvati con `bulk_add_papers()` 
5. Correlation analysis (Pearson/PCMCI)
6. Report generation

**Status**:
- ✅ Streaming via WebSocket
- ✅ Multi-source parallel collection
- ✅ Structured logging (JSON)
- ✅ Validation framework (Great Expectations style)
- ✅ Knowledge service integration fixed

#### 4. **Data Layer** 
```
src/surge_shazam/data/
├── cmems_client.py             # Copernicus Marine SSH
├── era5_client.py              # ECMWF Reanalysis
├── climate_indices.py          # NAO, AO, ENSO
└── geoid/                      # Geoid correction

src/data_manager/
├── manager.py                  # Briefing + download orchestrator
├── config.py                   # Resolution configs
└── loaders/
    ├── cmems_loader.py
    ├── era5_loader.py
    └── climate_loader.py

data/
├── aviso/                      # Altimetry
├── cmems/                      # SSH data
├── slcci/                      # Sea level
├── cache/                      # Downloaded data
└── pipeline/                   # Pipeline outputs
```

**Status**:
- ✅ CMEMS client operativo
- ✅ ERA5 client con humidity vars
- ✅ Cache management
- ⚠️ Data Explorer returns 503 (DataManager issue)

#### 5. **Knowledge Base** (Papers/Events/Patterns)
```
api/services/minimal_knowledge.py
├── Papers: {id: Paper}         # In-memory dict
├── Events: {id: HistoricalEvent}
├── Patterns: {id: CausalPattern}
└── Indices: {id: ClimateIndex}
```

**Current State**:
```json
{
  "papers": 0,
  "events": 0, 
  "patterns": 0,
  "climate_indices": 0,
  "relationships": 0
}
```

**Issue Identificato**:
- Investigation pre-fix usava SurrealDB (fallito → knowledge_service=None)
- Papers validati (30) ma NON salvati
- Fix: MinimalKnowledgeService default + tutti metodi implementati
- **Soluzione**: Rifare investigation con backend riavviato

#### 6. **Frontend** (React + TypeScript + Vite)
```
frontend/src/
├── pages/
│   ├── KnowledgeBasePage.tsx   # Papers/Events browser
│   ├── DataExplorerPage.tsx    # Dataset manager
│   ├── AIAssistantPage.tsx     # Chat + Investigation
│   └── ...
├── components/
│   ├── KnowledgeStats.tsx      # Stats cards
│   └── InvestigationInterface.tsx
└── services/
    └── api.ts                  # API client
```

**Status**:
- ✅ Running on port 5173
- ✅ Connected to backend API
- ✅ WebSocket streaming working
- ⚠️ Stats showing 0 (expected, waiting new investigation)

#### 7. **Pipeline Engine**
```
src/pipeline/
├── scraper.py                  # News/Papers scraper
├── raffinatore.py              # LLM extraction
├── correlatore.py              # Temporal correlation
├── knowledge_scorer.py         # Hypothesis scoring
└── research_pipeline.py        # Full orchestrator
```

**Status**:
- ⚠️ Non testato in questa sessione
- ⚠️ `/api/v1/pipeline/run` disponibile ma non verificato

---

## 🎯 GOAL ANALYSIS

### Obiettivo Primario
**"Sistema di causal discovery per eventi oceanografici con knowledge base integrata"**

#### Componenti Necessari
1. ✅ Data ingestion (CMEMS, ERA5, climate indices)
2. ✅ Causal discovery (PCMCI)
3. ✅ Knowledge base (Papers/Events/Patterns)
4. ✅ Investigation agent (orchestrazione automatica)
5. ✅ API modulare production-ready
6. ✅ Frontend interattivo
7. ⚠️ Persistence layer (in-memory → SurrealDB/Neo4j)
8. ❌ Test coverage insufficiente

### Gap Identificati

#### 🔴 CRITICAL
1. **Knowledge persistence**: In-memory → dati persi a restart
2. **Test coverage**: 40% → 80% target
3. **Data Explorer**: Returns 503 (DataManager issue)

#### 🟡 MEDIUM
1. **SurrealDB incomplete**: 18 abstract methods mancanti
2. **Neo4j no stats**: Missing `get_stats()` method
3. **Pipeline untested**: Research pipeline non verificato
4. **Cache stats**: Returns 503

#### 🟢 LOW
1. **Documentation**: API docs complete, architecture partial
2. **Monitoring**: Logs ok, metrics missing (Prometheus)
3. **Deployment**: No Docker compose, no CI/CD

---

## 🤖 MULTI-AGENT AUDIT ARCHITECTURE

### Topics Identificati

#### 1. **Data Flow & Integration** 🌊
**Scope**: CMEMS, ERA5, Climate Indices, cache management  
**Agent Specializz**: `DataFlowAuditor`

**Audit Points**:
- [ ] CMEMS client: connection, download, error handling
- [ ] ERA5 client: humidity vars, temporal resolution
- [ ] Climate indices: NAO/AO/ENSO availability
- [ ] Cache: read/write, cleanup, stats endpoint
- [ ] Data manager: briefing creation, download orchestration
- [ ] File formats: NetCDF, CSV, ZARR support
- [ ] Geoid correction: accuracy, edge cases

**Metriche**:
- Connection success rate
- Download time per GB
- Cache hit ratio
- Data quality scores

---

#### 2. **Investigation Pipeline** 🔍
**Scope**: Agent orchestration, WebSocket, validation  
**Agent Specializz**: `InvestigationAuditor`

**Audit Points**:
- [ ] Query parsing: location extraction, date ranges
- [ ] GeoResolver: bbox accuracy, fallback logic
- [ ] Parallel collection: asyncio.gather timing
- [ ] Literature scraper: arXiv API, rate limits
- [ ] Validation: paper metadata, critical failures
- [ ] Knowledge service: bulk_add success, rollback
- [ ] WebSocket: message order, error propagation
- [ ] Logging: completeness, correlation IDs

**Metriche**:
- Pipeline success rate
- Average investigation time
- Papers found per query
- Validation pass rate

---

#### 3. **Knowledge System** 📚
**Scope**: Services, persistence, search  
**Agent Specializz**: `KnowledgeAuditor`

**Audit Points**:
- [ ] MinimalKnowledgeService: thread-safety, memory leaks
- [ ] Abstract interface: method coverage, documentation
- [ ] SurrealDB impl: complete 18 missing methods
- [ ] Neo4j impl: add missing get_stats, test queries
- [ ] Factory fallback: error handling, logging clarity
- [ ] Paper operations: bulk_add, search, dedupe
- [ ] Event operations: correlations, temporal queries
- [ ] Pattern validation: confidence scoring

**Metriche**:
- Query response time
- Memory usage per 1000 papers
- Abstract method coverage
- Backend availability

---

#### 4. **API Layer & Security** 🔐
**Scope**: Routers, middleware, rate limiting  
**Agent Specializz**: `APIAuditor`

**Audit Points**:
- [ ] Router modularity: line count, complexity
- [ ] Endpoint coverage: all 7 routers tested
- [ ] Rate limiting: per-endpoint configs, Redis integration
- [ ] Security headers: CSP, HSTS, X-Frame-Options
- [ ] Input validation: Pydantic models, SQL injection
- [ ] Error handling: 4xx/5xx consistency, stacktraces
- [ ] CORS: whitelist, preflight handling
- [ ] Versioning: /api/v1 strategy, deprecation

**Metriche**:
- Request latency p95/p99
- Error rate per endpoint
- Rate limit violations
- Security scan score

---

#### 5. **Causal Discovery Engine** 🧠
**Scope**: PCMCI, Tigramite, correlations  
**Agent Specializz**: `CausalAuditor`

**Audit Points**:
- [ ] PCMCI implementation: lag selection, p-values
- [ ] Tigramite integration: version compatibility, fallbacks
- [ ] Correlation fallback: Pearson vs Spearman
- [ ] Graph construction: edge weights, DAG validation
- [ ] LLM interpretation: prompt engineering, reliability
- [ ] Performance: large datasets (10k+ points)
- [ ] Numerical stability: missing data, outliers

**Metriche**:
- Discovery success rate
- Graph complexity (edges/nodes)
- LLM explanation quality
- Computation time

---

#### 6. **Test Coverage & CI/CD** 🧪
**Scope**: Unit tests, integration tests, automation  
**Agent Specializz**: `QualityAuditor`

**Audit Points**:
- [ ] Unit test coverage: 40% → 80%
- [ ] Integration tests: E2E investigation flow
- [ ] Mock strategies: external APIs, LLM responses
- [ ] CI/CD pipeline: GitHub Actions, test reports
- [ ] Code quality: radon complexity, flake8
- [ ] Documentation tests: docstrings, examples
- [ ] Performance tests: load testing, profiling

**Metriche**:
- Line coverage %
- Branch coverage %
- Test execution time
- Flaky test ratio

---

#### 7. **Frontend Integration** 🎨
**Scope**: React components, API client, WebSocket  
**Agent Specializz**: `FrontendAuditor`

**Audit Points**:
- [ ] Component modularity: reusability, props
- [ ] State management: React hooks, context
- [ ] API client: error handling, retries
- [ ] WebSocket: reconnection, message buffering
- [ ] UI/UX: loading states, error messages
- [ ] Accessibility: ARIA labels, keyboard nav
- [ ] Performance: bundle size, lazy loading
- [ ] Type safety: TypeScript coverage

**Metriche**:
- Component render time
- Bundle size (KB)
- API error rate
- TypeScript errors

---

#### 8. **Deployment & Monitoring** 🚀
**Scope**: Docker, logging, metrics  
**Agent Specializz**: `OpsAuditor`

**Audit Points**:
- [ ] Docker compose: backend + frontend + SurrealDB
- [ ] Environment configs: dev/staging/prod
- [ ] Log aggregation: rotation, ELK stack
- [ ] Metrics: Prometheus + Grafana dashboards
- [ ] Health checks: liveness, readiness probes
- [ ] Backup strategy: knowledge base, data cache
- [ ] Secrets management: env vars, vault
- [ ] Scaling: horizontal/vertical strategies

**Metriche**:
- Uptime %
- Resource usage (CPU/RAM)
- Log volume per day
- Alert frequency

---

## 📋 EXECUTION PLAN

### Phase 1: Parallel Audit Preparation (2h)
```bash
# Create audit agents directory
mkdir -p audit_agents/

# Agent templates
agents/
├── data_flow_auditor.py
├── investigation_auditor.py
├── knowledge_auditor.py
├── api_auditor.py
├── causal_auditor.py
├── quality_auditor.py
├── frontend_auditor.py
├── ops_auditor.py
└── orchestrator.py
```

### Phase 2: MCP Server (Model Context Protocol) (3h)
```python
# audit_agents/mcp_server.py

from typing import List, Dict, Any
import asyncio
from concurrent.futures import ProcessPoolExecutor

class AuditAgent:
    def __init__(self, name: str, scope: List[str]):
        self.name = name
        self.scope = scope
        self.results = []
    
    async def run_audit(self) -> Dict[str, Any]:
        """Run audit and return structured report."""
        pass

class MCPOrchestrator:
    def __init__(self):
        self.agents = []
        self.results = {}
    
    def register_agent(self, agent: AuditAgent):
        self.agents.append(agent)
    
    async def run_parallel(self) -> Dict[str, Any]:
        """Execute all agents in parallel."""
        tasks = [agent.run_audit() for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                self.results[agent.name] = {"error": str(result)}
            else:
                self.results[agent.name] = result
        
        return self.aggregate_report()
    
    def aggregate_report(self) -> Dict[str, Any]:
        """Combine all agent reports."""
        return {
            "summary": self.generate_summary(),
            "agents": self.results,
            "critical_issues": self.extract_critical(),
            "recommendations": self.generate_recommendations()
        }
```

### Phase 3: Agent Implementation (8h)
**Parallel execution per agent**:
- DataFlowAuditor: Check CMEMS/ERA5/cache
- InvestigationAuditor: Test E2E pipeline
- KnowledgeAuditor: Verify all services
- APIAuditor: Security & performance tests
- CausalAuditor: PCMCI validation
- QualityAuditor: Run pytest, coverage
- FrontendAuditor: Component tests
- OpsAuditor: Docker setup, logs

### Phase 4: Report Generation (1h)
```markdown
# AUDIT_REPORT_2025-12-25.md

## Executive Summary
- Total checks: 156
- Passed: 89 (57%)
- Failed: 45 (29%)
- Warnings: 22 (14%)

## Critical Issues
1. Knowledge persistence (in-memory only)
2. Test coverage 40% (target 80%)
3. SurrealDB incomplete

## Recommendations
Priority matrix with effort estimates
```

---

## 🚀 LANCIO AGENTI

### Opzione A: Manuale (bash script)
```bash
#!/bin/bash
# run_audit.sh

echo "🔍 Starting Multi-Agent Audit..."

# Run agents in parallel using GNU parallel
parallel -j 8 ::: \
  "python audit_agents/data_flow_auditor.py > reports/data_flow.json" \
  "python audit_agents/investigation_auditor.py > reports/investigation.json" \
  "python audit_agents/knowledge_auditor.py > reports/knowledge.json" \
  "python audit_agents/api_auditor.py > reports/api.json" \
  "python audit_agents/causal_auditor.py > reports/causal.json" \
  "python audit_agents/quality_auditor.py > reports/quality.json" \
  "python audit_agents/frontend_auditor.py > reports/frontend.json" \
  "python audit_agents/ops_auditor.py > reports/ops.json"

# Aggregate results
python audit_agents/orchestrator.py aggregate reports/*.json > AUDIT_REPORT.md

echo "✅ Audit complete! See AUDIT_REPORT.md"
```

### Opzione B: VS Code Task
```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Multi-Agent Audit",
      "type": "shell",
      "command": "./run_audit.sh",
      "group": {
        "kind": "test",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "dedicated"
      }
    }
  ]
}
```

### Opzione C: Python asyncio (Recommended)
```python
# audit_agents/run_all.py

import asyncio
from orchestrator import MCPOrchestrator
from data_flow_auditor import DataFlowAuditor
from investigation_auditor import InvestigationAuditor
# ... altri imports

async def main():
    orchestrator = MCPOrchestrator()
    
    # Register agents
    orchestrator.register_agent(DataFlowAuditor())
    orchestrator.register_agent(InvestigationAuditor())
    orchestrator.register_agent(KnowledgeAuditor())
    orchestrator.register_agent(APIAuditor())
    orchestrator.register_agent(CausalAuditor())
    orchestrator.register_agent(QualityAuditor())
    orchestrator.register_agent(FrontendAuditor())
    orchestrator.register_agent(OpsAuditor())
    
    # Run in parallel
    print("🔍 Launching 8 audit agents in parallel...")
    report = await orchestrator.run_parallel()
    
    # Save report
    with open("AUDIT_REPORT.md", "w") as f:
        f.write(report.to_markdown())
    
    print("✅ Audit complete! See AUDIT_REPORT.md")

if __name__ == "__main__":
    asyncio.run(main())
```

**Esecuzione**:
```bash
python audit_agents/run_all.py
```

---

## 📊 METRICHE AGGREGATE

### Success Criteria
- [ ] **Coverage**: 80%+ test coverage
- [ ] **Performance**: p95 latency < 500ms
- [ ] **Reliability**: 99.9% uptime
- [ ] **Security**: 0 critical vulnerabilities
- [ ] **Knowledge**: 1000+ papers indexed
- [ ] **Investigation**: <30s average time

### Dashboard Proposta (Grafana)
- Request latency histogram
- Error rate by endpoint
- Knowledge base growth
- Investigation success rate
- Cache hit ratio
- Resource usage (CPU/RAM)

---

## 🎯 IMMEDIATE NEXT STEPS

1. **Riavviare investigation** (con backend fixed)
   ```bash
   # Frontend already open at localhost:5173
   # 1. Go to AI Assistant
   # 2. Enter: "Lago Maggiore flood 2000-10-10 to 2000-10-20"
   # 3. Verify papers saved: curl http://localhost:8000/api/v1/knowledge/stats
   ```

2. **Creare scheletro agenti**
   ```bash
   mkdir -p audit_agents
   touch audit_agents/{data_flow,investigation,knowledge,api,causal,quality,frontend,ops}_auditor.py
   touch audit_agents/orchestrator.py
   ```

3. **MCP orchestrator base**
   - Implementare MCPOrchestrator class
   - Async parallel execution
   - Report aggregation

4. **Primo agent (DataFlowAuditor)**
   - Check CMEMS connection
   - Test ERA5 download
   - Verify cache stats

---

## 📝 CONCLUSIONI

### Sistema Attuale: **B+ (85/100)**

**Punti di Forza**:
- ✅ Architettura modulare ben strutturata
- ✅ Investigation agent robusto con streaming
- ✅ API production-ready (logging, security, middleware)
- ✅ MinimalKnowledgeService funzionante
- ✅ Frontend intuitivo e responsive

**Aree Miglioramento**:
- ⚠️ Persistence layer (in-memory → database)
- ⚠️ Test coverage (40% → 80%)
- ⚠️ Monitoring (logs ok, metrics missing)
- ⚠️ Documentation (partial)

### Proposta Multi-Agent Audit
**8 agenti specializzati** per audit parallelo completo:
- Execution time: ~30min (vs 4h sequenziale)
- Coverage: 156+ check points
- Output: Markdown report con priorità e effort estimates

**Ready to implement**: Scheletro MCP + primi 2 agenti (DataFlow, Investigation)
