# 🤖 Multi-Agent Audit System

Sistema di audit parallelo per il progetto Causal Discovery.

## 🎯 Obiettivo

Audit completo del sistema attraverso **8 agenti specializzati** che lavorano in parallelo:

1. **DataFlowAuditor** 🌊 - CMEMS, ERA5, Climate Indices, Cache
2. **InvestigationAuditor** 🔍 - Pipeline E2E, WebSocket, Validation
3. **KnowledgeAuditor** 📚 - Services, Persistence, Search
4. **APIAuditor** 🔐 - Security, Performance, Rate Limiting
5. **CausalAuditor** 🧠 - PCMCI, Tigramite, Correlations
6. **QualityAuditor** 🧪 - Tests, Coverage, CI/CD
7. **FrontendAuditor** 🎨 - React, TypeScript, WebSocket
8. **OpsAuditor** 🚀 - Docker, Monitoring, Logs

## 📋 Status Implementazione

| Agente | Status | Checks | Completamento |
|--------|--------|--------|---------------|
| DataFlowAuditor | ✅ Complete | 13 | 100% |
| InvestigationAuditor | ⚪ Stub | 0 | 0% |
| KnowledgeAuditor | ⚪ Stub | 0 | 0% |
| APIAuditor | ⚪ Stub | 0 | 0% |
| CausalAuditor | ⚪ Stub | 0 | 0% |
| QualityAuditor | ⚪ Stub | 0 | 0% |
| FrontendAuditor | ⚪ Stub | 0 | 0% |
| OpsAuditor | ⚪ Stub | 0 | 0% |

## 🚀 Quick Start

### 1. Test singolo agente
```bash
# Backend deve essere running su port 8000
source .venv/bin/activate
python audit_agents/data_flow_auditor.py
```

### 2. Run audit completo (quando tutti implementati)
```bash
python audit_agents/run_all.py
```

### 3. Output
- `audit_reports/audit_report.json` - Report JSON completo
- `audit_reports/AUDIT_REPORT.md` - Report Markdown human-readable

## 📊 DataFlowAuditor Results

**Ultimo run**: 2025-12-25 16:49

```
Total Checks: 13
Passed:       11 ✅
Failed:       0 ❌
Warnings:     2 ⚠️
Pass Rate:    85%
Duration:     2463ms
```

### Checks Passed ✅
- Backend API accessible
- Backend components reported
- CMEMS client imports & instantiates
- ERA5 client imports & instantiates
- Climate indices client imports & instantiates
- Data Manager imports & instantiates
- File formats (NetCDF, CSV, ZARR) supported

### Warnings ⚠️
1. **ERA5 humidity vars** - Variables not verified (client instantiated but vars not checked)
2. **Cache stats** - Endpoint returns 503 (DataManager issue)

## 🏗️ Architettura

### MCP Orchestrator
```python
from audit_agents.orchestrator import MCPOrchestrator
from audit_agents.data_flow_auditor import DataFlowAuditor

orchestrator = MCPOrchestrator()
orchestrator.register_agent(DataFlowAuditor())
report = await orchestrator.run_parallel()
```

### Agent Base Class
```python
from audit_agents.orchestrator import AuditAgent

class MyAuditor(AuditAgent):
    def __init__(self):
        super().__init__(name="MyAuditor", scope="My scope")
    
    async def _run_checks(self):
        # Implement checks
        self.check(
            name="check_name",
            condition=True,  # boolean
            severity="critical",  # critical/high/medium/low
            message="Check passed",
            **extra_details
        )
```

## 📝 Implementare Nuovo Agente

1. **Crea file agente**
```bash
touch audit_agents/my_auditor.py
```

2. **Implementa classe**
```python
from audit_agents.orchestrator import AuditAgent

class MyAuditor(AuditAgent):
    def __init__(self):
        super().__init__(name="MyAuditor", scope="My scope")
    
    async def _run_checks(self):
        # Run your checks
        self.check(...)
```

3. **Registra in run_all.py**
```python
from audit_agents.my_auditor import MyAuditor

orchestrator.register_agent(MyAuditor())
```

4. **Test standalone**
```bash
python audit_agents/my_auditor.py
```

## 📈 Metriche

### Check Severity
- **critical** - Sistema non funziona (es. backend down)
- **high** - Funzionalità core rotta (es. CMEMS client fails)
- **medium** - Funzionalità secondaria issue (es. cache 503)
- **low** - Warning, best practice (es. missing docs)

### Report Structure
```json
{
  "summary": {
    "total_checks": 156,
    "passed": 120,
    "failed": 25,
    "warnings": 11,
    "pass_rate": "77%"
  },
  "agents": {
    "DataFlowAuditor": {...}
  },
  "critical_issues": [...],
  "recommendations": [...]
}
```

## 🎯 Next Steps

### Priority 1 (Critical)
1. **InvestigationAuditor** - Test E2E pipeline con real investigation
2. **KnowledgeAuditor** - Verify MinimalKnowledgeService + abstract methods
3. **QualityAuditor** - Run pytest, measure coverage

### Priority 2 (High)
4. **APIAuditor** - Security scan, rate limit tests
5. **CausalAuditor** - PCMCI validation con test data

### Priority 3 (Medium)
6. **FrontendAuditor** - TypeScript errors, component tests
7. **OpsAuditor** - Docker setup, log rotation
8. **Documentation** - Complete all agent implementations

## 📚 Documentation

- **PROJECT_AUDIT_ARCHITECTURE.md** - Analisi completa progetto + topic identificati
- **orchestrator.py** - MCP implementation details
- **data_flow_auditor.py** - Esempio agente completo

## 🤝 Contributing

Per implementare un agente:
1. Fork da `data_flow_auditor.py` come template
2. Implementa `_run_checks()` con i tuoi check
3. Test standalone: `python audit_agents/your_auditor.py`
4. PR con test results

## 📞 Support

Issues: GitHub Issues con tag `audit-system`
Docs: `docs/PROJECT_AUDIT_ARCHITECTURE.md`

---

**Version**: 1.0.0  
**Last Updated**: 2025-12-25  
**Status**: ✅ DataFlowAuditor complete, 7 agents pending
