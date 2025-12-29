# 🤖 AGENTS.md - AI Agent Instructions

> ⚠️ **STOP! READ THIS ENTIRE FILE BEFORE WRITING ANY CODE!**

---

## 🚨 CRITICAL: Architecture Compliance

This project has a **documented architecture** that MUST be followed.

### Read These First (IN ORDER):
1. `docs/ARCHITECTURE.md` - System architecture diagram
2. `docs/CHAT_HISTORY.md` - What's been done, what's pending
3. `.github/copilot-instructions.md` - Detailed instructions

---

## 🏗️ The NICO Unified Architecture

```
┌─────────────────────────────────────────────────────────┐
│  PRESENTATION: Streamlit / React / CLI                  │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  API GATEWAY: FastAPI (api/)                            │
│  /gates  /data  /analysis  /knowledge  /pipeline        │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  SERVICES: src/services/                                │
│  GateService | DataService | AnalysisService            │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  CORE: src/core/                                        │
│  models.py | coordinates.py | config.py                 │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  DATA ACCESS: Loaders + Config                          │
│  config/datasets.yaml | config/gates.yaml               │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE: SurrealDB | NetCDF | APIs              │
└─────────────────────────────────────────────────────────┘
```

---

## ❌ DO NOT

- **Hardcode file paths** (use config files)
- **Bypass Services layer** (UI must call Services, not loaders directly)
- **Create duplicate code** (check if it exists in src/services/ first)
- **Ignore config files** (datasets.yaml, gates.yaml define the system)
- **Start coding without reading docs** (you WILL break things)

---

## ✅ DO

- **Follow the data flow**: UI → Services → DataAccess → Infrastructure
- **Use existing services**: `from src.services import GateService, DataService`
- **Use existing models**: `from src.core.models import BoundingBox, GateModel`
- **Add to config files**: New datasets go in `config/datasets.yaml`
- **Update docs**: Add to CHAT_HISTORY.md when done

---

## 🔧 Quick Reference

### Load Data for a Gate:
```python
from src.services import GateService, DataService
from src.core.models import TimeRange

gs = GateService()
ds = DataService()

gate = gs.get_gate("fram_strait")
bbox = gate.bbox

request = ds.build_request(
    bbox=bbox,
    time_range=TimeRange(start=..., end=...),
    dataset_id="cmems_sealevel"  # From config/datasets.yaml
)
data = ds.load(request)  # Routes to correct provider automatically
```

### Get Gate Info:
```python
from src.services import GateService
gs = GateService()
gates = gs.list_gates()  # All gates from config/gates.yaml
gate = gs.get_gate("fram_strait")
print(gate.bbox, gate.datasets)
```

---

## 📂 Key Files

| File | Purpose |
|------|---------|
| `config/datasets.yaml` | Dataset providers & config |
| `config/gates.yaml` | Ocean gates definitions |
| `src/services/data_service.py` | Data loading orchestration |
| `src/services/gate_service.py` | Gate operations |
| `src/core/models.py` | Pydantic models |
| `docs/ARCHITECTURE.md` | Full architecture docs |

---

## 🌿 Current Branch

Check which branch you're on:
```bash
git branch --show-current
```

- `master` - React + API + Knowledge Graph
- `feature/gates-streamlit` - Streamlit + Gates integration

---

*Last updated: 2025-12-29*
