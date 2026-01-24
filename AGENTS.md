# 🤖 AGENTS.md - AI Agent Instructions

> ⚠️ **STOP! READ THIS ENTIRE FILE BEFORE WRITING ANY CODE!**

---

## 🚨 CRITICAL: Development Approach - "GARAGE FIRST"

**Prima di tutto, leggi:** `docs/DEVELOPMENT_APPROACH.md`

Questo progetto usa l'approccio **"Garage First"**:
1. **Prima** costruisci/estendi il layer API (il "garage")
2. **Poi** aggiungi data client (le "auto")
3. **Infine** costruisci i motori che consumano i dati (l'"officina")

**NON** creare "vertical slices" separati - i componenti sono accoppiati.

### Read These First (IN ORDER):
1. `docs/DEVELOPMENT_APPROACH.md` - **Filosofia e approccio di sviluppo**
2. `docs/ARCHITECTURE.md` - System architecture diagram
3. `data/knowledge/external_resources.json` - Librerie e papers esterni

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

---

## 🏠 DATA LAYER: "IL GARAGE"

Il layer dati è il fondamento. Ogni data client deve seguire un contratto unificato.

### Stato Attuale (Audit 2026-01-19)

```
FUNZIONANTI: CMEMS, ERA5, GPM, GRACE, Tide Gauges, ARGO, Climate Indices
PARZIALI:    Aircraft (MADIS stub), Sentinel (download stub), CYGNSS
MANCANTE:    Interfaccia unificata BaseDataClient
```

### Quando Aggiungi un Data Source

1. Aggiungi entry in `src/surge_shazam/data/api_registry.py`
2. Crea client in `src/surge_shazam/data/`
3. Implementa: `download()`, `list_products()`, `health_check()`
4. Aggiungi fallback sintetico realistico
5. Aggiungi test

### Risorse Esterne

Tutte le librerie utili sono catalogate in:
- `data/knowledge/external_resources.json`

Carica in SurrealDB con:
```bash
python scripts/seed_external_resources.py
```

---

*Last updated: 2026-01-19*
