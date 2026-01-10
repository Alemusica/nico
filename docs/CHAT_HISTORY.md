# 📜 Chat History & Session Context

> **Purpose**: Preserve context between AI agent sessions to prevent duplication and confusion.
> **Last Updated**: 2026-01-10 22:00

---

## 🔥 SESSIONE CORRENTE: 2026-01-10 (Visual Enhancement + Geostrophic Velocity)

### ✅ Ultimo Commit
```
(DA FARE - vedi sotto)
```

### 🎯 Obiettivi della Sessione
1. ✅ Riorganizzare Volume Transport e Geostrophic Velocity tabs
2. ✅ Implementare confronto v_perp vs v_geo con spatial averaging
3. ✅ Correggere segno formula geostrofica (+g/f invece di -g/f)
4. ✅ Aggiungere dual x-axis (km + gradi) a tutti i plot spaziali
5. ✅ Cambiare unità da Sv/mSv a m³/s standard SI
6. ✅ Visual enhancement: white background + elegant color scheme
7. ✅ Fix bug divided gates velocity filtering
8. ❌ **BLOCCO**: Plotly `secondary_x` error - tab Geostrophic Velocity crasha

### 🔴 PROBLEMA CRITICO APERTO
**Error**: `ValueError: Invalid key 'secondary_x' in make_subplots specs`
- **Location**: `app/components/tabs.py` line ~3981
- **Impact**: Geostrophic Velocity tab NON si apre
- **Fix**: Vedere `docs/ISSUES/ISSUE_2026-01-10_CRITICAL_FIXES.md`

### 🟡 ALTRI PROBLEMI DA SISTEMARE DOMANI
1. Monthly Analysis tab: mancano valori slope e R² sui grafici
2. Deprecation warning: `use_container_width` → sostituire con `width='stretch'`
3. Dual x-axis: standardizzare su tutti i plot spaziali

### 📁 Files Modificati
- `app/components/tabs.py` - ~600 lines changed (rewrite Volume Transport + new Geostrophic Velocity)
- `app/components/loaders/base.py` - Fix divided gates velocity filtering
- `src/services/transport_service.py` - New velocity calculation functions
- `.streamlit/config.toml` - NEW: Light theme configuration
- `app/components/chart_style.py` - NEW: Centralized styling module

### 🎨 Style Changes
- Background: White (#FFFFFF)
- Primary color: Navy Blue (#1E3A5F)
- Secondary color: Coral (#E07B53)
- Font: Inter, sans-serif
- No rounded corners
- Subtle gridlines

---

## 📖 SESSIONE PRECEDENTE: 2026-01-04 (CMEMS L3/L4 Strategy)

### ✅ Ultimo Commit
```
f9be240 - 🔧 Fix CMEMS crash: Remove All Tracks for local, use L4 for API
```

### 🟢 App Funzionante
- **URL**: http://localhost:8504
- **Stato**: Running senza crash
- **Dataset testati**: DTUSpace ✅, SLCCI ✅, CMEMS L3 local (con track) ✅

---

## 🐛 PROBLEMI NOTI / DA RISOLVERE

### 1. ⚠️ CMEMS L3 API Non Funziona (Priority: MEDIUM)
**Problema**: Il dataset L3 along-track di CMEMS **non supporta** `copernicusmarine.open_dataset()` con filtro geografico.

**Errore**: `'Command' object is not subscriptable`

**Workaround Attuale**: API mode usa **L4 gridded** invece di L3.
- L4 funziona perfettamente con l'API
- L4 ha risoluzione 0.125° (tutti gli altimetri merged)
- Ma L4 è GRIDDED, non along-track!

**Possibili Soluzioni Future**:
1. Usare `copernicusmarine.subset()` per L3 (scarica file, poi li apre) - testato, funziona ma lento
2. Mantenere L4 per API (attuale) - OK per analisi generale
3. Implementare download batch L3 con caching - complesso

**File Coinvolti**:
- `src/services/cmems_service.py` (`_load_from_api`)
- `app/components/sidebar.py` (`_render_cmems_params`)

### 2. ⚠️ Deprecation Warning Streamlit (Priority: LOW)
```
Please replace `use_container_width` with `width`.
For `use_container_width=True`, use `width='stretch'`.
```

### 3. ⚠️ Warning CMEMS Base Directory (Priority: LOW)
```
WARNING cmems_service: CMEMS base directory not found: /tmp/cmems_api_cache
```

---

## 📊 STRATEGIA DUAL-MODE CMEMS (IMPLEMENTATA)

| Mode | Dataset | Track Selection | Funziona? |
|------|---------|-----------------|-----------|
| **LOCAL** | L3 Along-Track | ✅ Obbligatoria | ✅ SI |
| **API** | L4 Gridded | ❌ Non disponibile | ✅ SI |

### Perché questa scelta?
1. **LOCAL L3**: 7093 file NetCDF - se carichi "All Tracks" → **CRASH** (segfault)
2. **API L3**: `open_dataset()` con bbox **non funziona** per dataset along-track
3. **API L4**: Funziona perfettamente, ma è gridded (non along-track)

---

## 📁 FILE CHIAVE MODIFICATI (2026-01-04)

### `app/components/sidebar.py`
- `_render_cmems_params()`: UI diversa per LOCAL vs API
- LOCAL: track selection obbligatoria
- API: skip track selection (L4 non ha tracks)

### `src/services/cmems_service.py`
- `_load_from_api()`: Ora usa L4 gridded invece di L3 along-track
- Dataset ID: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`

---

## 🎯 TODO - PRIORITÀ ALTA

### 1. Testare CMEMS L3 LOCAL con Track Specifico
```bash
# Seleziona un gate con track nel nome (es. bering_strait_pass_76)
# Seleziona CMEMS L3 → LOCAL
# Seleziona track suggerito (76)
# Verifica che carichi senza crash
```

### 2. Testare CMEMS API (L4)
```bash
# Seleziona un gate qualsiasi
# Seleziona CMEMS L3 → API
# Clicca Load
# Verifica che carichi dati L4 gridded
```

### 3. Verificare Comparison Mode
- [ ] SLCCI + CMEMS L3 local
- [ ] SLCCI + CMEMS L4 API
- [ ] DTUSpace + CMEMS

---

## 🎯 TODO - PRIORITÀ MEDIA

### 4. Fix Deprecation Warnings
Sostituire `use_container_width=True` con `width='stretch'`

### 5. Migliorare UX API Mode
- Aggiungere info box che spiega: "API mode usa L4 gridded (0.125°), non L3 along-track"
- Mostrare progress bar durante download API

### 6. Cache per API Mode
Implementare cache locale per download API.

---

## 🎯 TODO - PRIORITÀ BASSA

### 7. Implementare L3 API con subset()
Se serve davvero L3 via API, usare `copernicusmarine.subset()` invece di `open_dataset()`.

### 8. Cleanup Code
- Rimuovere codice morto
- Aggiungere type hints
- Documentare funzioni

---

## 🔧 COMANDI UTILI

### Avviare Streamlit
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
source .venv/bin/activate
streamlit run app/main.py --server.port 8504
```

### Testare CMEMS API
```python
import copernicusmarine

# L4 (FUNZIONA)
ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    minimum_longitude=-175,
    maximum_longitude=-162,
    minimum_latitude=60,
    maximum_latitude=66,
)

# L3 (NON FUNZIONA con open_dataset - usare subset())
```

---

## 🔑 CREDENZIALI E PATH

### CMEMS API
```bash
copernicusmarine login  # Una volta per salvare credenziali
```

### Path Dati Locali
```python
SLCCI_BASE_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
CMEMS_BASE_DIR = "/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA"
DTU_PATH = "/Users/nicolocaron/Desktop/ARCFRESH/arctic_ocean_prod_DTUSpace_v4.0.nc/..."
```

---

## 📝 NOTE PER IL PROSSIMO AGENTE

1. **SEMPRE fare `git pull` prima di iniziare!**
2. **Leggere `docs/PROGRESS.md` per lo stato attuale**
3. **L'app gira su porta 8504**, non 8501
4. **CMEMS L3 API non funziona** - usa L4 oppure local con track selection
5. **Il crash "All Tracks" è risolto** - rimossa l'opzione per LOCAL mode

---

## 🧪 ULTIMO TEST RIUSCITO (DTUSpace)

```
[18:05:05] INFO  dtu_service: Loading DTUSpace data...
[18:05:05] INFO  dtu_service: DOT shape: (93, 720, 144), 144 time steps
[18:05:06] INFO  dtu_service: DTUSpace data loaded successfully: 57600 synthetic observations
```

---

## ⚠️ CRITICAL WARNING FOR ALL AGENTS

**BEFORE writing ANY code, READ:**
1. `docs/ARCHITECTURE.md` - The NICO Unified Architecture diagram
2. `docs/PROGRESS.md` - Current progress and bugs
3. This file - Previous session context

**The data flow MUST be:**
```
UI → Services (src/services/) → DataAccess → Infrastructure
```

**DO NOT:**
- Hardcode file paths
- Bypass the Services layer
- Create new loaders without adding to DataService
- Ignore the config files

---

## 🔄 How to Use This File

### For AI Agents Starting a New Session:
1. Read this file FIRST after `git pull`
2. Understand what was accomplished in previous sessions
3. Don't redo completed work
4. Continue from where the last session ended

### For Agents Ending a Session:
1. Add a new entry at the TOP of the "Session Log" section
2. Include: Date, branch, what was done, what's pending
3. Commit this file with your changes

---

## 📋 Session Log

### 2025-12-29 (Session 2) - feature/gates-streamlit (Human: nicolocaron)

**MAJOR: Wired UI to Services Layer**

**Architecture Work:**
- Saved NICO Unified Architecture diagram to `docs/ARCHITECTURE.md`
- Created `docs/ARCHITECTURE_GAP.md` documenting the UI→Services gap
- Fixed the gap: sidebar now calls DataService properly

**Implementation:**
- `DataService.load()` now routes based on `config/datasets.yaml`
- Added `_load_noaa()` and `_load_nasa()` providers
- Updated `_load_cmems()` and `_load_era5()` to use config
- `_load_data_for_gate()` follows architecture:
  1. Gets dataset from user selection OR gate.datasets OR default
  2. Builds DataRequest
  3. Calls DataService.load() → routes to correct provider

**Data Flow Now Working:**
```
User selects gate → clicks "Load Data" → sidebar._load_data_for_gate()
  → DataService.build_request() → DataService.load()
  → routes to _load_cmems/_load_era5/_load_noaa/_load_nasa
  → returns xarray.Dataset → stored in session_state.datasets
  → graphs render!
```

**Pending:**
- Test with real CMEMS credentials
- Add dataset selector in catalog tab
- Time range selector in UI

---

### 2025-12-29 (Session 1) - feature/gates-streamlit (Human: nicolocaron)

**Completed:**
- ✅ Phase 0-8 of Unified Architecture implementation
- ✅ Created `src/core/models.py` with all Pydantic models (GateModel, DataRequest, BoundingBox, etc.)
- ✅ Created `config/gates.yaml`, `config/datasets.yaml`, `config/regions.yaml`
- ✅ Implemented `src/gates/catalog.py` - GateCatalog loading from YAML
- ✅ Implemented `src/services/gate_service.py` - Full gate operations
- ✅ Created `app/components/data_selector.py` - Unified data selection UI
- ✅ Enabled Gate selector in Streamlit sidebar
- ✅ Implemented centralized logging in `src/core/logging_config.py`
- ✅ Created GitHub Issues #12, #13, #14, #15
- ✅ Fixed multiple bugs:
  - BoundingBox.center property
  - TimeRange datetime parsing
  - DataRequest.dataset_id field
  - SpatialResolution float enum
  - GateService.get_gate() method
  - GateModel.datasets field

**In Progress:**
- 🔄 Connect gate selection to actual plot visualization
- 🔄 Data loading from selected gate bbox

**Pending:**
- ⬜ Graph visualization with selected gate data
- ⬜ ERA5/CMEMS data integration with gates
- ⬜ Time series analysis per gate

**Key Files Modified:**
- `src/core/models.py` - Added GateModel with bbox property
- `src/services/gate_service.py` - Added get_gate(), get_gate_geometry()
- `config/gates.yaml` - Added datasets, default_buffer_km fields
- `app/components/sidebar.py` - Enabled gate dropdown
- `docs/FEATURE_INVENTORY.md` - Created cross-branch feature list

**Context for Next Session:**
- Streamlit runs on port 8501
- Gate selector works but graphs not connected
- User wants: graphs to show data for selected gate
- Services layer is complete, need to wire up visualization

---

## 📊 Feature Status Overview

| Feature | Branch | Status | Last Updated |
|---------|--------|--------|--------------|
| Gate Selection UI | feature/gates-streamlit | ✅ Working | 2025-01-XX |
| GateCatalog | feature/gates-streamlit | ✅ Working | 2025-01-XX |
| Data Visualization | feature/gates-streamlit | 🔄 Partial | 2025-01-XX |
| Knowledge Graph | master | 🔄 Partial | - |
| React Frontend | master | ✅ Working | - |

---

## 🔗 Related Documentation

- `docs/PROGRESS.md` - Overall project progress
- `docs/FEATURE_INVENTORY.md` - All features across branches
- `docs/CHANGELOG.md` - Changes log
- `.github/copilot-instructions.md` - Agent instructions
