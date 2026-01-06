# 📊 Surge Shazam - Progress Tracker

> Last Updated: 2026-01-06 (Session - Unified Tabs + Geostrophic + Export)
> Agent: Use this file to track progress. Update after each task.

---

## 🔄 SESSION 2026-01-06 (Evening): Unified Rendering Functions

### 🎯 Obiettivo
Standardizzare l'output di tutti e 3 i dataset (SLCCI, CMEMS L4, DTUSpace) con **gli stessi tab e plot**.

### ✅ Funzioni Unificate Create (`app/components/tabs.py`)

| # | Funzione | Descrizione |
|---|----------|-------------|
| 1 | `_render_unified_dataset_tabs()` | Entry point - crea 6 tab identiche |
| 2 | `_render_unified_slope_timeline()` | Slope vs tempo + R² + trend line |
| 3 | `_render_unified_dot_profile()` | DOT vs lon/km + switch unità m/mm |
| 4 | `_render_unified_spatial_map()` | Mappa con gate line e osservazioni |
| 5 | `_render_unified_monthly_analysis()` | 12 subplot mensili con R² e slope |
| 6 | `_render_unified_geostrophic_velocity()` | v_geo time series + climatologia |
| 7 | `_render_unified_export_tab()` | Export CSV per tutti i dati |

### ✅ Tab Aggiornate per usare funzioni unificate
- `_render_cmems_tabs()` → Unified
- `_render_dtu_tabs()` → Unified  
- `_render_slcci_tabs()` → Unified

### ✅ Opzioni Uniformi per Tutti i Dataset
- **X-axis toggle**: Distance (km) vs Longitude (°)
- **Y-axis units**: m, cm, mm
- **Trend line**: con R² e pendenza
- **Monthly analysis**: 12 subplot con regressione lineare
- **Geostrophic velocity**: time series + monthly climatology
- **Export**: CSV per slope, DOT, v_geo

### ✅ CMEMS L4 Geostrophic Velocity (`src/services/cmems_l4_service.py`)
```python
def _compute_geostrophic_velocity(slope_series, mean_lat):
    """v = -g/f * (dη/dx)"""
    g = 9.81
    OMEGA = 7.2921e-5
    f = 2 * OMEGA * np.sin(np.deg2rad(mean_lat))
    slope_m_m = slope_series / 100000.0  # m/100km to m/m
    return -g / f * slope_m_m, f
```

### ✅ Cache Manager UI (`app/components/sidebar.py`)
- Nuovo bottone "🗑️ Manage Cache" in sidebar
- Expander con lista item cached
- Delete buttons per ogni item

### ⚠️ Problema: tabs.py troppo grande
- **4719 righe** - contiene codice duplicato
- Molte funzioni legacy non più usate
- **Prossimo step**: Shrink e cleanup

---

## 🔄 SESSION 2026-01-06 (Morning): Recovery + Integration

### 🚨 Problema Critico: File Persi
Durante una sessione precedente, `git checkout HEAD --` ha cancellato modifiche non committate.
**8 file recuperati dalla VS Code Local History** (~1546 righe).

### Commits della Giornata

| Hash | Descrizione |
|------|-------------|
| `0a29229` | Multi-dataset comparison mode + charts refactor |
| `1497aa3` | Recover lost services from VS Code History |
| `e2b268d` | Integrate recovered services into exports |
| `1d95ded` | Integrate Cache and Bathymetry services |
| `548b6a4` | Audit report + documentation updates |
| `7c5bdda` | Integrate lon_filter for divided gates |
| **PENDING** | Standardized Gate Division System |

### File Recuperati
| File | Righe | Funzionalità |
|------|-------|--------------|
| `src/services/cache_service.py` | 471 | DataCache con persistenza pickle |
| `src/services/bathymetry_service.py` | 229 | BathymetryService per GEBCO |
| `src/services/transport_service.py` | 243 | Calcolo volume transport |
| `app/components/loaders/base.py` | 220 | `apply_longitude_filter()` |
| `app/components/loaders/slcci_loader.py` | 140 | Loader SLCCI |
| `app/components/loaders/dtu_loader.py` | 114 | Loader DTU |
| `app/components/loaders/cmems_l4_loader.py` | 112 | Loader CMEMS L4 |
| `app/components/loaders/__init__.py` | 17 | Exports |

### Integrazioni Completate

#### ✅ Cache Service (`sidebar.py`)
```python
from src.services.cache_service import DataCache
_cache = DataCache()

# In _load_slcci_data():
cached_data = _cache.load("slcci", gate_name, pass_number=pass_number)
if cached_data is None:
    pass_data = service.load_pass_data(...)
    _cache.save("slcci", gate_name, pass_data, pass_number=pass_number)
```

#### ✅ Bathymetry Service (`volume_transport_chart.py`)
```python
from src.services.bathymetry_service import BathymetryService, BathymetryProfile

def _render_computed_volume_transport():
    profile = _get_bathymetry_profile(pass_data)
    if profile:
        gate_depth = profile.mean_depth  # From GEBCO
```

#### ✅ lon_filter Integration (`sidebar.py`)
```python
from app.components.loaders.base import apply_longitude_filter

# In all _load_*_data() functions:
lon_min, lon_max = _get_lon_filter_for_gate(config.selected_gate)
if lon_min is not None or lon_max is not None:
    pass_data = apply_longitude_filter(pass_data, lon_min, lon_max, config.selected_gate)
```

#### ✅ Standardized Gate Division System

**New GateModel Fields:**
- `parent_gate` - Reference to parent gate for divided sections
- `division_longitude` - Longitude where gate is split West/East

**Cache Key Strategy:**
```python
# Divided gates (Fram West, Fram East) share parent's cache
cache_gate = _get_parent_gate_id(config.selected_gate)  # Returns "fram_strait"
gate_name = cache_gate.replace(" ", "_").lower()
cached_data = _cache.load("slcci", gate_name)

# After loading, apply lon_filter for specific section
pass_data = apply_longitude_filter(pass_data, lon_min, lon_max, config.selected_gate)
```

**Shapefile Resolution:**
```python
# _get_gate_shapefile() now uses parent for divided gates
gate = service.get_gate("fram_strait_west")
actual_gate = service.get_gate(gate.parent_gate)  # "fram_strait"
shapefile = actual_gate.file  # "fram_strait_S3_pass_481.shp"
```

### ⚠️ Problemi Identificati (vedi AUDIT_REPORT_2026-01-06.md)

| Problema | Severity | Status |
|----------|----------|--------|
| ~~lon_filter non integrato in sidebar~~ | ~~🔴 HIGH~~ | ✅ DONE |
| Loaders module non usato | 🟠 MEDIUM | Dead code (OK) |
| GEBCO file mancante | 🟠 MEDIUM | User action |
| .pkl files in git | 🟡 LOW | .gitignore updated |

---

## 🚀 FEATURE: CMEMS L3/L4 Smart Loading Strategy (2026-01-04)

### Problema
Streamlit crashava con **Segmentation Fault** quando si selezionava "All Tracks" per CMEMS L3 locale:
- 7093 file NetCDF da caricare in parallelo
- Memory exhaustion → crash del processo

### Soluzione: Strategia Dual-Mode

#### LOCAL Mode → L3 Along-Track (Track Selection Required)
- Usa i file NetCDF locali (J1, J2, J3)
- **DEVE** selezionare un track specifico
- No "All Tracks" (causerebbe crash)
- Track options: suggested, closest, manual

#### API Mode → L4 Gridded (No Track Selection)  
- Usa l'API Copernicus Marine
- Dataset L4: `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`
- Risoluzione: 0.125° (~14km) daily
- **NO track selection** (L4 è gridded, non along-track)
- Filtra per bounding box geografico
- Tutti gli altimetri merged

### Perché L4 per API?
- L3 along-track datasets **non supportano** `open_dataset()` con filtro geografico
- Errore: `'Command' object is not subscriptable`
- L4 è già interpolato e funziona perfettamente con l'API

### Modifiche ai File

1. **`app/components/sidebar.py`** (`_render_cmems_params`):
   - LOCAL: track selection obbligatoria (no "all")
   - API: skip track selection UI (L4 non ha tracks)
   - Messaggi chiari per LOCAL vs API

2. **`src/services/cmems_service.py`** (`_load_from_api`):
   - Usa dataset L4 gridded
   - Conversione xarray → DataFrame
   - Supporta bounding box geografico

### Requisiti API
```bash
pip install copernicusmarine
copernicusmarine login  # Una volta per salvare credenziali
```

---

## 🐛 BUG FIX: Pass Extraction from Gate Filename (2026-01-04)

### Problema
Il pass suggerito non veniva estratto dal nome del gate shapefile.
- SLCCI: Nessun pass suggerito mostrato nella sidebar
- CMEMS: Nessun track suggerito mostrato

### Causa Root
**Il codice usava `config.selected_gate` (= gate_id) invece di `gate_path` (= filename)!**

```yaml
# config/gates.yaml
- id: denmark_strait          # ← config.selected_gate = questo
  file: denmark_strait_TPJ_pass_246.shp  # ← gate_path = questo (ha il pass!)
```

La funzione `_extract_pass_from_gate_name()` cercava pattern come `pass_XXX` nel nome,
ma riceveva solo `denmark_strait` invece di `denmark_strait_TPJ_pass_246.shp`.

### Soluzione
Modificato `app/components/sidebar.py`:

**Prima:**
```python
suggested_pass = _extract_pass_from_gate_name(config.selected_gate)  # ❌ gate_id
```

**Dopo:**
```python
gate_path = _get_gate_shapefile(config.selected_gate)
suggested_pass = _extract_pass_from_gate_name(gate_path)  # ✅ filename
```

Fix applicato in:
1. `_render_unified_pass_selection()` (linea ~180)
2. `_render_cmems_params()` (linea ~782)

### Lezione Appresa
- `gate_id` (es. "denmark_strait") ≠ `gate_file` (es. "denmark_strait_TPJ_pass_246.shp")
- Sempre verificare quale variabile contiene l'informazione necessaria

---

## 🐛 BUG FIX: Streamlit Black Screen (2026-01-04)

### Problema
L'app Streamlit mostrava **schermo completamente nero** in tutti i browser (Chrome, Safari).
- HTML caricato ma `<body>` vuoto
- Nessun errore nei log Python
- WebSocket connesso ma niente renderizzato

### Causa Root
**Mancava la chiamata `run_app()` alla fine di `app/main.py`!**

Il file definiva la funzione `run_app()` ma non la chiamava mai.
Streamlit caricava il modulo, eseguiva gli import, ma non eseguiva alcuna UI.

### Soluzione
Aggiunto alla fine di `app/main.py`:
```python
# === MAIN ENTRY POINT ===
# This is called when running: streamlit run app/main.py
run_app()
```

### Debugging Process
1. Test minimal app → ✅ funzionava
2. Test import singoli → ✅ tutti OK
3. Test esecuzione funzioni → ✅ tutti OK
4. Analisi `main.py` → ❌ `run_app()` mai chiamato

### Lezione Appresa
Quando Streamlit mostra nero ma nessun errore:
1. Verificare che ci sia codice **eseguito** (non solo definito)
2. Streamlit esegue il file top-to-bottom
3. Funzioni definite ma non chiamate = niente UI

---

## 🧠 Pre-Task: Awareness Check

**PRIMA di ogni task, verifica:**
- [ ] Letto `docs/TASKS/CONTEXT.md`?
- [ ] Letto `docs/UNIFIED_DATA_PIPELINE.md`? ← NEW!
- [ ] Verificato codice esistente?
- [ ] Usando `.venv/bin/python`?

---

## 🆕 4 DATASET ARCHITECTURE (2026-01-04 - LATEST)

### Dataset Comparison Table

| # | Dataset | Type | Filter Variable | Source | DOI/Link |
|---|---------|------|-----------------|--------|----------|
| 1 | **SLCCI** | Along-track (L2) | `pass` | Local | ESA CCI |
| 2 | **CMEMS L3** | Along-track (1Hz) | `track` | Local | [10.48670/moi-00149](https://doi.org/10.48670/moi-00149) |
| 3 | **CMEMS L4** | Gridded | ❌ none | **API** | [10.48670/moi-00148](https://doi.org/10.48670/moi-00148) |
| 4 | **DTUSpace** | Gridded | ❌ none | Local | DTU Space |

### Workflow per Tipo

**Along-Track (SLCCI, CMEMS L3):**
```
Gate → Find closest pass/track → Filter by pass/track → Scatter plot
```
- UI: Pass/Track selection (5 closest, manual, from filename)
- Spatial: Scatter points lungo la traccia satellite
- Slope: Calcolata su punti reali

**Gridded (CMEMS L4, DTUSpace):**
```
Gate → Sample gate geometry (N points) → KD-tree nearest grid → Extract DOT
```
- UI: Solo time range (NO pass selection)
- Spatial: Interpolazione sulla griglia → punti lungo il gate
- Slope: Calcolata su punti interpolati (synthetic pass)

### Files Created/Modified (2026-01-04)

| File | Action | Purpose |
|------|--------|---------|
| `src/services/cmems_l4_service.py` | **NEW** | CMEMS L4 via API (`copernicusmarine`) |
| `src/services/cmems_service.py` | Updated | Docstring con DOI link L3 |
| `src/services/dtu_service.py` | Updated | Docstring con comparison table |
| `src/services/__init__.py` | Updated | Export CMEMSL4Service |

### CMEMS L3 Dataset Info
- **Product**: SEALEVEL_GLO_PHY_L3_MY_008_062
- **Name**: Global Ocean Along-track L3 Sea Surface Heights
- **DOI**: https://doi.org/10.48670/moi-00149
- **URL**: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L3_MY_008_062/description
- **Type**: ALONG-TRACK (like SLCCI)
- **Filter**: `track` variable

### CMEMS L4 Dataset Info
- **Product**: SEALEVEL_GLO_PHY_L4_MY_008_047
- **Name**: Global Ocean Gridded L4 Sea Surface Heights
- **DOI**: https://doi.org/10.48670/moi-00148
- **URL**: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
- **Type**: GRIDDED (0.125° daily)
- **API**: `copernicusmarine.open_dataset()`

### API Usage (CMEMS L4)
```python
import copernicusmarine

ds = copernicusmarine.open_dataset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    variables=["adt", "sla"],
    minimum_longitude=bbox[0],
    maximum_longitude=bbox[2],
    minimum_latitude=bbox[1],
    maximum_latitude=bbox[3],
    start_datetime="2010-01-01",
    end_datetime="2020-12-31",
)
```

---

## 🐛 DTUSpace BUG FIXES (2026-01-03)

### Bug: DTUSpace Tabs Not Rendering After Load

**Problem**: Clicking "Load DTUSpace Data" loaded data successfully (logs showed 57600 observations) but tabs never appeared.

**Root Cause**: `app/main.py` line 64-67 only checked `slcci_pass_data` and `datasets`, NOT `dataset_dtu`:
```python
# OLD (broken):
if not slcci_data and not datasets:
    render_catalog_only_view()
    return
```

**Fix**: Added check for all dataset types:
```python
# NEW (fixed):
slcci_data = st.session_state.get("slcci_pass_data") or st.session_state.get("dataset_slcci")
cmems_data = st.session_state.get("dataset_cmems")
dtu_data = st.session_state.get("dataset_dtu")
datasets = st.session_state.get("datasets")

has_data = any([slcci_data, cmems_data, dtu_data, datasets])

if not has_data:
    render_catalog_only_view()
    return
```

### Files Modified

| File | Change |
|------|--------|
| `app/main.py` | Check all data types (SLCCI, CMEMS, DTU, generic) |
| `app/components/sidebar.py` | `gate_path = None` initialization (UnboundLocalError fix) |

---

## 🟢 DTUSpace v4 INTEGRATION (2026-01-03)

### Summary
Added DTUSpace v4 as third dataset (ISOLATED from SLCCI/CMEMS).

### Key Differences
| Aspect | SLCCI/CMEMS | DTUSpace |
|--------|-------------|----------|
| Type | Along-track | **Gridded** (lat × lon × time) |
| Pass/Track | Real satellite passes | **Synthetic** (from gate) |
| API | CEDA/Copernicus | **None** (local only) |
| Spatial | Scatter points | **Heatmap** grid |
| Color | 🟠/🔵 | 🟢 Green |

### Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `src/services/dtu_service.py` | **NEW** | DTUService, DTUConfig, DTUPassData |
| `app/state.py` | Modified | Added `dataset_dtu`, DTU functions |
| `app/components/sidebar.py` | Modified | DTUSpace option + config |
| `app/components/tabs.py` | Modified | 5 DTU tabs (ISOLATED) |
| `src/services/__init__.py` | Modified | Export DTU classes |

### DTUSpace Tabs
| Tab | Content |
|-----|---------|
| 🟢 Slope Timeline | Monthly slope time series |
| 🟢 DOT Profile | Mean DOT across gate with WEST/EAST |
| 🟢 Spatial Map | Heatmap of mean DOT grid |
| 🟢 Geostrophic Velocity | v_geo time series + climatology |
| 📥 Export | CSV export of synthetic data |

### Test Instructions
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
source .venv/bin/activate
streamlit run streamlit_app.py

# In sidebar:
# 1. Select gate
# 2. Choose "DTUSpace" dataset
# 3. Set NetCDF path
# 4. Click "Load DTUSpace Data"
```

---

## 📋 FULL AUDIT SESSION (2026-01-03)

### Summary
Full code review and audit of all tabs for SLCCI and CMEMS datasets.

### Changes Made

| Task | Status | Files |
|------|--------|-------|
| Export Tab for CMEMS | ✅ DONE | `tabs.py` - 5 tabs now (was 4) |
| Audit test scripts | ✅ DONE | `scripts/quick_test.py`, `tab_audit.py`, `test_full_audit.py` |
| Audit report | ✅ DONE | `docs/AUDIT_REPORT_2026-01-03.md` |
| Start script | ✅ DONE | `start_streamlit.sh` |

### Tab Configuration by Dataset

| Dataset | Tabs |
|---------|------|
| **SLCCI** | 6: Slope, DOT, Spatial, Monthly, Geostrophic, Export |
| **CMEMS** | 5: Slope, DOT, Spatial, Geostrophic, Export |
| **DTUSpace** | 5: 🟢Slope, 🟢DOT, 🟢Spatial(grid), 🟢Geostrophic, Export |
| **Comparison** | 7: Slope, DOT, Spatial, Geostrophic, Correlation, Difference, Export |

### Test Instructions
```bash
cd /Users/nicolocaron/Documents/GitHub/nico
./start_streamlit.sh
# Open http://localhost:8501
```

---

## 🚀 CMEMS PERFORMANCE OPTIMIZATIONS (2026-01-03)

### New Features Implemented

| Task | Status | Files |
|------|--------|-------|
| Parallel file loading | ✅ DONE | `cmems_service.py` - `_load_parallel()` |
| Caching with pickle | ✅ DONE | `cmems_service.py` - `CACHE_DIR` |
| CMEMS API support | ✅ DONE | `cmems_service.py` - `_load_from_api()` |
| Dynamic variables in Spatial Map | ✅ DONE | `tabs.py` - SLCCI vs CMEMS variables |
| Performance UI options | ✅ DONE | `sidebar.py` - Cache/Parallel toggles |
| README updated | ✅ DONE | Streamlit section added |

### CMEMS Variables vs SLCCI
| SLCCI | CMEMS |
|-------|-------|
| `corssh` | `sla_filtered` |
| `geoid` | `mdt` |
| `dot` | `dot` |
| `cycle` | `cycle` |
| `pass` | `track` |
| - | `satellite` |

### Performance Options (sidebar.py)
- ⚡ **Parallel Loading**: ThreadPoolExecutor (8 workers)
- 📦 **Cache**: Pickle files in `data/cache/cmems_processed/`
- 🗑️ **Clear Cache**: Button to reset

---

## 🆕 COMPARISON MODE & EXPORT (2026-01-02) 

### ✅ FUNCTIONAL TESTS PASSED (2026-01-02)

| Test | Status | Result |
|------|--------|--------|
| SLCCI Service Import | ✅ PASS | Config + Service work |
| CMEMS Service | ✅ PASS | 29010 rows, pass 481 extracted |
| Pass Extraction | ✅ PASS | All 5 patterns work |
| State Functions | ✅ PASS | store/get/clear work |
| Tabs Imports | ✅ PASS | All comparison functions load |

**Test Script**: `scripts/test_comparison_mode.py`

### New Features Implemented

| Task | Status | Files |
|------|--------|-------|
| Pass extraction from filename | ✅ DONE | `cmems_service.py` - `_extract_pass_from_gate_name()` |
| CMEMS buffer fix (5.0°) | ✅ DONE | From Copernicus notebook |
| Separate session state keys | ✅ DONE | `state.py` - `dataset_slcci`, `dataset_cmems` |
| Comparison mode toggle | ✅ DONE | `sidebar.py` - checkbox when both loaded |
| Comparison tabs overlay | ✅ DONE | `tabs.py` - `_render_comparison_tabs()` |
| Export tab (CSV + PNG) | ✅ DONE | `tabs.py` - `_render_export_tab()` |
| Git Commit & Push | ✅ DONE | Commits: 536dc80, a4bc166 |

### Comparison Mode Colors
- **SLCCI**: `darkorange` (🟠)
- **CMEMS**: `steelblue` (🔵)

### Comparison Mode Pattern (from COMPARISON_BATCH.ipynb)
- **SLCCI**: Orange (`tab:orange`) 
- **CMEMS**: Blue (`tab:blue`)
- Overlay plots on same figure
- Statistics comparison side-by-side

### Pass Number Extraction Patterns
- `_pass_XXX` at end → `("Strait Name", 248)`
- `_XXX` trailing number → `("Strait Name", 248)`
- `pass_XXX` anywhere → `("Strait Name", 248)`
- No pass found → `("Strait Name", None)` (synthetic pass)

---

## 🔄 UNIFIED DATA PIPELINE (2025-12-29)

### GitHub Issue: #16 (Architecture Agent)

| Task | Status | Notes |
|------|--------|-------|
| Document pipeline | ✅ DONE | docs/UNIFIED_DATA_PIPELINE.md |
| DataService uses IntakeCatalogBridge | ✅ DONE | Prioritized catalog.yaml routing |
| Mock data in altimetry format | ✅ DONE | corssh, mss, lat, lon variables |
| Fix TimeRange string handling | ✅ DONE | Handles both string and datetime |

**Key Insight**: Alemusica already has working API tokens in React!
- React calls `/api/v1/data/*` endpoints
- FastAPI routes to DataManager
- DataManager uses existing clients (ERA5, CMEMS, etc.)
- Credentials in environment variables

---

## 🎯 Phase 1: Catalog Foundation

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 1.1 catalog.yaml | ✅ DONE | 2024-12-28 | 2024-12-28 | 11 datasets, latency badges |
| 1.2 intake_bridge.py | ✅ DONE | 2024-12-28 | 2024-12-28 | Bridge + search + summary |
| 1.3 catalog API | ✅ DONE | 2024-12-28 | 2024-12-28 | 4 endpoints added |

## 🛰️ Phase 2: CYGNSS Client (PARALLELIZZABILE)

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 2.1 cygnss_client.py | ✅ DONE | 2024-12-28 | 2024-12-28 | HIGH priority, NASA near-RT |

## 🔗 Phase 3: Causal Graph (PARALLELIZZABILE)

| Task | Status | Started | Completed | Notes |
|------|--------|---------|-----------|-------|
| 3.1 causal_graph.py | ✅ DONE | 2024-12-28 | 2024-12-28 | SurrealDB + 4 known chains |

---

## 🔄 UNIFIED ARCHITECTURE REFACTORING (v2.0)

### GitHub Issue: #12
https://github.com/Alemusica/nico/issues/12

| Phase | Task | Status | Notes |
|-------|------|--------|-------|
| 0 | Documentation | ✅ DONE | ROADMAP, GATES_CATALOG, MODELS docs |
| 1 | Core Models | ✅ DONE | src/core/models.py (Pydantic) |
| 2 | Config Files | ✅ DONE | config/gates.yaml, datasets.yaml, regions.yaml, defaults.yaml |
| 3 | Gates Module | ✅ DONE | src/gates/{catalog,loader,buffer,passes}.py |
| 4 | Services Layer | ✅ DONE | src/services/{gate,data,analysis}_service.py |
| 5 | API Integration | ✅ DONE | api/routers/gates_router.py |
| 6 | Streamlit v2 | ✅ DONE | app/components/sidebar_v2.py, data_selector.py |
| 7 | Data Loaders | ✅ DONE | src/data/unified_loader.py |
| 8 | Tests | ✅ DONE | tests/test_core_models.py, test_gate_service.py |
| 9 | Docs Update | ✅ DONE | FEATURE_INVENTORY.md, CHANGELOG.md |
| 10 | Merge | ⬜ TODO | Merge to master, cleanup |

---

## 🐛 BUG FIXES (2025-12-29)

| Issue | Status | Description |
|-------|--------|-------------|
| #13 | ✅ CLOSED | GateService missing get_gate() method |
| #14 | ✅ CLOSED | TimeRange string vs datetime type error |
| #15 | ✅ CLOSED | Centralized Logging System implemented |

---

## 🔧 INFRASTRUCTURE (2025-12-29)

| Component | Status | Files |
|-----------|--------|-------|
| Logging System | ✅ DONE | src/core/logging_config.py |
| Feature Inventory | ✅ DONE | docs/FEATURE_INVENTORY.md |
| Issue Documentation | ✅ DONE | docs/ISSUES/BUG_001, BUG_002, FEATURE_003 |

---

## 🛰️ SLCCI INTEGRATION (2026-01-02) ✅ STATE OF THE ART

| Task | Status | Files |
|------|--------|-------|
| SLCCIService | ✅ DONE | src/services/slcci_service.py |
| Geoid Interpolation | ✅ DONE | Using TUM_ogmoc.nc |
| Pass Finding | ✅ DONE | find_closest_pass() |
| DOT Calculation | ✅ DONE | DOT = corssh - geoid |
| **Unified tabs.py** | ✅ DONE | app/components/tabs.py |
| lon_bin_size Config | ✅ DONE | SLCCIConfig in sidebar |
| Documentation | ✅ DONE | docs/VISUALIZATION_ARCHITECTURE.md |

---

## � CMEMS INTEGRATION (2026-01-02) ✅ NEW

| Task | Status | Files |
|------|--------|-------|
| CMEMSService | ✅ DONE | src/services/cmems_service.py |
| DOT Calculation | ✅ DONE | DOT = sla_filtered + mdt (MDT included) |
| Jason Merge | ✅ DONE | J1+J2+J3 merged automatically |
| Monthly Slopes | ✅ DONE | Binning + linear regression |
| Geostrophic Velocity | ✅ DONE | v = -g/f * (dη/dx) |
| 66°N Coverage Warning | ✅ DONE | check_gate_coverage() |

### 🎯 5 Tabs Implemented

| Tab | X-axis | Y-axis | Data Source |
|-----|--------|--------|-------------|
| **1. Slope Timeline** | `time_array` (dates) | `slope_series` (m/100km) | PassData attributes |
| **2. DOT Profile** | `x_km` (Distance km) | `profile_mean` (DOT m) | PassData attributes |
| **3. Spatial Map** | lon | lat | DataFrame + gate overlay |
| **4. Monthly Analysis** | Longitude (°) | DOT (m) | 12 subplots + regression |
| **5. Geostrophic Velocity** | time | v_geo (cm/s) | NEW! v = -g/f * slope |

### 🔑 Key Differences SLCCI vs CMEMS

| Aspect | SLCCI | CMEMS |
|--------|-------|-------|
| DOT Calculation | corssh - TUM_ogmoc | sla_filtered + mdt |
| Satellites | J2 single | J1+J2+J3 merged |
| Pass Selection | Auto/Manual | None (gate = synthetic pass) |
| lon_bin_size | 0.01-0.10° | 0.05-0.50° |
| External Geoid | ✅ Required | ❌ MDT included |
| Coverage | Global | ±66° latitude |

### 📄 PassData Interface Extended
```python
class PassData:
    strait_name: str
    pass_number: int
    slope_series: np.ndarray      # Shape: (n_periods,)
    time_array: np.ndarray        # Shape: (n_periods,)
    profile_mean: np.ndarray      # Shape: (n_lon_bins,)
    x_km: np.ndarray              # Shape: (n_lon_bins,)
    dot_matrix: np.ndarray        # Shape: (n_lon_bins, n_periods)
    df: pd.DataFrame              # Columns: lat, lon, dot, month, time
    gate_lon_pts, gate_lat_pts: np.ndarray
    # NEW for Tab 5 (Geostrophic):
    v_geostrophic_series: np.ndarray  # Shape: (n_periods,) in m/s
    mean_latitude: float              # For Coriolis display
    coriolis_f: float                 # f = 2Ω sin(lat)
```

**Logica tabs.py** (usa getattr per compatibilità):
```python
slope_series = getattr(slcci_data, 'slope_series', None)
profile_mean = getattr(slcci_data, 'profile_mean', None)
x_km = getattr(slcci_data, 'x_km', None)
```

### 📄 Documentazione Architettura
**Vedi**: `docs/VISUALIZATION_ARCHITECTURE.md` per:
- Specifiche complete dei 4 tabs
- Come aggiungere nuovi dataset
- Calcoli chiave (slope, lon_to_km)
- Checklist per nuovi dataset

---

## 📊 VISUALIZATION STATUS

| Feature | Location | Status | Notes |
|---------|----------|--------|-------|
| **tabs.py (UNIFIED)** | app/components/tabs.py | ✅ STATE OF THE ART | 4 tabs, SLCCI PLOTTER compatible |
| Slope Timeline | tabs.py → _render_slope_timeline | ✅ WORKING | Uses slope_series, time_array |
| DOT Profile | tabs.py → _render_dot_profile | ✅ WORKING | Uses profile_mean, x_km (NOT latitude!) |
| Spatial Map | tabs.py → _render_spatial_map | ✅ WORKING | MapBox + Gate overlay |
| Monthly Analysis | tabs.py → _render_monthly_analysis | ✅ WORKING | 12 subplots + linear regression |

**To see SLCCI/CMEMS graphs**: 
1. Select gate from sidebar
2. Expand "🛰️ SLCCI Data (ESA CCI)" section
3. Set paths to J2 data and TUM_ogmoc.nc
4. Click "Load SLCCI Data"
5. All 5 tabs now work correctly!

**Blockers**: NONE ✅

**Next Steps**:
- [x] Apply same architecture to CMEMS dataset ✅
- [x] Create CMEMSService with PassData interface ✅
- [x] Add Tab 5 (Geostrophic Velocity) ✅
- [ ] Integrate CMEMS into sidebar.py
- [ ] Apply same architecture to ERA5 dataset
- [ ] Create ERA5Service with PassData interface

---

### [DATE] - Task X.X
**Status**: ✅ / ❌ / 🔄
**What was done**:
- ...

**Blockers**:
- ...

**Next**:
- ...

---

## 🏗️ Architecture Refactoring (v2.0)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Documentation | ✅ DONE | Roadmap, Models, Gates docs |
| Phase 1: Core Models | ⬜ TODO | `src/core/models.py` |
| Phase 2: Config | ⬜ TODO | `config/` directory |
| Phase 3: Gates Module | ⬜ TODO | `src/gates/` |
| Phase 4: Services | ⬜ TODO | `src/services/` |
| Phase 5: API | ⬜ TODO | `gates_router.py` |
| Phase 6: Streamlit | ⬜ TODO | Refactor sidebar |
| Phase 7: React | ⬜ TODO | Gates component |
| Phase 8: Loaders | ⬜ TODO | Migrate from Legacy |
| Phase 9: Testing | ⬜ TODO | 80% coverage |
| Phase 10: Merge | ⬜ TODO | Final cleanup |

📄 See `docs/ROADMAP_UNIFIED_ARCHITECTURE.md` for details.

---

## ✅ Esistente (NON toccare)

| File | Linee | Cosa fa |
|------|-------|---------|
| `src/data_manager/catalog.py` | 736 | CopernicusCatalog (solo CMEMS) |
| `src/surge_shazam/data/era5_client.py` | ~200 | ERA5 download |
| `src/surge_shazam/data/cmems_client.py` | ~300 | CMEMS download |
| `src/surge_shazam/data/climate_indices.py` | ~150 | NOAA indices |

---

## ✅ Legend

- ⬜ TODO
- 🔄 IN PROGRESS  
- ✅ DONE
- ❌ BLOCKED
