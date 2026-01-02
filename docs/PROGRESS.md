# 📊 Surge Shazam - Progress Tracker

> Last Updated: 2026-01-02 (Session - SLCCI Visualization Complete)
> Agent: Use this file to track progress. Update after each task.

---

## 🧠 Pre-Task: Awareness Check

**PRIMA di ogni task, verifica:**
- [ ] Letto `docs/TASKS/CONTEXT.md`?
- [ ] Letto `docs/UNIFIED_DATA_PIPELINE.md`? ← NEW!
- [ ] Verificato codice esistente?
- [ ] Usando `.venv/bin/python`?

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

### 🎯 4 Tabs Implemented (Following SLCCI PLOTTER exactly)

| Tab | X-axis | Y-axis | Data Source |
|-----|--------|--------|-------------|
| **1. Slope Timeline** | `time_array` (dates) | `slope_series` (m/100km) | PassData attributes |
| **2. DOT Profile** | `x_km` (Distance km) | `profile_mean` (DOT m) | PassData attributes |
| **3. Spatial Map** | lon | lat | DataFrame + gate overlay |
| **4. Monthly Analysis** | Longitude (°) | DOT (m) | 12 subplots + regression |

### 🔑 Key Implementation Details

**PassData Interface** (standard per tutti i dataset):
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

**To see SLCCI graphs**: 
1. Select gate from sidebar
2. Expand "🛰️ SLCCI Data (ESA CCI)" section
3. Set paths to J2 data and TUM_ogmoc.nc
4. Click "Load SLCCI Data"
5. All 4 tabs now work correctly!

**Blockers**: NONE ✅

**Next Steps**:
- [ ] Apply same architecture to CMEMS dataset
- [ ] Apply same architecture to ERA5 dataset
- [ ] Create CMEMSService with PassData interface
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
