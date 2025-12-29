# 🔴 ARCHITECTURE GAP ANALYSIS

> **Status**: CRITICAL - Architecture exists but UI is not wired up!
> **Date**: 2025-12-29
> **Branch**: feature/gates-streamlit

---

## ✅ What EXISTS and WORKS

### 1. Core Layer (`src/core/`)
```python
from src.core.models import GateModel, BoundingBox, TimeRange, DataRequest
# ✅ All models work, tested
```

### 2. Services Layer (`src/services/`)
```python
from src.services import GateService, DataService, AnalysisService
# ✅ All services instantiate and work
gs = GateService()
gs.list_gates()  # Returns 8 gates ✅
```

### 3. Gates Module (`src/gates/`)
```python
from src.gates import GateCatalog
# ✅ Loads from config/gates.yaml
```

### 4. Config Files (`config/`)
- `gates.yaml` ✅ 8 gates with bbox, datasets, buffer
- `datasets.yaml` ✅ Dataset definitions
- `regions.yaml` ✅ Pre-defined regions
- `defaults.yaml` ✅ Default parameters

### 5. API Routers (`api/routers/`)
- `gates_router.py` ✅ /api/v1/gates endpoints

---

## ❌ What is BROKEN / NOT CONNECTED

### 1. Streamlit → DataService

**Problem**: `app/main.py` doesn't call `DataService` when user clicks "Load Data"

```python
# Current flow (BROKEN):
render_sidebar() → gate selected → NOTHING HAPPENS

# Expected flow:
render_sidebar() → gate selected → DataService.load_dataset() → session_state.datasets
```

### 2. data_selector.py Not Used

**Problem**: We have `app/components/data_selector.py` with full UI but it's NOT rendered!

```python
# main.py imports but never calls:
from app.components.data_selector import render_data_selector  # IMPORTED
# render_data_selector()  # NEVER CALLED
```

### 3. _handle_data_load Never Called

**Problem**: `main.py` has `_handle_data_load()` function but nothing triggers it!

---

## 🔧 IMMEDIATE FIX NEEDED

### Option A: Wire up existing data_selector.py

```python
# In main.py run_app():
if DATA_SELECTOR_AVAILABLE:
    selection = render_data_selector()
    if is_data_load_requested():
        _handle_data_load(selection)
        clear_load_request()
```

### Option B: Add Load button to sidebar.py

```python
# After gate selection in sidebar:
if st.sidebar.button("🚀 Load Gate Data"):
    data_service = DataService()
    datasets = data_service.load_for_gate(gate_id, time_range)
    st.session_state.datasets = datasets
```

---

## 📋 TODO to Fix This

1. [ ] Connect `render_data_selector()` in main.py
2. [ ] Wire `_handle_data_load()` to button click
3. [ ] Test DataService.load_dataset() with real bbox
4. [ ] Add progress bar during loading
5. [ ] Handle errors gracefully

---

## 🏗️ Architecture Flow (How it SHOULD work)

```
┌──────────────────────────────────────────────────────────────┐
│                         USER ACTION                           │
│          Select Gate → Select Dataset → Click Load            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app/main.py)                 │
│  render_data_selector() → is_data_load_requested() → True    │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    SERVICES LAYER                             │
│  DataService.load_dataset(dataset_id, bbox, time_range)      │
│       │                                                       │
│       ├── Check Intake catalog                                │
│       ├── Download/load NetCDF                                │
│       └── Return xarray.Dataset                               │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    SESSION STATE                              │
│  st.session_state.datasets = [xr.Dataset, ...]               │
│  st.session_state.cycle_info = [...]                          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    VISUALIZATION                              │
│  render_tabs(config) → render_slope_timeline_tab(datasets)   │
│                      → render_profiles_tab(datasets)          │
│                      → render_map_tab(datasets)               │
└──────────────────────────────────────────────────────────────┘
```

