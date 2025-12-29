# 🔄 Unified Data Pipeline

> **CRITICAL**: This document describes the SINGLE SOURCE OF TRUTH for data loading.
> ALL components (Streamlit, React, API, Notebooks) MUST use this pipeline.

---

## 🎯 The Problem We're Solving

Multiple implementations existed:
- `src/services/data_service.py` - Streamlit's data loading
- `src/data_manager/manager.py` - API's DataManager
- `src/data_manager/intake_bridge.py` - Catalog bridge
- Direct API calls scattered in React

This caused:
- Duplicated code
- Inconsistent behavior
- Hard-to-maintain codebase
- Confusion about which to use

---

## ✅ The Solution: Unified Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │  Streamlit   │   │    React     │   │       Notebooks          │ │
│  │  (sidebar)   │   │ (DataExplorer)│  │   (j2_utils imports)     │ │
│  └──────┬───────┘   └──────┬───────┘   └───────────┬──────────────┘ │
└─────────┼──────────────────┼───────────────────────┼────────────────┘
          │                  │                       │
          ▼                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       UNIFIED API LAYER                              │
│                                                                      │
│  FastAPI Router: /api/v1/data/*                                     │
│  - POST /data/load     (load data by catalog ID)                    │
│  - GET  /data/catalog  (list all datasets)                          │
│  - POST /data/download (download from remote API)                   │
│  - GET  /data/preview  (preview without full load)                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA SERVICE LAYER                               │
│                                                                      │
│  src/services/data_service.py                                       │
│  - Agnostic to data source                                          │
│  - Routes based on catalog.yaml config                              │
│  - NO hardcoded paths or API keys                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CATALOG & ROUTING LAYER                           │
│                                                                      │
│  catalog.yaml (source of truth for all datasets)                    │
│  │                                                                   │
│  ├── cmems_sealevel  → CopernicusCatalog client                     │
│  ├── cmems_sst       → CopernicusCatalog client                     │
│  ├── era5_reanalysis → ERA5Client                                   │
│  ├── noaa_tides      → NOAAClient                                   │
│  ├── local_slcci     → Local NetCDF loader                          │
│  └── demo_*          → Mock data generator                          │
│                                                                      │
│  src/data_manager/intake_bridge.py - IntakeCatalogBridge            │
│  - Reads catalog.yaml                                               │
│  - Instantiates appropriate client                                  │
│  - Returns xarray Dataset                                           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  CMEMS Client    │ │   ERA5 Client    │ │   Local Loader   │
│                  │ │                  │ │                  │
│ CopernicusCatalog│ │   ERA5Client     │ │  xr.open_dataset │
│ (736 lines!)     │ │                  │ │                  │
│                  │ │                  │ │                  │
│ Uses env vars:   │ │ Uses env vars:   │ │ Uses paths from  │
│ CMEMS_USERNAME   │ │ CDS_API_KEY      │ │ config/datasets  │
│ CMEMS_PASSWORD   │ │                  │ │ .yaml            │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

---

## 📁 Key Files

| File | Purpose | Status |
|------|---------|--------|
| `catalog.yaml` | Master dataset catalog | ✅ EXISTS |
| `config/datasets.yaml` | Secondary config | ✅ EXISTS |
| `src/data_manager/intake_bridge.py` | Catalog bridge | ✅ EXISTS |
| `src/data_manager/catalog.py` | CMEMS client (736 lines) | ✅ EXISTS |
| `src/surge_shazam/data/era5_client.py` | ERA5 client | ✅ EXISTS |
| `src/surge_shazam/data/cmems_client.py` | CMEMS simple client | ✅ EXISTS |
| `src/services/data_service.py` | Unified service | 🔄 NEEDS UPDATE |
| `api/routers/data_router.py` | API endpoints | ✅ EXISTS |

---

## 🔧 Environment Variables (REQUIRED)

```bash
# CMEMS (Copernicus Marine)
export CMEMS_USERNAME="your-username"
export CMEMS_PASSWORD="your-password"

# ERA5 (CDS)
export CDS_API_KEY="your-api-key"

# Optional: Override cache directory
export DATA_CACHE_DIR="/path/to/cache"
```

These are already configured in Alemusica's environment!

---

## 🚀 How to Load Data (The RIGHT Way)

### From Streamlit

```python
from src.services import DataService, GateService

# 1. Get gate (spatial bounds)
gs = GateService()
gate = gs.get_gate("fram_strait")

# 2. Build request from catalog
ds = DataService()
request = ds.build_request(
    gate=gate,
    dataset_id="cmems_sealevel",  # From catalog.yaml!
    time_range=TimeRange(start="2024-01-01", end="2024-12-31")
)

# 3. Load - DataService routes to correct client automatically
data = ds.load(request)
```

### From React

```typescript
// Call the API, NOT direct API calls to CMEMS/ERA5!
const response = await fetch(`${API_BASE}/data/load`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    dataset_id: 'cmems_sealevel',  // From catalog!
    bbox: { lat_min: 78, lat_max: 80, lon_min: -20, lon_max: 10 },
    time_range: { start: '2024-01-01', end: '2024-12-31' }
  })
});
```

### From Notebooks

```python
# Use the intake catalog directly
import intake
cat = intake.open_catalog('catalog.yaml')

# List available
print(cat.list())

# Load with metadata
ds = cat.cmems_sealevel.read()
```

---

## ❌ FORBIDDEN Patterns

```python
# ❌ NEVER DO THIS - Hardcoded paths
data_dir = "/Users/nicolocaron/Desktop/ARCFRESH/J2"

# ❌ NEVER DO THIS - Direct API calls bypassing catalog
import copernicusmarine
ds = copernicusmarine.open_dataset(...)

# ❌ NEVER DO THIS - Hardcoded credentials
username = "myuser"
password = "mypass"

# ❌ NEVER DO THIS - Duplicate client implementations
class MyOwnCMEMSClient:
    ...
```

---

## 📊 Adding a New Dataset

1. **Add to `catalog.yaml`**:
```yaml
sources:
  my_new_dataset:
    driver: intake_xarray.netcdf.NetCDFSource
    description: "My new data source"
    metadata:
      provider: MyProvider
      variables: [var1, var2]
      latency_badge: "🟡"
      client: "src.my_module.MyClient"
      status: available
```

2. **Create client if needed** (in `src/surge_shazam/data/`):
```python
class MyClient:
    def load(self, bbox, time_range, variables):
        # Implementation
        return xr.Dataset(...)
```

3. **Register in `intake_bridge.py`** if special handling needed.

4. **TEST IT**:
```python
from src.data_manager.intake_bridge import IntakeCatalogBridge
cat = IntakeCatalogBridge()
ds = cat.load("my_new_dataset", bbox=..., time_range=...)
```

---

## 🔄 Migration Path

Current state → Unified pipeline:

1. ✅ `catalog.yaml` exists with all datasets
2. ✅ `IntakeCatalogBridge` can route to clients
3. 🔄 `DataService` needs to use `IntakeCatalogBridge` instead of custom routing
4. 🔄 Streamlit sidebar needs to call DataService correctly
5. ✅ React already calls API endpoints

---

## 📝 Related Issues

- #16 - Architecture Agent (coordination)
- Need: Data pipeline unification issue

---

*Last updated: 29 Dec 2025 - Session 2*
