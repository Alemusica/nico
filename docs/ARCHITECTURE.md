# 🏗️ NICO Unified Architecture

> **Version**: 2.1 - Post-Recovery Architecture  
> **Updated**: 2026-01-06  
> **Status**: ✅ IMPLEMENTED + Services Integrated

---

## 📊 Master Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NICO UNIFIED ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     PRESENTATION LAYER                               │   │
│  │  ┌───────────────────┐  ┌───────────────────┐  ┌────────────────┐  │   │
│  │  │ React + Cosmograph │  │   Streamlit App   │  │    CLI/API     │  │   │
│  │  │  (master branch)   │  │  (gates branch)   │  │  (notebooks)   │  │   │
│  │  └─────────┬─────────┘  └─────────┬─────────┘  └───────┬────────┘  │   │
│  └────────────┼──────────────────────┼────────────────────┼────────────┘   │
│               │                      │                    │                 │
│               ▼                      ▼                    ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      API GATEWAY LAYER                               │   │
│  │                        FastAPI (api/)                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ /gates     /data     /analysis     /knowledge    /pipeline  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DOMAIN SERVICES LAYER                             │   │
│  │              src/services/ (NEW - shared by all)                     │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │   │
│  │  │ GateService  │ │ DataService  │ │AnalysisServ │ │ PipelineS │  │   │
│  │  │              │ │              │ │             │ │           │  │   │
│  │  │- select_gate │ │- load_data   │ │- compute_dot│ │- run_pipe │  │   │
│  │  │- get_bbox    │ │- filter_bbox │ │- bin_stats  │ │- get_state│  │   │
│  │  │- buffer_area │ │- merge_cycles│ │- find_causals│ │- resume   │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       CORE LAYER (SHARED)                            │   │
│  │                          src/core/                                   │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │     models.py      │   coordinates.py   │     config.py       │  │   │
│  │  │  (Pydantic models) │  (geo utilities)   │  (shared configs)   │  │   │
│  │  ├───────────────────┼───────────────────┼─────────────────────┤  │   │
│  │  │  • GateModel      │  • wrap_longitudes│  • load_yaml_config  │  │   │
│  │  │  • BoundingBox    │  • lon_in_bounds  │  • get_defaults      │  │   │
│  │  │  • TimeRange      │  • create_mask    │  • DatasetConfig     │  │   │
│  │  │  • DataRequest    │  • get_lon_lat    │  • AppConfig         │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DATA ACCESS LAYER                                 │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │   │
│  │  │ GateCatalog  │ │DatasetCatalog│ │    Loaders   │ │   Cache   │  │   │
│  │  │ src/gates/   │ │ catalog.yaml │ │ (xarray/nc)  │ │  (future) │  │   │
│  │  │              │ │              │ │              │ │           │  │   │
│  │  │- gates.yaml  │ │- intake      │ │- load_cycle  │ │- get/set  │  │   │
│  │  │- load_shape  │ │- cmems_client│ │- filter_bbox │ │- invalidate│ │   │
│  │  │- get_passes  │ │- era5_client │ │- merge       │ │- ttl      │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│               │                                                             │
│               ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     INFRASTRUCTURE LAYER                             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │   │
│  │  │  SurrealDB   │ │  NetCDF/nc   │ │  Shapefiles  │ │ External  │  │   │
│  │  │  (knowledge) │ │  (altimetry) │ │   (gates/)   │ │ APIs      │  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure (Implemented)

```
nico/
├── config/                         # ✅ CENTRALIZED CONFIG
│   ├── gates.yaml                  # ✅ 8 gates with bbox, datasets
│   ├── datasets.yaml               # ✅ Dataset definitions
│   ├── regions.yaml                # ✅ Pre-defined regions
│   └── defaults.yaml               # ✅ Default parameters
│
├── src/
│   ├── core/                       # ✅ SHARED CORE
│   │   ├── models.py               # ✅ Pydantic models (GateModel, BoundingBox, etc.)
│   │   ├── coordinates.py          # ✅ Geo utilities
│   │   ├── config.py               # ✅ Config loader
│   │   ├── logging_config.py       # ✅ Centralized logging
│   │   └── helpers.py              # ✅ General utilities
│   │
│   ├── gates/                      # ✅ GATES MODULE
│   │   ├── __init__.py             # ✅ Exports
│   │   ├── catalog.py              # ✅ GateCatalog class
│   │   ├── loader.py               # ✅ Shapefile loading
│   │   ├── passes.py               # ✅ Pass filtering
│   │   └── buffer.py               # ✅ Buffer calculations
│   │
│   ├── services/                   # ✅ DOMAIN SERVICES (Updated 2026-01-06)
│   │   ├── __init__.py             # ✅ All exports
│   │   ├── gate_service.py         # ✅ Gate operations
│   │   ├── data_service.py         # ✅ Data loading
│   │   ├── analysis_service.py     # ✅ Analysis pipelines
│   │   ├── slcci_service.py        # ✅ ESA SLCCI loading
│   │   ├── cmems_service.py        # ✅ CMEMS L3 along-track
│   │   ├── cmems_l4_service.py     # ✅ CMEMS L4 gridded (API)
│   │   ├── dtu_service.py          # ✅ DTUSpace gridded
│   │   ├── cache_service.py        # ✅ NEW - Persistent cache
│   │   ├── bathymetry_service.py   # ✅ NEW - GEBCO extraction
│   │   └── transport_service.py    # ✅ NEW - Volume transport
│   │
│   ├── data/                       # ✅ DATA UTILITIES
│   │   ├── loaders.py              # ✅ NetCDF loading
│   │   └── unified_loader.py       # ✅ Unified data loader
│   │
│   └── analysis/                   # ✅ ANALYSIS
│       ├── dot.py                  # ✅ DOT computation
│       └── slope.py                # ✅ Slope analysis
│
├── api/                            # ✅ API GATEWAY
│   ├── main.py                     # ✅ FastAPI app
│   └── routers/
│       ├── gates_router.py         # ✅ /api/v1/gates
│       ├── data_router.py          # ✅ /api/v1/data
│       ├── analysis_router.py      # ✅ /api/v1/analysis
│       └── knowledge_router.py     # ✅ /api/v1/knowledge
│
├── app/                            # ✅ STREAMLIT UI (Updated 2026-01-06)
│   ├── main.py                     # ✅ Entry point
│   ├── state.py                    # ✅ Session state + AppConfig
│   └── components/
│       ├── sidebar.py              # ✅ Gate/dataset selector + cache
│       ├── tabs.py                 # ✅ All visualization tabs
│       ├── data_selector.py        # ✅ Unified data selector
│       ├── charts/                 # ✅ Chart components
│       │   ├── slope_chart.py      # ✅ Slope timeline
│       │   ├── dot_profile_chart.py # ✅ DOT profile
│       │   ├── spatial_chart.py    # ✅ Spatial map
│       │   ├── geostrophic_chart.py # ✅ Geostrophic velocity
│       │   └── volume_transport_chart.py # ✅ +bathymetry
│       └── loaders/                # ⚠️ NOT YET WIRED
│           ├── base.py             # ✅ apply_longitude_filter
│           ├── slcci_loader.py     # ✅ SLCCI loader
│           ├── dtu_loader.py       # ✅ DTU loader
│           └── cmems_l4_loader.py  # ✅ CMEMS L4 loader
│
├── frontend/                       # ✅ REACT (master branch)
│   └── src/                        # ✅ React + Cosmograph
│
└── tests/
    ├── test_core_models.py         # ✅ Model tests
    └── test_gates/                 # ✅ Gate tests
```

---

## � CMEMS Service Architecture (NEW - 2026-01-03)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CMEMS DATA LOADING                                │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    CMEMSConfig (dataclass)                         │  │
│  │  - source_mode: "local" | "api"                                   │  │
│  │  - use_parallel: bool (ThreadPoolExecutor)                        │  │
│  │  - use_cache: bool (pickle files)                                 │  │
│  │  - base_dir: path to J1/J2/J3_netcdf folders                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      CMEMSService                                  │  │
│  │                                                                    │  │
│  │  load_pass_data(gate_path) ──┬── source_mode="local" ──┐          │  │
│  │                              │                          ▼          │  │
│  │                              │    ┌─────────────────────────┐     │  │
│  │                              │    │ _load_all_jason_files() │     │  │
│  │                              │    │   - Check cache first   │     │  │
│  │                              │    │   - Parallel/Sequential │     │  │
│  │                              │    │   - Save to cache       │     │  │
│  │                              │    └─────────────────────────┘     │  │
│  │                              │                                     │  │
│  │                              └── source_mode="api" ──┐             │  │
│  │                                                       ▼             │  │
│  │                                   ┌─────────────────────────┐      │  │
│  │                                   │ _load_from_api()        │      │  │
│  │                                   │   - Use CMEMSClient     │      │  │
│  │                                   │   - Direct download     │      │  │
│  │                                   │   - Convert to DataFrame│      │  │
│  │                                   └─────────────────────────┘      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼                                           │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    PassData (output)                               │  │
│  │  - df: DataFrame with columns:                                     │  │
│  │      CMEMS: sla_filtered, mdt, dot, satellite, cycle, track       │  │
│  │      SLCCI: corssh, geoid, dot, cycle, pass                       │  │
│  │  - slope_series, time_array, dot_matrix, x_km, etc.               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### CMEMS vs SLCCI Variable Mapping

| Purpose | SLCCI | CMEMS |
|---------|-------|-------|
| Sea Level Anomaly | `corssh` | `sla_filtered` |
| Geoid/MDT | `geoid` | `mdt` |
| Dynamic Ocean Topography | `dot = corssh - geoid` | `dot = sla_filtered + mdt` |
| Time identifier | `cycle` | `cycle` |
| Spatial track | `pass` | `track` |
| Satellite | (implicit J1/J2) | `satellite` (J1/J2/J3) |

---

## �🔴 CURRENT GAP: UI → Services Not Wired

**Problem**: The architecture EXISTS but Streamlit doesn't call the Services!

```
CURRENT FLOW (BROKEN):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Sidebar    │ ──► │  Gate Select │ ──► │   NOTHING    │
│   (UI)       │     │  (works!)    │     │   HAPPENS    │
└──────────────┘     └──────────────┘     └──────────────┘

EXPECTED FLOW:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Sidebar    │ ──► │  Gate Select │ ──► │ DataService  │ ──► │   Datasets   │
│   (UI)       │     │  + Load Btn  │     │ .load_data() │     │   + Graphs   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## ✅ Implementation Status

| Layer | Component | Status | Notes |
|-------|-----------|--------|-------|
| **Presentation** | Streamlit | ✅ | Gate selector works |
| **Presentation** | React | ✅ | On master branch |
| **API Gateway** | FastAPI | ✅ | All routers exist |
| **Services** | GateService | ✅ | Fully working |
| **Services** | DataService | ⚠️ | Exists but not wired to UI |
| **Services** | AnalysisService | ⚠️ | Exists but not wired to UI |
| **Core** | models.py | ✅ | All models defined |
| **Core** | logging | ✅ | Centralized logging |
| **Data Access** | GateCatalog | ✅ | Loads from YAML |
| **Data Access** | Loaders | ✅ | NetCDF loading works |
| **Infrastructure** | SurrealDB | ✅ | Knowledge graph |
| **Infrastructure** | Shapefiles | ✅ | 8 gates available |

---

## 🎯 Next Steps to Complete

1. **Wire DataService to Streamlit UI**
   - Add "Load Data" button in sidebar
   - Call `DataService.load_dataset()` on click
   - Store result in `st.session_state.datasets`

2. **Wire AnalysisService to graphs**
   - Pass datasets to analysis tabs
   - Use gate bbox for filtering

3. **Test end-to-end flow**
   - Select gate → Load data → See graphs

---

## 🔄 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERACTION                               │
│                                                                          │
│   1. Select Gate ──► 2. Choose Dataset ──► 3. Click "Load" ──► 4. View  │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                │
         ▼                    ▼                    ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI (app/)                              │
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │  sidebar.py │───►│data_selector│───►│  main.py    │───►│  tabs.py │ │
│  │             │    │    .py      │    │_handle_load │    │  graphs  │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SERVICES LAYER (src/services/)                      │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │  GateService    │    │   DataService   │    │  AnalysisService    │ │
│  │                 │    │                 │    │                     │ │
│  │ .get_gate(id)   │───►│ .load_dataset() │───►│ .compute_dot()      │ │
│  │ .get_bbox()     │    │ .filter_bbox()  │    │ .bin_by_longitude() │ │
│  │ .get_buffer()   │    │ .merge_cycles() │    │ .compute_slope()    │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DATA ACCESS (src/gates/, src/data/)                  │
│                                                                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────┐ │
│  │   GateCatalog   │    │  DatasetCatalog │    │     Loaders         │ │
│  │                 │    │                 │    │                     │ │
│  │ config/gates.yml│    │ catalog.yaml    │    │ xarray.open_dataset │ │
│  │ gates/*.shp     │    │ intake          │    │ filter, merge       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE                                    │
│                                                                          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────────────────┐│
│  │ SurrealDB │  │  NetCDF   │  │ Shapefiles│  │   External APIs       ││
│  │           │  │  Files    │  │  (gates/) │  │  CMEMS, ERA5, etc.    ││
│  └───────────┘  └───────────┘  └───────────┘  └───────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Service Layer API Reference

### GateService (`src/services/gate_service.py`)

```python
from src.services import GateService

gs = GateService()

# List all gates
gates = gs.list_gates()  # -> List[GateModel]

# Get specific gate
gate = gs.get_gate("fram_strait")  # -> GateModel

# Get gate bounding box
bbox = gate.bbox  # -> BoundingBox

# Get gates by region
atlantic_gates = gs.list_gates_by_region("Atlantic Sector")
```

### DataService (`src/services/data_service.py`)

```python
from src.services import DataService
from src.core.models import BoundingBox, TimeRange

ds = DataService()

# List datasets
datasets = ds.list_datasets()  # -> List[str]

# Load data for a bbox
data = ds.load_dataset(
    dataset_id="cmems_sla",
    bbox=bbox,
    time_range=TimeRange(start=..., end=...),
    variables=["sla", "adt"]
)  # -> xarray.Dataset
```

### AnalysisService (`src/services/analysis_service.py`)

```python
from src.services import AnalysisService

analysis = AnalysisService()

# Run slope analysis
result = analysis.run_slope_analysis(
    datasets=datasets,
    config=config
)  # -> Dict with slopes, errors, etc.
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YAML config files** | Human-readable, version controllable, Kedro-compatible |
| **Pydantic models** | Type safety, validation, FastAPI native |
| **Service layer** | Same logic for API and Streamlit |
| **Gates as module** | First-class citizen, not sidebar hack |
| **Centralized logging** | Debug across layers consistently |

---

## 📚 Related Documents

- `docs/ROADMAP_UNIFIED_ARCHITECTURE.md` - Migration plan
- `docs/MODELS.md` - Pydantic models reference
- `docs/GATES_CATALOG.md` - Gates documentation
- `docs/ARCHITECTURE_GAP.md` - Current gaps analysis
- `docs/FEATURE_INVENTORY.md` - Cross-branch features

---

*Last updated: 2025-12-29 - Unified Architecture v2.0*

```
User Action
    │
    ▼
┌─────────────────┐
│   Sidebar       │ ──── Load files, set params
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Session State   │ ──── Store datasets, config
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Tab Component │ ──── Process & render
└─────────────────┘
    │
    ├──► src/analysis/* ──► Compute
    │
    └──► src/visualization/* ──► Plot
```

## Extension Points

### Adding a New Analysis

1. Create `src/analysis/new_analysis.py`
2. Add exports to `src/analysis/__init__.py`
3. Create `app/components/new_tab.py`
4. Register in `app/components/tabs.py`

### Adding a New Visualization

1. Add function to `src/visualization/plotly_charts.py`
2. Import and use in relevant tab component

### Supporting New Data Format

1. Add loader in `src/data/loaders.py`
2. Add any new filters in `src/data/filters.py`
3. Update documentation

## Testing Strategy

```
tests/
├── unit/
│   ├── test_coordinates.py    # Pure function tests
│   ├── test_slope.py
│   └── test_statistics.py
├── integration/
│   ├── test_loaders.py        # Requires test data
│   └── test_analysis.py
└── fixtures/
    └── test_data.nc           # Small test dataset
```

## Performance Considerations

1. **Caching**: Use `@st.cache_data` for expensive computations
2. **Sampling**: Limit points for map visualization
3. **Lazy Loading**: Load cycles on-demand when possible
4. **Chunking**: Use dask for very large datasets

---

## 🚀 Architecture Evolution (v2.0)

> **Status**: In Progress  
> **Tracking**: See `docs/ROADMAP_UNIFIED_ARCHITECTURE.md`

The architecture is being refactored to support:
- **Unified Gates Module** (`src/gates/`)
- **Centralized Config** (`config/`)
- **Services Layer** (`src/services/`)
- **Shared Pydantic Models** (`src/core/models.py`)

### New Components (v2.0)

```
config/                    # Centralized YAML configs
├── gates.yaml            # Ocean gates catalog
├── datasets.yaml         # Dataset providers
└── defaults.yaml         # Default parameters

src/gates/                # Gates module
├── catalog.py            # GateCatalog class
├── loader.py             # Shapefile loading
└── buffer.py             # Buffer calculations

src/services/             # Business logic layer
├── gate_service.py       # Gate operations
├── data_service.py       # Data operations
└── analysis_service.py   # Analysis operations
```

### Related Documentation
- [ROADMAP_UNIFIED_ARCHITECTURE.md](ROADMAP_UNIFIED_ARCHITECTURE.md) - Full refactoring plan
- [MODELS.md](MODELS.md) - Pydantic models reference
- [GATES_CATALOG.md](GATES_CATALOG.md) - Gates documentation

---

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
