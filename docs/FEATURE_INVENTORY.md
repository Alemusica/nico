# 📚 Feature Inventory - Cross-Branch Documentation

> **Purpose**: Documentare tutte le feature implementate nei vari branch/layer per evitare duplicazioni e perdita di informazioni.
> **Last Updated**: 2026-01-06

---

## 🆕 Unified Tab System (2026-01-06)

### Location: `app/components/tabs.py`

**Status**: ✅ Implemented | **Used by**: All 3 datasets

All datasets now share identical rendering functions:

| Function | Purpose |
|----------|---------|
| `_render_unified_dataset_tabs()` | Main entry point - creates 6 tabs |
| `_render_unified_slope_timeline()` | Slope vs time + R² + trend |
| `_render_unified_dot_profile()` | DOT vs lon/km, m/cm/mm toggle |
| `_render_unified_spatial_map()` | Map with gate line |
| `_render_unified_monthly_analysis()` | 12 monthly subplots + R²/slope |
| `_render_unified_geostrophic_velocity()` | v = -g/f * (dη/dx) |
| `_render_unified_export_tab()` | CSV export |

**Helper Functions**:
- `_get_unified_dataset_info(data, dataset_type)` → Returns emoji, name, color, type
- `_render_monthly_from_df(df, strait_name, ds_info)` → For along-track data

---

## 🗺️ Location & Geo Features

### GeoResolver (`src/agent/tools/geo_resolver.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit, API, Agent

Risoluzione geografica di nomi luoghi in coordinate.

```python
from src.agent.tools.geo_resolver import GeoResolver, GeoLocation

resolver = GeoResolver()
location = await resolver.resolve("Lago Maggiore")
# Returns: GeoLocation(name="Lago Maggiore", lat=45.95, lon=8.65, bbox=(...))
```

**Features**:
- Nominatim (OpenStreetMap) API integration
- Cache locale per performance
- Known locations pre-definite (laghi italiani, stretti oceanici)
- Bounding box calculation
- Rate limiting (1 req/sec per Nominatim TOS)

**Files**:
- `src/agent/tools/geo_resolver.py` (493 lines)
- `tests/test_geo_resolver.py`

---

## 🚪 Gate System

### GateService (`src/services/gate_service.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit, API

Gestione centralizzata dei gate oceanici.

```python
from src.services import GateService

service = GateService()
gates = service.list_gates()
gate = service.get_gate("fram_strait")
bbox = service.get_bbox("fram_strait", buffer_km=50)
geometry = service.get_gate_geometry("fram_strait")
```

**Features**:
- Caricamento da `config/gates.yaml`
- Caricamento shapefile
- Buffer geografici
- Filtraggio satellite passes

**Files**:
- `src/services/gate_service.py`
- `src/gates/catalog.py`
- `src/gates/loader.py`
- `src/gates/buffer.py`
- `config/gates.yaml`

---

## �️ SLCCI Data Service

### SLCCIService (`src/services/slcci_service.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit

Service for loading and processing ESA Sea Level CCI (SLCCI) data.

```python
from src.services import SLCCIService, SLCCIConfig

# Initialize with config
config = SLCCIConfig(
    base_dir="/path/to/J2",
    geoid_path="/path/to/TUM_ogmoc.nc",
    cycles=list(range(1, 282)),
)
service = SLCCIService(config)

# Find closest passes to gate
closest_passes = service.find_closest_pass(gate_path="/path/to/gate.shp", n_passes=5)

# Load pass data with DOT computation
pass_data = service.load_pass_data(
    gate_path="/path/to/gate.shp",
    pass_number=248,
)

# Access results (PassData interface)
print(f"Observations: {len(pass_data.df)}")
print(f"Slope series: {pass_data.slope_series}")      # Shape: (n_periods,)
print(f"Time array: {pass_data.time_array}")          # Shape: (n_periods,)
print(f"DOT profile mean: {pass_data.profile_mean}")  # Shape: (n_lon_bins,)
print(f"Distance x_km: {pass_data.x_km}")             # Shape: (n_lon_bins,)
print(f"DOT matrix: {pass_data.dot_matrix.shape}")    # Shape: (n_lon_bins, n_periods)
```

**PassData Attributes** (Standard Interface for all datasets):
| Attribute | Type | Description |
|-----------|------|-------------|
| `strait_name` | str | Name of the gate/strait |
| `pass_number` | int | Satellite pass number |
| `slope_series` | np.ndarray | Slope per time period (m/100km) |
| `time_array` | np.ndarray | Dates for each period |
| `time_periods` | list | Period labels (YYYY-MM) |
| `profile_mean` | np.ndarray | Mean DOT per lon bin |
| `x_km` | np.ndarray | Distance in km along longitude |
| `dot_matrix` | np.ndarray | DOT values [space, time] |
| `df` | DataFrame | Raw data (lat, lon, dot, month, time) |
| `gate_lon_pts`, `gate_lat_pts` | np.ndarray | Gate line coordinates |

**Features**:
- Load SLCCI NetCDF files (SLCCI_ALTDB_J2_CycleXXX_V2.nc)
- Geoid interpolation using TUM_ogmoc.nc
- DOT calculation: corssh - geoid
- Pass filtering (auto-detect or manual)
- Slope computation along gate (m/100km)
- DOT matrix building for temporal analysis
- `lon_bin_size` configurable (default 0.05°)

**Files**:
- `src/services/slcci_service.py` (600+ lines)
- Uses: `legacy/j2_utils.py` patterns (migrated)

---

### SLCCI Visualization (tabs.py) ✅ STATE OF THE ART
**Status**: ✅ Complete | **Used in**: Streamlit | **Date**: 2026-01-02

**Single unified file**: `app/components/tabs.py` (450+ lines)

Following SLCCI PLOTTER notebook workflow exactly:

| Tab | Function | X-axis | Y-axis |
|-----|----------|--------|--------|
| 1. Slope Timeline | `_render_slope_timeline()` | `time_array` | `slope_series` (m/100km) |
| 2. DOT Profile | `_render_dot_profile()` | `x_km` (Distance km) | `profile_mean` (DOT m) |
| 3. Spatial Map | `_render_spatial_map()` | lon | lat (MapBox) |
| 4. Monthly Analysis | `_render_monthly_analysis()` | Longitude (°) | DOT (m) + regression |

**Key Implementation Pattern**:
```python
# tabs.py uses getattr for flexible data access
slope_series = getattr(slcci_data, 'slope_series', None)
time_array = getattr(slcci_data, 'time_array', None)
profile_mean = getattr(slcci_data, 'profile_mean', None)
x_km = getattr(slcci_data, 'x_km', None)  # Distance in km, NOT latitude!
```

**Tab Features**:
- **Slope Timeline**: Trend line, statistics, unit conversion (m/100km ↔ cm/km)
- **DOT Profile**: Mean profile, ±1 std band, individual periods view, WEST/EAST labels
- **Spatial Map**: Color by dot/corssh/geoid, gate overlay, 5000 point sampling
- **Monthly Analysis**: 12 subplots, linear regression per month, slopes summary table

**Architecture Documentation**: `docs/VISUALIZATION_ARCHITECTURE.md`

---

## 🌊 CMEMS Data Service

### CMEMSService (`src/services/cmems_service.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit | **Date**: 2026-01-02

Service for loading Copernicus Marine (CMEMS) L3 1Hz along-track altimetry data.

```python
from src.services.cmems_service import CMEMSService, CMEMSConfig

# Initialize with config
config = CMEMSConfig(
    base_dir="/path/to/COPERNICUS DATA",
    start_date=date(2002, 1, 1),
    end_date=date(2024, 12, 31),
    lon_bin_size=0.1,  # 0.05-0.50° (coarser than SLCCI)
    max_latitude=66.0,  # Jason coverage limit
)
service = CMEMSService(config)

# Check gate coverage
coverage = service.check_gate_coverage("/path/to/gate.shp")
if coverage["warning"]:
    print(f"⚠️ {coverage['warning']}")

# Load pass data
pass_data = service.load_pass_data(gate_path="/path/to/gate.shp")

# Access results (same PassData interface as SLCCI + geostrophic)
print(f"Observations: {len(pass_data.df)}")
print(f"Slope series: {pass_data.slope_series}")
print(f"v_geostrophic: {pass_data.v_geostrophic_series}")  # NEW! m/s
print(f"Mean latitude: {pass_data.mean_latitude}")
print(f"Coriolis f: {pass_data.coriolis_f}")
```

**Key Differences from SLCCI**:
| Aspect | SLCCI | CMEMS |
|--------|-------|-------|
| DOT | corssh - TUM_ogmoc | sla_filtered + mdt |
| Satellites | J2 single | J1+J2+J3 merged |
| Pass Selection | Auto/Manual | Gate name = synthetic pass |
| lon_bin_size | 0.01-0.10° | 0.05-0.50° |
| External Geoid | ✅ Required | ❌ MDT included |
| Coverage | Global | ±66° latitude |

**Extended PassData Attributes** (CMEMS adds):
| Attribute | Type | Description |
|-----------|------|-------------|
| `v_geostrophic_series` | np.ndarray | Geostrophic velocity (m/s) |
| `mean_latitude` | float | Mean lat for Coriolis display |
| `coriolis_f` | float | Coriolis parameter f = 2Ω sin(lat) |

**Files**:
- `src/services/cmems_service.py` (520+ lines)

---

### Tab 5: Geostrophic Velocity (NEW)
**Status**: ✅ Implemented | **Date**: 2026-01-02

Function: `_render_geostrophic_velocity()` in `app/components/tabs.py`

**Formula**: v = -g/f × (dη/dx)
- g = 9.81 m/s² (gravity)
- f = 2Ω sin(lat) (Coriolis parameter)
- dη/dx = DOT slope along gate

**Features**:
- Time series plot (cm/s)
- Monthly climatology bar chart
- Statistics (mean, std, max, min)
- Physical interpretation expander
- Works for both SLCCI and CMEMS data

---

### Legacy Tab Files (DEPRECATED)
**Status**: ⚠️ Deprecated - Use tabs.py instead

These files are no longer used:
- ~~`app/components/slcci_slope_tab.py`~~
- ~~`app/components/slcci_profile_tab.py`~~ 
- ~~`app/components/slcci_spatial_tab.py`~~

All functionality consolidated in `app/components/tabs.py`

---

## � DTUSpace v4 Service (NEW!)

### DTUService (`src/services/dtu_service.py`)
**Status**: ✅ Implemented | **Date**: 2026-01-03 | **Branch**: `feature/gates-streamlit`

Service for loading and processing DTUSpace v4 **gridded** DOT products.

**ISOLATED** from SLCCI/CMEMS - completely separate data handling.

```python
from src.services import DTUService, DTUConfig, DTUPassData

service = DTUService()
pass_data = service.load_gate_data(
    nc_path="/path/to/arctic_ocean_prod_DTUSpace_v4.0.nc",
    gate_path="/path/to/gate.shp",
    start_year=2006,
    end_year=2017,
    n_gate_pts=400
)
```

**Key Differences from SLCCI/CMEMS:**

| Aspect | SLCCI/CMEMS | DTUSpace |
|--------|-------------|----------|
| Data Type | Along-track | **Gridded** (lat × lon × time) |
| Pass/Track | Real satellite passes | **Synthetic** (from gate geometry) |
| API Access | CEDA/Copernicus | **None** (local files only) |
| Spatial Plot | Scatter points | **Heatmap** (pcolormesh style) |
| Color | 🟠 Orange / 🔵 Blue | 🟢 Green |

**DTUPassData Attributes:**
| Attribute | Type | Description |
|-----------|------|-------------|
| `strait_name` | str | Gate name (from filename) |
| `dataset_name` | str | "arctic ocean prod DTUSpace v4.0" |
| `slope_series` | np.ndarray | Monthly slopes (m/100km) |
| `time_array` | np.ndarray | Monthly dates |
| `profile_mean` | np.ndarray | Mean DOT along gate |
| `x_km` | np.ndarray | Distance along gate (km) |
| `dot_matrix` | np.ndarray | DOT [n_gate_pts × n_time] |
| `v_geostrophic_series` | np.ndarray | Pre-computed velocities |
| `dot_mean_grid` | xr.DataArray | Mean DOT grid for spatial map |
| `df` | DataFrame | Synthetic "observations" |

**Processing Flow:**
1. Load NetCDF → `xr.open_dataset()`
2. Load gate → shapefile to GeoDataFrame
3. Interpolate 400 points along gate line
4. KD-tree match → each gate point to nearest grid cell
5. Extract DOT along gate for all times
6. Compute slope series + geostrophic velocity
7. Build synthetic DataFrame for compatibility

**Files:**
- `src/services/dtu_service.py` (400+ lines)
- `app/state.py` - `store_dtu_data()`, `get_dtu_data()`
- `app/components/sidebar.py` - DTUSpace section
- `app/components/tabs.py` - `_render_dtu_*()` functions

---

## �🔀 Comparison Mode (NEW!)

### SLCCI vs CMEMS Overlay
**Status**: ✅ Implemented | **Date**: 2026-01-02 | **Branch**: `feature/gates-streamlit`

Compare SLCCI satellite altimetry with CMEMS L3 data on the same plots.

```python
# In Streamlit, load both datasets then enable comparison
# sidebar.py handles the toggle automatically

# Colors defined in tabs.py
COLOR_SLCCI = "darkorange"  # 🟠
COLOR_CMEMS = "steelblue"   # 🔵
```

**Workflow**:
1. Select SLCCI → Load SLCCI Data
2. Select CMEMS → Load CMEMS Data  
3. Both loaded? Checkbox "🔀 Comparison Mode" appears
4. Enable comparison → 5 overlay tabs appear

**Comparison Tabs** (in `tabs.py`):

| Tab | Function | Description |
|-----|----------|-------------|
| 1. Slope Timeline | `_render_slope_comparison()` | Both slopes on same plot |
| 2. DOT Profile | `_render_dot_profile_comparison()` | Both profiles overlaid |
| 3. Spatial Map | `_render_spatial_map_comparison()` | Points with different colors |
| 4. Geostrophic Velocity | `_render_geostrophic_comparison()` | v_geo + monthly climatology |
| 5. Export | `_render_export_tab()` | CSV downloads for both datasets |

**Session State Keys** (in `state.py`):
```python
st.session_state["dataset_slcci"]    # SLCCI PassData
st.session_state["dataset_cmems"]    # CMEMS PassData  
st.session_state["comparison_mode"]  # bool
```

**State Functions**:
- `store_slcci_data(pass_data)` - Store SLCCI separately
- `store_cmems_data(pass_data)` - Store CMEMS separately
- `get_slcci_data()` / `get_cmems_data()` - Retrieve
- `is_comparison_mode()` / `set_comparison_mode(bool)` - Toggle

**Pass Number Extraction** (from gate filename):
```python
# In cmems_service.py: _extract_pass_from_gate_name()
# Patterns detected:
"barents_sea_opening_S3_pass_481.shp"  → ("Barents Sea Opening", 481)
"denmark_strait_TPJ_pass_248.shp"      → ("Denmark Strait", 248)  
"gate_name_481.shp"                    → ("Gate Name", 481)
"fram_strait.shp"                      → ("Fram Strait", None)
```

**Files**:
- `app/components/tabs.py` - All comparison rendering (1367 lines)
- `app/state.py` - Session state management
- `app/components/sidebar.py` - Comparison toggle UI
- `src/services/cmems_service.py` - Pass extraction

**Test Script**: `scripts/test_comparison_mode.py`

---

## 📊 Data Processing & Visualization

### DOT Calculation (`legacy/j2_utils.py`)
**Status**: ✅ Migrated to SLCCIService | **Used in**: Streamlit

```python
# Now use SLCCIService instead:
from src.services import SLCCIService
pass_data = service.load_pass_data(gate_path, pass_number)
# DOT already computed in pass_data.df["dot"]
```

### Slope Analysis (`src/analysis/slope.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit

```python
from src.analysis.slope import bin_by_longitude, compute_slope

bin_centers, bin_means, bin_stds, bin_counts = bin_by_longitude(lon, dot, bin_size=0.01)
slope, intercept, r2, slope_err = compute_slope(bin_centers, bin_means)
```

### Visualization (`src/visualization/plotly_charts.py`)
**Status**: ✅ Implemented | **Used in**: Streamlit

- `create_slope_timeline_plot()` - DOT slope evolution
- Monthly 12-subplot analysis

---

## 🔍 Search & Discovery

### Investigation Agent (`src/agent/investigation_agent.py`)
**Status**: ✅ Implemented | **Used in**: API, React

Agente AI per investigare eventi climatici.

**Capabilities**:
- Event parsing (estrae location, date, type da testo)
- Auto geo-resolution via GeoResolver
- Multi-source data gathering
- Causal analysis coordination

### Knowledge Graph (`api/services/knowledge_service.py`)
**Status**: ✅ Implemented | **Used in**: React

Sistema di knowledge graph con SurrealDB.

---

## 🎨 React Components (Frontend)

### Available Components

| Component | Purpose | API Endpoints Used |
|-----------|---------|-------------------|
| `ChatPanel.tsx` | Chat interface for investigation | `/api/v1/agent/investigate` |
| `InvestigationBriefing.tsx` | Shows investigation summary | Investigation results |
| `KnowledgeGraphView.tsx` | 2D knowledge graph | `/api/v1/knowledge/graph` |
| `KnowledgeGraph3DView.tsx` | 3D knowledge graph (Cosmograph) | Same |
| `CausalGraphView.tsx` | Causal relationships view | `/api/v1/causal/graph` |
| `DataExplorer.tsx` | Browse datasets | `/api/v1/data/catalog` |
| `PCMCIPanel.tsx` | PCMCI causal analysis | `/api/v1/causal/pcmci` |
| `HistoricalAnalysis.tsx` | Historical event analysis | Various |

### Key Features in React NOT in Streamlit

1. **3D Knowledge Graph** - Cosmograph visualization
2. **Chat-based Investigation** - Natural language queries
3. **Real-time PCMCI** - Causal analysis UI
4. **Investigation Workflow** - Step-by-step guided analysis

---

## 📦 Data Services

### Intake Bridge (`src/data_manager/intake_bridge.py`)
**Status**: ✅ Implemented

```python
from src.data_manager.intake_bridge import get_catalog

catalog = get_catalog()
datasets = catalog.list_datasets()
data = catalog.load_dataset("slcci_altimetry")
```

### DataService (`src/services/data_service.py`)
**Status**: ✅ Implemented

Unified data loading with bbox/time filtering.

---

## 🔧 Infrastructure

### Logging (`src/core/logging_config.py`)
**Status**: ✅ Implemented

```python
from src.core.logging_config import setup_logging, get_logger

setup_logging(level="DEBUG", env="development")
logger = get_logger(__name__)
```

### Models (`src/core/models.py`)
**Status**: ✅ Implemented

Pydantic models: `BoundingBox`, `TimeRange`, `GateModel`, `DataRequest`, etc.

---

## 💾 Cache Service (NEW 2026-01-06)

### DataCache (`src/services/cache_service.py`)
**Status**: ✅ Recovered & Integrated | **Used in**: Streamlit sidebar

Persistent cache for processed PassData objects using pickle.

```python
from src.services.cache_service import DataCache

cache = DataCache()

# Save data
cache.save("slcci", "fram_strait", pass_data, pass_number=248)

# Load data (returns None if not cached)
pass_data = cache.load("slcci", "fram_strait", pass_number=248)

# Check if cached
if cache.exists("slcci", "fram_strait", pass_number=248):
    print("Data is cached!")

# List cached items
items = cache.list_cached("slcci")

# Clear cache
cache.clear("slcci", "fram_strait")  # Clear specific
cache.clear_all()  # Clear everything
```

**Features**:
- Pickle-based persistence
- Automatic index.json tracking
- Size and metadata tracking
- Dataset-specific subdirectories

**Cache Structure**:
```
data/cache/processed/
├── index.json
├── slcci/
│   └── fram_strait_pass_248.pkl
├── cmems_l4/
│   └── fram_strait.pkl
└── dtuspace/
    └── fram_strait.pkl
```

**Files**:
- `src/services/cache_service.py` (471 lines)

---

## 🌊 Bathymetry Service (NEW 2026-01-06)

### BathymetryService (`src/services/bathymetry_service.py`)
**Status**: ✅ Recovered & Integrated | **Used in**: Volume Transport Chart

GEBCO bathymetry extraction along gate lines.

```python
from src.services.bathymetry_service import BathymetryService

service = BathymetryService("data/bathymetry/gebco_2024.nc")
profile = service.extract_profile(gate_lon, gate_lat, x_km)

# Returns BathymetryProfile dataclass:
# - depth: np.ndarray (positive = below sea level)
# - lon, lat: np.ndarray
# - x_km: np.ndarray (distance along gate)
# - sill_depth: float (minimum depth)
# - mean_depth: float
# - max_depth: float
# - source: str = "GEBCO"
```

**Features**:
- Lazy loading (loads GEBCO on first use)
- Bounding box subsetting (reduces memory)
- RegularGridInterpolator for fast lookups
- Sill depth detection

**Requirements**:
- GEBCO NetCDF file (~11GB global, or regional subset)
- Download from: https://www.gebco.net/

**Files**:
- `src/services/bathymetry_service.py` (229 lines)

---

## 🚢 Transport Service (NEW 2026-01-06)

### VolumeTransport (`src/services/transport_service.py`)
**Status**: ✅ Recovered | **Used in**: Planned for volume transport tab

Volume transport calculation from geostrophic velocity.

```python
from src.services.transport_service import (
    compute_perpendicular_velocity,
    calculate_volume_transport,
    VolumeTransportResult
)

# Compute perpendicular velocity component
v_perp = compute_perpendicular_velocity(
    v_geo=geostrophic_velocity,
    gate_lon=lon_array,
    gate_lat=lat_array
)

# Calculate volume transport
result = calculate_volume_transport(
    v_perp=v_perp,
    depth_profile=bathymetry_depth,
    x_km=distance_km
)
# result.transport_sv: float (Sverdrup)
# result.transport_m3s: float (m³/s)
```

**Formula**: Q = ∫∫ v⊥ dA
- v⊥ = velocity perpendicular to gate
- dA = cross-sectional area element
- 1 Sv = 10⁶ m³/s

**Files**:
- `src/services/transport_service.py` (243 lines)

---

## 📥 Loaders Module (NEW 2026-01-06)

### Data Loaders (`app/components/loaders/`)
**Status**: ⚠️ Recovered but NOT YET USED | **Planned for**: Refactored sidebar

Modular data loading with longitude filtering support.

```python
from app.components.loaders import (
    load_slcci_data,
    load_dtu_data,
    load_cmems_l4_data,
    apply_longitude_filter
)

# Load with longitude filter (for divided gates)
result = load_slcci_data(
    gate_id="fram_strait_west",
    lon_filter_min=None,
    lon_filter_max=0.0,  # West of 0°
    cycles=range(1, 282),
)
```

**Files**:
- `app/components/loaders/__init__.py` (17 lines)
- `app/components/loaders/base.py` (220 lines) - `apply_longitude_filter()`
- `app/components/loaders/slcci_loader.py` (140 lines)
- `app/components/loaders/dtu_loader.py` (114 lines)
- `app/components/loaders/cmems_l4_loader.py` (112 lines)

**Note**: These loaders are implemented but sidebar.py still uses internal functions. Integration pending.

---

## 🎯 Migration Priority

### High Priority (Needed for Streamlit)
1. ~~GeoResolver integration~~ ✅ Done
2. ~~Gate selector~~ ✅ Done
3. DOT/Slope visualization (in `analysis_tab.py`)
4. Monthly analysis (in `monthly_tab.py`)

### Medium Priority
5. Knowledge graph integration
6. Investigation agent UI
7. PCMCI UI

### Low Priority (React-only for now)
8. 3D Cosmograph
9. Chat interface

---

## 📝 Notes

- React and Streamlit share the same API backend
- Services layer (`src/services/`) works with both
- Legacy code in `legacy/` needs gradual migration
- All new features should be in services layer first
