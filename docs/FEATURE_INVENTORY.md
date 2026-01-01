# 📚 Feature Inventory - Cross-Branch Documentation

> **Purpose**: Documentare tutte le feature implementate nei vari branch/layer per evitare duplicazioni e perdita di informazioni.

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

# Access results
print(f"Observations: {len(pass_data.df)}")
print(f"Slope series: {pass_data.slope_series}")
print(f"DOT profile: {pass_data.profile_mean}")
```

**Features**:
- Load SLCCI NetCDF files (SLCCI_ALTDB_J2_CycleXXX_V2.nc)
- Geoid interpolation using TUM_ogmoc.nc
- DOT calculation: corssh - geoid
- Pass filtering (auto-detect or manual)
- Slope computation along gate (m/100km)
- DOT matrix building for temporal analysis

**Files**:
- `src/services/slcci_service.py` (600+ lines)
- Uses: `legacy/j2_utils.py` patterns (migrated)

---

### SLCCI Visualization Tabs
**Status**: ✅ Implemented | **Used in**: Streamlit

Three separate tabs for SLCCI analysis:

1. **Slope Timeline** (`app/components/slcci_slope_tab.py`)
   - Interactive Plotly timeline
   - Trend line overlay
   - Statistics summary
   - CSV download

2. **DOT Profile** (`app/components/slcci_profile_tab.py`)
   - Profile across gate (distance in km)
   - West/East labels
   - Linear fit overlay
   - Temporal variation explorer

3. **Spatial Map** (`app/components/slcci_spatial_tab.py`)
   - Interactive MapBox map
   - DOT color-coded points
   - Gate geometry overlay
   - Multiple basemap styles

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
