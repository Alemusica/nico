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

## 📊 Data Processing & Visualization

### DOT Calculation (`legacy/j2_utils.py`)
**Status**: 🔄 In Legacy | **Needs**: Migration to services

```python
def calculate_dot(ds_cycle):
    """DOT = corssh - geoid"""
    return ds_cycle["corssh"] - ds_cycle["geoid"]
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
