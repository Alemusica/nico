# 🏗️ Architecture

## Overview

The SLCCI Altimetry project follows a **modular, layered architecture** designed for:
- **Separation of concerns** - Each module has a single responsibility
- **Scalability** - Easy to add new analyses or visualizations
- **Testability** - Pure functions can be unit tested
- **Reusability** - Core modules can be used in notebooks, scripts, or apps

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              streamlit_app.py                        │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ Sidebar │ │  Tabs   │ │ Styles  │ │  State  │   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  ┌───────────────────┐  ┌───────────────────────────────┐  │
│  │  app/components/  │  │         app/                   │  │
│  │  ├─ analysis_tab  │  │  ├─ main.py (orchestration)   │  │
│  │  ├─ profiles_tab  │  │  ├─ state.py (session)        │  │
│  │  ├─ monthly_tab   │  │  └─ styles.py (CSS)           │  │
│  │  ├─ spatial_tab   │  │                               │  │
│  │  └─ explorer_tab  │  │                               │  │
│  └───────────────────┘  └───────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     BUSINESS LAYER                           │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  │
│  │ src/analysis/  │  │ src/visualiz./ │  │  src/data/   │  │
│  │ ├─ dot.py      │  │ ├─ plotly_*   │  │ ├─ loaders   │  │
│  │ ├─ slope.py    │  │ └─ mpl_*      │  │ ├─ geoid     │  │
│  │ └─ statistics  │  │               │  │ └─ filters   │  │
│  └────────────────┘  └────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       CORE LAYER                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │                    src/core/                        │    │
│  │   ├─ satellite.py   (mission detection)            │    │
│  │   ├─ coordinates.py (geo utilities)                │    │
│  │   └─ helpers.py     (general utilities)            │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              NetCDF Files (*.nc)                    │    │
│  │   ├─ SLCCI_ALTDB_J1_Cycle*.nc                      │    │
│  │   ├─ SLCCI_ALTDB_J2_Cycle*.nc                      │    │
│  │   └─ TUM_ogmoc.nc (geoid)                          │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### Core Layer (`src/core/`)

**Purpose**: Low-level utilities with no external dependencies.

| Module | Responsibility |
|--------|---------------|
| `satellite.py` | Detect J1/J2 from path, mission parameters |
| `coordinates.py` | Longitude wrapping, spatial masks |
| `helpers.py` | File parsing, string utilities |

### Data Layer (`src/data/`)

**Purpose**: Data loading, filtering, and preprocessing.

| Module | Responsibility |
|--------|---------------|
| `loaders.py` | Load NetCDF files, combine cycles |
| `geoid.py` | Geoid interpolation |
| `filters.py` | Quality flags, spatial/temporal filters |

### Analysis Layer (`src/analysis/`)

**Purpose**: Scientific computations.

| Module | Responsibility |
|--------|---------------|
| `dot.py` | DOT computation (SSH - reference) |
| `slope.py` | Binning, linear regression, slope conversion |
| `statistics.py` | Descriptive stats, monthly aggregation |

### Visualization Layer (`src/visualization/`)

**Purpose**: Create plots and figures.

| Module | Responsibility |
|--------|---------------|
| `plotly_charts.py` | Interactive charts for Streamlit |
| `matplotlib_charts.py` | Publication-quality static figures |

### Application Layer (`app/`)

**Purpose**: Streamlit-specific UI components.

| Module | Responsibility |
|--------|---------------|
| `main.py` | App entry point, orchestration |
| `state.py` | Session state management |
| `styles.py` | Custom CSS |
| `components/*.py` | Individual UI tabs/widgets |

## Design Principles

### 1. Single Responsibility
Each module does ONE thing well:
```python
# Good: dot.py only handles DOT computation
def compute_dot(ds, ssh_var, reference_var): ...

# Bad: mixing concerns
def compute_dot_and_plot_and_save(): ...
```

### 2. Dependency Injection
Pass dependencies rather than importing globally:
```python
# Good: explicit dependencies
def analyze_cycle(ds, config: AppConfig): ...

# Bad: hidden dependencies
def analyze_cycle():
    config = get_global_config()  # Hidden!
```

### 3. Pure Functions
Prefer pure functions without side effects:
```python
# Good: pure function
def bin_by_longitude(lon, values, bin_size) -> tuple:
    return bin_centers, bin_means, ...

# Bad: modifies global state
def bin_by_longitude(lon, values):
    global LAST_BIN_RESULT  # Side effect!
    LAST_BIN_RESULT = ...
```

### 4. Data Classes for Configuration
```python
@dataclass
class AppConfig:
    mss_var: str = "mean_sea_surface"
    bin_size: float = 0.01
    lat_range: tuple | None = None
```

## Data Flow

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
