# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2024-12-23

### Added
- 🔮 **Historical Episode Analysis**
  - Frontend component for analyzing well-documented Arctic events
  - 4 historical episodes: Arctic Ice 2007, Atlantic Intrusion 2015, Fram Export 2012, Marine Heatwave 2018
  - Precursor signal detection with physics validation
  - Up to 147 days advance warning capability
  - Cross-episode pattern discovery

- 🔌 **New API Endpoints**
  - `GET /historical/episodes` - List all historical episodes
  - `GET /historical/episodes/{id}` - Get episode details
  - `POST /historical/analyze/{id}` - Run precursor analysis
  - `GET /historical/cross-patterns` - Cross-episode patterns
  - Enhanced `/health` endpoint with component status

- 🛡️ **System Robustness**
  - All core packages now installed: tigramite, neo4j, surrealdb
  - Graceful fallback when services unavailable
  - Health endpoint shows status of all components
  - SYSTEM_STATUS.md documentation

### Fixed
- Duplicate import in HistoricalAnalysis.tsx causing white page
- Tigramite now properly installed and available

## [1.3.0] - 2024-12-23

### Added
- 🐳 **Docker Deployment Stack**
  - docker-compose.yml with Neo4j + SurrealDB + API + Frontend
  - Dockerfile.api for FastAPI backend
  - frontend/Dockerfile for React + nginx

- 🌐 **Cross-Region Experiments**
  - Teleconnection discovery between non-overlapping regions
  - Norwegian Sea → Fram Strait propagation (63-77 day lag)
  - Physics validation against ocean current speeds
  - Experience engine for pattern learning

- 🤖 **Agent Layer for Multi-Layer Causal Discovery**
  - agent_layer.py with operators, infrastructure, physical processes
  - Knowledge service extensions
  - Neo4j agent methods
  - React frontend base components

## [1.2.0] - 2024-12-22

### Added
- 🗺️ **Ocean Gate Selection System**
  - Interactive gate selector in sidebar
  - 8 pre-configured Arctic gates (Fram, Bering, Davis, Denmark, Nares, etc.)
  - Visual gate info cards with region/description
  - Buffer slider for gate area selection
  - Gate geometry overlay on spatial map
  - Shapefile loading with SHAPE_RESTORE_SHX support

- ⚡ **Performance Optimizations**
  - Data sampling slider (1-100%, default 10%)
  - Progress bar during cycle analysis
  - Minimum 1000 points guarantee for statistics
  - Fixed seed for reproducibility

- 🐛 **Bug Fixes**
  - Fixed datetime64 handling in analysis_tab.py and monthly_tab.py
  - Fixed relative imports in main.py for Streamlit compatibility
  - Fixed path resolution for data directory
  - Added pandas import for Timestamp conversion

- 🧪 **Test Suite**
  - `tests/test_loaders.py` - 25+ data loading tests
  - `tests/test_analysis.py` - 21 analysis function tests
  - `tests/conftest.py` - Pytest configuration and fixtures
  - `tests/test_resolver.py` - VariableResolver tests

- 🎨 **UI Improvements**
  - Spatial View with cycle/variable/sample controls in columns
  - Gate info display on spatial map
  - Better metric formatting in stats rows
  - Error handling with user-friendly messages

### Changed
- Spatial tab completely refactored for better UX
- AppConfig extended with gate_geometry, sample_fraction, gate_buffer_km
- Sidebar now shows gate selection at top priority

## [1.1.0] - 2024-12-22

### Added
- 🔧 **Dataset Configuration System** (`src/core/config.py`, `src/core/resolver.py`)
  - Centralized variable mapping across different data formats
  - Support for SLCCI (J1/J2), CMEMS (L3/L4), AVISO
  - Auto-detection of dataset format
  - Canonical variable names (ssh, mss, dot, etc.)
  - `VariableResolver` class for unified data access

- 📁 **Project Reorganization**
  - `gates/` - Strait gate shapefiles
  - `data/slcci/` - SLCCI NetCDF cycles
  - `data/geoid/` - Geoid reference files
  - `notebooks/` - Jupyter notebooks
  - `legacy/` - Original j2_utils.py

- 📖 **New Documentation**
  - `DATASET_CONFIG.md` - Variable mapping system guide
  - README files for each directory

### Changed
- Files reorganized from root into proper directories
- Updated main README with new structure

## [1.0.0] - 2024-12-22

### Added
- 🏗️ **Modular Architecture**
  - Split monolithic code into layered modules
  - Core utilities (`src/core/`)
  - Data handling (`src/data/`)
  - Analysis functions (`src/analysis/`)
  - Visualization (`src/visualization/`)
  - Streamlit app (`app/`)

- 📊 **Analysis Features**
  - DOT computation (SSH - MSS/Geoid)
  - Slope timeline with error bars
  - Monthly analysis (12 subplots)
  - Longitude binning
  - Statistics computation

- 🎨 **Visualization**
  - Interactive Plotly charts
  - Publication-quality Matplotlib figures
  - Mapbox spatial visualization

- 📱 **Streamlit Dashboard**
  - Drag & drop file upload
  - 5 analysis tabs
  - Configurable parameters
  - Session state management

- 📖 **Documentation**
  - README with quick start
  - Architecture documentation
  - Contributing guidelines
  - Changelog

### Changed
- Refactored `j2_utils.py` into modular structure
- Improved error handling throughout

### Technical Details
- Python 3.10+ required
- Type hints throughout
- Dataclasses for structured data
- Consistent docstring format

## [0.1.0] - 2024-12-21 (Initial)

### Added
- Initial Streamlit app (single file)
- Basic NetCDF loading
- DOT visualization

---

## Planned Features

### [1.1.0] (Future)
- [ ] Multi-strait comparison
- [ ] Export analysis results
- [ ] Batch processing CLI
- [ ] Unit tests

### [1.2.0] (Future)
- [ ] Machine learning trend detection
- [ ] Anomaly detection
- [ ] API endpoints
