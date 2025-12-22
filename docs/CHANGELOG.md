# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
