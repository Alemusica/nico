# 🛰️ SLCCI Satellite Altimetry Analysis

A modular Python toolkit for analyzing satellite altimetry data from the **Sea Level CCI (SLCCI)** project, specifically Jason-1 and Jason-2 missions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Overview

This project provides tools for:
- **DOT Analysis** - Dynamic Ocean Topography computation (SSH - MSS/Geoid)
- **Slope Timeline** - Monthly DOT slope evolution with error bars
- **Monthly Analysis** - Seasonal patterns in 12-subplot format
- **Spatial Visualization** - Interactive maps with Plotly
- **Strait Analysis** - Gate-based analysis for ocean straits

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd nico

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## 📁 Project Structure

```
nico/
├── streamlit_app.py          # 🚀 Main entry point
├── app/                      # 📱 Streamlit application
│   ├── main.py              # App orchestration
│   ├── state.py             # Session state management
│   ├── styles.py            # Custom CSS
│   └── components/          # UI components
│       ├── sidebar.py       # Data loading & config
│       ├── tabs.py          # Tab container
│       ├── analysis_tab.py  # Slope timeline
│       ├── profiles_tab.py  # DOT profiles
│       ├── monthly_tab.py   # Monthly analysis
│       ├── spatial_tab.py   # Map visualization
│       └── explorer_tab.py  # Data explorer
├── src/                      # 📚 Core library
│   ├── core/                # Base utilities
│   │   ├── satellite.py     # Satellite detection
│   │   ├── coordinates.py   # Geo utilities
│   │   └── helpers.py       # General helpers
│   ├── data/                # Data handling
│   │   ├── loaders.py       # NetCDF loading
│   │   ├── geoid.py         # Geoid interpolation
│   │   └── filters.py       # Data filtering
│   ├── analysis/            # Scientific analysis
│   │   ├── dot.py           # DOT computation
│   │   ├── slope.py         # Slope analysis
│   │   └── statistics.py    # Statistical functions
│   └── visualization/       # Plotting
│       ├── plotly_charts.py # Interactive plots
│       └── matplotlib_charts.py  # Static plots
├── data/                     # 📊 Data files (see data/README.md)
│   ├── slcci/               # SLCCI NetCDF cycles
│   └── geoid/               # Geoid reference files
├── gates/                    # 🗺️ Strait gate shapefiles
├── notebooks/                # 📓 Jupyter notebooks
├── legacy/                   # 📜 Legacy code (j2_utils.py)
└── docs/                     # 📖 Documentation
    ├── ARCHITECTURE.md
    ├── CONTRIBUTING.md
    ├── CHANGELOG.md
    └── CMEMS-SL-PUM-*.pdf   # CMEMS reference docs
```

## 📊 Data Format

The toolkit works with **SLCCI Altimeter Database V2.0** NetCDF files:

```
SLCCI_ALTDB_J1_CycleXXX_V2.nc  # Jason-1
SLCCI_ALTDB_J2_CycleXXX_V2.nc  # Jason-2
```

### Key Variables:
| Variable | Description | Units |
|----------|-------------|-------|
| `corssh` | Corrected Sea Surface Height | m |
| `mean_sea_surface` | Mean Sea Surface | m |
| `latitude` | Latitude | degrees |
| `longitude` | Longitude | degrees |
| `TimeDay` | Days since 2000-01-01 | days |
| `validation_flag` | Quality flag (0=valid) | - |

## 🔬 Scientific Methods

### DOT Computation
```
DOT = SSH - Reference Surface
```
Where Reference Surface is either Mean Sea Surface (MSS) or Geoid.

### Slope Analysis
1. **Longitude Binning** - Data binned by 0.01° longitude
2. **Linear Regression** - `scipy.stats.linregress`
3. **Unit Conversion** - m/deg → mm/m (latitude corrected)

```python
slope_mm_per_m = (slope_m_per_deg / meters_per_deg) * 1000
meters_per_deg = 111320 * cos(latitude)
```

## 🖥️ Dashboard Features

### 📈 Slope Timeline
- Error bars from regression standard error
- Trend line with rate
- Mean ± std reference line

### 🌊 DOT Profiles
- Multi-cycle comparison
- Longitude-binned profiles
- Interactive selection

### 📅 Monthly Analysis
- 12-subplot grid
- Linear fit per month
- R² and slope statistics

### 🗺️ Spatial View
- Interactive Mapbox maps
- Variable selection
- Dynamic sampling for performance

## 🧪 Usage Examples

### Programmatic Usage

```python
from src.data.loaders import load_filtered_cycles
from src.analysis.dot import compute_dot
from src.analysis.slope import compute_slope_timeline

# Load data
ds = load_filtered_cycles(
    cycles=range(1, 100),
    base_dir="/path/to/data",
    lat_range=(60, 80),
)

# Compute DOT
dot = compute_dot(ds, reference_var="mean_sea_surface")

# Slope analysis
timeline = compute_slope_timeline(df, bin_size=0.01)
```

## 📦 Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
xarray>=2023.1.0
netCDF4>=1.6.0
scipy>=1.10.0
plotly>=5.14.0
streamlit>=1.28.0
geopandas>=0.14.0
cartopy>=0.22.0
```

## 🤝 Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See LICENSE file.

## 🙏 Acknowledgments

- ESA Climate Change Initiative - Sea Level CCI
- CNES/NASA Jason-1 and Jason-2 missions
- TUM for geoid data (TUM_ogmoc)

---

**Built with ❤️ for ocean science**
