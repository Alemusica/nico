# 📊 Export Service - Function Audit & Documentation

> **Last Updated:** 2026-01-26
> **File:** `src/services/export_service.py`
> **Purpose:** Generate CSV files and PNG images (300 DPI) for data export

---

## 🎯 Quick Reference - Export Functions

### DOT & Slope Analysis

| Function | Description | Output Folder | Emoji |
|----------|-------------|---------------|-------|
| `export_slope_timeline()` | DOT slope (mm/km) nel tempo | `dot_analysis/` | 📈 |
| `export_dot_profile_along_gate()` | Mean DOT profile con ±std e regression | `dot_analysis/` | 📊 |
| `export_monthly_dot_profiles_grid()` | 3×4 monthly DOT con slope (mm/km) e R² | `monthly_analysis/` | 📊 |

### Velocity Analysis

| Function | Description | Output Folder | Emoji |
|----------|-------------|---------------|-------|
| `export_monthly_velocity_profiles_grid()` | 3×4 monthly velocity con slope e R² | `velocity/` | 📊 |
| `export_velocity_comparison_timeseries()` | v_perp vs v_geo nello stesso plot | `velocity/` | 📈 |

### Volume Transport

| Function | Description | Output Folder | Emoji |
|----------|-------------|---------------|-------|
| `export_total_transport_timeseries()` | Transport time series colorato (north/south) | `volume_transport/` | 📈 |
| `export_volume_transport_statistics()` | Monthly boxplot statistics | `volume_transport/` | 📊 |
| `export_volume_transport_timeseries()` | Basic transport timeseries | `volume_transport/` | 📈 |

### Geography & Bathymetry

| Function | Description | Output Folder | Emoji |
|----------|-------------|---------------|-------|
| `export_spatial_map()` | Mappa con coastlines, confini, griglia lat/lon | `spatial/` | 🗺️ |
| `export_bathymetry_profile_clean()` | Batimetria senza marrone, zero in alto | `bathymetry/` | 🏔️ |

### Salt Flux & Properties

| Function | Description | Output Folder | Emoji |
|----------|-------------|---------------|-------|
| `export_salinity_density_along_gate()` | Salinity & Density profiles | `salt_flux/` | 🌡️ |
| `export_salt_flux_timeseries()` | Salt flux time series | `salt_flux/` | 📈 |

---

## 📁 ZIP Structure Generated

```
export_{gate_name}_{start_year}-{end_year}_{date}/
├── csv/
│   ├── {gate}_volume_transport_raw.csv
│   ├── {gate}_volume_transport_climatology.csv
│   ├── {gate}_volume_transport_annual.csv
│   └── {gate}_salt_flux_raw.csv
├── dot_analysis/
│   ├── {gate}_slope_timeline.png          📈 Slope nel tempo
│   └── {gate}_dot_profile.png             📊 DOT profile mean±std
├── monthly_analysis/
│   └── {gate}_monthly_dot_grid.png        📊 3×4 DOT con R²
├── velocity/
│   ├── {gate}_monthly_velocity_grid.png   📊 3×4 velocity con R²
│   └── {gate}_velocity_comparison.png     📈 v_perp vs v_geo
├── volume_transport/
│   ├── {gate}_total_transport_timeseries.png  📈 Colorato
│   └── {gate}_monthly_statistics.png      📊 Boxplot
├── spatial/
│   └── {gate}_geographic_map.png          🗺️ Con coastlines
├── bathymetry/
│   └── {gate}_bathymetry.png              🏔️ Zero in alto
└── salt_flux/
    └── {gate}_salinity_density.png        🌡️ Due pannelli
```

---

## 🔧 Function Signatures

### Main Export Functions

```python
def generate_full_export(
    gate_data: Dict[str, Any],
    include_images: bool = True,
    include_csv: bool = True,
    dpi: int = 300,
    export_options: Dict[str, bool] = None
) -> bytes:
    """
    Generate complete export ZIP.
    
    export_options keys:
        - slope_timeline
        - dot_profile
        - spatial_map
        - monthly_dot
        - monthly_velocity
        - velocity_comparison
        - total_transport
        - bathymetry
        - salinity_density
        - volume_transport_stats
    """
```

### gate_data Required Keys

```python
gate_data = {
    'gate_name': str,           # Required
    'time_array': np.ndarray,   # Required
    'dataset': str,             # Default: 'cmems_l4'
    'v_perp': np.ndarray,       # Shape (n_pts, n_time)
    'v_geo': np.ndarray,        # Optional, same shape
    'x_km': np.ndarray,         # Distance along gate
    'dot_matrix': np.ndarray,   # DOT data (n_pts, n_time)
    'transport_sv': np.ndarray, # Volume transport (Sv)
    'depth_profile': np.ndarray,
    'gate_lon': np.ndarray,
    'gate_lat': np.ndarray,
    'salinity_profile': np.ndarray,  # Optional
    'density_profile': np.ndarray,   # Optional
    'slope_values': np.ndarray,      # DOT slope per timestep
}
```

---

## 📈 Individual Export Functions

### export_slope_timeline()
```python
def export_slope_timeline(
    slope_values: np.ndarray,  # m/km values
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📈 Slope Timeline - Pendenza DOT nel tempo (mm/km).
    Shows 30-day rolling mean and overall mean.
    """
```

### export_dot_profile_along_gate()
```python
def export_dot_profile_along_gate(
    dot_matrix: np.ndarray,    # (n_pts, n_time)
    x_km: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    start_year: int = None,
    end_year: int = None,
    dpi: int = 300
) -> bytes:
    """
    📊 DOT Profile - Mean ± std with linear regression.
    Slope shown in mm/km, R² annotated.
    """
```

### export_spatial_map()
```python
def export_spatial_map(
    gate_lon: np.ndarray,
    gate_lat: np.ndarray,
    gate_name: str,
    dpi: int = 300
) -> bytes:
    """
    🗺️ Spatial Map - Cartopy map with:
    - Coastlines, borders, rivers, lakes
    - Lat/lon grid with labels
    - Start/end markers with coordinates
    """
```

### export_monthly_dot_profiles_grid()
```python
def export_monthly_dot_profiles_grid(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 3×4 Monthly DOT Grid.
    Each panel shows: mean ± std, regression line,
    slope (mm/km), R² in annotation box.
    """
```

### export_monthly_velocity_profiles_grid()
```python
def export_monthly_velocity_profiles_grid(
    v_perp: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 3×4 Monthly Velocity Grid.
    Colors match Streamlit (blue=north, red=south).
    Slope and R² annotated.
    """
```

### export_velocity_comparison_timeseries()
```python
def export_velocity_comparison_timeseries(
    v_perp: np.ndarray,        # Full matrix (n_pts, n_time)
    v_geo: np.ndarray,         # Can be None
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📈 v_perp vs v_geo comparison.
    Both velocities in same plot with 30-day rolling means.
    """
```

### export_total_transport_timeseries()
```python
def export_total_transport_timeseries(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📈 Total Transport - Colored by direction.
    Blue=northward, red=southward (matching Streamlit).
    30-day rolling mean, statistics annotated.
    """
```

### export_bathymetry_profile_clean()
```python
def export_bathymetry_profile_clean(
    depth_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    gate_lon: np.ndarray = None,
    gate_lat: np.ndarray = None,
    dpi: int = 300
) -> bytes:
    """
    🏔️ Bathymetry Profile.
    - NO brown fill
    - Zero (sea level) at TOP
    - Navy line for seafloor
    - Light blue fill for water
    - Coordinates annotated
    """
```

### export_salinity_density_along_gate()
```python
def export_salinity_density_along_gate(
    salinity_profile: np.ndarray,
    density_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    dpi: int = 300
) -> bytes:
    """
    🌡️ Salinity & Density - Two panels.
    Mean lines annotated with values.
    """
```

---

## 🗂️ CSV Export Functions

| Function | Description | Columns |
|----------|-------------|---------|
| `generate_volume_transport_raw_csv()` | Raw monthly data | gate_name, year, month, mean/std/min/max Sv |
| `generate_volume_transport_climatology_csv()` | Monthly climatology | month, mean/std/median Sv |
| `generate_volume_transport_annual_csv()` | Annual stats | year, mean/std/trend Sv |
| `generate_salt_flux_raw_csv()` | Salt flux time series | time, salt_flux_kg_s |

---

## ⚠️ REMOVED Functions

The following function was **REMOVED** as per user request:

- ~~`export_velocity_hovmoller()`~~ - Hovmöller diagram removed

---

## 🔄 Git History

| Commit | Date | Description |
|--------|------|-------------|
| e1cbf20 | 2026-01-26 | Added 7 new export functions |
| (current) | 2026-01-26 | Rewrote export functions per user specs |

---

## 💡 Usage Example

```python
from src.services.export_service import generate_full_export

# Prepare gate data
gate_data = {
    'gate_name': 'Fram Strait',
    'dataset': 'cmems_l4',
    'time_array': time_array,
    'v_perp': v_perp,
    'v_geo': v_geo,
    'x_km': x_km,
    'dot_matrix': dot_matrix,
    'transport_sv': transport_sv,
    'depth_profile': depth_profile,
    'gate_lon': gate_lon,
    'gate_lat': gate_lat,
}

# Export options (all True by default)
export_options = {
    'slope_timeline': True,
    'dot_profile': True,
    'spatial_map': True,
    'monthly_dot': True,
    'monthly_velocity': True,
    'velocity_comparison': True,
    'total_transport': True,
    'bathymetry': True,
    'salinity_density': False,  # Skip if no data
    'volume_transport_stats': True,
}

# Generate ZIP
zip_bytes = generate_full_export(
    gate_data,
    include_images=True,
    include_csv=True,
    dpi=300,
    export_options=export_options
)

# Save or return
with open('export.zip', 'wb') as f:
    f.write(zip_bytes)
```

---

## 🎨 Color Scheme (matching Streamlit)

| Element | Color | Hex |
|---------|-------|-----|
| Northward flow | Blue | `#1f77b4` |
| Southward flow | Red | `#d62728` |
| DOT profile | Dark Blue | `darkblue` |
| Regression line | Red dashed | `r--` |
| Seafloor | Navy | `navy` |
| Sea surface | Steel Blue | `steelblue` |
| Salinity | Green | `g-` |
| Density | Purple | `purple` |

---

## 📝 Notes for Future Agents

1. **All images are 300 DPI** - This is hardcoded as per user requirement
2. **Slope units are mm/km** - Convert from m/km by multiplying by 1000
3. **Bathymetry has zero at TOP** - Y-axis is inverted, no brown fill
4. **v_perp and v_geo in same plot** - Both velocities shown together
5. **Colors match Streamlit** - Blue=north, Red=south for transport
6. **No Hovmöller** - This was explicitly removed

---

*Document generated by Copilot Agent - 2026-01-26*
