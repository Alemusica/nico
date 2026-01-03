# 🚀 Enhancement Plan: Advanced Visualization & Export

> **Date**: 2026-01-02
> **Branch**: `feature/gates-streamlit`
> **Priority**: 🟢 (After current comparison mode is tested)

---

## 🎯 Overview

This document outlines the plan for implementing advanced visualizations and export features.

---

## 📊 5.1 UI Improvements

### 5.1.1 Progress Bar During CMEMS Loading
**Problem**: Loading 7000+ CMEMS files takes ~2 minutes with no feedback.

**Solution**:
```python
# In cmems_service.py, emit progress events
# In sidebar.py, show st.progress()

with st.spinner("Loading CMEMS data..."):
    progress_bar = st.progress(0)
    for i, result in enumerate(service.load_pass_data_with_progress(gate_path)):
        progress_bar.progress(i / total_files)
```

**Files to modify**:
- `src/services/cmems_service.py` - Add generator version
- `app/components/sidebar.py` - Add progress bar

### 5.1.2 Cache Loaded Data
**Problem**: Re-loading same gate data is slow.

**Solution**: Use `st.cache_data` decorator
```python
@st.cache_data(ttl=3600)  # 1 hour cache
def load_slcci_cached(gate_path: str, pass_number: int):
    return service.load_pass_data(gate_path, pass_number)
```

### 5.1.3 Date Filter for Comparison
**Problem**: SLCCI and CMEMS may have different time ranges.

**Solution**: Add date range picker in sidebar
```python
date_range = st.date_input(
    "Date Range", 
    value=(start_date, end_date),
    min_value=date(2002, 1, 1),
    max_value=date(2024, 12, 31)
)
```

---

## 📈 5.2 New Visualizations

### 5.2.1 Correlation Plot (SLCCI vs CMEMS)
**Purpose**: Show correlation between SLCCI and CMEMS slope values.

**Implementation**:
```python
def _render_correlation_plot(slcci_data, cmems_data, config):
    """Scatter plot of SLCCI slope vs CMEMS slope for same dates."""
    
    # Align time series to common dates
    common_dates = align_time_series(slcci_data.time_array, cmems_data.time_array)
    
    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=slcci_slopes_aligned,
        y=cmems_slopes_aligned,
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.6)
    ))
    
    # Add 1:1 line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='1:1 Line'
    ))
    
    # Compute R² and add annotation
    r_squared = np.corrcoef(slcci_aligned, cmems_aligned)[0,1]**2
    fig.add_annotation(text=f"R² = {r_squared:.3f}", ...)
```

**Tab**: Add as Tab 6 in comparison mode

### 5.2.2 Difference Plot (SLCCI - CMEMS)
**Purpose**: Show systematic differences between datasets.

**Implementation**:
```python
def _render_difference_plot(slcci_data, cmems_data, config):
    """Time series of SLCCI - CMEMS difference."""
    
    # Compute difference for aligned dates
    diff = slcci_slopes - cmems_slopes
    
    # Plot difference
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=common_dates,
        y=diff,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(100,100,100,0.2)'
    ))
    
    # Add mean difference line
    mean_diff = np.mean(diff)
    fig.add_hline(y=mean_diff, line_dash="dash", ...)
    
    # Statistics
    st.metric("Mean Bias", f"{mean_diff:.4f}")
    st.metric("RMSE", f"{rmse:.4f}")
```

### 5.2.3 DOT Scatter Comparison
**Purpose**: Compare DOT values point-by-point spatially.

**Implementation**:
```python
def _render_dot_scatter(slcci_data, cmems_data, config):
    """Scatter plot of DOT values at matching locations."""
    
    # Spatial matching within tolerance
    matched_points = spatial_match(slcci_df, cmems_df, tolerance_km=10)
    
    # Scatter plot
    fig = px.scatter(
        matched_points,
        x='dot_slcci',
        y='dot_cmems',
        color='distance_km',
        title='DOT Comparison at Matched Locations'
    )
```

---

## 📥 5.3 Advanced Export

### 5.3.1 Export NetCDF
**Purpose**: Export processed data in NetCDF format for further analysis.

**Implementation**:
```python
def export_to_netcdf(pass_data, filename: str):
    """Export PassData to NetCDF format."""
    import xarray as xr
    
    ds = xr.Dataset({
        'slope': (['time'], pass_data.slope_series),
        'profile_mean': (['x'], pass_data.profile_mean),
        'dot_matrix': (['x', 'time'], pass_data.dot_matrix),
    }, coords={
        'time': pass_data.time_array,
        'x_km': pass_data.x_km,
    }, attrs={
        'strait_name': pass_data.strait_name,
        'pass_number': pass_data.pass_number,
    })
    
    ds.to_netcdf(filename)
```

### 5.3.2 Export Multiple Plots as ZIP
**Purpose**: Batch export all plots at once.

**Implementation**:
```python
import zipfile
from io import BytesIO

def export_plots_as_zip(figures: List[go.Figure], names: List[str]) -> BytesIO:
    """Create ZIP file with all plots as PNG."""
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        for fig, name in zip(figures, names):
            img_bytes = fig.to_image(format='png', scale=2)
            zf.writestr(f"{name}.png", img_bytes)
    
    zip_buffer.seek(0)
    return zip_buffer

# In Export tab
st.download_button(
    label="📦 Download All Plots (ZIP)",
    data=export_plots_as_zip(figures, names),
    file_name="comparison_plots.zip",
    mime="application/zip"
)
```

### 5.3.3 PDF Report with 3D Earth
**Purpose**: Generate professional PDF report with all visualizations.

**Structure**:
```
📄 Comparison Report
├── Cover Page
│   └── 3D Earth Globe (rotated to show gate location)
├── Executive Summary
│   └── Key metrics, date range, data sources
├── Slope Timeline Comparison
│   └── Plot + statistics table
├── DOT Profile Comparison
│   └── Plot + profile statistics
├── Geostrophic Velocity
│   └── Time series + monthly climatology
├── Correlation Analysis
│   └── Scatter plot + R² + RMSE
├── Data Tables
│   └── Monthly summary tables
└── Appendix
    └── Data sources, methodology, references
```

**Implementation**:
```python
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table
import plotly.io as pio

def generate_pdf_report(slcci_data, cmems_data, config, output_path: str):
    """Generate professional PDF report."""
    
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    story = []
    
    # 1. Cover page with 3D Earth
    earth_img = generate_3d_earth(
        center_lat=pass_data.mean_latitude,
        center_lon=pass_data.mean_longitude,
        gate_coords=pass_data.gate_lon_pts, pass_data.gate_lat_pts
    )
    story.append(Image(earth_img, width=400, height=400))
    
    # 2. Executive summary
    story.append(Paragraph("Executive Summary", styles['Heading1']))
    summary_table = [
        ["Metric", "SLCCI", "CMEMS"],
        ["Time Range", f"{slcci_range}", f"{cmems_range}"],
        ["Mean Slope", f"{slcci_mean:.4f}", f"{cmems_mean:.4f}"],
        ["Correlation", f"{r_squared:.3f}", ""],
    ]
    story.append(Table(summary_table))
    
    # 3. Plots (convert Plotly to image)
    for fig, title in plots:
        img_bytes = pio.to_image(fig, format='png', scale=2)
        story.append(Image(BytesIO(img_bytes), width=500))
    
    doc.build(story)
```

**3D Earth Globe**:
```python
import plotly.graph_objects as go
import numpy as np

def generate_3d_earth(center_lat, center_lon, gate_coords):
    """Generate 3D rotating Earth with gate highlighted."""
    
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    # Earth texture (simplified with continents)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=earth_texture,
        colorscale='Earth',
        showscale=False
    ))
    
    # Add gate as red line on sphere
    gate_x, gate_y, gate_z = latlon_to_xyz(gate_coords)
    fig.add_trace(go.Scatter3d(
        x=gate_x, y=gate_y, z=gate_z,
        mode='lines',
        line=dict(color='red', width=5),
        name='Gate'
    ))
    
    # Camera position to focus on gate
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=eye_x, y=eye_y, z=eye_z)
        )
    )
    
    return fig.to_image(format='png', scale=3)
```

---

## 📋 Implementation Priority

| Feature | Priority | Effort | Dependencies |
|---------|----------|--------|--------------|
| Progress bar | 🔴 High | Low | None |
| Data caching | 🔴 High | Low | None |
| Correlation plot | 🟠 Medium | Medium | Time alignment |
| Difference plot | 🟠 Medium | Medium | Time alignment |
| Date filter | 🟡 Low | Low | None |
| NetCDF export | 🟡 Low | Medium | xarray |
| ZIP export | 🟡 Low | Medium | kaleido |
| PDF report | 🟢 Nice-to-have | High | reportlab, kaleido |
| 3D Earth | 🟢 Nice-to-have | High | plotly, earth texture |

---

## 📦 Required Packages

```bash
# For advanced export
pip install kaleido  # Plotly static export
pip install reportlab  # PDF generation
pip install xarray netcdf4  # NetCDF export (probably already installed)
```

---

## 🎬 Next Steps

1. **Implement Progress Bar** (quick win, improves UX significantly)
2. **Add Data Caching** (reduces load time for repeated analysis)
3. **Add Correlation Plot** (most requested visualization)
4. **Implement NetCDF Export** (useful for researchers)
5. **PDF Report** (final deliverable)

---

*Document created: 2026-01-02*
*Author: AI Agent*
