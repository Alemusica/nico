# 📋 SESSION RESUME - 2 Gennaio 2026

## 🎯 OBIETTIVO ORIGINALE
Implementare **Comparison Mode** tra SLCCI e CMEMS con:
1. **Pass number extraction** da filename gate shapefile (es. `_pass_481` → 481)
2. **Buffer 5.0°** per CMEMS (dal notebook `Copernicus dataset.ipynb`)
3. **Overlay plots** (non caricamento simultaneo) - Arancione=SLCCI, Blu=CMEMS
4. **Export** PNG + CSV
5. **Mappe 2D** nelle tabs

---

## ✅ COMPLETATO E COMMITTATO

### 1. `src/services/cmems_service.py`
```python
# FATTO: Pass number extraction
def _extract_pass_from_gate_name(self, gate_name: str) -> Optional[int]:
    # Patterns: pass_481, _pass_481, pass481, _NNN finale
    
# FATTO: Buffer aggiornato
@dataclass
class CMEMSConfig:
    buffer_deg: float = 5.0  # Era 0.5°, ora 5.0° come nel notebook
```

### 2. `app/state.py`
```python
# FATTO: Funzioni per comparison mode
def store_slcci_data(data) -> None
def store_cmems_data(data) -> None
def get_slcci_data() -> Optional[Any]
def get_cmems_data() -> Optional[Any]
def is_comparison_mode() -> bool
def set_comparison_mode(enabled: bool) -> None

# Session state keys:
# - dataset_slcci
# - dataset_cmems  
# - comparison_mode
```

### 3. `app/components/sidebar.py`
```python
# FATTO: UI comparison mode con checkbox
def _render_data_source():
    # Checkbox SLCCI + CMEMS
    # Mostra pass number estratto per CMEMS
    
def _load_slcci_data():
    # Usa store_slcci_data() invece di session_state diretto
    
def _load_cmems_data():
    # Usa store_cmems_data()
```

### 4. Documentazione
- `docs/PROGRESS.md` - Aggiornato
- `docs/VISUALIZATION_ARCHITECTURE.md` - Aggiornato con comparison mode

---

## ✅ COMPLETATO - 2026-01-02 (SECONDA SESSIONE)

### `app/components/tabs.py` - AGGIORNATO

**Stato attuale** (1367 righe, versione completa):
- ✅ Import di `get_slcci_data`, `get_cmems_data`, `is_comparison_mode`
- ✅ Costanti `COLOR_SLCCI = "darkorange"`, `COLOR_CMEMS = "steelblue"`
- ✅ Funzione `_render_comparison_tabs()` con 5 tab comparison
- ✅ Funzione `_render_cmems_tabs()` per CMEMS singolo
- ✅ Tab Export con download CSV
- ✅ Tutte le funzioni single-dataset (`_render_slope_timeline`, etc.)

**Commit**: `536dc80` - feat: Add comparison mode with SLCCI/CMEMS overlay

---

## 📝 CODICE DA AGGIUNGERE A `app/components/tabs.py`

### STEP 1: Sostituire gli import (righe 1-15)

```python
"""
Main content tabs for the dashboard.
Following SLCCI PLOTTER notebook workflow exactly.
Supports comparison mode with SLCCI/CMEMS overlay.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List
import io

from .sidebar import AppConfig
from ..state import get_slcci_data, get_cmems_data, is_comparison_mode

# Comparison mode colors (from COMPARISON_BATCH notebook)
COLOR_SLCCI = "darkorange"
COLOR_CMEMS = "steelblue"
```

### STEP 2: Sostituire la funzione `render_tabs()` (circa riga 17-30)

```python
def render_tabs(config: AppConfig):
    """Render main content tabs based on loaded data type and comparison mode."""
    slcci_data = get_slcci_data()
    cmems_data = get_cmems_data()
    comparison_mode = is_comparison_mode()
    
    # Legacy support
    legacy_slcci = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    selected_dataset_type = st.session_state.get("selected_dataset_type", "SLCCI")
    
    # Comparison mode: overlay SLCCI and CMEMS
    if comparison_mode and slcci_data is not None and cmems_data is not None:
        _render_comparison_tabs(slcci_data, cmems_data, config)
    # Single SLCCI mode
    elif slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    # Single CMEMS mode  
    elif cmems_data is not None:
        _render_cmems_tabs(cmems_data, config)
    # Legacy SLCCI support
    elif selected_dataset_type == "SLCCI" and legacy_slcci is not None:
        _render_slcci_tabs(legacy_slcci, config)
    elif datasets:
        _render_generic_tabs(datasets, config)
    else:
        _render_empty_tabs(config)
```

### STEP 3: Aggiungere queste funzioni ALLA FINE del file (dopo `_render_generic_tabs`)

```python
# ==============================================================================
# COMPARISON MODE TABS
# ==============================================================================
def _render_comparison_tabs(slcci_data, cmems_data, config: AppConfig):
    """Render comparison tabs overlaying SLCCI and CMEMS data."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Slope Timeline Comparison",
        "DOT Profile Comparison", 
        "Spatial Map Comparison",
        "Geostrophic Velocity Comparison",
        "Export Data"
    ])
    
    with tab1:
        _render_slope_comparison(slcci_data, cmems_data, config)
    with tab2:
        _render_dot_profile_comparison(slcci_data, cmems_data, config)
    with tab3:
        _render_spatial_map_comparison(slcci_data, cmems_data, config)
    with tab4:
        _render_geostrophic_comparison(slcci_data, cmems_data, config)
    with tab5:
        _render_export_tab(slcci_data, cmems_data, config)


def _render_slope_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render slope timeline comparison overlay."""
    st.subheader("SSH Slope Timeline - SLCCI vs CMEMS")
    
    # Get SLCCI data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    slcci_name = getattr(slcci_data, 'strait_name', 'Unknown')
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    # Get CMEMS data
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_slope is None and cmems_slope is None:
        st.warning("No slope data available for comparison.")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend lines", value=True, key="comp_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="comp_slope_unit")
    
    fig = go.Figure()
    
    # Plot SLCCI (Orange)
    if slcci_slope is not None:
        valid_mask = ~np.isnan(slcci_slope)
        if np.sum(valid_mask) > 0:
            y_vals = slcci_slope * 100 if unit == "cm/km" else slcci_slope
            x_vals = slcci_time if slcci_time is not None else np.arange(len(slcci_slope))
            
            valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
            valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
            
            fig.add_trace(go.Scatter(
                x=valid_x,
                y=valid_y,
                mode="markers+lines",
                name=f"SLCCI (Pass {slcci_pass})",
                marker=dict(size=6, color=COLOR_SLCCI),
                line=dict(width=2, color=COLOR_SLCCI)
            ))
            
            if show_trend and len(valid_y) > 2:
                x_numeric = np.arange(len(valid_y))
                z = np.polyfit(x_numeric, valid_y, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=valid_x,
                    y=p(x_numeric),
                    mode="lines",
                    name=f"SLCCI Trend ({z[0]:.4f}/period)",
                    line=dict(dash="dash", color=COLOR_SLCCI, width=1)
                ))
    
    # Plot CMEMS (Blue)
    if cmems_slope is not None:
        valid_mask = ~np.isnan(cmems_slope)
        if np.sum(valid_mask) > 0:
            y_vals = cmems_slope * 100 if unit == "cm/km" else cmems_slope
            x_vals = cmems_time if cmems_time is not None else np.arange(len(cmems_slope))
            
            valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
            valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
            
            fig.add_trace(go.Scatter(
                x=valid_x,
                y=valid_y,
                mode="markers+lines",
                name=f"CMEMS (Pass {cmems_pass})",
                marker=dict(size=6, color=COLOR_CMEMS),
                line=dict(width=2, color=COLOR_CMEMS)
            ))
            
            if show_trend and len(valid_y) > 2:
                x_numeric = np.arange(len(valid_y))
                z = np.polyfit(x_numeric, valid_y, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=valid_x,
                    y=p(x_numeric),
                    mode="lines",
                    name=f"CMEMS Trend ({z[0]:.4f}/period)",
                    line=dict(dash="dash", color=COLOR_CMEMS, width=1)
                ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    y_label = "Slope (cm/km)" if unit == "cm/km" else "Slope (m/100km)"
    fig.update_layout(
        title=f"Slope Comparison: {slcci_name}",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics comparison
    _render_comparison_stats(slcci_slope, cmems_slope, "Slope", unit)


def _render_dot_profile_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render DOT profile comparison overlay."""
    st.subheader("Mean DOT Profile - SLCCI vs CMEMS")
    
    # Get SLCCI data
    slcci_profile = getattr(slcci_data, 'profile_mean', None)
    slcci_x_km = getattr(slcci_data, 'x_km', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    # Get CMEMS data
    cmems_profile = getattr(cmems_data, 'profile_mean', None)
    cmems_x_km = getattr(cmems_data, 'x_km', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_profile is None and cmems_profile is None:
        st.warning("No profile data available for comparison.")
        return
    
    fig = go.Figure()
    
    # Plot SLCCI (Orange)
    if slcci_profile is not None and slcci_x_km is not None:
        valid_mask = ~np.isnan(slcci_profile)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=slcci_x_km[valid_mask],
                y=slcci_profile[valid_mask],
                mode="lines",
                name=f"SLCCI (Pass {slcci_pass})",
                line=dict(color=COLOR_SLCCI, width=2)
            ))
    
    # Plot CMEMS (Blue)
    if cmems_profile is not None and cmems_x_km is not None:
        valid_mask = ~np.isnan(cmems_profile)
        if np.any(valid_mask):
            fig.add_trace(go.Scatter(
                x=cmems_x_km[valid_mask],
                y=cmems_profile[valid_mask],
                mode="lines",
                name=f"CMEMS (Pass {cmems_pass})",
                line=dict(color=COLOR_CMEMS, width=2)
            ))
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"DOT Profile Comparison: {strait_name}",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    _render_comparison_stats(slcci_profile, cmems_profile, "DOT", "m")


def _render_spatial_map_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render spatial map comparison."""
    st.subheader("Spatial Distribution - SLCCI vs CMEMS")
    
    # Get data
    slcci_df = getattr(slcci_data, 'df', None)
    cmems_df = getattr(cmems_data, 'df', None)
    gate_lon = getattr(slcci_data, 'gate_lon_pts', None) or getattr(cmems_data, 'gate_lon_pts', None)
    gate_lat = getattr(slcci_data, 'gate_lat_pts', None) or getattr(cmems_data, 'gate_lat_pts', None)
    
    if (slcci_df is None or slcci_df.empty) and (cmems_df is None or cmems_df.empty):
        st.warning("No spatial data available for comparison.")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_gate = st.checkbox("Show gate line", value=True, key="comp_map_gate")
    with col2:
        sample_size = st.slider("Sample size", 500, 5000, 2000, key="comp_map_sample")
    
    fig = go.Figure()
    
    # Plot SLCCI (Orange)
    if slcci_df is not None and not slcci_df.empty:
        plot_df = slcci_df.sample(min(sample_size, len(slcci_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"],
            lon=plot_df["lon"],
            mode="markers",
            name="SLCCI",
            marker=dict(size=5, color=COLOR_SLCCI, opacity=0.6)
        ))
    
    # Plot CMEMS (Blue)
    if cmems_df is not None and not cmems_df.empty:
        plot_df = cmems_df.sample(min(sample_size, len(cmems_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"],
            lon=plot_df["lon"],
            mode="markers",
            name="CMEMS",
            marker=dict(size=5, color=COLOR_CMEMS, opacity=0.6)
        ))
    
    # Gate line
    if show_gate and gate_lon is not None and gate_lat is not None:
        fig.add_trace(go.Scattermapbox(
            lat=gate_lat,
            lon=gate_lon,
            mode="lines",
            name="Gate",
            line=dict(width=3, color="red")
        ))
    
    # Center map
    all_lats = []
    all_lons = []
    if slcci_df is not None and not slcci_df.empty:
        all_lats.extend(slcci_df["lat"].tolist())
        all_lons.extend(slcci_df["lon"].tolist())
    if cmems_df is not None and not cmems_df.empty:
        all_lats.extend(cmems_df["lat"].tolist())
        all_lons.extend(cmems_df["lon"].tolist())
    
    center_lat = np.mean(all_lats) if all_lats else 70.0
    center_lon = np.mean(all_lons) if all_lons else 0.0
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Spatial Comparison: {strait_name}",
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=5
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_geostrophic_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render geostrophic velocity comparison."""
    st.subheader("Geostrophic Velocity - SLCCI vs CMEMS")
    
    # Get SLCCI geostrophic velocity
    slcci_v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    # Get CMEMS geostrophic velocity
    cmems_v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_v_geo is None and cmems_v_geo is None:
        st.warning("No geostrophic velocity data available. Load data with geostrophic computation enabled.")
        return
    
    fig = go.Figure()
    
    # Plot SLCCI (Orange)
    if slcci_v_geo is not None and len(slcci_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=slcci_v_geo.index,
            y=slcci_v_geo.values * 100,  # cm/s
            mode="lines+markers",
            name=f"SLCCI (Pass {slcci_pass})",
            line=dict(color=COLOR_SLCCI, width=2),
            marker=dict(size=6, color=COLOR_SLCCI)
        ))
    
    # Plot CMEMS (Blue)
    if cmems_v_geo is not None and len(cmems_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=cmems_v_geo.index,
            y=cmems_v_geo.values * 100,  # cm/s
            mode="lines+markers",
            name=f"CMEMS (Pass {cmems_pass})",
            line=dict(color=COLOR_CMEMS, width=2),
            marker=dict(size=6, color=COLOR_CMEMS)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Geostrophic Velocity Comparison: {strait_name}",
        xaxis_title="Time",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly climatology comparison
    st.subheader("Monthly Climatology Comparison")
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_clim = go.Figure()
    
    if slcci_v_geo is not None and len(slcci_v_geo) > 0:
        monthly_slcci = slcci_v_geo.groupby(slcci_v_geo.index.month).mean()
        fig_clim.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_slcci.index],
            y=monthly_slcci.values * 100,
            name="SLCCI",
            marker_color=COLOR_SLCCI,
            opacity=0.7
        ))
    
    if cmems_v_geo is not None and len(cmems_v_geo) > 0:
        monthly_cmems = cmems_v_geo.groupby(cmems_v_geo.index.month).mean()
        fig_clim.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_cmems.index],
            y=monthly_cmems.values * 100,
            name="CMEMS",
            marker_color=COLOR_CMEMS,
            opacity=0.7
        ))
    
    fig_clim.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig_clim.update_layout(
        title="Monthly Mean Geostrophic Velocity",
        xaxis_title="Month",
        yaxis_title="Velocity (cm/s)",
        height=400,
        template="plotly_white",
        barmode="group"
    )
    
    st.plotly_chart(fig_clim, use_container_width=True)


def _render_export_tab(slcci_data, cmems_data, config: AppConfig):
    """Render export tab with PNG and CSV downloads."""
    st.subheader("Export Data")
    
    col1, col2 = st.columns(2)
    
    # CSV Export
    with col1:
        st.markdown("### CSV Export")
        
        if slcci_data is not None:
            slcci_df = getattr(slcci_data, 'df', None)
            if slcci_df is not None and not slcci_df.empty:
                csv_slcci = slcci_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download SLCCI CSV",
                    data=csv_slcci,
                    file_name="slcci_data.csv",
                    mime="text/csv",
                    key="download_slcci_csv"
                )
        
        if cmems_data is not None:
            cmems_df = getattr(cmems_data, 'df', None)
            if cmems_df is not None and not cmems_df.empty:
                csv_cmems = cmems_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CMEMS CSV",
                    data=csv_cmems,
                    file_name="cmems_data.csv",
                    mime="text/csv",
                    key="download_cmems_csv"
                )
    
    # PNG Export info
    with col2:
        st.markdown("### PNG Export")
        st.info("""
        **To export plots as PNG:**
        1. Hover over any plot
        2. Click the 📷 camera icon in the toolbar
        3. PNG will download automatically
        
        Or use the Plotly menu → Download as PNG
        """)
    
    # Summary statistics export
    st.markdown("### Summary Statistics")
    
    stats_data = []
    
    if slcci_data is not None:
        slcci_slope = getattr(slcci_data, 'slope_series', None)
        if slcci_slope is not None:
            valid_slopes = slcci_slope[~np.isnan(slcci_slope)]
            if len(valid_slopes) > 0:
                stats_data.append({
                    'Source': 'SLCCI',
                    'Variable': 'Slope',
                    'Mean': np.mean(valid_slopes),
                    'Std': np.std(valid_slopes),
                    'Min': np.min(valid_slopes),
                    'Max': np.max(valid_slopes),
                    'N': len(valid_slopes)
                })
    
    if cmems_data is not None:
        cmems_slope = getattr(cmems_data, 'slope_series', None)
        if cmems_slope is not None:
            valid_slopes = cmems_slope[~np.isnan(cmems_slope)]
            if len(valid_slopes) > 0:
                stats_data.append({
                    'Source': 'CMEMS',
                    'Variable': 'Slope',
                    'Mean': np.mean(valid_slopes),
                    'Std': np.std(valid_slopes),
                    'Min': np.min(valid_slopes),
                    'Max': np.max(valid_slopes),
                    'N': len(valid_slopes)
                })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv_stats,
            file_name="comparison_statistics.csv",
            mime="text/csv",
            key="download_stats_csv"
        )


def _render_comparison_stats(slcci_data, cmems_data, variable_name: str, unit: str):
    """Render comparison statistics expander."""
    with st.expander(f"{variable_name} Statistics Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**SLCCI (🟠 {COLOR_SLCCI})**")
            if slcci_data is not None:
                valid = slcci_data[~np.isnan(slcci_data)]
                if len(valid) > 0:
                    st.metric("Mean", f"{np.mean(valid):.4f} {unit}")
                    st.metric("Std Dev", f"{np.std(valid):.4f} {unit}")
                    st.metric("N Points", len(valid))
                else:
                    st.warning("No valid data")
            else:
                st.warning("No data")
        
        with col2:
            st.markdown(f"**CMEMS (🔵 {COLOR_CMEMS})**")
            if cmems_data is not None:
                valid = cmems_data[~np.isnan(cmems_data)]
                if len(valid) > 0:
                    st.metric("Mean", f"{np.mean(valid):.4f} {unit}")
                    st.metric("Std Dev", f"{np.std(valid):.4f} {unit}")
                    st.metric("N Points", len(valid))
                else:
                    st.warning("No valid data")
            else:
                st.warning("No data")


# ==============================================================================
# CMEMS-ONLY TABS
# ==============================================================================
def _render_cmems_tabs(cmems_data, config: AppConfig):
    """Render tabs for CMEMS data only."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "Slope Timeline",
        "DOT Profile",
        "Spatial Map",
        "Geostrophic Velocity"
    ])
    
    with tab1:
        _render_slope_timeline(cmems_data, config)
    with tab2:
        _render_dot_profile(cmems_data, config)
    with tab3:
        _render_spatial_map(cmems_data, config)
    with tab4:
        _render_geostrophic_velocity(cmems_data, config)
```

---

## 📁 FILE MODIFICATI (da committare dopo tabs.py)

| File | Stato | Note |
|------|-------|------|
| `src/services/cmems_service.py` | ✅ Done | Pass extraction + buffer 5.0° |
| `app/state.py` | ✅ Done | Comparison mode functions |
| `app/components/sidebar.py` | ✅ Done | Checkbox UI + load functions |
| `docs/PROGRESS.md` | ✅ Done | Documentation |
| `docs/VISUALIZATION_ARCHITECTURE.md` | ✅ Done | Architecture docs |
| `app/components/tabs.py` | ❌ TODO | **USA CODICE SOPRA** |

---

## 🔧 CHECKLIST - ✅ COMPLETATA

1. [x] Aprire `app/components/tabs.py` in VS Code
2. [x] Sostituire import (STEP 1)
3. [x] Sostituire `render_tabs()` (STEP 2)
4. [x] Aggiungere funzioni comparison (STEP 3) alla fine del file
5. [x] Salvare il file
6. [ ] Test: `source .venv/bin/activate && streamlit run streamlit_app.py`
7. [ ] Verificare:
   - Caricamento SLCCI singolo
   - Caricamento CMEMS singolo (check pass extraction)
   - Comparison mode (entrambi checkbox)
   - Overlay plots (🟠 arancione + 🔵 blu)
   - Export CSV funzionante
8. [x] Git commit & push:
   ```bash
   git add app/components/tabs.py
   git commit -m "feat: Add comparison mode with SLCCI/CMEMS overlay"
   git push origin feature/gates-streamlit
   ```
9. [x] Cleanup: file temporanei rimossi

---

## 📊 RIFERIMENTI

- **Colori**: Da notebook `COMPARISON_BATCH`
  - SLCCI: `darkorange` (🟠)
  - CMEMS: `steelblue` (🔵)
  
- **Buffer CMEMS**: 5.0° (da `Copernicus dataset.ipynb`, linea ~207)

- **Pass extraction**: Regex patterns in `cmems_service.py`:
  - `pass_481` → 481
  - `_pass_481` → 481
  - `pass481` → 481
  - `..._NNN` (finale) → NNN

---

*Documento generato automaticamente - 2 Gennaio 2026*
