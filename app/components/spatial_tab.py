"""
Spatial View Tab
================
Interactive map visualization with gate overlay.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .sidebar import AppConfig


def render_spatial_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render spatial visualization tab."""
    
    if not datasets:
        st.warning("⚠️ No data loaded. Use the sidebar to load NetCDF files.")
        return
    
    st.subheader("🗺️ Spatial Distribution")
    
    # Controls in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Cycle selector
        cycle_options = [f"Cycle {c['cycle']}" for c in cycle_info]
        selected_idx = st.selectbox(
            "Select Cycle",
            range(len(cycle_options)),
            format_func=lambda x: cycle_options[x],
            key="spatial_cycle_select"
        )
    
    ds = datasets[selected_idx]
    
    with col2:
        # Variable selector
        var_options = ["DOT (SSH - MSS)", "corssh (SSH)"]
        if "mean_sea_surface" in ds.data_vars:
            var_options.append("mean_sea_surface")
        if "bathymetry" in ds.data_vars:
            var_options.append("bathymetry")
        if "swh" in ds.data_vars:
            var_options.append("swh (Wave Height)")
        
        var_name = st.selectbox("Variable", var_options, key="spatial_var_select")
    
    with col3:
        # Sample size
        sample_pct = st.slider("Sample %", 1, 100, 10, key="spatial_sample")
    
    # Get data
    try:
        if "DOT" in var_name:
            if config.mss_var in ds.data_vars:
                values = ds["corssh"].values.flatten() - ds[config.mss_var].values.flatten()
                colorbar_title = "DOT (m)"
            else:
                st.error(f"MSS variable '{config.mss_var}' not found")
                return
        elif "corssh" in var_name:
            values = ds["corssh"].values.flatten()
            colorbar_title = "SSH (m)"
        elif "swh" in var_name:
            values = ds["swh"].values.flatten()
            colorbar_title = "SWH (m)"
        else:
            var_key = var_name.split()[0] if " " in var_name else var_name
            values = ds[var_key].values.flatten()
            colorbar_title = var_name
        
        lat = ds["latitude"].values.flatten()
        lon = ds["longitude"].values.flatten()
        
    except Exception as e:
        st.error(f"Error reading data: {e}")
        return
    
    # Sample for performance
    n_points = len(lat)
    n_sample = max(1000, int(n_points * sample_pct / 100))
    
    if n_sample < n_points:
        np.random.seed(42)
        idx = np.random.choice(n_points, n_sample, replace=False)
        lat = lat[idx]
        lon = lon[idx]
        values = values[idx]
    
    # Remove NaN
    mask = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    
    if mask.sum() == 0:
        st.warning("No valid data points after filtering")
        return
    
    df_map = pd.DataFrame({
        "lat": lat[mask],
        "lon": lon[mask],
        "value": values[mask],
    })
    
    # Show gate info if selected
    if config.selected_gate and config.gate_geometry is not None:
        st.info(f"🎯 Gate selected: **{config.selected_gate}** (buffer: {config.gate_buffer_km} km)")
    
    # Create map
    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="value",
        color_continuous_scale="RdYlBu_r",
        zoom=2,
        height=550,
        labels={"value": colorbar_title},
    )
    
    # Add gate geometry if available
    if config.gate_geometry is not None:
        try:
            gdf = config.gate_geometry
            # Get geometry bounds for gate line
            for _, row in gdf.iterrows():
                geom = row.geometry
                if geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    fig.add_trace(go.Scattermapbox(
                        lon=lons,
                        lat=lats,
                        mode='lines',
                        line=dict(width=4, color='red'),
                        name=f'Gate: {config.selected_gate}',
                        showlegend=True,
                    ))
                elif geom.geom_type == 'Point':
                    fig.add_trace(go.Scattermapbox(
                        lon=[geom.x],
                        lat=[geom.y],
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name=f'Gate: {config.selected_gate}',
                    ))
        except Exception as e:
            st.warning(f"Could not plot gate: {e}")
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        title=f"Cycle {cycle_info[selected_idx]['cycle']} - {colorbar_title}",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats row
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📍 Points", f"{len(df_map):,}")
    with col2:
        st.metric("📉 Min", f"{df_map['value'].min():.4f}")
    with col3:
        st.metric("📈 Max", f"{df_map['value'].max():.4f}")
    with col4:
        st.metric("📊 Mean", f"{df_map['value'].mean():.4f}")
