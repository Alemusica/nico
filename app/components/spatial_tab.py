"""
Spatial View Tab
================
Interactive map visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from .sidebar import AppConfig


def render_spatial_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render spatial visualization tab."""
    
    st.subheader("🗺️ Spatial Distribution")
    
    # Cycle selector
    cycle_options = [
        f"Cycle {c['cycle']} - {c['filename']}" 
        for c in cycle_info
    ]
    
    selected_idx = st.selectbox(
        "Select Cycle",
        range(len(cycle_options)),
        format_func=lambda x: cycle_options[x],
    )
    
    ds = datasets[selected_idx]
    
    # Variable selector
    var_options = ["DOT (computed)", "corssh"]
    if "mean_sea_surface" in ds.data_vars:
        var_options.append("mean_sea_surface")
    if "bathymetry" in ds.data_vars:
        var_options.append("bathymetry")
    
    var_name = st.selectbox("Variable", var_options)
    
    # Get data
    if var_name == "DOT (computed)":
        if config.mss_var in ds.data_vars:
            values = ds["corssh"].values.flatten() - ds[config.mss_var].values.flatten()
            colorbar_title = "DOT (m)"
        else:
            values = ds["corssh"].values.flatten()
            colorbar_title = "SSH (m)"
    else:
        values = ds[var_name].values.flatten()
        colorbar_title = f"{var_name}"
    
    lat = ds["latitude"].values.flatten()
    lon = ds["longitude"].values.flatten()
    
    # Sample for performance
    n_points = len(lat)
    max_points = 50000
    
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        lat = lat[idx]
        lon = lon[idx]
        values = values[idx]
        st.info(f"Showing {max_points:,} of {n_points:,} points for performance")
    
    # Remove NaN
    mask = ~np.isnan(values) & ~np.isnan(lat) & ~np.isnan(lon)
    
    df_map = pd.DataFrame({
        "lat": lat[mask],
        "lon": lon[mask],
        "value": values[mask],
    })
    
    # Create map
    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        color="value",
        color_continuous_scale="RdYlBu_r",
        zoom=3,
        height=600,
        title=f"Spatial Distribution - {var_name}",
        labels={"value": colorbar_title},
    )
    
    fig.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Points Displayed", f"{len(df_map):,}")
    with col2:
        st.metric("Value Range", f"{df_map['value'].min():.3f} to {df_map['value'].max():.3f}")
    with col3:
        st.metric("Mean Value", f"{df_map['value'].mean():.4f}")
