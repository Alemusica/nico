"""
Data Explorer Tab
=================
Raw data inspection and statistics.
"""

import streamlit as st
import numpy as np
import pandas as pd

from .sidebar import AppConfig


def render_explorer_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render data explorer tab."""
    
    st.subheader("📊 Data Explorer")
    
    ds = datasets[0]
    available_vars = list(ds.data_vars)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Info**")
        st.write(f"- Files loaded: {len(datasets)}")
        st.write(f"- Variables: {', '.join(available_vars)}")
        
        # Coordinates
        if "latitude" in ds.coords or "latitude" in ds.data_vars:
            lat = ds["latitude"].values.flatten()
            lon = ds["longitude"].values.flatten()
            st.write(f"- Latitude range: {np.nanmin(lat):.2f}° to {np.nanmax(lat):.2f}°")
            st.write(f"- Longitude range: {np.nanmin(lon):.2f}° to {np.nanmax(lon):.2f}°")
    
    with col2:
        st.markdown("**Quick Statistics**")
        
        var_for_stats = st.selectbox("Variable for Stats", available_vars, key="stats_var")
        
        data = ds[var_for_stats].values.flatten()
        valid = data[np.isfinite(data)]
        
        if len(valid) > 0:
            st.write(f"- Count: {len(valid):,}")
            st.write(f"- Mean: {np.mean(valid):.6f}")
            st.write(f"- Std: {np.std(valid):.6f}")
            st.write(f"- Min: {np.min(valid):.6f}")
            st.write(f"- Max: {np.max(valid):.6f}")
        else:
            st.write("No valid data")
    
    # Raw data preview
    st.subheader("📋 Raw Data Preview")
    
    preview_var = st.selectbox("Select Variable", available_vars, key="preview_var")
    
    preview_data = ds[preview_var].values.flatten()[:1000]
    lat_preview = ds["latitude"].values.flatten()[:1000]
    lon_preview = ds["longitude"].values.flatten()[:1000]
    
    df_preview = pd.DataFrame({
        preview_var: preview_data,
        "latitude": lat_preview,
        "longitude": lon_preview,
    })
    
    st.dataframe(df_preview.head(100), use_container_width=True)
    
    # Variable attributes
    st.subheader("📝 Variable Attributes")
    
    attr_var = st.selectbox("Select Variable", available_vars, key="attr_var")
    
    attrs = ds[attr_var].attrs
    if attrs:
        for key, val in attrs.items():
            st.write(f"- **{key}**: {val}")
    else:
        st.write("No attributes available")
