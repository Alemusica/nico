"""
SLCCI Spatial Map Tab
=====================
Geographic map showing DOT values and satellite ground tracks.

This tab shows:
- Map with DOT values color-coded
- Gate geometry overlay
- Coastlines and geographic features
- Interactive zoom and pan

Uses Plotly Scattermapbox with OpenStreetMap tiles for best visibility.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

from src.services.slcci_service import PassData


def render_slcci_spatial_map_tab(pass_data: Optional[PassData] = None):
    """
    Render the spatial map tab for SLCCI data.
    
    Parameters
    ----------
    pass_data : PassData, optional
        Pre-loaded pass data from SLCCIService. If None, shows instructions.
    """
    st.subheader("🗺️ Spatial DOT Map")
    
    # Check if data is loaded
    if pass_data is None:
        pass_data = st.session_state.get("slcci_pass_data")
    
    if pass_data is None:
        st.info("👆 Select a gate and load data from the sidebar to see the spatial map.")
        _render_map_explainer()
        return
    
    # Header info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🛰️ Satellite", pass_data.satellite)
    with col2:
        st.metric("🎯 Pass", pass_data.pass_number)
    with col3:
        st.metric("📊 Points", f"{len(pass_data.df):,}")
    with col4:
        st.metric("🔄 Cycles", pass_data.df["cycle"].nunique())
    
    st.divider()
    
    # Map options in 3 columns
    col1, col2, col3 = st.columns(3)
    with col1:
        color_var = st.selectbox(
            "🎨 Color by",
            ["Mean DOT", "DOT Std Dev", "Observation Count"],
            index=0,
            key="slcci_map_color_var"
        )
    with col2:
        map_style = st.selectbox(
            "🗺️ Map Style",
            ["carto-positron", "carto-darkmatter", "open-street-map", "stamen-terrain", "stamen-watercolor"],
            index=0,
            key="slcci_map_style"
        )
    with col3:
        marker_size = st.slider("📍 Marker Size", 5, 20, 10, key="slcci_marker_size")
    
    # === MAIN MAP ===
    fig = _create_mapbox_map(pass_data, color_var, map_style, marker_size)
    st.plotly_chart(fig, use_container_width=True, key="slcci_spatial_map")
    
    # === DATA SUMMARY TABLE ===
    _render_spatial_summary(pass_data)


def _create_mapbox_map(
    pass_data: PassData,
    color_var: str = "Mean DOT",
    map_style: str = "carto-positron",
    marker_size: int = 10,
) -> go.Figure:
    """
    Create interactive map with Mapbox tiles (no API key needed for open styles).
    
    Features:
    - Clear visible coastlines from tile layer
    - High-quality map tiles for Arctic regions
    - Interactive zoom, pan, hover
    - Gate line overlay
    """
    
    df = pass_data.df.copy()
    
    # Aggregate by location (mean per lat/lon cell)
    # Round to 2 decimal places for grouping
    df["lat_bin"] = df["lat"].round(2)
    df["lon_bin"] = df["lon"].round(2)
    
    agg_df = df.groupby(["lat_bin", "lon_bin"]).agg({
        "dot": ["mean", "std", "count"],
        "corssh": "mean",
        "geoid": "mean",
    }).reset_index()
    
    agg_df.columns = ["lat", "lon", "dot_mean", "dot_std", "obs_count", "corssh_mean", "geoid_mean"]
    
    # Choose color variable and scale
    if color_var == "Mean DOT":
        color_col = "dot_mean"
        color_label = "DOT (m)"
        colorscale = "Viridis"
    elif color_var == "DOT Std Dev":
        color_col = "dot_std"
        color_label = "DOT Std (m)"
        colorscale = "Plasma"
    else:  # Observation Count
        color_col = "obs_count"
        color_label = "Count"
        colorscale = "Blues"
    
    # Create figure
    fig = go.Figure()
    
    # Add satellite data points with Scattermapbox
    fig.add_trace(go.Scattermapbox(
        lat=agg_df["lat"],
        lon=agg_df["lon"],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=agg_df[color_col],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=color_label, font=dict(size=12)),
                thickness=15,
                len=0.7,
                x=1.02,
            ),
            opacity=0.85,
        ),
        text=[
            f"<b>DOT:</b> {row['dot_mean']:.4f} m<br>"
            f"<b>Std:</b> {row['dot_std']:.4f} m<br>"
            f"<b>Count:</b> {row['obs_count']}<br>"
            f"<b>Lat:</b> {row['lat']:.3f}°<br>"
            f"<b>Lon:</b> {row['lon']:.3f}°"
            for _, row in agg_df.iterrows()
        ],
        hoverinfo='text',
        name='DOT Data',
    ))
    
    # Add gate line
    fig.add_trace(go.Scattermapbox(
        lat=pass_data.gate_lat_pts,
        lon=pass_data.gate_lon_pts,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=10, color='red', symbol='circle'),
        name='Gate',
        hoverinfo='name',
    ))
    
    # Calculate map center and zoom
    lat_center = agg_df["lat"].mean()
    lon_center = agg_df["lon"].mean()
    
    lat_range = agg_df["lat"].max() - agg_df["lat"].min()
    lon_range = agg_df["lon"].max() - agg_df["lon"].min()
    max_range = max(lat_range, lon_range)
    
    # Compute zoom level based on data extent
    if max_range > 40:
        zoom = 2
    elif max_range > 20:
        zoom = 3
    elif max_range > 10:
        zoom = 4
    elif max_range > 5:
        zoom = 5
    elif max_range > 2:
        zoom = 6
    else:
        zoom = 7
    
    # Layout with mapbox settings
    fig.update_layout(
        title=dict(
            text=f"<b>Spatial DOT Map</b> - {pass_data.strait_name} - Pass {pass_data.pass_number}",
            font=dict(size=16),
            x=0.5,
        ),
        mapbox=dict(
            style=map_style,
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom,
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=600,
        legend=dict(
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            font=dict(size=11),
        ),
        showlegend=True,
    )
    
    return fig


def _render_spatial_summary(pass_data: PassData):
    """Render spatial data summary."""
    
    with st.expander("📊 Spatial Data Summary"):
        df = pass_data.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Geographic Extent:**")
            st.write(f"- Latitude: {df['lat'].min():.3f}° to {df['lat'].max():.3f}°")
            st.write(f"- Longitude: {df['lon'].min():.3f}° to {df['lon'].max():.3f}°")
            st.write(f"- Gate length: {pass_data.x_km.max():.1f} km")
        
        with col2:
            st.markdown("**Data Coverage:**")
            st.write(f"- Total observations: {len(df):,}")
            st.write(f"- Unique cycles: {df['cycle'].nunique()}")
            st.write(f"- Time span: {df['time'].min().year} - {df['time'].max().year}")
        
        # DOT histogram
        st.markdown("**DOT Distribution:**")
        
        fig_hist = px.histogram(
            df, x="dot", nbins=50,
            title="DOT Value Distribution",
            labels={"dot": "DOT (m)", "count": "Frequency"},
            color_discrete_sequence=["#4CAF50"],
        )
        fig_hist.update_layout(height=250, margin=dict(l=40, r=20, t=40, b=40))
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Download aggregated data
        st.markdown("**Download Spatial Data:**")
        
        agg_df = df.groupby(["lat", "lon"]).agg({
            "dot": ["mean", "std", "count"],
            "corssh": "mean",
            "geoid": "mean",
        }).reset_index()
        agg_df.columns = ["lat", "lon", "dot_mean", "dot_std", "obs_count", "corssh_mean", "geoid_mean"]
        
        csv = agg_df.to_csv(index=False)
        filename = f"spatial_dot_{pass_data.strait_name.replace(' ', '_')}_pass{pass_data.pass_number}.csv"
        
        st.download_button(
            label="📥 Download Aggregated Spatial Data (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv",
        )


def _render_map_explainer():
    """Render explanation when no data is loaded."""
    
    st.markdown("""
    ### What does the Spatial Map show?
    
    The **Spatial Map** visualizes DOT (Dynamic Ocean Topography) values geographically:
    
    🔵 **Colored points**: Satellite altimetry measurements  
    🔴 **Red line/diamonds**: Gate geometry defining the strait  
    🌍 **Background**: Coastlines, continents, and ocean bathymetry  
    
    ---
    
    ### 🎨 Color Options
    
    | Variable | Description |
    |----------|-------------|
    | **Mean DOT** | Average sea surface height anomaly |
    | **DOT Std Dev** | Variability in DOT measurements |
    | **Observation Count** | Number of satellite passes per location |
    
    ---
    
    ### 🌍 Map Projections
    
    | Projection | Best For |
    |------------|----------|
    | **Natural Earth** | General overview, balanced view |
    | **Orthographic** | 3D globe-like view |
    | **Equirectangular** | Flat map, distortion at poles |
    | **Mercator** | Navigation, web maps style |
    | **Stereographic** | Polar regions (Arctic/Antarctic) |
    
    ---
    
    ### 🖱️ Interactive Features
    
    - **🔍 Zoom**: Scroll or pinch to zoom in/out
    - **🖐️ Pan**: Click and drag to move the view
    - **📍 Hover**: See detailed data for each point
    - **📏 Rotate**: For orthographic/stereographic projections
    - **📸 Download**: Use the camera icon to save as PNG
    
    ---
    
    ### 🚀 To get started
    
    1. Select a gate from the sidebar
    2. Adjust the bin size if needed (smaller = more detail)
    3. Click "Load SLCCI Data"
    4. Explore the map with different color options and projections!
    """)
