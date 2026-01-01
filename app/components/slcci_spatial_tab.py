"""
SLCCI Spatial Map Tab
=====================
Geographic map showing DOT values and satellite ground tracks.

This tab shows:
- Map with DOT values color-coded
- Gate geometry overlay
- Coastlines and geographic features
- Interactive zoom and pan

Uses Plotly for interactivity instead of Cartopy (which requires backend rendering).
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
    
    # Map options
    col1, col2 = st.columns(2)
    with col1:
        color_var = st.selectbox(
            "Color by",
            ["Mean DOT", "DOT Std Dev", "Observation Count"],
            index=0,
            key="slcci_map_color_var"
        )
    with col2:
        map_style = st.selectbox(
            "Map Style",
            ["open-street-map", "carto-positron", "carto-darkmatter"],
            index=1,
            key="slcci_map_style"
        )
    
    # === MAIN MAP ===
    fig = _create_spatial_map(pass_data, color_var, map_style)
    st.plotly_chart(fig, use_container_width=True, key="slcci_spatial_map")
    
    # === DATA SUMMARY TABLE ===
    _render_spatial_summary(pass_data)


def _create_spatial_map(
    pass_data: PassData,
    color_var: str = "Mean DOT",
    map_style: str = "carto-positron",
) -> go.Figure:
    """Create interactive spatial map with Plotly."""
    
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
    
    # Choose color variable
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
    
    # Create map
    fig = go.Figure()
    
    # Add satellite data points
    fig.add_trace(go.Scattermapbox(
        lat=agg_df["lat"],
        lon=agg_df["lon"],
        mode='markers',
        marker=dict(
            size=8,
            color=agg_df[color_col],
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title=color_label,
                thickness=15,
                len=0.7,
            ),
            opacity=0.8,
        ),
        text=[
            f"DOT: {row['dot_mean']:.4f} m<br>"
            f"Std: {row['dot_std']:.4f} m<br>"
            f"Count: {row['obs_count']}<br>"
            f"Lat: {row['lat']:.3f}°<br>"
            f"Lon: {row['lon']:.3f}°"
            for _, row in agg_df.iterrows()
        ],
        hoverinfo='text',
        name='DOT Data',
    ))
    
    # Add gate line
    fig.add_trace(go.Scattermapbox(
        lat=pass_data.gate_lat_pts,
        lon=pass_data.gate_lon_pts,
        mode='lines',
        line=dict(color='red', width=4),
        name='Gate',
        hoverinfo='name',
    ))
    
    # Calculate map center and zoom
    lat_center = agg_df["lat"].mean()
    lon_center = agg_df["lon"].mean()
    
    lat_range = agg_df["lat"].max() - agg_df["lat"].min()
    lon_range = agg_df["lon"].max() - agg_df["lon"].min()
    max_range = max(lat_range, lon_range)
    
    # Estimate zoom level
    if max_range > 20:
        zoom = 2
    elif max_range > 10:
        zoom = 3
    elif max_range > 5:
        zoom = 4
    elif max_range > 2:
        zoom = 5
    else:
        zoom = 6
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"Spatial DOT Map - {pass_data.strait_name} - Pass {pass_data.pass_number}",
            font=dict(size=16),
        ),
        mapbox=dict(
            style=map_style,
            center=dict(lat=lat_center, lon=lon_center),
            zoom=zoom,
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
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
    
    The **Spatial Map** visualizes DOT values geographically:
    
    - **Colored points**: Satellite altimetry measurements
    - **Red line**: Gate geometry
    - **Color scale**: DOT value (or other variable)
    
    ---
    
    **Interactive Features:**
    
    - 🔍 **Zoom**: Scroll or pinch
    - 🖐️ **Pan**: Click and drag
    - 📍 **Hover**: See point details
    - 📏 **Measure**: Double-click to set marker
    
    ---
    
    **To get started:**
    1. Select a gate from the sidebar
    2. Load SLCCI data
    3. The map will show all observations colored by DOT
    """)
