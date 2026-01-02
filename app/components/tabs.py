"""
Main content tabs for the dashboard.
Supports both SLCCI data and generic datasets with different tab layouts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any, List

from .sidebar import AppConfig


def render_tabs(config: AppConfig):
    """
    Render main content tabs based on loaded data type.
    """
    slcci_data = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    selected_dataset_type = st.session_state.get("selected_dataset_type", "SLCCI")
    cycle_info = st.session_state.get("cycle_info", {})
    
    if selected_dataset_type == "SLCCI" and slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    elif datasets:
        _render_generic_tabs(datasets, cycle_info, config)
    else:
        _render_empty_tabs(config)


def _render_empty_tabs(config: AppConfig):
    """Render welcome tabs when no data is loaded."""
    tab1, tab2 = st.tabs(["Welcome", "Help"])
    
    with tab1:
        st.markdown("## Welcome to NICO Dashboard")
        st.info("""
        **Getting Started:**
        
        1. **Select a Region** from the sidebar
        2. **Choose a Gate** for analysis
        3. **Load Data** using SLCCI or API
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Datasets", len(st.session_state.get("datasets", {})))
        with col2:
            st.metric("Gate", st.session_state.get("selected_gate", "None"))
        with col3:
            st.metric("Type", st.session_state.get("selected_dataset_type", "None"))
    
    with tab2:
        st.markdown("## Help")
        st.markdown("""
        ### SLCCI Data
        - Uses local NetCDF files from J2 satellite
        
        ### Other Datasets
        - CMEMS: Ocean data via API
        - ERA5: Atmospheric reanalysis
        """)


def _render_slcci_tabs(slcci_data, config: AppConfig):
    """Render tabs for SLCCI satellite data."""
    tab1, tab2, tab3, tab4 = st.tabs(["Slope Timeline", "DOT Profile", "Spatial Map", "Monthly Analysis"])
    
    with tab1:
        _render_slope_timeline(slcci_data, config)
    with tab2:
        _render_dot_profile(slcci_data, config)
    with tab3:
        _render_spatial_map(slcci_data, config)

    with tab4:
        _render_slcci_monthly_analysis(slcci_data, config)

def _render_slope_timeline(slcci_data, config: AppConfig):
    """Render slope timeline chart."""
    st.subheader("SSH Slope Timeline")
    
    if hasattr(slcci_data, 'df'):
        df = slcci_data.df
    elif isinstance(slcci_data, dict):
        df = slcci_data.get("df") or slcci_data.get("slopes_df")
    else:
        df = None
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No slope data available.")
        return
    
    if 'cycle' not in df.columns:
        st.warning("Data missing 'cycle' column")
        return
    
    slope_cols = [c for c in df.columns if 'slope' in c.lower()]
    if not slope_cols:
        st.warning("No slope columns found")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        y_column = st.selectbox("Y-axis variable", options=slope_cols, index=0)
    with col2:
        show_trend = st.checkbox("Show trend line", value=True)
    
    if df['cycle'].duplicated().any():
        plot_df = df.groupby('cycle')[y_column].mean().reset_index()
    else:
        plot_df = df[['cycle', y_column]].copy()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["cycle"], y=plot_df[y_column],
        mode="markers+lines", name="SSH Slope",
        marker=dict(size=8, color="steelblue"),
        line=dict(width=1, color="steelblue")
    ))
    
    if show_trend and len(plot_df) > 2:
        z = np.polyfit(plot_df["cycle"], plot_df[y_column], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=plot_df["cycle"], y=p(plot_df["cycle"]),
            mode="lines", name=f"Trend ({z[0]:.4f}/cycle)",
            line=dict(dash="dash", color="red")
        ))
    
    fig.update_layout(
        title="Cross-Gate SSH Slope Over Time",
        xaxis_title="Cycle", yaxis_title="Slope",
        height=500, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("Statistics"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{plot_df[y_column].mean():.3f}")
        c2.metric("Std", f"{plot_df[y_column].std():.3f}")
        c3.metric("Min", f"{plot_df[y_column].min():.3f}")
        c4.metric("Max", f"{plot_df[y_column].max():.3f}")


def _render_dot_profile(slcci_data, config: AppConfig):
    """Render DOT profile plot."""
    st.subheader("Dynamic Ocean Topography Profile")
    
    if hasattr(slcci_data, 'df'):
        df = slcci_data.df
    elif isinstance(slcci_data, dict):
        df = slcci_data.get("df")
    else:
        df = None
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No DOT profiles available.")
        return
    
    dot_cols = [c for c in df.columns if 'dot' in c.lower()]
    if not dot_cols:
        st.warning("No DOT columns found")
        return
    
    if 'cycle' not in df.columns:
        st.warning("No cycle information")
        return
    
    available_cycles = sorted(df['cycle'].unique())
    selected_cycles = st.multiselect(
        "Select cycles", options=available_cycles,
        default=available_cycles[:min(5, len(available_cycles))]
    )
    
    if not selected_cycles:
        st.info("Select at least one cycle.")
        return
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    
    fig = go.Figure()
    colors = px.colors.sample_colorscale("viridis", len(selected_cycles))
    
    for i, cycle in enumerate(selected_cycles):
        cycle_df = df[df['cycle'] == cycle].copy()
        if len(cycle_df) == 0:
            continue
        if lat_col:
            cycle_df = cycle_df.sort_values(lat_col)
            x_vals = cycle_df[lat_col]
        else:
            x_vals = range(len(cycle_df))
        
        fig.add_trace(go.Scatter(
            x=x_vals, y=cycle_df[dot_cols[0]],
            mode="lines", name=f"Cycle {cycle}",
            line=dict(color=colors[i] if i < len(colors) else None)
        ))
    
    fig.update_layout(
        title="DOT Profiles", xaxis_title="Latitude" if lat_col else "Index",
        yaxis_title="DOT (m)", height=500, template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_spatial_map(slcci_data, config: AppConfig):
    """Render spatial map."""
    st.subheader("Spatial Distribution")
    
    if hasattr(slcci_data, 'df'):
        df = slcci_data.df
    elif isinstance(slcci_data, dict):
        df = slcci_data.get("df")
    else:
        df = None
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No spatial data available.")
        return
    
    lat_col = next((c for c in df.columns if c.lower() in ["lat", "latitude"]), None)
    lon_col = next((c for c in df.columns if c.lower() in ["lon", "longitude"]), None)
    
    if not lat_col or not lon_col:
        st.error(f"Missing lat/lon columns. Available: {list(df.columns[:10])}")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    color_options = [c for c in numeric_cols if c not in [lat_col, lon_col]]
    color_var = st.selectbox("Color by", options=color_options) if color_options else None
    
    plot_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
    
    fig = px.scatter_mapbox(
        plot_df, lat=lat_col, lon=lon_col, color=color_var,
        zoom=6, height=600
    )
    fig.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)


def _render_generic_tabs(datasets, cycle_info, config: AppConfig):
    """Render tabs for generic datasets."""
    tab1, tab2, tab3, tab4 = st.tabs(["Monthly", "Profiles", "Spatial", "Explorer"])
    
    with tab1:
        _render_monthly_analysis(datasets, config)
    with tab2:
        _render_profiles(datasets, config)
    with tab3:
        _render_generic_spatial(datasets, config)
    with tab4:
        _render_data_explorer(datasets, config)


def _render_monthly_analysis(datasets, config):
    st.subheader("Monthly Analysis")
    if not datasets:
        st.info("Load data to see monthly analysis.")
        return
    
    dataset_name = st.selectbox("Dataset", options=list(datasets.keys()))
    if dataset_name and dataset_name in datasets:
        data = datasets[dataset_name]
        if hasattr(data, 'to_dataframe'):
            df = data.to_dataframe().reset_index()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            st.warning(f"Cannot display {type(data)}")
            return
        st.write(f"**Shape:** {df.shape}")
        st.dataframe(df.head(100))


def _render_profiles(datasets, config):
    st.subheader("Profile Analysis")
    if not datasets:
        st.info("Load data to see profiles.")
        return
    dataset_name = st.selectbox("Dataset for profiles", list(datasets.keys()), key="prof_ds")
    if dataset_name:
        st.info(f"Profile view for {dataset_name}")


def _render_generic_spatial(datasets, config):
    st.subheader("Spatial View")
    if not datasets:
        st.info("Load data to see spatial view.")
        return
    
    dataset_name = st.selectbox("Dataset", list(datasets.keys()), key="spat_ds")
    if dataset_name and dataset_name in datasets:
        data = datasets[dataset_name]
        if hasattr(data, 'to_dataframe'):
            df = data.to_dataframe().reset_index()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            st.warning("Cannot display")
            return
        
        lat_cols = [c for c in df.columns if 'lat' in c.lower()]
        lon_cols = [c for c in df.columns if 'lon' in c.lower()]
        
        if lat_cols and lon_cols:
            fig = px.scatter_mapbox(df.head(1000), lat=lat_cols[0], lon=lon_cols[0], zoom=4, height=500)
            fig.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No lat/lon columns")


def _render_data_explorer(datasets, config):
    st.subheader("Data Explorer")
    if not datasets:
        st.info("Load data to explore.")
        return
    
    dataset_name = st.selectbox("Dataset", list(datasets.keys()), key="expl_ds")
    if dataset_name and dataset_name in datasets:
        data = datasets[dataset_name]
        st.write("**Type:**", type(data).__name__)
        if hasattr(data, 'shape'):
            st.write("**Shape:**", data.shape)
        
        with st.expander("Sample Data"):
            if hasattr(data, 'to_dataframe'):
                st.dataframe(data.to_dataframe().reset_index().head(100))
            elif isinstance(data, pd.DataFrame):
                st.dataframe(data.head(100))
            else:
                st.write(data)


def _render_slcci_monthly_analysis(slcci_data, config):
    """
    Render 12-month DOT analysis (like SLCCI PLOTTER notebook).
    Shows DOT vs Longitude for each month (1-12) with linear regression.
    """
    st.subheader("12 Months DOT Analysis")
    
    df = getattr(slcci_data, 'df', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No data available for monthly analysis.")
        return
    
    required = ['lon', 'dot', 'month']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key="monthly_reg")
    with col2:
        slope_units = st.selectbox("Slope units", ["mm/m", "m/100km"], index=0, key="monthly_units")
    
    from plotly.subplots import make_subplots
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    R_earth = 6371.0
    mean_lat = df['lat'].mean() if 'lat' in df.columns else 70.0
    lat_rad = np.deg2rad(mean_lat)
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        month_df = df[df['month'] == month].copy()
        if len(month_df) < 2:
            continue
        
        lon = month_df['lon'].values
        dot = month_df['dot'].values
        
        mask = np.isfinite(lon) & np.isfinite(dot)
        if np.sum(mask) < 2:
            continue
        
        lon_valid = lon[mask]
        dot_valid = dot[mask]
        
        fig.add_trace(
            go.Scatter(
                x=lon_valid, y=dot_valid, mode='markers',
                marker=dict(size=3, color='steelblue', opacity=0.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        if show_regression and len(lon_valid) > 2:
            try:
                lon_rad_arr = np.deg2rad(lon_valid)
                dlon_rad = lon_rad_arr - lon_rad_arr.min()
                x_km = R_earth * dlon_rad * np.cos(lat_rad)
                
                slope_m_km, intercept = np.polyfit(x_km, dot_valid, 1)
                
                if slope_units == "mm/m":
                    slope_display = slope_m_km * 1000
                else:
                    slope_display = slope_m_km * 100
                
                slopes_info.append({
                    'month': month,
                    'name': month_names[month-1],
                    'slope': slope_display,
                    'n_points': len(lon_valid)
                })
                
                lon_line = np.linspace(lon_valid.min(), lon_valid.max(), 50)
                lon_line_rad = np.deg2rad(lon_line)
                dlon_line_rad = lon_line_rad - lon_rad_arr.min()
                x_km_line = R_earth * dlon_line_rad * np.cos(lat_rad)
                dot_line = slope_m_km * x_km_line + intercept
                
                fig.add_trace(
                    go.Scatter(
                        x=lon_line, y=dot_line, mode='lines',
                        line=dict(color='red', width=2),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            except Exception:
                pass
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - Monthly DOT vs Longitude",
        height=700,
        template="plotly_white",
        showlegend=False
    )
    
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text="Lon (°)", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="DOT (m)", row=row, col=col)
    
    st.plotly_chart(fig, use_container_width=True)
    
    if slopes_info:
        with st.expander("Monthly Slopes Summary"):
            slopes_df = pd.DataFrame(slopes_info)
            slopes_df.columns = ['Month #', 'Month', f'Slope ({slope_units})', 'N Points']
            st.dataframe(slopes_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Slope", f"{slopes_df[f'Slope ({slope_units})'].mean():.4f}")
            with col2:
                st.metric("Std Dev", f"{slopes_df[f'Slope ({slope_units})'].std():.4f}")
            with col3:
                st.metric("Months with Data", len(slopes_df))
