"""
Main content tabs for the dashboard.
Following SLCCI PLOTTER notebook workflow exactly.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Dict, Any, List

from .sidebar import AppConfig


def render_tabs(config: AppConfig):
    """Render main content tabs based on loaded data type."""
    slcci_data = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    selected_dataset_type = st.session_state.get("selected_dataset_type", "SLCCI")
    
    if selected_dataset_type == "SLCCI" and slcci_data is not None:
        _render_slcci_tabs(slcci_data, config)
    elif datasets:
        _render_generic_tabs(datasets, config)
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
    
    with tab2:
        st.markdown("## Help")
        st.markdown("""
        ### SLCCI Data
        - Uses local NetCDF files from J2 satellite
        - Shows Slope Timeline, DOT Profile, Spatial Map, Monthly Analysis
        """)


def _render_slcci_tabs(slcci_data, config: AppConfig):
    """Render tabs for SLCCI satellite data using PassData attributes."""
    tab1, tab2, tab3, tab4 = st.tabs([
        "Slope Timeline", 
        "DOT Profile", 
        "Spatial Map", 
        "Monthly Analysis"
    ])
    
    with tab1:
        _render_slope_timeline(slcci_data, config)
    with tab2:
        _render_dot_profile(slcci_data, config)
    with tab3:
        _render_spatial_map(slcci_data, config)
    with tab4:
        _render_monthly_analysis(slcci_data, config)


# ==============================================================================
# TAB 1: SLOPE TIMELINE (from SLCCI PLOTTER Panel 1)
# ==============================================================================
def _render_slope_timeline(slcci_data, config: AppConfig):
    """
    Render slope timeline using PassData.slope_series and time_array.
    
    From SLCCI PLOTTER:
    - X-axis: time_array (dates)
    - Y-axis: slope_series (m/100km)
    """
    st.subheader("SSH Slope Timeline")
    
    # Get attributes from PassData
    slope_series = getattr(slcci_data, 'slope_series', None)
    time_array = getattr(slcci_data, 'time_array', None)
    time_periods = getattr(slcci_data, 'time_periods', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if slope_series is None:
        st.error("❌ No slope_series in PassData. Check SLCCIService.")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(slope_series)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        st.warning("⚠️ All slope values are NaN. The satellite pass may not intersect the gate.")
        st.info("💡 Try selecting a different pass number closer to the gate.")
        return
    
    # Build time axis
    if time_array is not None and len(time_array) > 0:
        x_vals = time_array
        x_label = "Date"
    elif time_periods is not None:
        x_vals = np.arange(len(slope_series))
        x_label = "Time Period Index"
    else:
        x_vals = np.arange(len(slope_series))
        x_label = "Index"
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend line", value=True, key="slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="slope_unit")
    
    # Convert units
    if unit == "cm/km":
        y_vals = slope_series * 100
        y_label = "Slope (cm/km)"
    else:
        y_vals = slope_series
        y_label = "Slope (m/100km)"
    
    # Create figure
    fig = go.Figure()
    
    # Plot only valid values
    valid_x = [x_vals[i] for i in range(len(x_vals)) if valid_mask[i]]
    valid_y = [y_vals[i] for i in range(len(y_vals)) if valid_mask[i]]
    
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        mode="markers+lines",
        name="SSH Slope",
        marker=dict(size=6, color="steelblue"),
        line=dict(width=1, color="steelblue")
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8)
    
    # Trend line
    if show_trend and len(valid_y) > 2:
        x_numeric = np.arange(len(valid_y))
        z = np.polyfit(x_numeric, valid_y, 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=valid_x,
            y=p(x_numeric),
            mode="lines",
            name=f"Trend ({z[0]:.4f}/period)",
            line=dict(dash="dash", color="red")
        ))
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - Monthly DOT Slope",
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    with st.expander("Statistics"):
        valid_slopes = slope_series[valid_mask]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(valid_slopes):.4f}")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_slopes):.4f}")
        with col3:
            st.metric("Valid Points", f"{n_valid}/{len(slope_series)}")
        with col4:
            st.metric("Pass", pass_number)


# ==============================================================================
# TAB 2: DOT PROFILE (from SLCCI PLOTTER Panel 2)
# ==============================================================================
def _render_dot_profile(slcci_data, config: AppConfig):
    """
    Render DOT profile using PassData.profile_mean and x_km.
    
    From SLCCI PLOTTER Panel 2:
    - X-axis: x_km (Distance along longitude in km)
    - Y-axis: profile_mean (Mean DOT in m)
    """
    st.subheader("Mean DOT Profile Across Gate")
    
    # Get attributes from PassData
    profile_mean = getattr(slcci_data, 'profile_mean', None)
    x_km = getattr(slcci_data, 'x_km', None)
    dot_matrix = getattr(slcci_data, 'dot_matrix', None)
    time_periods = getattr(slcci_data, 'time_periods', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if profile_mean is None or x_km is None:
        st.error("❌ No profile_mean or x_km in PassData.")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(profile_mean)
    if not np.any(valid_mask):
        st.warning("⚠️ All DOT values are NaN.")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        view_mode = st.radio(
            "View mode",
            ["Mean Profile", "Individual Periods"],
            horizontal=True,
            key="dot_view_mode"
        )
    with col2:
        show_std = st.checkbox("Show ±1 Std Dev", value=True, key="dot_std")
    
    fig = go.Figure()
    
    if view_mode == "Mean Profile":
        # Plot mean profile (like SLCCI PLOTTER Panel 2)
        fig.add_trace(go.Scatter(
            x=x_km[valid_mask],
            y=profile_mean[valid_mask],
            mode="lines",
            name="Mean DOT",
            line=dict(color="steelblue", width=2)
        ))
        
        # Add std band
        if show_std and dot_matrix is not None:
            std = np.nanstd(dot_matrix, axis=1)
            upper = profile_mean + std
            lower = profile_mean - std
            
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_km[valid_mask], x_km[valid_mask][::-1]]),
                y=np.concatenate([upper[valid_mask], lower[valid_mask][::-1]]),
                fill='toself',
                fillcolor='rgba(70,130,180,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev'
            ))
    
    else:  # Individual Periods
        if dot_matrix is None or time_periods is None:
            st.warning("No time period data available")
            return
        
        n_periods = dot_matrix.shape[1]
        period_labels = [str(p)[:7] for p in time_periods]
        
        selected = st.multiselect(
            "Select periods",
            options=list(range(n_periods)),
            default=list(range(min(5, n_periods))),
            format_func=lambda i: period_labels[i],
            key="dot_periods"
        )
        
        if not selected:
            st.info("Select at least one period")
            return
        
        colors = px.colors.qualitative.Plotly
        for i, idx in enumerate(selected):
            profile = dot_matrix[:, idx]
            mask = ~np.isnan(profile)
            if np.any(mask):
                fig.add_trace(go.Scatter(
                    x=x_km[mask],
                    y=profile[mask],
                    mode="lines",
                    name=period_labels[idx],
                    line=dict(color=colors[i % len(colors)])
                ))
    
    # Add WEST/EAST labels
    fig.add_annotation(
        x=x_km[valid_mask].min(),
        y=np.nanmax(profile_mean[valid_mask]),
        text="WEST",
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor="left"
    )
    fig.add_annotation(
        x=x_km[valid_mask].max(),
        y=np.nanmax(profile_mean[valid_mask]),
        text="EAST",
        showarrow=False,
        font=dict(size=12, color="black"),
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - DOT Profile",
        xaxis_title="Distance along longitude (km)",
        yaxis_title="DOT (m)",
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    with st.expander("Profile Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean DOT", f"{np.nanmean(profile_mean):.4f} m")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(profile_mean) - np.nanmin(profile_mean):.4f} m")
        with col3:
            st.metric("Gate Length", f"{x_km.max():.1f} km")


# ==============================================================================
# TAB 3: SPATIAL MAP (from SLCCI PLOTTER Panel 3)
# ==============================================================================
def _render_spatial_map(slcci_data, config: AppConfig):
    """Render spatial map of DOT measurements."""
    st.subheader("Spatial Distribution")
    
    df = getattr(slcci_data, 'df', None)
    gate_lon_pts = getattr(slcci_data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(slcci_data, 'gate_lat_pts', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if df is None or df.empty:
        st.warning("No spatial data available.")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        color_var = st.selectbox(
            "Color by",
            ["dot", "corssh", "geoid"] if "geoid" in df.columns else ["dot", "corssh"],
            key="map_color"
        )
    with col2:
        show_gate = st.checkbox("Show gate line", value=True, key="map_gate")
    
    # Sample for performance
    if len(df) > 5000:
        plot_df = df.sample(5000)
        st.caption("Showing 5000 random points for performance")
    else:
        plot_df = df
    
    # Create map
    fig = px.scatter_mapbox(
        plot_df,
        lat="lat",
        lon="lon",
        color=color_var,
        color_continuous_scale="viridis",
        zoom=5,
        height=600,
        title=f"{strait_name} - Pass {pass_number}"
    )
    
    # Add gate line
    if show_gate and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scattermapbox(
            lat=gate_lat_pts,
            lon=gate_lon_pts,
            mode="lines",
            name="Gate",
            line=dict(width=3, color="red")
        ))
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# TAB 4: MONTHLY ANALYSIS (from SLCCI PLOTTER 12-subplot figure)
# ==============================================================================
def _render_monthly_analysis(slcci_data, config: AppConfig):
    """
    Render 12-month DOT analysis like SLCCI PLOTTER.
    Shows DOT vs Longitude for each month (1-12) with linear regression.
    """
    st.subheader("12 Months DOT Analysis")
    
    df = getattr(slcci_data, 'df', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    if df is None or df.empty:
        st.warning("No data available for monthly analysis.")
        return
    
    if 'month' not in df.columns or 'lon' not in df.columns or 'dot' not in df.columns:
        st.error("Missing required columns: month, lon, dot")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_regression = st.checkbox("Show linear regression", value=True, key="monthly_reg")
    with col2:
        slope_units = st.selectbox("Slope units", ["mm/m", "m/100km"], key="monthly_units")
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{month_names[i]} ({i+1})" for i in range(12)],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )
    
    R_earth = 6371.0
    mean_lat = df['lat'].mean()
    lat_rad = np.deg2rad(mean_lat)
    
    slopes_info = []
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        month_df = df[df['month'] == month]
        if len(month_df) < 2:
            continue
        
        lon = month_df['lon'].values
        dot = month_df['dot'].values
        
        mask = np.isfinite(lon) & np.isfinite(dot)
        if np.sum(mask) < 2:
            continue
        
        lon_valid = lon[mask]
        dot_valid = dot[mask]
        
        # Scatter
        fig.add_trace(
            go.Scatter(
                x=lon_valid, y=dot_valid, mode='markers',
                marker=dict(size=3, color='steelblue', opacity=0.5),
                showlegend=False
            ),
            row=row, col=col
        )
        
        # Regression
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
                
                # Regression line
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
    
    # Axis labels
    for i in range(1, 13):
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1
        if row == 3:
            fig.update_xaxes(title_text="Lon (°)", row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text="DOT (m)", row=row, col=col)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary table
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


# ==============================================================================
# GENERIC TABS (for CMEMS/ERA5)
# ==============================================================================
def _render_generic_tabs(datasets, config: AppConfig):
    """Render tabs for generic datasets."""
    tab1, tab2 = st.tabs(["Data View", "Explorer"])
    
    with tab1:
        st.subheader("Dataset Overview")
        if datasets:
            for name, data in datasets.items():
                st.write(f"**{name}**: {type(data).__name__}")
        else:
            st.info("No datasets loaded")
    
    with tab2:
        st.subheader("Data Explorer")
        st.info("Load CMEMS or ERA5 data to explore")
