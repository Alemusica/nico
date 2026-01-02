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


def _render_empty_tabs(config: AppConfig):
    """Render welcome tabs when no data is loaded."""
    tab1, tab2 = st.tabs(["Welcome", "Help"])
    
    with tab1:
        st.markdown("## Welcome to NICO Dashboard")
        st.info("""
        **Getting Started:**
        
        1. **Select a Region** from the sidebar
        2. **Choose a Gate** for analysis
        3. **Load Data** using:
           - SLCCI local files (J2 satellite passes)
           - CMEMS/ERA5 via API
        
        Data will appear in the appropriate tabs once loaded.
        """)
        
        # Quick status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Datasets Loaded", len(st.session_state.get("datasets", {})))
        with col2:
            st.metric("Selected Gate", st.session_state.get("selected_gate", "None"))
        with col3:
            st.metric("Data Type", st.session_state.get("selected_dataset_type", "None"))
    
    with tab2:
        st.markdown("## Help")
        st.markdown("""
        ### SLCCI Data
        - Uses local NetCDF files from J2 satellite
        - Shows Slope Timeline, DOT Profile, Spatial Map, Monthly Analysis
        
        ### CMEMS Data
        - Uses Copernicus Marine Service data
        - Similar visualizations to SLCCI
        
        ### Comparison Mode
        - Load both SLCCI and CMEMS data
        - Enable "Comparison Mode" checkbox
        - View overlay plots with orange (SLCCI) and blue (CMEMS)
        """)


def _render_slcci_tabs(slcci_data: Dict[str, Any], config: AppConfig):
    """Render tabs for SLCCI satellite data."""
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Slope Timeline", 
        "DOT Profile", 
        "Spatial Map", 
        "Monthly Analysis",
        "Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_slope_timeline(slcci_data, config)
    with tab2:
        _render_dot_profile(slcci_data, config)
    with tab3:
        _render_spatial_map(slcci_data, config)
    with tab4:
        _render_slcci_monthly_analysis(slcci_data, config)
    with tab5:
        _render_geostrophic_velocity(slcci_data, config)
    with tab6:
        _render_export_tab(slcci_data, None, config)


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
def _render_slcci_monthly_analysis(slcci_data, config: AppConfig):
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
# TAB 5: GEOSTROPHIC VELOCITY
# ==============================================================================
def _render_geostrophic_velocity(slcci_data, config):
    """
    Render geostrophic velocity analysis.
    Uses the formula: v = -g/f * (dη/dx) where f = 2Ω sin(lat)
    """
    st.subheader("Geostrophic Velocity Analysis")
    
    # Constants
    g = 9.81  # m/s²
    OMEGA = 7.2921e-5  # Earth's angular velocity (rad/s)
    R_earth = 6371.0  # km
    
    # Get data from PassData
    df = getattr(slcci_data, 'df', None)
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    pass_number = getattr(slcci_data, 'pass_number', 0)
    
    # Check if v_geostrophic_series is already computed (from CMEMS service)
    v_geostrophic_series = getattr(slcci_data, 'v_geostrophic_series', None)
    mean_latitude = getattr(slcci_data, 'mean_latitude', None)
    coriolis_f = getattr(slcci_data, 'coriolis_f', None)
    
    if v_geostrophic_series is not None and len(v_geostrophic_series) > 0:
        # Use pre-computed values (CMEMS style)
        st.info(f"Using pre-computed geostrophic velocities at lat={mean_latitude:.2f}° (f={coriolis_f:.2e} s⁻¹)")
        
        time_index = v_geostrophic_series.index
        v_values = v_geostrophic_series.values
        
        # Plot time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_index,
            y=v_values * 100,  # Convert m/s to cm/s
            mode='lines+markers',
            name='v_geostrophic',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"{strait_name} - Geostrophic Velocity Time Series",
            xaxis_title="Time",
            yaxis_title="Geostrophic Velocity (cm/s)",
            height=450,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly climatology
        st.subheader("Monthly Climatology")
        monthly_clim = v_geostrophic_series.groupby(v_geostrophic_series.index.month).mean()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig_clim = go.Figure()
        fig_clim.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_clim.index],
            y=monthly_clim.values * 100,
            marker_color=['steelblue' if v >= 0 else 'coral' for v in monthly_clim.values],
            name='Mean Velocity'
        ))
        
        fig_clim.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        fig_clim.update_layout(
            title=f"{strait_name} - Monthly Mean Geostrophic Velocity",
            xaxis_title="Month",
            yaxis_title="Velocity (cm/s)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_clim, use_container_width=True)
        
        # Statistics
        with st.expander("Geostrophic Velocity Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{v_values.mean() * 100:.2f} cm/s")
            with col2:
                st.metric("Std Dev", f"{v_values.std() * 100:.2f} cm/s")
            with col3:
                st.metric("Max", f"{v_values.max() * 100:.2f} cm/s")
            with col4:
                st.metric("Min", f"{v_values.min() * 100:.2f} cm/s")
        
        return
    
    # Otherwise compute from raw data (SLCCI style)
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.warning("No data available for geostrophic velocity calculation.")
        return
    
    # Check required columns
    required = ['lon', 'dot', 'month']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return
    
    # Get mean latitude for Coriolis parameter
    if 'lat' in df.columns:
        mean_lat = df['lat'].mean()
    else:
        mean_lat = st.number_input("Mean latitude (°)", value=70.0, step=0.1)
    
    lat_rad = np.deg2rad(mean_lat)
    f = 2 * OMEGA * np.sin(lat_rad)
    
    st.info(f"Computing geostrophic velocity at lat={mean_lat:.2f}° (f={f:.2e} s⁻¹)")
    
    # Check for year column for time series
    has_year = 'year' in df.columns
    
    if has_year:
        df['year_month'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
        groups = df.groupby('year_month')
    else:
        groups = df.groupby('month')
    
    # Calculate slope and geostrophic velocity for each period
    results = []
    
    for period, group_df in groups:
        lon = group_df['lon'].values
        dot = group_df['dot'].values
        
        mask = np.isfinite(lon) & np.isfinite(dot)
        if np.sum(mask) < 3:
            continue
        
        lon_valid = lon[mask]
        dot_valid = dot[mask]
        
        # Convert longitude to meters
        lon_rad_arr = np.deg2rad(lon_valid)
        dlon_rad = lon_rad_arr - lon_rad_arr.min()
        x_m = R_earth * 1000 * dlon_rad * np.cos(lat_rad)
        
        try:
            slope_m_m, _ = np.polyfit(x_m, dot_valid, 1)
            v_geo = -g / f * slope_m_m
            
            results.append({
                'period': period,
                'slope_m_m': slope_m_m,
                'v_geostrophic_m_s': v_geo,
                'v_geostrophic_cm_s': v_geo * 100,
                'n_points': len(lon_valid)
            })
        except Exception:
            continue
    
    if not results:
        st.warning("Could not compute geostrophic velocities. Check data quality.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Plot time series or monthly values
    fig = go.Figure()
    
    if has_year:
        x_vals = results_df['period']
        title_suffix = "Time Series"
    else:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x_vals = [month_names[int(m)-1] for m in results_df['period']]
        title_suffix = "Monthly"
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=results_df['v_geostrophic_cm_s'],
        mode='lines+markers',
        name='v_geostrophic',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{strait_name} - Pass {pass_number} - Geostrophic Velocity ({title_suffix})",
        xaxis_title="Time" if has_year else "Month",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    with st.expander("Geostrophic Velocity Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        v_vals = results_df['v_geostrophic_cm_s'].values
        with col1:
            st.metric("Mean", f"{np.mean(v_vals):.2f} cm/s")
        with col2:
            st.metric("Std Dev", f"{np.std(v_vals):.2f} cm/s")
        with col3:
            st.metric("Max", f"{np.max(v_vals):.2f} cm/s")
        with col4:
            st.metric("Min", f"{np.min(v_vals):.2f} cm/s")
        
        st.subheader("Detailed Results")
        display_df = results_df.copy()
        display_df.columns = ['Period', 'Slope (m/m)', 'v_geo (m/s)', 'v_geo (cm/s)', 'N Points']
        st.dataframe(display_df, use_container_width=True)
    
    # Physical interpretation
    with st.expander("Physical Interpretation"):
        mean_v = np.mean(v_vals)
        st.markdown(f"""
        **Geostrophic Balance Formula:**
        
        v = -g/f × (dη/dx)
        
        Where:
        - g = 9.81 m/s² (gravity)
        - f = 2Ω sin(lat) = {f:.2e} s⁻¹ (Coriolis at {mean_lat:.1f}°)
        - dη/dx = DOT slope along gate
        
        **Results:**
        - Mean geostrophic velocity: **{mean_v:.2f} cm/s**
        - Positive values → flow northward
        - Negative values → flow southward
        """)


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
