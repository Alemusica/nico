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
from ..state import get_slcci_data, get_cmems_data, is_comparison_mode, get_dtu_data

# Comparison mode colors (from COMPARISON_BATCH notebook)
COLOR_SLCCI = "darkorange"
COLOR_CMEMS = "steelblue"
COLOR_DTU = "seagreen"  # DTUSpace color (green)


def render_tabs(config: AppConfig):
    """Render main content tabs based on loaded data type and comparison mode."""
    slcci_data = get_slcci_data()
    cmems_data = get_cmems_data()
    dtu_data = get_dtu_data()  # DTUSpace (ISOLATED)
    comparison_mode = is_comparison_mode()
    
    # Legacy support
    legacy_slcci = st.session_state.get("slcci_pass_data")
    datasets = st.session_state.get("datasets", {})
    # Read from both radio key and explicit state (for persistence after rerun)
    selected_dataset_type = st.session_state.get("sidebar_datasource") or st.session_state.get("selected_dataset_type", "SLCCI")
    
    # DEBUG: Show what data is available (uncomment to debug)
    st.caption(f"🔍 Data status: DTU={dtu_data is not None}, SLCCI={slcci_data is not None}, CMEMS={cmems_data is not None}, type={selected_dataset_type}")
    
    # DTUSpace mode (ISOLATED - separate tabs) - PRIORITY if DTU data exists
    if dtu_data is not None:
        _render_dtu_tabs(dtu_data, config)
        return  # Exit early - DTU is isolated
    # Comparison mode: overlay SLCCI and CMEMS
    elif comparison_mode and slcci_data is not None and cmems_data is not None:
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
    """Render welcome tabs when no data is loaded - with 3D Globe."""
    tab1, tab2, tab3 = st.tabs(["🌍 Globe", "Welcome", "Help"])
    
    with tab1:
        # Import and render the globe component
        try:
            from .globe import render_globe_landing
            render_globe_landing()
        except ImportError as e:
            st.error(f"Globe component not available: {e}")
            st.info("Install plotly: `pip install plotly`")
    
    with tab2:
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
    
    with tab3:
        st.markdown("## Help")
        st.markdown("""
        ### SLCCI Data
        - Uses local NetCDF files from J2 satellite
        - Select pass number and cycle range
        - Shows slope, DOT profiles, and spatial maps
        
        ### Other Datasets
        - CMEMS: Ocean data via API
        - ERA5: Atmospheric reanalysis
        - Monthly aggregations and profiles
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
    
    # Build available variables dynamically based on DataFrame columns
    # SLCCI has: corssh, geoid, dot
    # CMEMS has: sla_filtered, mdt, dot, satellite, cycle, track
    available_vars = []
    
    # Common variables
    if "dot" in df.columns:
        available_vars.append("dot")
    
    # SLCCI-specific
    if "corssh" in df.columns:
        available_vars.append("corssh")
    if "geoid" in df.columns:
        available_vars.append("geoid")
    
    # CMEMS-specific
    if "sla_filtered" in df.columns:
        available_vars.append("sla_filtered")
    if "mdt" in df.columns:
        available_vars.append("mdt")
    if "satellite" in df.columns:
        available_vars.append("satellite")
    if "track" in df.columns:
        available_vars.append("track")
    if "cycle" in df.columns:
        available_vars.append("cycle")
    
    # Fallback
    if not available_vars:
        available_vars = ["dot"]
    
    # Reset session state if current selection is not valid for this dataset
    if "map_color" in st.session_state and st.session_state.map_color not in available_vars:
        st.session_state.map_color = available_vars[0]
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        color_var = st.selectbox(
            "Color by",
            available_vars,
            index=0,  # Default to first available variable
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
    
    # Determine if color variable is categorical or numeric
    categorical_vars = ["satellite", "track", "cycle"]
    is_categorical = color_var in categorical_vars
    
    # Create map with appropriate color handling
    if is_categorical:
        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color=color_var,
            zoom=5,
            height=600,
            title=f"{strait_name} - Pass {pass_number}"
        )
    else:
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
        
        # Handle both numpy arrays and pandas Series
        if hasattr(v_geostrophic_series, 'index'):
            time_index = v_geostrophic_series.index
            v_values = v_geostrophic_series.values
        else:
            # Convert numpy array to series using time_array from PassData
            time_array = getattr(slcci_data, 'time_array', None)
            if time_array is not None:
                time_index = pd.to_datetime(time_array)
            else:
                time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
            v_values = v_geostrophic_series
        
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
        
        # Create Series for groupby - use time_index from above
        v_series = pd.Series(v_values, index=time_index)
        monthly_clim = v_series.groupby(v_series.index.month).mean()
        
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Slope Timeline Comparison",
        "DOT Profile Comparison", 
        "Spatial Map Comparison",
        "Geostrophic Velocity Comparison",
        "📈 Correlation Analysis",
        "📊 Difference Plot",
        "📥 Export Data"
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
        _render_correlation_analysis(slcci_data, cmems_data, config)
    with tab6:
        _render_difference_plot(slcci_data, cmems_data, config)
    with tab7:
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
                x=valid_x, y=valid_y,
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
                    x=valid_x, y=p(x_numeric),
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
                x=valid_x, y=valid_y,
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
                    x=valid_x, y=p(x_numeric),
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
    
    slcci_profile = getattr(slcci_data, 'profile_mean', None)
    slcci_x_km = getattr(slcci_data, 'x_km', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    cmems_profile = getattr(cmems_data, 'profile_mean', None)
    cmems_x_km = getattr(cmems_data, 'x_km', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_profile is None and cmems_profile is None:
        st.warning("No profile data available for comparison.")
        return
    
    fig = go.Figure()
    
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
    _render_comparison_stats(slcci_profile, cmems_profile, "DOT", "m")


def _render_spatial_map_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render spatial map comparison."""
    st.subheader("Spatial Distribution - SLCCI vs CMEMS")
    
    slcci_df = getattr(slcci_data, 'df', None)
    cmems_df = getattr(cmems_data, 'df', None)
    gate_lon = getattr(slcci_data, 'gate_lon_pts', None) or getattr(cmems_data, 'gate_lon_pts', None)
    gate_lat = getattr(slcci_data, 'gate_lat_pts', None) or getattr(cmems_data, 'gate_lat_pts', None)
    
    if (slcci_df is None or slcci_df.empty) and (cmems_df is None or cmems_df.empty):
        st.warning("No spatial data available for comparison.")
        return
    
    col1, col2 = st.columns([2, 1])
    with col1:
        show_gate = st.checkbox("Show gate line", value=True, key="comp_map_gate")
    with col2:
        sample_size = st.slider("Sample size", 500, 5000, 2000, key="comp_map_sample")
    
    fig = go.Figure()
    
    if slcci_df is not None and not slcci_df.empty:
        plot_df = slcci_df.sample(min(sample_size, len(slcci_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"], lon=plot_df["lon"],
            mode="markers", name="SLCCI",
            marker=dict(size=5, color=COLOR_SLCCI, opacity=0.6)
        ))
    
    if cmems_df is not None and not cmems_df.empty:
        plot_df = cmems_df.sample(min(sample_size, len(cmems_df)))
        fig.add_trace(go.Scattermapbox(
            lat=plot_df["lat"], lon=plot_df["lon"],
            mode="markers", name="CMEMS",
            marker=dict(size=5, color=COLOR_CMEMS, opacity=0.6)
        ))
    
    if show_gate and gate_lon is not None and gate_lat is not None:
        fig.add_trace(go.Scattermapbox(
            lat=gate_lat, lon=gate_lon,
            mode="lines", name="Gate",
            line=dict(width=3, color="red")
        ))
    
    all_lats, all_lons = [], []
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
        mapbox=dict(style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=5),
        height=600, margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_geostrophic_comparison(slcci_data, cmems_data, config: AppConfig):
    """Render geostrophic velocity comparison."""
    st.subheader("Geostrophic Velocity - SLCCI vs CMEMS")
    
    slcci_v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
    slcci_pass = getattr(slcci_data, 'pass_number', 0)
    
    cmems_v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
    cmems_pass = getattr(cmems_data, 'pass_number', 0)
    
    if slcci_v_geo is None and cmems_v_geo is None:
        st.warning("No geostrophic velocity data available.")
        return
    
    fig = go.Figure()
    
    if slcci_v_geo is not None and len(slcci_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=slcci_v_geo.index, y=slcci_v_geo.values * 100,
            mode="lines+markers", name=f"SLCCI (Pass {slcci_pass})",
            line=dict(color=COLOR_SLCCI, width=2),
            marker=dict(size=6, color=COLOR_SLCCI)
        ))
    
    if cmems_v_geo is not None and len(cmems_v_geo) > 0:
        fig.add_trace(go.Scatter(
            x=cmems_v_geo.index, y=cmems_v_geo.values * 100,
            mode="lines+markers", name=f"CMEMS (Pass {cmems_pass})",
            line=dict(color=COLOR_CMEMS, width=2),
            marker=dict(size=6, color=COLOR_CMEMS)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Geostrophic Velocity Comparison: {strait_name}",
        xaxis_title="Time", yaxis_title="Geostrophic Velocity (cm/s)",
        height=500, template="plotly_white", legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly climatology
    st.subheader("Monthly Climatology Comparison")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_clim = go.Figure()
    
    if slcci_v_geo is not None and len(slcci_v_geo) > 0:
        monthly_slcci = slcci_v_geo.groupby(slcci_v_geo.index.month).mean()
        fig_clim.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_slcci.index],
            y=monthly_slcci.values * 100,
            name="SLCCI", marker_color=COLOR_SLCCI, opacity=0.7
        ))
    
    if cmems_v_geo is not None and len(cmems_v_geo) > 0:
        monthly_cmems = cmems_v_geo.groupby(cmems_v_geo.index.month).mean()
        fig_clim.add_trace(go.Bar(
            x=[month_names[m-1] for m in monthly_cmems.index],
            y=monthly_cmems.values * 100,
            name="CMEMS", marker_color=COLOR_CMEMS, opacity=0.7
        ))
    
    fig_clim.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    fig_clim.update_layout(
        title="Monthly Mean Geostrophic Velocity",
        xaxis_title="Month", yaxis_title="Velocity (cm/s)",
        height=400, template="plotly_white", barmode="group"
    )
    
    st.plotly_chart(fig_clim, use_container_width=True)


def _render_export_tab(slcci_data, cmems_data, config: AppConfig):
    """Render comprehensive export tab with CSV, PNG, and statistics."""
    st.subheader("📤 Export Data & Results")
    
    # Check what data is available
    has_slcci = slcci_data is not None
    has_cmems = cmems_data is not None
    
    if not has_slcci and not has_cmems:
        st.warning("No data loaded. Load SLCCI or CMEMS data first.")
        return
    
    # Create tabs for different export types
    export_tabs = st.tabs(["📊 Raw Data", "📈 Time Series", "📉 Statistics", "🖼️ Plots"])
    
    # ==========================================================================
    # TAB 1: RAW DATA EXPORT
    # ==========================================================================
    with export_tabs[0]:
        st.markdown("### Raw Observation Data")
        st.caption("Full DataFrame with all observations (lat, lon, time, variables)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if has_slcci:
                slcci_df = getattr(slcci_data, 'df', None)
                if slcci_df is not None and not slcci_df.empty:
                    st.markdown(f"**SLCCI**: {len(slcci_df):,} observations")
                    st.caption(f"Columns: {', '.join(slcci_df.columns[:8])}...")
                    csv_slcci = slcci_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download SLCCI Raw Data (CSV)",
                        data=csv_slcci,
                        file_name=f"slcci_raw_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_slcci_raw"
                    )
                else:
                    st.info("SLCCI: No raw data available")
            else:
                st.info("SLCCI: Not loaded")
        
        with col2:
            if has_cmems:
                cmems_df = getattr(cmems_data, 'df', None)
                if cmems_df is not None and not cmems_df.empty:
                    st.markdown(f"**CMEMS**: {len(cmems_df):,} observations")
                    st.caption(f"Columns: {', '.join(cmems_df.columns[:8])}...")
                    csv_cmems = cmems_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CMEMS Raw Data (CSV)",
                        data=csv_cmems,
                        file_name=f"cmems_raw_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_cmems_raw"
                    )
                else:
                    st.info("CMEMS: No raw data available")
            else:
                st.info("CMEMS: Not loaded")
    
    # ==========================================================================
    # TAB 2: TIME SERIES EXPORT
    # ==========================================================================
    with export_tabs[1]:
        st.markdown("### Monthly Time Series")
        st.caption("Slope, geostrophic velocity, and other derived time series")
        
        # Build combined time series DataFrame
        ts_data = []
        
        if has_slcci:
            time_array = getattr(slcci_data, 'time_array', None)
            slope_series = getattr(slcci_data, 'slope_series', None)
            v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
            
            if time_array is not None and slope_series is not None:
                for i, t in enumerate(time_array):
                    row = {
                        'time': pd.Timestamp(t),
                        'source': 'SLCCI',
                        'slope_m_100km': slope_series[i] if i < len(slope_series) else np.nan,
                    }
                    if v_geo is not None and i < len(v_geo):
                        row['v_geostrophic_m_s'] = v_geo[i]
                    ts_data.append(row)
        
        if has_cmems:
            time_array = getattr(cmems_data, 'time_array', None)
            slope_series = getattr(cmems_data, 'slope_series', None)
            v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
            
            if time_array is not None and slope_series is not None:
                for i, t in enumerate(time_array):
                    row = {
                        'time': pd.Timestamp(t),
                        'source': 'CMEMS',
                        'slope_m_100km': slope_series[i] if i < len(slope_series) else np.nan,
                    }
                    if v_geo is not None and i < len(v_geo):
                        row['v_geostrophic_m_s'] = v_geo[i]
                    ts_data.append(row)
        
        if ts_data:
            ts_df = pd.DataFrame(ts_data)
            ts_df = ts_df.sort_values(['source', 'time'])
            
            st.dataframe(ts_df.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(ts_df)} rows")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv_ts = ts_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Time Series (Long Format)",
                    data=csv_ts,
                    file_name=f"timeseries_long_{config.selected_gate or 'data'}.csv",
                    mime="text/csv",
                    key="export_ts_long"
                )
            
            with col2:
                # Pivot to wide format for easier analysis
                try:
                    ts_wide = ts_df.pivot(index='time', columns='source', values='slope_m_100km')
                    ts_wide = ts_wide.reset_index()
                    csv_wide = ts_wide.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Time Series (Wide Format)",
                        data=csv_wide,
                        file_name=f"timeseries_wide_{config.selected_gate or 'data'}.csv",
                        mime="text/csv",
                        key="export_ts_wide"
                    )
                except Exception:
                    st.caption("Wide format not available (need both datasets)")
        else:
            st.warning("No time series data available")
    
    # ==========================================================================
    # TAB 3: STATISTICS EXPORT
    # ==========================================================================
    with export_tabs[2]:
        st.markdown("### Summary Statistics")
        
        stats_data = []
        
        # SLCCI stats
        if has_slcci:
            slcci_slope = getattr(slcci_data, 'slope_series', None)
            slcci_vgeo = getattr(slcci_data, 'v_geostrophic_series', None)
            strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
            pass_num = getattr(slcci_data, 'pass_number', None)
            
            if slcci_slope is not None:
                valid = slcci_slope[~np.isnan(slcci_slope)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'SLCCI',
                        'Variable': 'Slope (m/100km)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
            
            if slcci_vgeo is not None:
                valid = slcci_vgeo[~np.isnan(slcci_vgeo)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'SLCCI',
                        'Variable': 'V_geo (m/s)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
        
        # CMEMS stats
        if has_cmems:
            cmems_slope = getattr(cmems_data, 'slope_series', None)
            cmems_vgeo = getattr(cmems_data, 'v_geostrophic_series', None)
            strait_name = getattr(cmems_data, 'strait_name', 'Unknown')
            pass_num = getattr(cmems_data, 'pass_number', None)
            
            if cmems_slope is not None:
                valid = cmems_slope[~np.isnan(cmems_slope)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'CMEMS',
                        'Variable': 'Slope (m/100km)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
            
            if cmems_vgeo is not None:
                valid = cmems_vgeo[~np.isnan(cmems_vgeo)]
                if len(valid) > 0:
                    stats_data.append({
                        'Source': 'CMEMS',
                        'Variable': 'V_geo (m/s)',
                        'Mean': f"{np.mean(valid):.4f}",
                        'Std': f"{np.std(valid):.4f}",
                        'Min': f"{np.min(valid):.4f}",
                        'Max': f"{np.max(valid):.4f}",
                        'N': len(valid),
                        'Strait': strait_name,
                        'Pass/Track': pass_num or 'N/A'
                    })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name=f"statistics_{config.selected_gate or 'data'}.csv",
                mime="text/csv",
                key="export_stats"
            )
        
        # Comparison metrics (if both datasets available)
        if has_slcci and has_cmems:
            st.markdown("### Comparison Metrics")
            
            slcci_slope = getattr(slcci_data, 'slope_series', None)
            cmems_slope = getattr(cmems_data, 'slope_series', None)
            slcci_time = getattr(slcci_data, 'time_array', None)
            cmems_time = getattr(cmems_data, 'time_array', None)
            
            if all(x is not None for x in [slcci_slope, cmems_slope, slcci_time, cmems_time]):
                # Align time series
                try:
                    slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
                    cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
                    
                    # Find common time range
                    common_idx = slcci_series.index.intersection(cmems_series.index)
                    
                    if len(common_idx) > 5:
                        slcci_aligned = slcci_series.loc[common_idx]
                        cmems_aligned = cmems_series.loc[common_idx]
                        
                        # Remove NaN pairs
                        mask = ~(slcci_aligned.isna() | cmems_aligned.isna())
                        slcci_clean = slcci_aligned[mask]
                        cmems_clean = cmems_aligned[mask]
                        
                        if len(slcci_clean) > 5:
                            # Calculate metrics
                            corr = np.corrcoef(slcci_clean, cmems_clean)[0, 1]
                            bias = np.mean(cmems_clean - slcci_clean)
                            rmse = np.sqrt(np.mean((cmems_clean - slcci_clean)**2))
                            mae = np.mean(np.abs(cmems_clean - slcci_clean))
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Correlation (r)", f"{corr:.3f}")
                            col2.metric("Bias (CMEMS-SLCCI)", f"{bias:.4f}")
                            col3.metric("RMSE", f"{rmse:.4f}")
                            col4.metric("MAE", f"{mae:.4f}")
                            
                            st.caption(f"Based on {len(slcci_clean)} overlapping monthly values")
                            
                            # Export comparison data
                            comp_df = pd.DataFrame({
                                'time': common_idx[mask],
                                'SLCCI_slope': slcci_clean.values,
                                'CMEMS_slope': cmems_clean.values,
                                'difference': (cmems_clean - slcci_clean).values
                            })
                            csv_comp = comp_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Comparison Data (CSV)",
                                data=csv_comp,
                                file_name=f"comparison_{config.selected_gate or 'data'}.csv",
                                mime="text/csv",
                                key="export_comparison"
                            )
                        else:
                            st.info("Not enough overlapping data points for comparison")
                    else:
                        st.info("No overlapping time periods found")
                except Exception as e:
                    st.warning(f"Could not compute comparison: {e}")
        else:
            st.info("Load both SLCCI and CMEMS to see comparison metrics")
    
    # ==========================================================================
    # TAB 4: PLOT EXPORT
    # ==========================================================================
    with export_tabs[3]:
        st.markdown("### Export Plots")
        
        st.info("""
        **Method 1: Camera Icon (Recommended)**
        - Hover over any Plotly chart in the app
        - Click the 📷 camera icon in the top-right toolbar
        - PNG will download automatically
        
        **Method 2: Generate PNG Below**
        - Select a plot type below
        - Click "Generate PNG"
        - Download the generated image
        """)
        
        plot_type = st.selectbox(
            "Select Plot to Export",
            ["Slope Time Series", "Geostrophic Velocity", "DOT Profile"],
            key="export_plot_type"
        )
        
        if st.button("🖼️ Generate PNG", key="generate_png"):
            fig = None
            
            if plot_type == "Slope Time Series":
                fig = go.Figure()
                
                if has_slcci:
                    time_arr = getattr(slcci_data, 'time_array', None)
                    slope = getattr(slcci_data, 'slope_series', None)
                    if time_arr is not None and slope is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=slope,
                            mode='lines+markers',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    time_arr = getattr(cmems_data, 'time_array', None)
                    slope = getattr(cmems_data, 'slope_series', None)
                    if time_arr is not None and slope is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=slope,
                            mode='lines+markers',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.update_layout(
                    title=f"Slope Time Series - {config.selected_gate or 'Data'}",
                    xaxis_title="Time",
                    yaxis_title="Slope (m/100km)",
                    template="plotly_white",
                    height=500,
                    width=900
                )
            
            elif plot_type == "Geostrophic Velocity":
                fig = go.Figure()
                
                if has_slcci:
                    time_arr = getattr(slcci_data, 'time_array', None)
                    v_geo = getattr(slcci_data, 'v_geostrophic_series', None)
                    if time_arr is not None and v_geo is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=v_geo * 100,  # cm/s
                            mode='lines+markers',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    time_arr = getattr(cmems_data, 'time_array', None)
                    v_geo = getattr(cmems_data, 'v_geostrophic_series', None)
                    if time_arr is not None and v_geo is not None:
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(time_arr),
                            y=v_geo * 100,  # cm/s
                            mode='lines+markers',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(
                    title=f"Geostrophic Velocity - {config.selected_gate or 'Data'}",
                    xaxis_title="Time",
                    yaxis_title="Velocity (cm/s)",
                    template="plotly_white",
                    height=500,
                    width=900
                )
            
            elif plot_type == "DOT Profile":
                fig = go.Figure()
                
                if has_slcci:
                    x_km = getattr(slcci_data, 'x_km', None)
                    profile = getattr(slcci_data, 'profile_mean', None)
                    if x_km is not None and profile is not None:
                        fig.add_trace(go.Scatter(
                            x=x_km,
                            y=profile,
                            mode='lines',
                            name='SLCCI',
                            line=dict(color='darkorange', width=2)
                        ))
                
                if has_cmems:
                    x_km = getattr(cmems_data, 'x_km', None)
                    profile = getattr(cmems_data, 'profile_mean', None)
                    if x_km is not None and profile is not None:
                        fig.add_trace(go.Scatter(
                            x=x_km,
                            y=profile,
                            mode='lines',
                            name='CMEMS',
                            line=dict(color='steelblue', width=2)
                        ))
                
                fig.update_layout(
                    title=f"Mean DOT Profile - {config.selected_gate or 'Data'}",
                    xaxis_title="Distance (km)",
                    yaxis_title="DOT (m)",
                    template="plotly_white",
                    height=500,
                    width=900
                )
            
            if fig is not None and len(fig.data) > 0:
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate PNG bytes
                try:
                    import io
                    img_bytes = fig.to_image(format="png", width=900, height=500, scale=2)
                    
                    st.download_button(
                        label="📥 Download PNG",
                        data=img_bytes,
                        file_name=f"{plot_type.lower().replace(' ', '_')}_{config.selected_gate or 'data'}.png",
                        mime="image/png",
                        key="download_png"
                    )
                except Exception as e:
                    st.warning(f"PNG export requires kaleido: `pip install kaleido`")
                    st.caption(f"Error: {e}")
            else:
                st.warning("No data available for this plot type")


def _render_comparison_stats(slcci_data, cmems_data, variable_name: str, unit: str):
    """Render comparison statistics expander."""
    with st.expander(f"{variable_name} Statistics Comparison"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**SLCCI (orange)**")
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
            st.markdown(f"**CMEMS (blue)**")
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
# CORRELATION ANALYSIS (NEW!)
# ==============================================================================
def _render_correlation_analysis(slcci_data, cmems_data, config: AppConfig):
    """
    Render correlation analysis between SLCCI and CMEMS data.
    
    Shows scatter plot of slope values with correlation metrics.
    """
    st.subheader("📈 Correlation Analysis: SLCCI vs CMEMS")
    
    # Get slope data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    
    if slcci_slope is None or cmems_slope is None:
        st.warning("Both SLCCI and CMEMS slope data required for correlation analysis.")
        return
    
    # Convert to pandas for alignment
    if slcci_time is not None:
        slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
    else:
        st.warning("SLCCI time array not available.")
        return
    
    if cmems_time is not None:
        cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
    else:
        st.warning("CMEMS time array not available.")
        return
    
    # Align by month (YYYY-MM)
    slcci_monthly = slcci_series.groupby(slcci_series.index.to_period('M')).mean()
    cmems_monthly = cmems_series.groupby(cmems_series.index.to_period('M')).mean()
    
    # Find common periods
    common_periods = slcci_monthly.index.intersection(cmems_monthly.index)
    
    if len(common_periods) < 3:
        st.warning(f"Not enough common periods for correlation. Found {len(common_periods)} common months.")
        st.info("Try loading data with overlapping time ranges.")
        return
    
    # Extract aligned values
    slcci_aligned = slcci_monthly.loc[common_periods].values
    cmems_aligned = cmems_monthly.loc[common_periods].values
    
    # Remove NaN
    mask = ~np.isnan(slcci_aligned) & ~np.isnan(cmems_aligned)
    slcci_clean = slcci_aligned[mask]
    cmems_clean = cmems_aligned[mask]
    
    if len(slcci_clean) < 3:
        st.warning("Not enough valid data points for correlation after removing NaN.")
        return
    
    # Calculate correlation metrics
    correlation = np.corrcoef(slcci_clean, cmems_clean)[0, 1]
    r_squared = correlation ** 2
    
    # Calculate RMSE and bias
    diff = slcci_clean - cmems_clean
    rmse = np.sqrt(np.mean(diff ** 2))
    bias = np.mean(diff)
    
    # Create scatter plot
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=cmems_clean,
        y=slcci_clean,
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(100, 149, 237, 0.7)',  # Cornflower blue
            line=dict(width=1, color='darkblue')
        ),
        name='Monthly Mean Slope',
        hovertemplate='CMEMS: %{x:.4f}<br>SLCCI: %{y:.4f}<extra></extra>'
    ))
    
    # 1:1 line
    min_val = min(cmems_clean.min(), slcci_clean.min())
    max_val = max(cmems_clean.max(), slcci_clean.max())
    margin = (max_val - min_val) * 0.1
    
    fig.add_trace(go.Scatter(
        x=[min_val - margin, max_val + margin],
        y=[min_val - margin, max_val + margin],
        mode='lines',
        line=dict(dash='dash', color='gray', width=1),
        name='1:1 Line'
    ))
    
    # Linear regression line
    slope_reg, intercept_reg = np.polyfit(cmems_clean, slcci_clean, 1)
    x_line = np.array([min_val - margin, max_val + margin])
    y_line = slope_reg * x_line + intercept_reg
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Linear Fit (y = {slope_reg:.2f}x + {intercept_reg:.4f})'
    ))
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Slope Correlation: {strait_name}",
        xaxis_title="CMEMS Slope (m/100km)",
        yaxis_title="SLCCI Slope (m/100km)",
        height=500,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    st.markdown("### Correlation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (r)", f"{correlation:.3f}")
    with col2:
        st.metric("R²", f"{r_squared:.3f}")
    with col3:
        st.metric("RMSE", f"{rmse:.4f} m/100km")
    with col4:
        st.metric("Bias (SLCCI-CMEMS)", f"{bias:.4f} m/100km")
    
    # Interpretation
    with st.expander("📖 Interpretation"):
        if r_squared > 0.7:
            quality = "**Strong**"
            color = "green"
        elif r_squared > 0.4:
            quality = "**Moderate**"
            color = "orange"
        else:
            quality = "**Weak**"
            color = "red"
        
        st.markdown(f"""
        **Correlation Quality**: {quality}
        
        - **R² = {r_squared:.3f}**: {r_squared*100:.1f}% of variance explained
        - **Bias = {bias:.4f}**: {'SLCCI higher' if bias > 0 else 'CMEMS higher'} on average
        - **RMSE = {rmse:.4f}**: Typical difference between datasets
        - **N = {len(slcci_clean)}**: Number of monthly periods compared
        
        **Note**: Correlation is computed on monthly-averaged slope values to reduce noise.
        """)
    
    # Show data table
    with st.expander("📋 Data Table"):
        comparison_df = pd.DataFrame({
            'Period': [str(p) for p in common_periods[mask]],
            'SLCCI Slope': slcci_clean,
            'CMEMS Slope': cmems_clean,
            'Difference': diff
        })
        st.dataframe(comparison_df, use_container_width=True)


# ==============================================================================
# DIFFERENCE PLOT (NEW!)
# ==============================================================================
def _render_difference_plot(slcci_data, cmems_data, config: AppConfig):
    """
    Render difference plot showing SLCCI - CMEMS over time.
    
    Useful for identifying systematic biases and temporal patterns.
    """
    st.subheader("📊 Difference Analysis: SLCCI - CMEMS")
    
    # Get slope data
    slcci_slope = getattr(slcci_data, 'slope_series', None)
    slcci_time = getattr(slcci_data, 'time_array', None)
    cmems_slope = getattr(cmems_data, 'slope_series', None)
    cmems_time = getattr(cmems_data, 'time_array', None)
    
    if slcci_slope is None or cmems_slope is None:
        st.warning("Both SLCCI and CMEMS slope data required for difference analysis.")
        return
    
    # Convert to pandas for alignment
    if slcci_time is not None:
        slcci_series = pd.Series(slcci_slope, index=pd.to_datetime(slcci_time))
    else:
        st.warning("SLCCI time array not available.")
        return
    
    if cmems_time is not None:
        cmems_series = pd.Series(cmems_slope, index=pd.to_datetime(cmems_time))
    else:
        st.warning("CMEMS time array not available.")
        return
    
    # Align by month
    slcci_monthly = slcci_series.groupby(slcci_series.index.to_period('M')).mean()
    cmems_monthly = cmems_series.groupby(cmems_series.index.to_period('M')).mean()
    
    # Find common periods
    common_periods = slcci_monthly.index.intersection(cmems_monthly.index)
    
    if len(common_periods) < 2:
        st.warning("Not enough common periods for difference analysis.")
        return
    
    # Calculate difference
    diff_series = slcci_monthly.loc[common_periods] - cmems_monthly.loc[common_periods]
    
    # Remove NaN
    diff_clean = diff_series.dropna()
    
    if len(diff_clean) < 2:
        st.warning("Not enough valid data points after removing NaN.")
        return
    
    # Create time series plot
    fig = go.Figure()
    
    # Difference line with fill
    x_dates = diff_clean.index.to_timestamp()
    y_vals = diff_clean.values
    
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=y_vals,
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(100, 149, 237, 0.3)',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6),
        name='SLCCI - CMEMS'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    # Mean bias line
    mean_diff = np.mean(y_vals)
    fig.add_hline(
        y=mean_diff, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean Bias: {mean_diff:.4f}",
        annotation_position="top right"
    )
    
    # ±1 std bands
    std_diff = np.std(y_vals)
    fig.add_hrect(
        y0=mean_diff - std_diff,
        y1=mean_diff + std_diff,
        fillcolor="rgba(255, 0, 0, 0.1)",
        line_width=0,
    )
    
    strait_name = getattr(slcci_data, 'strait_name', 'Unknown')
    fig.update_layout(
        title=f"Slope Difference Time Series: {strait_name}",
        xaxis_title="Date",
        yaxis_title="Difference (SLCCI - CMEMS) [m/100km]",
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly climatology of differences
    st.subheader("Monthly Climatology of Difference")
    
    monthly_clim = diff_clean.groupby(diff_clean.index.month).mean()
    monthly_std = diff_clean.groupby(diff_clean.index.month).std()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_clim = go.Figure()
    
    # Bar chart with error bars
    fig_clim.add_trace(go.Bar(
        x=[month_names[m-1] for m in monthly_clim.index],
        y=monthly_clim.values,
        error_y=dict(type='data', array=monthly_std.values, visible=True),
        marker_color=['steelblue' if v >= 0 else 'coral' for v in monthly_clim.values],
        name='Mean Difference'
    ))
    
    fig_clim.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig_clim.update_layout(
        title="Monthly Mean Difference (SLCCI - CMEMS)",
        xaxis_title="Month",
        yaxis_title="Difference (m/100km)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_clim, use_container_width=True)
    
    # Statistics
    st.markdown("### Difference Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Bias", f"{mean_diff:.4f} m/100km")
    with col2:
        st.metric("Std Dev", f"{std_diff:.4f} m/100km")
    with col3:
        st.metric("Max Diff", f"{np.max(y_vals):.4f} m/100km")
    with col4:
        st.metric("Min Diff", f"{np.min(y_vals):.4f} m/100km")
    
    # Interpretation
    with st.expander("📖 Interpretation"):
        bias_direction = "SLCCI shows higher slopes" if mean_diff > 0 else "CMEMS shows higher slopes"
        seasonal_range = monthly_clim.max() - monthly_clim.min()
        
        st.markdown(f"""
        **Systematic Bias**: {bias_direction} on average ({abs(mean_diff):.4f} m/100km)
        
        **Seasonal Pattern**: 
        - Range: {seasonal_range:.4f} m/100km
        - Maximum: {month_names[int(monthly_clim.idxmax())-1]} ({monthly_clim.max():.4f})
        - Minimum: {month_names[int(monthly_clim.idxmin())-1]} ({monthly_clim.min():.4f})
        
        **Possible Causes**:
        - Different DOT calculation methods (SLCCI: corssh-geoid vs CMEMS: sla+mdt)
        - Different satellite coverage (SLCCI: J2 only vs CMEMS: J1+J2+J3)
        - Different temporal sampling
        - Processing differences (orbit, corrections)
        
        **N = {len(diff_clean)}** monthly periods analyzed
        """)


# ==============================================================================
# CMEMS-ONLY TABS
# ==============================================================================
def _render_cmems_tabs(cmems_data, config: AppConfig):
    """Render tabs for CMEMS data only."""
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Slope Timeline",
        "DOT Profile",
        "Spatial Map",
        "Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_slope_timeline(cmems_data, config)
    with tab2:
        _render_dot_profile(cmems_data, config)
    with tab3:
        _render_spatial_map(cmems_data, config)
    with tab4:
        _render_geostrophic_velocity(cmems_data, config)
    with tab5:
        _render_export_tab(None, cmems_data, config)


# ==============================================================================
# DTUSpace TABS (ISOLATED - does not share code with SLCCI/CMEMS)
# ==============================================================================

def _render_dtu_tabs(dtu_data, config: AppConfig):
    """
    Render tabs for DTUSpace gridded data.
    
    ISOLATED from SLCCI/CMEMS - uses DTU-specific rendering functions.
    """
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🟢 Slope Timeline",
        "🟢 DOT Profile",
        "🟢 Spatial Map",
        "🟢 Geostrophic Velocity",
        "📥 Export"
    ])
    
    with tab1:
        _render_dtu_slope_timeline(dtu_data, config)
    with tab2:
        _render_dtu_dot_profile(dtu_data, config)
    with tab3:
        _render_dtu_spatial_map(dtu_data, config)
    with tab4:
        _render_dtu_geostrophic_velocity(dtu_data, config)
    with tab5:
        _render_dtu_export_tab(dtu_data, config)


# ==============================================================================
# DTU TAB 1: SLOPE TIMELINE
# ==============================================================================

def _render_dtu_slope_timeline(dtu_data, config: AppConfig):
    """
    Render DTUSpace slope timeline.
    
    From DTUSpace_plotter notebook Panel 1:
    - X-axis: time_array (monthly dates)
    - Y-axis: slope_series (m/100km)
    """
    st.subheader("🟢 DTUSpace - Slope Timeline")
    
    slope_series = getattr(dtu_data, 'slope_series', None)
    time_array = getattr(dtu_data, 'time_array', None)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace v4')
    start_year = getattr(dtu_data, 'start_year', 2006)
    end_year = getattr(dtu_data, 'end_year', 2017)
    
    if slope_series is None or time_array is None:
        st.error("❌ No slope data available in DTUPassData")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(slope_series)
    n_valid = np.sum(valid_mask)
    
    if n_valid == 0:
        st.warning("⚠️ All slope values are NaN")
        return
    
    # Display info
    st.info(f"📊 **{dataset_name}** | {strait_name} | {start_year}–{end_year}")
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_trend = st.checkbox("Show trend line", value=True, key="dtu_slope_trend")
    with col2:
        unit = st.selectbox("Units", ["m/100km", "cm/km"], key="dtu_slope_unit")
    
    # Convert units
    if unit == "cm/km":
        y_vals = slope_series * 100
        y_label = "Slope (cm/km)"
    else:
        y_vals = slope_series
        y_label = "Slope (m/100km)"
    
    # Create figure
    fig = go.Figure()
    
    # Convert time to pandas datetime for plotting
    time_pd = pd.to_datetime(time_array)
    
    # Plot valid values only
    valid_x = time_pd[valid_mask]
    valid_y = y_vals[valid_mask]
    
    fig.add_trace(go.Scatter(
        x=valid_x,
        y=valid_y,
        mode="markers+lines",
        name="DOT Slope",
        marker=dict(size=6, color=COLOR_DTU),
        line=dict(width=2, color=COLOR_DTU)
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
            name=f"Trend ({z[0]:.4f}/month)",
            line=dict(dash="dash", color="darkgreen", width=1.5)
        ))
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Monthly DOT Slope ({start_year}–{end_year})</sup>",
        xaxis_title="Date",
        yaxis_title=y_label,
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    with st.expander("📊 Statistics"):
        valid_slopes = slope_series[valid_mask]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{np.mean(valid_slopes):.4f} m/100km")
        with col2:
            st.metric("Std Dev", f"{np.std(valid_slopes):.4f} m/100km")
        with col3:
            st.metric("Min", f"{np.min(valid_slopes):.4f} m/100km")
        with col4:
            st.metric("Max", f"{np.max(valid_slopes):.4f} m/100km")
        
        st.caption(f"Valid time steps: {n_valid}/{len(slope_series)}")


# ==============================================================================
# DTU TAB 2: DOT PROFILE
# ==============================================================================

def _render_dtu_dot_profile(dtu_data, config: AppConfig):
    """
    Render DTUSpace mean DOT profile across gate.
    
    From DTUSpace_plotter notebook Panel 2:
    - X-axis: x_km (Distance along gate in km)
    - Y-axis: profile_mean (Mean DOT in m)
    - With WEST/EAST labels
    """
    st.subheader("🟢 DTUSpace - Mean DOT Profile")
    
    profile_mean = getattr(dtu_data, 'profile_mean', None)
    x_km = getattr(dtu_data, 'x_km', None)
    dot_matrix = getattr(dtu_data, 'dot_matrix', None)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace v4')
    
    if profile_mean is None or x_km is None:
        st.error("❌ No profile data available")
        return
    
    # Check for valid data
    valid_mask = ~np.isnan(profile_mean)
    if not np.any(valid_mask):
        st.warning("⚠️ All DOT values are NaN")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        view_mode = st.radio(
            "View mode",
            ["Mean Profile", "Individual Time Steps"],
            horizontal=True,
            key="dtu_dot_view_mode"
        )
    with col2:
        show_std = st.checkbox("Show ±1 Std Dev", value=True, key="dtu_dot_std")
    
    fig = go.Figure()
    
    if view_mode == "Mean Profile":
        # Plot mean profile
        fig.add_trace(go.Scatter(
            x=x_km[valid_mask],
            y=profile_mean[valid_mask],
            mode="lines",
            name="Mean DOT",
            line=dict(color=COLOR_DTU, width=2)
        ))
        
        # Add std band if requested
        if show_std and dot_matrix is not None:
            profile_std = np.nanstd(dot_matrix, axis=1)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_km[valid_mask], x_km[valid_mask][::-1]]),
                y=np.concatenate([
                    (profile_mean + profile_std)[valid_mask],
                    (profile_mean - profile_std)[valid_mask][::-1]
                ]),
                fill='toself',
                fillcolor='rgba(46, 139, 87, 0.2)',  # seagreen with alpha
                line=dict(color='rgba(0,0,0,0)'),
                name='±1 Std Dev'
            ))
    
    else:  # Individual Time Steps
        if dot_matrix is None:
            st.warning("No time step data available")
            return
        
        n_time = dot_matrix.shape[1]
        time_array = getattr(dtu_data, 'time_array', None)
        
        # Let user select time steps
        max_select = min(10, n_time)
        selected = st.multiselect(
            "Select time steps",
            options=list(range(n_time)),
            default=list(range(min(5, n_time))),
            format_func=lambda i: str(pd.Timestamp(time_array[i]).strftime('%Y-%m')) if time_array is not None else f"Step {i}",
            key="dtu_time_steps",
            max_selections=max_select
        )
        
        if not selected:
            st.info("Select at least one time step")
            return
        
        # Use a green color scale
        colors = px.colors.sequential.Greens[2:]  # Skip lightest greens
        
        for i, idx in enumerate(selected):
            profile = dot_matrix[:, idx]
            mask = ~np.isnan(profile)
            if np.any(mask):
                color = colors[i % len(colors)]
                label = str(pd.Timestamp(time_array[idx]).strftime('%Y-%m')) if time_array is not None else f"Step {idx}"
                fig.add_trace(go.Scatter(
                    x=x_km[mask],
                    y=profile[mask],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.5)
                ))
    
    # Add WEST/EAST labels (like DTUSpace_plotter notebook)
    y_max = np.nanmax(profile_mean[valid_mask])
    y_min = np.nanmin(profile_mean[valid_mask])
    y_text = y_max - 0.05 * (y_max - y_min)
    
    fig.add_annotation(
        x=x_km[valid_mask].min(),
        y=y_text,
        text="WEST",
        showarrow=False,
        font=dict(size=12, color="black", weight="bold"),
        xanchor="left"
    )
    fig.add_annotation(
        x=x_km[valid_mask].max(),
        y=y_text,
        text="EAST",
        showarrow=False,
        font=dict(size=12, color="black", weight="bold"),
        xanchor="right"
    )
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Mean DOT Profile Across Gate</sup>",
        xaxis_title="Distance along gate (km)",
        yaxis_title="DOT (m)",
        yaxis_tickformat=".3f",  # 3 decimal places like notebook
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    with st.expander("📊 Profile Statistics"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean DOT", f"{np.nanmean(profile_mean):.4f} m")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(profile_mean) - np.nanmin(profile_mean):.4f} m")
        with col3:
            st.metric("Gate Length", f"{x_km.max():.1f} km")


# ==============================================================================
# DTU TAB 3: SPATIAL MAP (GRIDDED - uses pcolormesh style)
# ==============================================================================

def _render_dtu_spatial_map(dtu_data, config: AppConfig):
    """
    Render DTUSpace spatial map with mean DOT.
    
    Unlike SLCCI/CMEMS (scatter), DTU uses a gridded pcolormesh-style display.
    """
    st.subheader("🟢 DTUSpace - Spatial Map")
    
    dot_mean_grid = getattr(dtu_data, 'dot_mean_grid', None)
    gate_lon_pts = getattr(dtu_data, 'gate_lon_pts', None)
    gate_lat_pts = getattr(dtu_data, 'gate_lat_pts', None)
    lat_grid = getattr(dtu_data, 'lat_grid', None)
    lon_grid = getattr(dtu_data, 'lon_grid', None)
    map_extent = getattr(dtu_data, 'map_extent', {})
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace v4')
    
    if dot_mean_grid is None:
        st.error("❌ No gridded DOT data available")
        return
    
    # Options
    col1, col2 = st.columns([2, 1])
    with col1:
        show_gate = st.checkbox("Show gate line", value=True, key="dtu_map_gate")
    with col2:
        colorscale = st.selectbox(
            "Colorscale",
            ["viridis", "RdBu_r", "Plasma", "Cividis"],
            key="dtu_map_colorscale"
        )
    
    # Get grid data
    if hasattr(dot_mean_grid, 'values'):
        z_data = dot_mean_grid.values
    else:
        z_data = dot_mean_grid
    
    # Compute colorbar limits (5th-95th percentile like notebook)
    vmin = np.nanpercentile(z_data, 5)
    vmax = np.nanpercentile(z_data, 95)
    
    # Create heatmap figure
    fig = go.Figure()
    
    # Add gridded DOT as heatmap
    fig.add_trace(go.Heatmap(
        x=lon_grid,
        y=lat_grid,
        z=z_data,
        colorscale=colorscale,
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title="DOT (m)"),
        name="Mean DOT"
    ))
    
    # Add gate line
    if show_gate and gate_lon_pts is not None and gate_lat_pts is not None:
        fig.add_trace(go.Scatter(
            x=gate_lon_pts,
            y=gate_lat_pts,
            mode="lines",
            name="Gate",
            line=dict(color="red", width=3)
        ))
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Mean DOT (Gridded)</sup>",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1)  # Equal aspect ratio
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    with st.expander("📊 Grid Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Grid Size", f"{lat_grid.shape[0]} × {lon_grid.shape[0]}")
        with col2:
            st.metric("DOT Range", f"{np.nanmax(z_data) - np.nanmin(z_data):.3f} m")
        with col3:
            st.metric("Mean DOT", f"{np.nanmean(z_data):.4f} m")
        with col4:
            st.metric("Coverage", f"{(~np.isnan(z_data)).sum() / z_data.size * 100:.1f}%")


# ==============================================================================
# DTU TAB 4: GEOSTROPHIC VELOCITY
# ==============================================================================

def _render_dtu_geostrophic_velocity(dtu_data, config: AppConfig):
    """
    Render DTUSpace geostrophic velocity.
    
    Uses pre-computed v_geostrophic_series from DTUService.
    """
    st.subheader("🟢 DTUSpace - Geostrophic Velocity")
    
    v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
    time_array = getattr(dtu_data, 'time_array', None)
    mean_lat = getattr(dtu_data, 'mean_latitude', 70.0)
    coriolis_f = getattr(dtu_data, 'coriolis_f', 1e-4)
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace v4')
    
    if v_geo is None or len(v_geo) == 0:
        st.warning("⚠️ No geostrophic velocity data available")
        return
    
    st.info(f"📍 Computing at lat={mean_lat:.2f}° (f={coriolis_f:.2e} s⁻¹)")
    
    # Convert to pandas for easier handling
    time_pd = pd.to_datetime(time_array)
    v_series = pd.Series(v_geo, index=time_pd)
    
    # Time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_pd,
        y=v_geo * 100,  # Convert m/s to cm/s
        mode="lines+markers",
        name="v_geostrophic",
        line=dict(color=COLOR_DTU, width=2),
        marker=dict(size=6, color=COLOR_DTU)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{dataset_name} - {strait_name}<br><sup>Geostrophic Velocity Time Series</sup>",
        xaxis_title="Time",
        yaxis_title="Geostrophic Velocity (cm/s)",
        height=450,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly climatology
    st.subheader("Monthly Climatology")
    
    monthly_clim = v_series.groupby(v_series.index.month).mean()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig_clim = go.Figure()
    
    fig_clim.add_trace(go.Bar(
        x=[month_names[m-1] for m in monthly_clim.index],
        y=monthly_clim.values * 100,
        marker_color=[COLOR_DTU if v >= 0 else 'lightcoral' for v in monthly_clim.values],
        name="Mean Velocity"
    ))
    
    fig_clim.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig_clim.update_layout(
        title="Monthly Mean Geostrophic Velocity",
        xaxis_title="Month",
        yaxis_title="Velocity (cm/s)",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig_clim, use_container_width=True)
    
    # Statistics
    with st.expander("📊 Geostrophic Velocity Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{v_geo.mean() * 100:.2f} cm/s")
        with col2:
            st.metric("Std Dev", f"{v_geo.std() * 100:.2f} cm/s")
        with col3:
            st.metric("Max", f"{v_geo.max() * 100:.2f} cm/s")
        with col4:
            st.metric("Min", f"{v_geo.min() * 100:.2f} cm/s")
    
    # Physical interpretation
    with st.expander("📖 Physical Interpretation"):
        mean_v = v_geo.mean() * 100
        st.markdown(f"""
        **Geostrophic Balance Formula:**
        
        v = -g/f × (dη/dx)
        
        Where:
        - g = 9.81 m/s² (gravity)
        - f = 2Ω sin(lat) = {coriolis_f:.2e} s⁻¹ (Coriolis at {mean_lat:.1f}°)
        - dη/dx = DOT slope along gate
        
        **Results:**
        - Mean geostrophic velocity: **{mean_v:.2f} cm/s**
        - Positive values → flow in one direction
        - Negative values → flow in opposite direction
        """)


# ==============================================================================
# DTU TAB 5: EXPORT
# ==============================================================================

def _render_dtu_export_tab(dtu_data, config: AppConfig):
    """Render export tab for DTUSpace data."""
    st.subheader("📤 Export DTUSpace Data")
    
    # Info
    strait_name = getattr(dtu_data, 'strait_name', 'Unknown')
    dataset_name = getattr(dtu_data, 'dataset_name', 'DTUSpace')
    
    st.info(f"🟢 Exporting **{dataset_name}** data for **{strait_name}**")
    
    # Create tabs for different export types
    export_tabs = st.tabs(["📊 Synthetic Data", "📈 Time Series", "📉 Statistics"])
    
    # ==========================================================================
    # TAB 1: SYNTHETIC DATA EXPORT
    # ==========================================================================
    with export_tabs[0]:
        st.markdown("### Synthetic Observation Data")
        st.caption("DTUSpace is gridded - this creates synthetic 'observations' along the gate")
        
        df = getattr(dtu_data, 'df', None)
        
        if df is not None and not df.empty:
            st.markdown(f"**Rows**: {len(df):,} (gate points × time steps)")
            st.caption(f"Columns: {', '.join(df.columns)}")
            
            # Preview
            st.dataframe(df.head(20), use_container_width=True)
            
            # Download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download DTUSpace Synthetic Data (CSV)",
                data=csv_data,
                file_name=f"dtuspace_synthetic_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_synthetic"
            )
        else:
            st.warning("No synthetic data available")
    
    # ==========================================================================
    # TAB 2: TIME SERIES EXPORT
    # ==========================================================================
    with export_tabs[1]:
        st.markdown("### Monthly Time Series")
        
        time_array = getattr(dtu_data, 'time_array', None)
        slope_series = getattr(dtu_data, 'slope_series', None)
        v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
        
        if time_array is not None and slope_series is not None:
            ts_df = pd.DataFrame({
                'time': pd.to_datetime(time_array),
                'slope_m_100km': slope_series,
            })
            
            if v_geo is not None:
                ts_df['v_geostrophic_m_s'] = v_geo
                ts_df['v_geostrophic_cm_s'] = v_geo * 100
            
            ts_df['source'] = 'DTUSpace'
            ts_df['strait'] = strait_name
            
            st.dataframe(ts_df.head(20), use_container_width=True)
            st.caption(f"Showing first 20 of {len(ts_df)} rows")
            
            csv_ts = ts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Time Series (CSV)",
                data=csv_ts,
                file_name=f"dtuspace_timeseries_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_timeseries"
            )
        else:
            st.warning("No time series data available")
    
    # ==========================================================================
    # TAB 3: STATISTICS EXPORT
    # ==========================================================================
    with export_tabs[2]:
        st.markdown("### Summary Statistics")
        
        slope_series = getattr(dtu_data, 'slope_series', None)
        v_geo = getattr(dtu_data, 'v_geostrophic_series', None)
        start_year = getattr(dtu_data, 'start_year', 2006)
        end_year = getattr(dtu_data, 'end_year', 2017)
        x_km = getattr(dtu_data, 'x_km', None)
        
        stats_data = []
        
        if slope_series is not None:
            valid = slope_series[~np.isnan(slope_series)]
            if len(valid) > 0:
                stats_data.append({
                    'Source': 'DTUSpace',
                    'Variable': 'Slope (m/100km)',
                    'Mean': f"{np.mean(valid):.4f}",
                    'Std': f"{np.std(valid):.4f}",
                    'Min': f"{np.min(valid):.4f}",
                    'Max': f"{np.max(valid):.4f}",
                    'N_valid': len(valid),
                    'N_total': len(slope_series),
                    'Strait': strait_name,
                    'Period': f"{start_year}-{end_year}"
                })
        
        if v_geo is not None:
            valid = v_geo[~np.isnan(v_geo)]
            if len(valid) > 0:
                stats_data.append({
                    'Source': 'DTUSpace',
                    'Variable': 'V_geo (m/s)',
                    'Mean': f"{np.mean(valid):.6f}",
                    'Std': f"{np.std(valid):.6f}",
                    'Min': f"{np.min(valid):.6f}",
                    'Max': f"{np.max(valid):.6f}",
                    'N_valid': len(valid),
                    'N_total': len(v_geo),
                    'Strait': strait_name,
                    'Period': f"{start_year}-{end_year}"
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
            
            # Additional info
            if x_km is not None and len(x_km) > 0:
                st.metric("Gate Length", f"{x_km.max():.1f} km")
            
            csv_stats = stats_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv_stats,
                file_name=f"dtuspace_stats_{strait_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                key="export_dtu_stats"
            )
        else:
            st.warning("No statistics available")
