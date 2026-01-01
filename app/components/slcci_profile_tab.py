"""
SLCCI DOT Profile Tab
=====================
Mean DOT profile across the gate.

This tab shows:
- DOT profile along the gate (distance in km on x-axis)
- West/East labels for orientation
- Interactive plot with zoom capabilities

The profile shows the mean DOT computed across all time periods,
giving a static picture of the ocean topography gradient.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional

from src.services.slcci_service import PassData


def render_slcci_dot_profile_tab(pass_data: Optional[PassData] = None):
    """
    Render the DOT profile analysis tab for SLCCI data.
    
    Parameters
    ----------
    pass_data : PassData, optional
        Pre-loaded pass data from SLCCIService. If None, shows instructions.
    """
    st.subheader("🌊 DOT Profile Across Gate")
    
    # Check if data is loaded
    if pass_data is None:
        pass_data = st.session_state.get("slcci_pass_data")
    
    if pass_data is None:
        st.info("👆 Select a gate and load data from the sidebar to see DOT profile.")
        _render_profile_explainer()
        return
    
    # Header info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🚪 Gate", pass_data.strait_name)
    with col2:
        st.metric("📏 Gate Length", f"{pass_data.x_km.max():.1f} km")
    with col3:
        st.metric("📊 Profile Points", len(pass_data.profile_mean))
    
    st.divider()
    
    # === MAIN PROFILE PLOT ===
    fig = _create_dot_profile_plot(pass_data)
    st.plotly_chart(fig, use_container_width=True, key="slcci_dot_profile")
    
    # === PROFILE STATISTICS ===
    _render_profile_statistics(pass_data)
    
    # === TEMPORAL VARIATION (optional) ===
    with st.expander("📅 Show Temporal Variation"):
        _render_temporal_profiles(pass_data)


def _create_dot_profile_plot(pass_data: PassData) -> go.Figure:
    """Create interactive DOT profile plot with Plotly."""
    
    valid_mask = np.isfinite(pass_data.profile_mean)
    x_km = pass_data.x_km[valid_mask]
    profile = pass_data.profile_mean[valid_mask]
    
    if len(x_km) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid DOT data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Main profile line
    fig.add_trace(go.Scatter(
        x=x_km,
        y=profile,
        mode='lines',
        name='Mean DOT',
        line=dict(color='#4CAF50', width=3),
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.2)',
        hovertemplate=(
            "<b>Distance: %{x:.1f} km</b><br>"
            "DOT: %{y:.4f} m<br>"
            "<extra></extra>"
        ),
    ))
    
    # Add linear fit
    if len(x_km) >= 2:
        slope, intercept = np.polyfit(x_km, profile, 1)
        fit_line = slope * x_km + intercept
        
        fig.add_trace(go.Scatter(
            x=x_km,
            y=fit_line,
            mode='lines',
            name=f'Linear Fit (slope: {slope*100:.4f} m/100km)',
            line=dict(color='#FF5722', width=2, dash='dash'),
            hoverinfo='skip',
        ))
    
    # Add West/East annotations
    y_range = np.nanmax(profile) - np.nanmin(profile)
    y_text = np.nanmax(profile) - 0.05 * y_range if y_range > 0 else np.nanmean(profile)
    
    fig.add_annotation(
        x=x_km.min(),
        y=y_text,
        text="<b>WEST</b>",
        showarrow=False,
        font=dict(size=14, color='navy'),
        xanchor='left',
    )
    
    fig.add_annotation(
        x=x_km.max(),
        y=y_text,
        text="<b>EAST</b>",
        showarrow=False,
        font=dict(size=14, color='navy'),
        xanchor='right',
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"Mean DOT Profile - {pass_data.strait_name} - Pass {pass_data.pass_number}",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Distance along gate (km)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
        ),
        yaxis=dict(
            title="DOT (m)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        height=400,
    )
    
    return fig


def _render_profile_statistics(pass_data: PassData):
    """Render profile statistics."""
    
    st.subheader("📊 Profile Statistics")
    
    valid_profile = pass_data.profile_mean[np.isfinite(pass_data.profile_mean)]
    valid_x = pass_data.x_km[np.isfinite(pass_data.profile_mean)]
    
    if len(valid_profile) == 0:
        st.warning("No valid profile data.")
        return
    
    # Compute statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean DOT",
            f"{np.mean(valid_profile):.4f} m",
            help="Average DOT along the gate"
        )
    
    with col2:
        st.metric(
            "DOT Range",
            f"{np.ptp(valid_profile):.4f} m",
            help="Max - Min DOT (peak-to-peak)"
        )
    
    with col3:
        # Compute slope
        if len(valid_x) >= 2:
            slope, _ = np.polyfit(valid_x, valid_profile, 1)
            st.metric(
                "Slope",
                f"{slope*100:.4f} m/100km",
                help="Linear slope across gate"
            )
        else:
            st.metric("Slope", "N/A")
    
    with col4:
        st.metric(
            "Coverage",
            f"{100 * len(valid_profile) / len(pass_data.profile_mean):.1f}%",
            help="Percentage of gate with valid data"
        )


def _render_temporal_profiles(pass_data: PassData):
    """Render profiles for different time periods."""
    
    dot_matrix = pass_data.dot_matrix
    x_km = pass_data.x_km
    time_periods = pass_data.time_periods
    
    # Let user select time periods to compare
    if len(time_periods) == 0:
        st.warning("No time periods available.")
        return
    
    period_strs = [str(p) for p in time_periods]
    
    # Select specific periods
    selected_periods = st.multiselect(
        "Select periods to compare",
        options=period_strs,
        default=period_strs[:3] if len(period_strs) >= 3 else period_strs,
        max_selections=6,
    )
    
    if not selected_periods:
        return
    
    # Create multi-line plot
    fig = go.Figure()
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
    
    for i, period_str in enumerate(selected_periods):
        idx = period_strs.index(period_str)
        profile = dot_matrix[:, idx]
        valid_mask = np.isfinite(profile)
        
        if np.sum(valid_mask) < 2:
            continue
        
        fig.add_trace(go.Scatter(
            x=x_km[valid_mask],
            y=profile[valid_mask],
            mode='lines',
            name=period_str,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    
    fig.update_layout(
        title="DOT Profiles by Time Period",
        xaxis_title="Distance along gate (km)",
        yaxis_title="DOT (m)",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_profile_explainer():
    """Render explanation when no data is loaded."""
    
    st.markdown("""
    ### What is the DOT Profile?
    
    The **DOT Profile** shows how Dynamic Ocean Topography varies along the gate:
    
    - **X-axis**: Distance along the gate in kilometers (West to East)
    - **Y-axis**: DOT value in meters
    
    ---
    
    **Physical Interpretation:**
    
    - A **sloping profile** indicates a pressure gradient
    - The **gradient direction** determines current direction (via geostrophic balance)
    - **Higher DOT** → higher pressure → water flows perpendicular to gradient (to the right in NH)
    
    ---
    
    **To get started:**
    1. Select a gate from the sidebar
    2. Load SLCCI data
    3. The profile will show the mean DOT across all time periods
    """)
