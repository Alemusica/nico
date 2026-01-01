"""
SLCCI Slope Timeline Tab
========================
Monthly DOT slope evolution analysis for SLCCI data.

This tab shows:
- Interactive slope timeline (mm/m over time)
- Error bars from linear regression
- Statistics summary

The slope represents the DOT gradient across the gate, 
which indicates geostrophic current direction and intensity.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional
from datetime import datetime

from src.services.slcci_service import SLCCIService, PassData


def render_slcci_slope_timeline_tab(pass_data: Optional[PassData] = None):
    """
    Render the slope timeline analysis tab for SLCCI data.
    
    Parameters
    ----------
    pass_data : PassData, optional
        Pre-loaded pass data from SLCCIService. If None, shows instructions.
    """
    st.subheader("📈 DOT Slope Evolution")
    
    # Check if data is loaded
    if pass_data is None:
        pass_data = st.session_state.get("slcci_pass_data")
    
    if pass_data is None:
        st.info("👆 Select a gate and load data from the sidebar to see slope analysis.")
        _render_slope_explainer()
        return
    
    # Display header with pass info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🛰️ Satellite", pass_data.satellite)
    with col2:
        st.metric("🎯 Pass", pass_data.pass_number)
    with col3:
        st.metric("📊 Observations", f"{len(pass_data.df):,}")
    
    # Time range info
    if len(pass_data.time_array) > 0:
        time_start = pass_data.time_array[0]
        time_end = pass_data.time_array[-1]
        st.caption(f"📅 Time range: {time_start.strftime('%Y-%m')} to {time_end.strftime('%Y-%m')}")
    
    st.divider()
    
    # === MAIN PLOT ===
    fig = _create_slope_timeline_plot(pass_data)
    st.plotly_chart(fig, use_container_width=True, key="slcci_slope_timeline")
    
    # === STATISTICS ===
    _render_slope_statistics(pass_data)
    
    # === DOWNLOADABLE DATA ===
    _render_slope_data_table(pass_data)


def _create_slope_timeline_plot(pass_data: PassData) -> go.Figure:
    """Create interactive slope timeline plot with Plotly."""
    
    valid_mask = np.isfinite(pass_data.slope_series)
    times = pass_data.time_array[valid_mask]
    slopes = pass_data.slope_series[valid_mask]
    
    # Create figure
    fig = go.Figure()
    
    # Main trace with markers
    fig.add_trace(go.Scatter(
        x=times,
        y=slopes,
        mode='lines+markers',
        name=f'Pass {pass_data.pass_number}',
        line=dict(color='#2196F3', width=1.5),
        marker=dict(size=6, color='#2196F3', line=dict(width=1, color='white')),
        hovertemplate=(
            "<b>%{x|%Y-%m}</b><br>"
            "Slope: %{y:.4f} m/100km<br>"
            "<extra></extra>"
        ),
    ))
    
    # Zero line
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray", 
        line_width=1,
        annotation_text="Zero slope",
        annotation_position="left",
    )
    
    # Add trend line if enough points
    if len(times) >= 3:
        try:
            # Convert to numeric for regression
            times_numeric = pd.to_datetime(times).astype(np.int64) / 1e18
            z = np.polyfit(times_numeric, slopes, 1)
            trend = np.poly1d(z)(times_numeric)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=trend,
                mode='lines',
                name='Linear Trend',
                line=dict(color='rgba(255, 87, 34, 0.5)', width=2, dash='dot'),
                hoverinfo='skip',
            ))
        except Exception:
            pass
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"DOT Slope Timeline - {pass_data.strait_name} - {pass_data.satellite} Pass {pass_data.pass_number}",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            tickformat="%Y-%m",
        ),
        yaxis=dict(
            title="Slope (m / 100 km)",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
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


def _render_slope_statistics(pass_data: PassData):
    """Render slope statistics summary."""
    
    st.subheader("📊 Slope Statistics")
    
    valid_slopes = pass_data.slope_series[np.isfinite(pass_data.slope_series)]
    
    if len(valid_slopes) == 0:
        st.warning("No valid slope values to compute statistics.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Slope",
            f"{np.mean(valid_slopes):.4f}",
            help="Average DOT slope in m/100km"
        )
    
    with col2:
        st.metric(
            "Std Dev",
            f"{np.std(valid_slopes):.4f}",
            help="Standard deviation of slope values"
        )
    
    with col3:
        st.metric(
            "Min",
            f"{np.min(valid_slopes):.4f}",
            help="Minimum slope value"
        )
    
    with col4:
        st.metric(
            "Max",
            f"{np.max(valid_slopes):.4f}",
            help="Maximum slope value"
        )
    
    # Additional info
    with st.expander("📖 Interpretation Guide"):
        st.markdown("""
        **What does the slope mean?**
        
        - **Positive slope** → Higher DOT on the EAST side → Northward geostrophic current
        - **Negative slope** → Higher DOT on the WEST side → Southward geostrophic current
        - **Slope magnitude** → Current intensity (higher = stronger)
        
        **Units**: m / 100 km
        
        To convert to geostrophic velocity (approximate):
        ```
        V (cm/s) ≈ slope × g / f
        ```
        where g=9.8 m/s² and f is Coriolis parameter (~1.2×10⁻⁴ s⁻¹ at 60°N)
        """)


def _render_slope_data_table(pass_data: PassData):
    """Render downloadable slope data table."""
    
    with st.expander("📋 View/Download Slope Data"):
        # Build DataFrame
        valid_mask = np.isfinite(pass_data.slope_series)
        
        df = pd.DataFrame({
            "Date": [pd.Timestamp(str(p)).strftime("%Y-%m") for p in pass_data.time_periods],
            "Slope (m/100km)": pass_data.slope_series,
        })
        
        df["Valid"] = valid_mask
        df = df.dropna(subset=["Slope (m/100km)"])
        
        # Display table
        st.dataframe(
            df.style.format({"Slope (m/100km)": "{:.6f}"}),
            use_container_width=True,
            height=300,
        )
        
        # Download button
        csv = df.to_csv(index=False)
        filename = f"slope_timeline_{pass_data.strait_name.replace(' ', '_')}_pass{pass_data.pass_number}.csv"
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
        )


def _render_slope_explainer():
    """Render explanation when no data is loaded."""
    
    st.markdown("""
    ### What is DOT Slope?
    
    **DOT (Dynamic Ocean Topography)** = SSH - Geoid
    
    The **slope** of DOT across a strait indicates the geostrophic current direction 
    and intensity based on the geostrophic balance equation.
    
    ---
    
    **To get started:**
    1. Select a gate from the sidebar
    2. Choose a satellite pass
    3. Click "Load SLCCI Data"
    
    The slope timeline will show monthly variations in the DOT gradient.
    """)
