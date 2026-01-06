"""
Volume Transport Chart Components.
Renders volume transport visualizations for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .utils import DATASET_COLORS, get_pass_data_attributes


def render_volume_transport_tab(pass_data, height: int = 500):
    """
    Render volume transport analysis tab.
    
    Volume transport Q = v * A where:
    - v = geostrophic velocity (m/s)
    - A = cross-sectional area (m²)
    
    Args:
        pass_data: PassData object with volume transport data
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    pass_number = getattr(pass_data, 'pass_number', 0)
    
    # Check for pre-computed volume transport
    volume_transport = getattr(pass_data, 'volume_transport', None)
    volume_transport_series = getattr(pass_data, 'volume_transport_series', None)
    time_array = getattr(pass_data, 'time_array', None)
    
    if volume_transport_series is not None and len(volume_transport_series) > 0:
        return _render_volume_transport_timeseries(pass_data, height)
    
    # Try to compute from geostrophic velocity
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    gate_depth = getattr(pass_data, 'gate_depth', None)
    gate_width_km = getattr(pass_data, 'gate_width_km', None)
    
    if v_geostrophic_series is not None and gate_depth is not None and gate_width_km is not None:
        return _render_computed_volume_transport(pass_data, height)
    
    # Show placeholder with formula explanation
    return _render_volume_transport_info(strait_name, pass_number, height)


def _render_volume_transport_timeseries(pass_data, height: int = 500):
    """Render pre-computed volume transport time series."""
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    volume_transport_series = getattr(pass_data, 'volume_transport_series', None)
    time_array = getattr(pass_data, 'time_array', None)
    
    # Handle both numpy arrays and pandas Series
    if hasattr(volume_transport_series, 'index'):
        time_index = volume_transport_series.index
        vt_values = volume_transport_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            time_index = pd.date_range('2000-01', periods=len(volume_transport_series), freq='MS')
        vt_values = volume_transport_series
    
    fig = go.Figure()
    
    # Main time series
    fig.add_trace(go.Scatter(
        x=time_index,
        y=vt_values,  # Usually in Sv (Sverdrup = 10^6 m³/s)
        mode='lines+markers',
        name='Volume Transport',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{strait_name} - Volume Transport Time Series",
        xaxis_title="Time",
        yaxis_title="Volume Transport (Sv)",
        height=height,
        template="plotly_white"
    )
    
    return fig


def _render_computed_volume_transport(pass_data, height: int = 500):
    """Compute volume transport from geostrophic velocity and gate geometry."""
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    v_geostrophic_series = getattr(pass_data, 'v_geostrophic_series', None)
    gate_depth = getattr(pass_data, 'gate_depth', 200)  # Default 200m
    gate_width_km = getattr(pass_data, 'gate_width_km', None)
    time_array = getattr(pass_data, 'time_array', None)
    x_km = getattr(pass_data, 'x_km', None)
    
    # Estimate gate width from x_km if not provided
    if gate_width_km is None and x_km is not None:
        gate_width_km = x_km.max() - x_km.min()
    elif gate_width_km is None:
        gate_width_km = 100  # Default
    
    gate_width_m = gate_width_km * 1000
    cross_section_area = gate_width_m * gate_depth
    
    # Handle both numpy arrays and pandas Series
    if hasattr(v_geostrophic_series, 'index'):
        time_index = v_geostrophic_series.index
        v_values = v_geostrophic_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            time_index = pd.date_range('2000-01', periods=len(v_geostrophic_series), freq='MS')
        v_values = v_geostrophic_series
    
    # Compute volume transport: Q = v * A (m³/s), convert to Sv (10^6 m³/s)
    vt_sv = v_values * cross_section_area / 1e6
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_index,
        y=vt_sv,
        mode='lines+markers',
        name='Volume Transport',
        line=dict(color='steelblue', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=f"{strait_name} - Volume Transport (estimated)",
        xaxis_title="Time",
        yaxis_title="Volume Transport (Sv)",
        height=height,
        template="plotly_white",
        annotations=[
            dict(
                text=f"Gate: {gate_width_km:.0f} km × {gate_depth:.0f} m",
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)"
            )
        ]
    )
    
    return fig


def _render_volume_transport_info(strait_name: str, pass_number: int, height: int = 500):
    """Render informational chart when no volume transport data available."""
    fig = go.Figure()
    
    fig.add_annotation(
        text=f"<b>{strait_name} - Pass {pass_number}</b><br><br>" +
             "Volume Transport requires:<br>" +
             "• Geostrophic velocity time series<br>" +
             "• Gate geometry (width, depth)<br><br>" +
             "Formula: Q = v × A<br>" +
             "Where:<br>" +
             "• v = geostrophic velocity (m/s)<br>" +
             "• A = cross-sectional area (m²)<br>" +
             "• Q = volume transport (m³/s)",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14),
        align="center"
    )
    
    fig.update_layout(
        height=height,
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig


def render_volume_transport_climatology(pass_data, height: int = 400):
    """
    Render monthly climatology of volume transport.
    
    Args:
        pass_data: PassData with volume_transport_series
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    strait_name = getattr(pass_data, 'strait_name', 'Unknown')
    volume_transport_series = getattr(pass_data, 'volume_transport_series', None)
    time_array = getattr(pass_data, 'time_array', None)
    
    if volume_transport_series is None or len(volume_transport_series) == 0:
        return None
    
    # Handle both numpy arrays and pandas Series
    if hasattr(volume_transport_series, 'index'):
        time_index = volume_transport_series.index
        vt_values = volume_transport_series.values
    else:
        if time_array is not None:
            time_index = pd.to_datetime(time_array)
        else:
            return None
        vt_values = volume_transport_series
    
    # Create Series for groupby
    vt_series = pd.Series(vt_values, index=time_index)
    monthly_clim = vt_series.groupby(vt_series.index.month).mean()
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[month_names[m-1] for m in monthly_clim.index],
        y=monthly_clim.values,
        marker_color=['steelblue' if v >= 0 else 'coral' for v in monthly_clim.values],
        name='Mean Transport'
    ))
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    fig.update_layout(
        title=f"{strait_name} - Monthly Mean Volume Transport",
        xaxis_title="Month",
        yaxis_title="Volume Transport (Sv)",
        height=height,
        template="plotly_white"
    )
    
    return fig


def render_multi_volume_transport(datasets: dict, height: int = 500):
    """
    Render volume transport comparison for multiple datasets.
    
    Args:
        datasets: Dict of {name: PassData} objects
        height: Chart height in pixels
        
    Returns:
        go.Figure or None if no data
    """
    if not datasets:
        return None
    
    fig = go.Figure()
    has_data = False
    
    for name, data in datasets.items():
        volume_transport_series = getattr(data, 'volume_transport_series', None)
        time_array = getattr(data, 'time_array', None)
        
        if volume_transport_series is None or len(volume_transport_series) == 0:
            continue
        
        has_data = True
        color = DATASET_COLORS.get(name.lower(), 'gray')
        
        # Handle both numpy arrays and pandas Series
        if hasattr(volume_transport_series, 'index'):
            time_index = volume_transport_series.index
            vt_values = volume_transport_series.values
        else:
            if time_array is not None:
                time_index = pd.to_datetime(time_array)
            else:
                time_index = pd.date_range('2000-01', periods=len(volume_transport_series), freq='MS')
            vt_values = volume_transport_series
        
        fig.add_trace(go.Scatter(
            x=time_index,
            y=vt_values,
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4)
        ))
    
    if not has_data:
        return None
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title="Volume Transport Comparison",
        xaxis_title="Time",
        yaxis_title="Volume Transport (Sv)",
        height=height,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def get_volume_transport_stats(pass_data) -> dict:
    """
    Get statistics for volume transport.
    
    Args:
        pass_data: PassData with volume_transport_series
        
    Returns:
        Dict with mean, std, max, min in Sv
    """
    volume_transport_series = getattr(pass_data, 'volume_transport_series', None)
    
    if volume_transport_series is None or len(volume_transport_series) == 0:
        return {}
    
    if hasattr(volume_transport_series, 'values'):
        vt_values = volume_transport_series.values
    else:
        vt_values = volume_transport_series
    
    return {
        'mean_sv': float(vt_values.mean()),
        'std_sv': float(vt_values.std()),
        'max_sv': float(vt_values.max()),
        'min_sv': float(vt_values.min())
    }
