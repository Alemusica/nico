"""
Plotly Charts
=============
Interactive visualization functions using Plotly.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_slope_timeline_plot(
    timeline_df: pd.DataFrame,
    title: str = "DOT Slope Timeline",
    show_trend: bool = True,
    show_mean: bool = True,
) -> go.Figure:
    """
    Create slope timeline plot with error bars.
    
    Parameters
    ----------
    timeline_df : pd.DataFrame
        DataFrame with columns: date, slope_mm_per_m, slope_err_mm_per_m
    title : str
        Plot title
    show_trend : bool
        Add linear trend line
    show_mean : bool
        Add mean horizontal line
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    if timeline_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    fig = go.Figure()
    
    # Main data with error bars
    fig.add_trace(go.Scatter(
        x=timeline_df["date"],
        y=timeline_df["slope_mm_per_m"],
        error_y=dict(
            type="data",
            array=timeline_df["slope_err_mm_per_m"],
            visible=True,
            color="rgba(31, 119, 180, 0.5)",
        ),
        mode="lines+markers",
        name="DOT Slope",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=6),
    ))
    
    slopes = timeline_df["slope_mm_per_m"].values
    
    # Trend line
    if show_trend and len(slopes) > 2:
        x_numeric = np.arange(len(slopes))
        z = np.polyfit(x_numeric, slopes, 1)
        trend_y = np.polyval(z, x_numeric)
        
        fig.add_trace(go.Scatter(
            x=timeline_df["date"],
            y=trend_y,
            mode="lines",
            name=f"Trend ({z[0]:.4f} mm/m per period)",
            line=dict(color="red", dash="dash", width=2),
        ))
    
    # Mean line
    if show_mean:
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        
        fig.add_hline(
            y=mean_slope,
            line_dash="dot",
            line_color="green",
            annotation_text=f"Mean: {mean_slope:.4f} ± {std_slope:.4f} mm/m",
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Slope (mm/m)",
        template="plotly_white",
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    
    return fig


def create_dot_profile_plot(
    profiles: list[dict],
    title: str = "DOT Profiles",
) -> go.Figure:
    """
    Create DOT profile comparison plot.
    
    Parameters
    ----------
    profiles : list[dict]
        List of profile dictionaries with keys:
        - lon: longitude array
        - dot: DOT array
        - label: profile label
    title : str
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    
    for i, profile in enumerate(profiles):
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=profile["lon"],
            y=profile["dot"],
            mode="lines",
            name=profile.get("label", f"Profile {i+1}"),
            line=dict(color=color, width=2),
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Longitude (°)",
        yaxis_title="DOT (m)",
        template="plotly_white",
        height=500,
        showlegend=True,
    )
    
    return fig


def create_spatial_scatter_plot(
    df: pd.DataFrame,
    color_col: str = "dot",
    title: str = "Spatial Distribution",
    colorscale: str = "RdYlBu_r",
) -> go.Figure:
    """
    Create spatial scatter plot on map.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lat, lon, and color column
    color_col : str
        Column to use for color
    title : str
        Plot title
    colorscale : str
        Plotly colorscale name
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color=color_col,
        color_continuous_scale=colorscale,
        zoom=3,
        height=600,
        title=title,
    )
    
    fig.update_layout(mapbox_style="carto-positron")
    
    return fig


def create_monthly_subplots(
    monthly_data: dict[int, pd.DataFrame],
    value_col: str = "dot",
    title: str = "Monthly Analysis",
) -> go.Figure:
    """
    Create 12-subplot monthly analysis figure.
    
    Parameters
    ----------
    monthly_data : dict
        Dictionary mapping month (1-12) to DataFrame
    value_col : str
        Column to plot
    title : str
        Main figure title
        
    Returns
    -------
    go.Figure
        Plotly figure with 3x4 subplots
    """
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f"{name}" for name in month_names],
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        if month in monthly_data and not monthly_data[month].empty:
            data = monthly_data[month]
            
            fig.add_trace(
                go.Scatter(
                    x=data["lon"],
                    y=data[value_col],
                    mode="markers",
                    marker=dict(size=4, opacity=0.6),
                    name=month_names[month - 1],
                    showlegend=False,
                ),
                row=row, col=col,
            )
    
    fig.update_layout(
        title=title,
        height=600,
        template="plotly_white",
    )
    
    return fig


def create_histogram(
    data: np.ndarray,
    title: str = "Distribution",
    xlabel: str = "Value",
    nbins: int = 50,
) -> go.Figure:
    """
    Create histogram of data distribution.
    
    Parameters
    ----------
    data : np.ndarray
        Data to plot
    title : str
        Plot title
    xlabel : str
        X-axis label
    nbins : int
        Number of bins
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure(data=[go.Histogram(x=data, nbinsx=nbins)])
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Count",
        template="plotly_white",
        height=400,
    )
    
    return fig
