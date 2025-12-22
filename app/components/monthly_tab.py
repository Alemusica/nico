"""
Monthly Analysis Tab
====================
12-subplot monthly DOT analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime, timedelta

from .sidebar import AppConfig
from src.analysis.slope import bin_by_longitude


def render_monthly_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render monthly analysis tab."""
    
    st.subheader("📅 Monthly DOT Evolution")
    
    # Build combined DataFrame
    df = _build_combined_dataframe(datasets, cycle_info, config)
    
    if df.empty:
        st.warning("No data available for monthly analysis.")
        return
    
    # Create monthly subplots
    fig = _create_monthly_subplots(df, config)
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly statistics table
    st.subheader("📊 Monthly Statistics")
    stats_df = _compute_monthly_stats(df)
    st.dataframe(stats_df, use_container_width=True)


def _build_combined_dataframe(
    datasets: list, cycle_info: list, config: AppConfig
) -> pd.DataFrame:
    """Build combined DataFrame from all cycles."""
    
    all_data = []
    
    for i, ds in enumerate(datasets):
        if "corssh" not in ds.data_vars:
            continue
        if config.mss_var not in ds.data_vars:
            continue
        
        cycle_num = cycle_info[i]["cycle"] if i < len(cycle_info) else i + 1
        
        # Compute DOT
        dot = ds["corssh"].values.flatten() - ds[config.mss_var].values.flatten()
        lat = ds["latitude"].values.flatten()
        lon = ds["longitude"].values.flatten()
        
        # Get time
        if "TimeDay" in ds.data_vars:
            time_vals = ds["TimeDay"].values.flatten()
            ref_date = datetime(2000, 1, 1)
            dates = [ref_date + timedelta(days=float(t)) if not np.isnan(t) else None 
                    for t in time_vals]
        else:
            dates = [None] * len(dot)
        
        for j in range(len(dot)):
            if np.isnan(dot[j]) or np.isnan(lat[j]) or np.isnan(lon[j]):
                continue
            if dates[j] is None:
                continue
            
            all_data.append({
                "cycle": cycle_num,
                "lat": lat[j],
                "lon": lon[j],
                "dot": dot[j],
                "date": dates[j],
                "month": dates[j].month,
                "year": dates[j].year,
            })
    
    return pd.DataFrame(all_data)


def _create_monthly_subplots(df: pd.DataFrame, config: AppConfig) -> go.Figure:
    """Create 12-subplot monthly analysis figure."""
    
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=month_names,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )
    
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1
        
        month_data = df[df["month"] == month]
        
        if month_data.empty:
            continue
        
        # Bin by longitude
        bin_centers, bin_means, _, _ = bin_by_longitude(
            month_data["lon"].values,
            month_data["dot"].values,
            config.bin_size,
        )
        
        if len(bin_centers) < 2:
            continue
        
        fig.add_trace(
            go.Scatter(
                x=bin_centers,
                y=bin_means,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2),
                name=month_names[month - 1],
                showlegend=False,
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        title="Monthly DOT Evolution",
        height=700,
        template="plotly_white",
    )
    
    return fig


def _compute_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly statistics."""
    
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    
    stats = []
    
    for month in range(1, 13):
        month_data = df[df["month"] == month]
        
        if month_data.empty:
            stats.append({
                "Month": month_names[month],
                "N Obs": 0,
                "N Cycles": 0,
            })
            continue
        
        stats.append({
            "Month": month_names[month],
            "N Obs": len(month_data),
            "N Cycles": month_data["cycle"].nunique(),
            "Mean DOT (m)": f"{month_data['dot'].mean():.4f}",
            "Std DOT (m)": f"{month_data['dot'].std():.4f}",
        })
    
    return pd.DataFrame(stats)
