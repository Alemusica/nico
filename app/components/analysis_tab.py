"""
Slope Timeline Tab
==================
DOT slope evolution analysis.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .sidebar import AppConfig
from src.analysis.slope import bin_by_longitude, compute_slope
from src.visualization.plotly_charts import create_slope_timeline_plot


def render_slope_timeline_tab(datasets: list, cycle_info: list, config: AppConfig):
    """Render slope timeline analysis tab."""
    
    st.subheader("📈 DOT Slope Evolution")
    
    with st.spinner("Computing DOT analysis..."):
        results = _compute_all_cycles(datasets, cycle_info, config)
    
    if not results:
        st.warning("No results available. Check your data and filters.")
        return
    
    # Summary metrics
    slopes = [r["slope_mm_m"] for r in results if r["slope_mm_m"] is not None]
    
    if slopes:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Slope", f"{np.mean(slopes):.4f} mm/m")
        with col2:
            st.metric("Std Dev", f"{np.std(slopes):.4f} mm/m")
        with col3:
            st.metric("Min", f"{np.min(slopes):.4f} mm/m")
        with col4:
            st.metric("Max", f"{np.max(slopes):.4f} mm/m")
        
        # Timeline plot
        timeline_df = pd.DataFrame(results)
        timeline_df = timeline_df.rename(columns={
            "slope_mm_m": "slope_mm_per_m",
            "slope_err": "slope_err_mm_per_m",
        })
        
        fig = create_slope_timeline_plot(timeline_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Results table
        st.subheader("📋 Detailed Results")
        
        display_df = pd.DataFrame([{
            "Cycle": r["cycle"],
            "Date": r["date"].strftime("%Y-%m-%d") if r["date"] else "N/A",
            "Slope (mm/m)": f"{r['slope_mm_m']:.6f}",
            "Error (mm/m)": f"{r['slope_err']:.6f}" if r["slope_err"] else "N/A",
            "R²": f"{r['r2']:.4f}" if r["r2"] else "N/A",
            "N Points": r["n_points"],
        } for r in results])
        
        st.dataframe(display_df, use_container_width=True)


def _compute_all_cycles(datasets: list, cycle_info: list, config: AppConfig) -> list:
    """Compute analysis for all cycles with progress bar."""
    
    results = []
    progress_bar = st.progress(0, text="Analyzing cycles...")
    
    for i, ds in enumerate(datasets):
        cycle_num = cycle_info[i]["cycle"] if i < len(cycle_info) else i + 1
        progress_bar.progress((i + 1) / len(datasets), text=f"Analyzing cycle {cycle_num}...")
        
        analysis = _analyze_single_cycle(ds, config)
        
        if analysis:
            analysis["cycle"] = cycle_num
            results.append(analysis)
    
    progress_bar.empty()
    return results


def _analyze_single_cycle(ds, config: AppConfig) -> dict | None:
    """Analyze a single cycle with sampling for performance."""
    
    # Check required variables
    if "corssh" not in ds.data_vars:
        return None
    if config.mss_var not in ds.data_vars:
        return None
    
    # Compute DOT
    dot = ds["corssh"] - ds[config.mss_var]
    
    # Get coordinates
    lat = ds["latitude"].values.flatten()
    lon = ds["longitude"].values.flatten()
    dot_vals = dot.values.flatten()
    
    # Apply spatial filter
    mask = np.ones(len(lat), dtype=bool)
    
    if config.use_spatial_filter:
        if config.lat_range:
            mask &= (lat >= config.lat_range[0]) & (lat <= config.lat_range[1])
        if config.lon_range:
            mask &= (lon >= config.lon_range[0]) & (lon <= config.lon_range[1])
    
    lat_filt = lat[mask]
    lon_filt = lon[mask]
    dot_filt = dot_vals[mask]
    
    if len(lat_filt) == 0:
        return None
    
    # Apply sampling for performance (if sample_fraction < 1.0)
    if hasattr(config, 'sample_fraction') and config.sample_fraction < 1.0:
        n_total = len(lat_filt)
        n_samples = max(1000, int(n_total * config.sample_fraction))  # Min 1000 points
        if n_samples < n_total:
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(n_total, n_samples, replace=False)
            lat_filt = lat_filt[indices]
            lon_filt = lon_filt[indices]
            dot_filt = dot_filt[indices]
    
    # Bin by longitude
    bin_centers, bin_means, bin_stds, bin_counts = bin_by_longitude(
        lon_filt, dot_filt, config.bin_size
    )
    
    if len(bin_centers) < 3:
        return None
    
    # Compute slope
    lat_mean = np.nanmean(lat_filt)
    result = compute_slope(lon_filt, dot_filt, lat_mean, config.bin_size)
    
    if result is None:
        return None
    
    # Get time info
    date = None
    if "TimeDay" in ds.data_vars:
        time_vals = ds["TimeDay"].values.flatten()
        
        # Handle both numeric (days since epoch) and datetime64 formats
        if np.issubdtype(time_vals.dtype, np.datetime64):
            # Convert datetime64 to datetime, take mean of valid values
            valid_mask = ~np.isnat(time_vals)
            if valid_mask.any():
                valid_times = time_vals[valid_mask]
                # Convert to int64 (nanoseconds), compute mean, convert back
                mean_ns = np.mean(valid_times.astype('datetime64[ns]').astype(np.int64))
                date = pd.Timestamp(mean_ns, unit='ns').to_pydatetime()
        else:
            # Numeric: days since 2000-01-01
            valid_times = time_vals[~np.isnan(time_vals)]
            if len(valid_times) > 0:
                ref_date = datetime(2000, 1, 1)
                mean_days = np.nanmean(valid_times)
                date = ref_date + timedelta(days=float(mean_days))
    
    return {
        "slope_mm_m": result.slope_mm_per_m,
        "slope_err": result.slope_err_mm_per_m,
        "r2": result.r_squared,
        "p_value": result.p_value,
        "lat_mean": lat_mean,
        "date": date,
        "n_points": result.n_points,
        "lon_centers": bin_centers,
        "dot_mean": bin_means,
    }
