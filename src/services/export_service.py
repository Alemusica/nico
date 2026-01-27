"""
Export Service - Generate CSV files and PNG images for data export.

Handles:
1. Volume Transport CSV exports (raw, monthly climatology, annual)
2. Salt Flux CSV exports
3. PNG image generation with matplotlib for all visualizations
4. ZIP archive creation for bulk export

Dataset Full Names:
- CMEMS L4: "CMEMS Global Ocean Gridded L4 Sea Surface Heights (SEALEVEL_GLO_PHY_L4_NRT_008_046)"
- GEBCO: "GEBCO 2023 Global Bathymetry Grid"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import io
import zipfile
import logging

# Matplotlib for high-quality export
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from scipy import stats

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS - Colors matching Streamlit/Plotly tabs
# ==============================================================================

# Primary colors (matching tabs.py Plotly charts)
COLOR_PRIMARY = '#1E3A5F'       # Dark blue - main data lines
COLOR_PRIMARY_FILL = '#1E3A5F'  # Same, used with alpha for fills
COLOR_SECONDARY = '#E74C3C'     # Red - mean lines, depth caps
COLOR_SEA_LEVEL = '#3498DB'     # Light blue - sea level
COLOR_ZERO_LINE = '#7F8C8D'     # Gray - zero reference
COLOR_GRID = '#E8E8E8'          # Light gray - grid lines
COLOR_POSITIVE = '#1E3A5F'      # Dark blue - northward/positive
COLOR_NEGATIVE = '#d62728'      # Red - southward/negative
COLOR_TREND_LINE = 'darkred'    # Trend/regression lines

# Dataset-specific colors (matching DATASET_COLORS in tabs.py)
COLOR_CMEMS_L4 = 'mediumpurple'
COLOR_CMEMS = 'steelblue'
COLOR_SLCCI = 'darkorange'
COLOR_DTU = 'seagreen'

# ==============================================================================
# CONSTANTS - Full Dataset Names
# ==============================================================================

DATASET_FULL_NAMES = {
    "cmems_l4": "CMEMS Global Ocean Gridded L4 Sea Surface Heights (SEALEVEL_GLO_PHY_L4_NRT_008_046)",
    "cmems_l3": "CMEMS Along-Track L3 Sea Surface Heights",
    "slcci": "ESA Sea Level CCI (SLCCI) Along-Track Product",
    "dtu": "DTU Space Mean Sea Surface and Tides",
    "gebco": "GEBCO 2023 Global Bathymetry Grid",
}

MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
MONTH_ABBREV = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ==============================================================================
# CSV EXPORT - VOLUME TRANSPORT
# ==============================================================================

def generate_volume_transport_raw_csv(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset_velocity: str = "cmems_l4",
    dataset_bathymetry: str = "gebco",
    gate_coords: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate raw volume transport CSV with monthly data for each year.
    
    Returns DataFrame with columns:
    - gate_name, dataset_velocity, dataset_bathymetry
    - year, month, month_name
    - mean_transport_sv, std_transport_sv, min_transport_sv, max_transport_sv
    - n_observations, positive_flux_sv, negative_flux_sv
    """
    time_pd = pd.to_datetime(time_array)
    
    # Create dataframe with time and transport
    df_raw = pd.DataFrame({
        'time': time_pd,
        'transport_sv': transport_sv
    })
    df_raw['year'] = df_raw['time'].dt.year
    df_raw['month'] = df_raw['time'].dt.month
    
    # Group by year and month
    monthly_stats = []
    
    for (year, month), group in df_raw.groupby(['year', 'month']):
        values = group['transport_sv'].dropna()
        if len(values) == 0:
            continue
            
        positive_flux = values[values > 0].sum() if len(values[values > 0]) > 0 else 0
        negative_flux = values[values < 0].sum() if len(values[values < 0]) > 0 else 0
        
        monthly_stats.append({
            'gate_name': gate_name,
            'dataset_velocity': DATASET_FULL_NAMES.get(dataset_velocity, dataset_velocity),
            'dataset_bathymetry': DATASET_FULL_NAMES.get(dataset_bathymetry, dataset_bathymetry),
            'year': int(year),
            'month': int(month),
            'month_name': MONTH_NAMES[int(month) - 1],
            'mean_transport_sv': float(values.mean()),
            'std_transport_sv': float(values.std()),
            'min_transport_sv': float(values.min()),
            'max_transport_sv': float(values.max()),
            'n_observations': len(values),
            'positive_flux_sv': float(positive_flux),
            'negative_flux_sv': float(negative_flux),
        })
    
    df_monthly = pd.DataFrame(monthly_stats)
    
    # Add gate coordinates if provided
    if gate_coords:
        df_monthly['gate_start_lon'] = gate_coords.get('start_lon', np.nan)
        df_monthly['gate_start_lat'] = gate_coords.get('start_lat', np.nan)
        df_monthly['gate_end_lon'] = gate_coords.get('end_lon', np.nan)
        df_monthly['gate_end_lat'] = gate_coords.get('end_lat', np.nan)
        df_monthly['gate_length_km'] = gate_coords.get('length_km', np.nan)
    
    return df_monthly


def generate_volume_transport_climatology_csv(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset_velocity: str = "cmems_l4",
    dataset_bathymetry: str = "gebco"
) -> pd.DataFrame:
    """
    Generate monthly climatology CSV (mean across all years).
    
    Returns DataFrame with columns:
    - gate_name, month, month_name
    - climatology_mean_sv, climatology_std_sv
    - climatology_min_sv, climatology_max_sv
    - n_years, total_observations
    """
    time_pd = pd.to_datetime(time_array)
    
    df_raw = pd.DataFrame({
        'time': time_pd,
        'transport_sv': transport_sv,
        'year': time_pd.year,
        'month': time_pd.month
    })
    
    climatology = []
    
    for month in range(1, 13):
        month_data = df_raw[df_raw['month'] == month]['transport_sv'].dropna()
        years_with_data = df_raw[df_raw['month'] == month]['year'].nunique()
        
        if len(month_data) == 0:
            continue
            
        climatology.append({
            'gate_name': gate_name,
            'dataset_velocity': DATASET_FULL_NAMES.get(dataset_velocity, dataset_velocity),
            'dataset_bathymetry': DATASET_FULL_NAMES.get(dataset_bathymetry, dataset_bathymetry),
            'month': month,
            'month_name': MONTH_NAMES[month - 1],
            'climatology_mean_sv': float(month_data.mean()),
            'climatology_std_sv': float(month_data.std()),
            'climatology_min_sv': float(month_data.min()),
            'climatology_max_sv': float(month_data.max()),
            'n_years': years_with_data,
            'total_observations': len(month_data),
        })
    
    return pd.DataFrame(climatology)


def generate_volume_transport_annual_csv(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset_velocity: str = "cmems_l4",
    dataset_bathymetry: str = "gebco"
) -> pd.DataFrame:
    """
    Generate annual statistics CSV.
    
    Returns DataFrame with:
    - gate_name, year
    - annual_mean_sv, annual_std_sv
    - annual_total_km3 (integrated over year)
    """
    time_pd = pd.to_datetime(time_array)
    
    df_raw = pd.DataFrame({
        'time': time_pd,
        'transport_sv': transport_sv,
        'year': time_pd.year
    })
    
    annual_stats = []
    
    for year, group in df_raw.groupby('year'):
        values = group['transport_sv'].dropna()
        if len(values) == 0:
            continue
        
        # Annual total volume: integrate transport over time
        # Assuming daily data: total_km3 = mean_sv * 1e6 * seconds_per_year / 1e9
        mean_sv = values.mean()
        seconds_per_year = 365.25 * 24 * 3600
        annual_km3 = mean_sv * 1e6 * seconds_per_year / 1e9  # km³
        
        annual_stats.append({
            'gate_name': gate_name,
            'dataset_velocity': DATASET_FULL_NAMES.get(dataset_velocity, dataset_velocity),
            'dataset_bathymetry': DATASET_FULL_NAMES.get(dataset_bathymetry, dataset_bathymetry),
            'year': int(year),
            'annual_mean_sv': float(mean_sv),
            'annual_std_sv': float(values.std()),
            'annual_min_sv': float(values.min()),
            'annual_max_sv': float(values.max()),
            'annual_total_km3': float(annual_km3),
            'n_observations': len(values),
        })
    
    return pd.DataFrame(annual_stats)


# ==============================================================================
# CSV EXPORT - SALT FLUX
# ==============================================================================

def generate_salt_flux_raw_csv(
    salt_flux_kg_s: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    salinity_psu: float = 34.8,
    density_kg_m3: float = 1027.0,
    dataset_velocity: str = "cmems_l4",
    dataset_bathymetry: str = "gebco"
) -> pd.DataFrame:
    """
    Generate raw salt flux CSV with monthly data.
    
    Salt flux in kg/s, also converted to:
    - Gt/yr equivalent
    - Freshwater flux (mSv) using S_ref = 34.8 PSU
    """
    time_pd = pd.to_datetime(time_array)
    
    df_raw = pd.DataFrame({
        'time': time_pd,
        'salt_flux_kg_s': salt_flux_kg_s,
        'year': time_pd.year,
        'month': time_pd.month
    })
    
    monthly_stats = []
    
    for (year, month), group in df_raw.groupby(['year', 'month']):
        values = group['salt_flux_kg_s'].dropna()
        if len(values) == 0:
            continue
        
        mean_kg_s = values.mean()
        # Convert to Gt/yr: kg/s * seconds_per_year / 1e12
        seconds_per_year = 365.25 * 24 * 3600
        mean_gt_yr = mean_kg_s * seconds_per_year / 1e12
        
        monthly_stats.append({
            'gate_name': gate_name,
            'dataset_velocity': DATASET_FULL_NAMES.get(dataset_velocity, dataset_velocity),
            'dataset_bathymetry': DATASET_FULL_NAMES.get(dataset_bathymetry, dataset_bathymetry),
            'salinity_psu': salinity_psu,
            'density_kg_m3': density_kg_m3,
            'year': int(year),
            'month': int(month),
            'month_name': MONTH_NAMES[int(month) - 1],
            'mean_salt_flux_kg_s': float(mean_kg_s),
            'std_salt_flux_kg_s': float(values.std()),
            'mean_salt_flux_gt_yr': float(mean_gt_yr),
            'n_observations': len(values),
        })
    
    return pd.DataFrame(monthly_stats)


# ==============================================================================
# IMAGE EXPORT - MATPLOTLIB FIGURES
# ==============================================================================

def create_figure_title(
    gate_name: str,
    plot_type: str,
    dataset: str,
    start_year: int,
    end_year: int,
    n_observations: int,
    gate_length_km: Optional[float] = None,
    month: Optional[str] = None
) -> str:
    """Create detailed figure title with all metadata."""
    title = f"{gate_name} - {plot_type}"
    if month:
        title += f" ({month})"
    title += f"\nDataset: {DATASET_FULL_NAMES.get(dataset, dataset)}"
    title += f"\nPeriod: {start_year}-{end_year} | Observations: {n_observations:,}"
    if gate_length_km:
        title += f" | Gate Length: {gate_length_km:.0f} km"
    return title


def export_volume_transport_timeseries(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    Generate Volume Transport time series plot as PNG bytes.
    Colors match Streamlit/Plotly charts exactly.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(transport_sv[~np.isnan(transport_sv)])
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot time series with matching color
    ax.plot(time_pd, transport_sv, color=COLOR_PRIMARY, alpha=0.7, linewidth=0.8)
    
    # Add rolling mean (red, matching Plotly)
    df = pd.DataFrame({'time': time_pd, 'transport': transport_sv})
    df = df.set_index('time')
    rolling_mean = df['transport'].rolling(window=30, center=True).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color=COLOR_SECONDARY, linewidth=2, label='30-day mean')
    
    # Zero line (gray)
    ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='--', linewidth=1)
    
    # Labels and title
    title = create_figure_title(gate_name, "Volume Transport Time Series", dataset, 
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_volume_transport_statistics(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    Generate Volume Transport statistics boxplot (monthly) as PNG bytes.
    Colors match Streamlit/Plotly charts.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(transport_sv[~np.isnan(transport_sv)])
    
    # Create monthly data
    df = pd.DataFrame({
        'transport': transport_sv,
        'month': time_pd.month
    })
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Boxplot per month
    monthly_data = [df[df['month'] == m]['transport'].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(monthly_data, labels=MONTH_ABBREV, patch_artist=True)
    
    # Color boxes with primary color (matching Streamlit)
    for patch in bp['boxes']:
        patch.set_facecolor(COLOR_PRIMARY)
        patch.set_alpha(0.6)
    
    # Zero line (red, matching Streamlit)
    ax.axhline(y=0, color=COLOR_SECONDARY, linestyle='--', linewidth=1)
    
    # Monthly means as line (red)
    monthly_means = [np.nanmean(d) if len(d) > 0 else np.nan for d in monthly_data]
    ax.plot(range(1, 13), monthly_means, 'o-', color=COLOR_SECONDARY, markersize=8, linewidth=2, label='Monthly Mean')
    
    title = create_figure_title(gate_name, "Volume Transport Monthly Statistics", dataset,
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', color=COLOR_GRID)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_monthly_profiles_grid(
    monthly_profiles: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    gate_name: str,
    plot_type: str,  # "Volume Transport" or "Salt Flux"
    dataset: str,
    start_year: int,
    end_year: int,
    n_observations: int,
    y_label: str = "Velocity (cm/s)",
    y_scale: float = 100.0,  # Multiplier for display (e.g., m/s to cm/s)
    show_regression: bool = True,
    dpi: int = 300
) -> bytes:
    """
    Generate 3x4 grid of monthly profiles with slope and R² annotations.
    
    Args:
        monthly_profiles: Dict month -> (bin_centers, bin_means, bin_stds)
        gate_name: Name of the gate
        plot_type: Type of plot for title
        dataset: Dataset name
        start_year, end_year: Time range
        n_observations: Total observations
        y_label: Y-axis label
        y_scale: Multiplier for y values
        show_regression: Whether to show regression line and stats
        dpi: Image resolution
    
    Returns:
        PNG image bytes
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    # Calculate global y-axis limits for consistency
    all_y = []
    for month in range(1, 13):
        if month in monthly_profiles:
            _, means, _ = monthly_profiles[month]
            if len(means) > 0:
                # Filter out NaN values before extending
                valid_means = means[np.isfinite(means)]
                if len(valid_means) > 0:
                    all_y.extend(valid_means * y_scale)
    
    if all_y and len(all_y) > 0:
        # Filter out any remaining NaN/Inf
        all_y_clean = [y for y in all_y if np.isfinite(y)]
        if all_y_clean:
            y_min = min(all_y_clean) * 1.2
            y_max = max(all_y_clean) * 1.2
            # Ensure zero is visible
            y_min = min(y_min, -abs(y_max) * 0.1) if np.isfinite(y_max) else -10
            y_max = max(y_max, abs(y_min) * 0.1) if np.isfinite(y_min) else 10
        else:
            y_min, y_max = -10, 10
    else:
        y_min, y_max = -10, 10
    
    # Final safety check
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        y_min, y_max = -10, 10
    
    for month in range(1, 13):
        ax = axes[month - 1]
        ax.set_facecolor('white')
        
        if month not in monthly_profiles:
            ax.set_title(f"{MONTH_NAMES[month-1]}", fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        bin_centers, bin_means, bin_stds = monthly_profiles[month]
        
        if len(bin_centers) == 0:
            ax.set_title(f"{MONTH_NAMES[month-1]}", fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Plot data
        y_values = bin_means * y_scale
        y_errors = bin_stds * y_scale
        
        # Bar chart with error bars (using Streamlit-matching colors)
        colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in y_values]
        ax.bar(bin_centers, y_values, width=bin_centers[1]-bin_centers[0] if len(bin_centers)>1 else 5,
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(bin_centers, y_values, yerr=y_errors, fmt='none', color='black', 
                    capsize=2, alpha=0.5)
        
        # Zero line
        ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='--', linewidth=1)
        
        # Linear regression
        if show_regression and len(bin_centers) > 2:
            valid = np.isfinite(y_values)
            if np.sum(valid) > 2:
                slope, intercept, r_value, _, _ = stats.linregress(
                    bin_centers[valid], y_values[valid]
                )
                x_fit = np.array([bin_centers.min(), bin_centers.max()])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'k--', linewidth=1.5, alpha=0.8)
                
                # Annotation
                ax.text(0.05, 0.95, f"slope={slope:.2e}\nR²={r_value**2:.3f}",
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f"{MONTH_NAMES[month-1]}", fontsize=10, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
        
        # Only show axis labels on edge plots
        if month > 8:  # Bottom row
            ax.set_xlabel('Distance (km)', fontsize=9)
        if month in [1, 5, 9]:  # Left column
            ax.set_ylabel(y_label, fontsize=9)
    
    # Main title
    fig.suptitle(f"{gate_name} - {plot_type} Monthly Profiles\n"
                 f"Dataset: {DATASET_FULL_NAMES.get(dataset, dataset)}\n"
                 f"Period: {start_year}-{end_year} | Observations: {n_observations:,}",
                 fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_bathymetry_profile(
    depth_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    gate_lon: Optional[np.ndarray] = None,
    gate_lat: Optional[np.ndarray] = None,
    depth_cap: Optional[float] = 250.0,
    dpi: int = 300
) -> bytes:
    """
    Generate bathymetry profile plot as PNG bytes.
    Colors and style match Streamlit/Plotly charts exactly.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot depth (negative = below sea level, matching Plotly)
    depth_negative = -np.abs(depth_profile)
    
    # Fill area for bathymetry (matching Plotly rgba(30, 58, 95, 0.4))
    ax.fill_between(x_km, 0, depth_negative, color=COLOR_PRIMARY, alpha=0.4)
    ax.plot(x_km, depth_negative, color=COLOR_PRIMARY, linewidth=2)
    
    # Sea surface (light blue, matching Plotly)
    ax.axhline(y=0, color=COLOR_SEA_LEVEL, linewidth=2, label='Sea Level')
    
    # Depth cap line (red, dashed - matching Plotly)
    if depth_cap:
        ax.axhline(y=-depth_cap, color=COLOR_SECONDARY, linewidth=2, linestyle='--',
                   label=f'Depth Cap: {depth_cap:.0f}m')
    
    # Labels
    ax.set_title(f"{gate_name} — Cross-Section Bathymetry\n"
                 f"Dataset: {DATASET_FULL_NAMES['gebco']}\n"
                 f"Gate Length: {x_km.max():.0f} km | Max Depth: {np.nanmax(depth_profile):.0f} m",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    ax.legend(loc='lower right')
    
    # Set y-axis range similar to Plotly
    ax.set_ylim(min(depth_negative.min() * 1.1, -depth_cap * 1.5 if depth_cap else depth_negative.min() * 1.1), 50)
    
    # Add coordinate info if available
    if gate_lon is not None and gate_lat is not None:
        ax.text(0.02, 0.02, f"Start: {gate_lon[0]:.2f}°E, {gate_lat[0]:.2f}°N\n"
                           f"End: {gate_lon[-1]:.2f}°E, {gate_lat[-1]:.2f}°N",
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


def export_salt_flux_timeseries(
    salt_flux_kg_s: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    Generate Salt Flux time series plot as PNG bytes.
    Colors match Streamlit/Plotly charts.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(salt_flux_kg_s[~np.isnan(salt_flux_kg_s)])
    
    # Convert to Gt/yr for display
    seconds_per_year = 365.25 * 24 * 3600
    salt_flux_gt_yr = salt_flux_kg_s * seconds_per_year / 1e12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    ax.plot(time_pd, salt_flux_gt_yr, color=COLOR_CMEMS_L4, alpha=0.7, linewidth=0.8)
    
    # Rolling mean (red)
    df = pd.DataFrame({'time': time_pd, 'flux': salt_flux_gt_yr})
    df = df.set_index('time')
    rolling_mean = df['flux'].rolling(window=30, center=True).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, color=COLOR_SECONDARY, linewidth=2, label='30-day mean')
    
    ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='--', linewidth=1)
    
    title = create_figure_title(gate_name, "Salt Flux Time Series", dataset,
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Salt Flux (Gt/yr equivalent)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return buf.getvalue()


# ==============================================================================
# EXPORT FUNCTIONS - Richieste specifiche utente
# ==============================================================================

def export_slope_timeline(
    slope_values: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    show_trend: bool = True,
    dpi: int = 300
) -> bytes:
    """
    📈 Slope Timeline - EXACT replica of Streamlit tab.
    Shows DOT slope time series with optional trend line.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    
    # Filter valid values
    valid_mask = ~np.isnan(slope_values)
    valid_time = time_pd[valid_mask]
    valid_slope = slope_values[valid_mask]
    
    # Convert to m/100km (same as Streamlit)
    y_vals = valid_slope  # already in m/100km
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot data with markers+lines (matching Plotly)
    ax.plot(valid_time, y_vals, 'o-', color=COLOR_CMEMS_L4, markersize=4, linewidth=1.5, label='DOT Slope')
    
    # Zero line (solid black, matching Plotly)
    ax.axhline(y=0, color='black', linewidth=0.8)
    
    # Trend line (if enabled)
    if show_trend and len(valid_slope) > 2:
        x_numeric = np.arange(len(valid_slope))
        z = np.polyfit(x_numeric, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(valid_time, p(x_numeric), '--', color=COLOR_TREND_LINE, linewidth=1.5,
                label=f'Trend ({z[0]:.4f}/step)')
    
    # Title matching Streamlit format
    ax.set_title(f"{DATASET_FULL_NAMES.get(dataset, dataset)} - {gate_name}\nDOT Slope Time Series",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Slope (m/100km)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_dot_profile_along_gate(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    start_year: int = None,
    end_year: int = None,
    dpi: int = 300
) -> bytes:
    """
    📊 DOT Profile Along Gate - EXACT replica of Streamlit tab.
    Shows mean DOT with ±std fill.
    """
    dot_mean = np.nanmean(dot_matrix, axis=1)
    dot_std = np.nanstd(dot_matrix, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot mean line with fill for std (matching Plotly style)
    ax.plot(x_km, dot_mean, color=COLOR_PRIMARY, linewidth=2, label='Mean DOT')
    ax.fill_between(x_km, dot_mean - dot_std, dot_mean + dot_std,
                   alpha=0.3, color=COLOR_PRIMARY, label='±1 std')
    
    # Linear regression (matching Plotly red dashed line)
    valid = ~np.isnan(dot_mean)
    if valid.sum() > 2:
        slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], dot_mean[valid])
        ax.plot(x_km, slope * x_km + intercept, color=COLOR_SECONDARY, linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Fit: slope={slope*1000:.2f} mm/km, R²={r_value**2:.3f}')
    
    title = f"{gate_name} - DOT Profile Along Gate\n{DATASET_FULL_NAMES.get(dataset, dataset)}"
    if start_year and end_year:
        title += f"\n{start_year}-{end_year}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('DOT (m)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_spatial_map(
    gate_lon: np.ndarray,
    gate_lat: np.ndarray,
    gate_name: str,
    gate_length_km: float = None,
    dpi: int = 300
) -> bytes:
    """
    🗺️ Spatial Map - Geographic map with coastlines, borders, gridlines.
    Zoomed on gate with geographic context around it.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        
        # Calculate bounds with buffer (~2° around gate for context)
        lon_range = gate_lon.max() - gate_lon.min()
        lat_range = gate_lat.max() - gate_lat.min()
        buffer_lon = max(2.0, lon_range * 0.5)  # At least 2° buffer
        buffer_lat = max(2.0, lat_range * 0.5)
        
        lon_min = gate_lon.min() - buffer_lon
        lon_max = gate_lon.max() + buffer_lon
        lat_min = gate_lat.min() - buffer_lat
        lat_max = gate_lat.max() + buffer_lat
        
        # Projection
        central_lon = (gate_lon.min() + gate_lon.max()) / 2
        central_lat = (gate_lat.min() + gate_lat.max()) / 2
        
        # Use appropriate projection based on latitude
        if central_lat > 70:
            proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        elif central_lat < -70:
            proj = ccrs.SouthPolarStereo(central_longitude=central_lon)
        else:
            proj = ccrs.PlateCarree()  # Simple for mid-latitudes
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi, subplot_kw={'projection': proj})
        fig.patch.set_facecolor('white')
        
        # Set extent
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add features
        ax.add_feature(cfeature.LAND, facecolor='#f5f5dc', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#e6f3ff')
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Gridlines with labels on borders
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = True
        gl.right_labels = True
        gl.bottom_labels = True
        gl.left_labels = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        
        # Plot gate line (thick red)
        ax.plot(gate_lon, gate_lat, 'r-', linewidth=4, transform=ccrs.PlateCarree(),
               label=gate_name, zorder=10)
        
        # Start marker (green circle)
        ax.scatter(gate_lon[0], gate_lat[0], c='lime', s=200, marker='o', edgecolor='black', linewidth=2,
                  transform=ccrs.PlateCarree(), label=f'Start ({gate_lon[0]:.2f}°, {gate_lat[0]:.2f}°)', zorder=11)
        
        # End marker (red square)
        ax.scatter(gate_lon[-1], gate_lat[-1], c='red', s=200, marker='s', edgecolor='black', linewidth=2,
                  transform=ccrs.PlateCarree(), label=f'End ({gate_lon[-1]:.2f}°, {gate_lat[-1]:.2f}°)', zorder=11)
        
        # Title with gate info
        title_lines = [f'{gate_name} - Geographic Location']
        if gate_length_km:
            title_lines.append(f'Gate Length: {gate_length_km:.0f} km')
        ax.set_title('\n'.join(title_lines), fontsize=14, fontweight='bold')
        
        ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
        
    except ImportError:
        # Fallback senza cartopy
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi)
        ax.plot(gate_lon, gate_lat, 'r-', linewidth=3, label=gate_name)
        ax.scatter(gate_lon[0], gate_lat[0], c='lime', s=150, marker='o', edgecolor='black', label='Start')
        ax.scatter(gate_lon[-1], gate_lat[-1], c='red', s=150, marker='s', edgecolor='black', label='End')
        ax.set_xlabel('Longitude (°)', fontsize=10)
        ax.set_ylabel('Latitude (°)', fontsize=10)
        ax.set_title(f'{gate_name} - Geographic Location', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_monthly_dot_profiles_grid(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 Monthly DOT Analysis (3×4) - 12 plot mensili con slope (mm/km) e R².
    Colors match Streamlit/Plotly charts.
    """
    time_pd = pd.to_datetime(time_array)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for month_idx in range(12):
        ax = axes[month_idx]
        ax.set_facecolor('white')
        month_mask = time_pd.month == (month_idx + 1)
        
        if month_mask.sum() > 0:
            dot_month = dot_matrix[:, month_mask]
            dot_mean = np.nanmean(dot_month, axis=1)
            dot_std = np.nanstd(dot_month, axis=1)
            
            # Plot mean with std (primary color)
            ax.plot(x_km, dot_mean, color=COLOR_PRIMARY, linewidth=1.5)
            ax.fill_between(x_km, dot_mean - dot_std, dot_mean + dot_std, alpha=0.3, color=COLOR_PRIMARY)
            
            # Linear regression (secondary color = red)
            valid = ~np.isnan(dot_mean)
            if valid.sum() > 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], dot_mean[valid])
                slope_mm_km = slope * 1000  # mm/km
                ax.plot(x_km, slope * x_km + intercept, color=COLOR_SECONDARY, linestyle='--', linewidth=1, alpha=0.7)
                ax.text(0.05, 0.95, f'slope={slope_mm_km:.2f} mm/km\nR²={r_value**2:.3f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(MONTH_ABBREV[month_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (km)' if month_idx >= 8 else '', fontsize=8)
        ax.set_ylabel('DOT (m)' if month_idx % 4 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    fig.suptitle(f'{gate_name} - Monthly DOT Profiles [{dataset.upper()}]',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_monthly_velocity_profiles_grid(
    v_perp: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 Velocity Profile Along Gate (12 plot mensili) con slope e R².
    Colori consistenti con Streamlit (PRIMARY = #1E3A5F).
    """
    time_pd = pd.to_datetime(time_array)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for month_idx in range(12):
        ax = axes[month_idx]
        ax.set_facecolor('white')
        month_mask = time_pd.month == (month_idx + 1)
        
        if month_mask.sum() > 0:
            v_month = v_perp[:, month_mask]
            v_mean = np.nanmean(v_month, axis=1) * 100  # cm/s
            v_std = np.nanstd(v_month, axis=1) * 100
            
            # Color based on mean direction (matching Streamlit)
            color = COLOR_POSITIVE if np.nanmean(v_mean) >= 0 else COLOR_NEGATIVE
            
            ax.plot(x_km, v_mean, color=color, linewidth=1.5)
            ax.fill_between(x_km, v_mean - v_std, v_mean + v_std, alpha=0.3, color=color)
            
            # Linear regression (dark line)
            valid = ~np.isnan(v_mean)
            if valid.sum() > 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], v_mean[valid])
                ax.plot(x_km, slope * x_km + intercept, 'k--', linewidth=1, alpha=0.7)
                ax.text(0.05, 0.95, f'slope={slope:.4f}\nR²={r_value**2:.3f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='-', linewidth=0.5)
        ax.set_title(MONTH_ABBREV[month_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (km)' if month_idx >= 8 else '', fontsize=8)
        ax.set_ylabel('v (cm/s)' if month_idx % 4 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    fig.suptitle(f'{gate_name} - Monthly Velocity Profiles [{dataset.upper()}]',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_velocity_comparison_timeseries(
    v_perp: np.ndarray,
    v_geo: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    gate_lon: np.ndarray = None,
    gate_lat: np.ndarray = None,
    dpi: int = 300
) -> bytes:
    """
    📈 Time Series: v_perp vs v_geo - EXACT replica of Streamlit tab.
    NO 30-day mean - just raw data as in the tab.
    v_perp = from ugos/vgos, v_geo = from DOT slope
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(time_array)
    
    # Mean along gate for each time step (matching Streamlit)
    v_perp_mean = np.nanmean(v_perp, axis=0) * 100  # cm/s
    v_geo_ts = v_geo * 100 if v_geo is not None else None  # Already 1D (time,)
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # v_perp line (dark blue, matching Streamlit #1E3A5F)
    ax.plot(time_pd, v_perp_mean, color=COLOR_PRIMARY, linewidth=1.5, label='v_perp (ugos/vgos)')
    
    # v_geo line (orange-red, matching Streamlit #E07B53)
    if v_geo_ts is not None and not np.all(np.isnan(v_geo_ts)):
        ax.plot(time_pd, v_geo_ts, color='#E07B53', linewidth=1.5, label='v_geo (DOT slope)')
    
    # Zero line (gray dashed, matching Streamlit)
    ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='--', linewidth=1)
    
    # Build detailed title
    title_lines = [f"{gate_name} - Geostrophic Velocity Comparison"]
    title_lines.append(f"{DATASET_FULL_NAMES.get(dataset, dataset)}")
    title_lines.append(f"{start_year}-{end_year} | N={n_obs:,}")
    
    ax.set_title('\n'.join(title_lines), fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Velocity (cm/s)', fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor=COLOR_GRID)
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # Add stats annotation
    v_perp_mean_val = np.nanmean(v_perp_mean)
    info_text = [f"v_perp mean: {v_perp_mean_val:.2f} cm/s"]
    if v_geo_ts is not None and not np.all(np.isnan(v_geo_ts)):
        v_geo_mean_val = np.nanmean(v_geo_ts)
        info_text.append(f"v_geo mean: {v_geo_mean_val:.2f} cm/s")
    
    if gate_lon is not None and gate_lat is not None:
        info_text.append(f"Gate: ({gate_lon[0]:.1f}°,{gate_lat[0]:.1f}°) → ({gate_lon[-1]:.1f}°,{gate_lat[-1]:.1f}°)")
    
    ax.text(0.02, 0.02, '\n'.join(info_text), transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_total_transport_timeseries(
    transport_sv: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    gate_lon: np.ndarray = None,
    gate_lat: np.ndarray = None,
    gate_length_km: float = None,
    dpi: int = 300
) -> bytes:
    """
    📈 Total Transport Time Series - Clean line plot WITHOUT fill or rolling mean.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(transport_sv[~np.isnan(transport_sv)])
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Transport line ONLY (dark blue, NO fill, NO rolling mean)
    ax.plot(time_pd, transport_sv, color=COLOR_PRIMARY, linewidth=1.2, label='Total Transport')
    
    # Zero line (light blue, solid)
    ax.axhline(y=0, color=COLOR_SEA_LEVEL, linewidth=1.5)
    
    # Mean line (green dashed with value annotation)
    mean_transport = np.nanmean(transport_sv)
    std_transport = np.nanstd(transport_sv)
    ax.axhline(y=mean_transport, color='green', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_transport:.2f} ± {std_transport:.2f} Sv')
    
    # Build detailed title
    title_lines = [f"{gate_name} - Total Volume Transport"]
    title_lines.append(f"{DATASET_FULL_NAMES.get(dataset, dataset)}")
    title_lines.append(f"{start_year}-{end_year} | N={n_obs:,}")
    
    ax.set_title('\n'.join(title_lines), fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor=COLOR_GRID)
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
    # Add coordinate info if available
    info_text = []
    if gate_lon is not None and gate_lat is not None:
        info_text.append(f"Start: ({gate_lon[0]:.2f}°, {gate_lat[0]:.2f}°)")
        info_text.append(f"End: ({gate_lon[-1]:.2f}°, {gate_lat[-1]:.2f}°)")
    if gate_length_km:
        info_text.append(f"Gate length: {gate_length_km:.0f} km")
    
    if info_text:
        ax.text(0.02, 0.02, '\n'.join(info_text), transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_monthly_transport_profiles_grid(
    monthly_profiles: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    x_km: np.ndarray,
    gate_lon: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 Volume Transport Along Gate - 12 monthly plots (3×4 grid).
    EXACT replica of Streamlit tab style with bars colored by sign.
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for month_idx in range(12):
        ax = axes[month_idx]
        ax.set_facecolor('white')
        month = month_idx + 1
        
        if month in monthly_profiles:
            bin_centers, bin_means, bin_stds = monthly_profiles[month]
            
            if len(bin_centers) > 0:
                # Bar colors: blue positive, red negative (matching Streamlit #3498DB / #E74C3C)
                colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in bin_means]
                
                # Bar width
                bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 5
                
                ax.bar(bin_centers, bin_means, width=bar_width * 0.8, color=colors,
                       edgecolor='black', linewidth=0.3, alpha=0.8)
                
                # Error bars
                ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none',
                           color='black', capsize=2, alpha=0.5, linewidth=0.5)
                
                # Zero line
                ax.axhline(y=0, color=COLOR_ZERO_LINE, linewidth=0.8)
                
                # Total for this month
                total_sv = np.nansum(bin_means)
                ax.text(0.95, 0.95, f'Σ={total_sv:.3f}', transform=ax.transAxes,
                       fontsize=8, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(MONTH_NAMES[month_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (km)' if month_idx >= 8 else '', fontsize=8)
        ax.set_ylabel('Transport (×10⁶ m³/s)' if month_idx % 4 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    fig.suptitle(f'{gate_name} — Monthly Volume Transport Along Gate',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_single_velocity_profile(
    bin_centers: np.ndarray,
    bin_means: np.ndarray,
    bin_stds: np.ndarray,
    x_km: np.ndarray,
    gate_lon: np.ndarray,
    gate_name: str,
    month_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 Single Velocity Profile Along Gate for one month.
    EXACT replica of Streamlit tab style.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Interpolate longitude for bin centers
    bin_lon = np.interp(bin_centers, x_km, gate_lon)
    
    # v_perp profile (cm/s) - matching Streamlit style
    ax.plot(bin_centers, bin_means * 100, 'o-', color=COLOR_PRIMARY, linewidth=2.5,
            markersize=7, label='v_perp (ugos/vgos)')
    
    # Error fill
    ax.fill_between(bin_centers, 
                    (bin_means - bin_stds) * 100, 
                    (bin_means + bin_stds) * 100,
                    alpha=0.3, color=COLOR_PRIMARY)
    
    # Zero line
    ax.axhline(y=0, color=COLOR_ZERO_LINE, linestyle='--', linewidth=1)
    
    # Mean velocity for this month
    mean_v = np.nanmean(bin_means) * 100
    
    ax.set_title(f"Perpendicular Velocity Along Gate — {month_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Velocity (cm/s)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    # Add mean annotation
    ax.text(0.02, 0.98, f'Mean v_perp: {mean_v:.2f} cm/s', transform=ax.transAxes,
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_single_transport_profile(
    bin_centers: np.ndarray,
    bin_means: np.ndarray,
    bin_stds: np.ndarray,
    x_km: np.ndarray,
    gate_lon: np.ndarray,
    gate_name: str,
    month_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📊 Single Volume Transport Along Gate for one month.
    EXACT replica of Streamlit tab style with colored bars.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Bar colors: blue positive, red negative (matching Streamlit #3498DB / #E74C3C)
    colors = ['#3498DB' if v >= 0 else '#E74C3C' for v in bin_means]
    
    # Bar width
    bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 5
    
    ax.bar(bin_centers, bin_means, width=bar_width * 0.85, color=colors,
           edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Error bars (matplotlib format: tuple for color with alpha)
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='none',
               color=(0, 0, 0, 0.5), capsize=3, linewidth=1)
    
    # Zero line
    ax.axhline(y=0, color=COLOR_ZERO_LINE, linewidth=1)
    
    # Total transport for this month
    total_sv = np.nansum(bin_means)
    total_m3s = total_sv * 1e6
    
    ax.set_title(f"Transport Along Gate — {month_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Transport (×10⁶ m³/s)', fontsize=10)
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    # Add total annotation
    ax.text(0.02, 0.98, f'{month_name} Total: {total_m3s:.2e} m³/s ({total_sv:.3f} ×10⁶ m³/s)',
            transform=ax.transAxes, fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_bathymetry_profile_clean(
    depth_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    gate_lon: np.ndarray = None,
    gate_lat: np.ndarray = None,
    depth_cap: Optional[float] = 250.0,
    dpi: int = 300
) -> bytes:
    """
    🏔️ Bathymetry Profile (clean style) - Matching Streamlit/Plotly.
    Zero at top, depth going down (negative y).
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Depth as negative values (matching Plotly style where 0 is sea level)
    depth_negative = -np.abs(depth_profile)
    
    # Fill area for bathymetry (matching Plotly rgba(30, 58, 95, 0.4))
    ax.fill_between(x_km, 0, depth_negative, color=COLOR_PRIMARY, alpha=0.4)
    ax.plot(x_km, depth_negative, color=COLOR_PRIMARY, linewidth=2)
    
    # Sea level at y=0 (light blue, matching Plotly)
    ax.axhline(y=0, color=COLOR_SEA_LEVEL, linewidth=2, label='Sea Level')
    
    # Depth cap line (red, dashed - matching Plotly)
    if depth_cap:
        ax.axhline(y=-depth_cap, color=COLOR_SECONDARY, linewidth=2, linestyle='--',
                   label=f'Depth Cap: {depth_cap:.0f}m')
    
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    
    # Set y-axis range similar to Plotly
    ax.set_ylim(min(depth_negative.min() * 1.1, -depth_cap * 1.5 if depth_cap else depth_negative.min() * 1.1), 50)
    
    title = f"{gate_name} — Cross-Section Bathymetry\n{DATASET_FULL_NAMES['gebco']}"
    if gate_lon is not None and gate_lat is not None:
        title += f"\nStart: ({gate_lon[0]:.2f}°, {gate_lat[0]:.2f}°) → End: ({gate_lon[-1]:.2f}°, {gate_lat[-1]:.2f}°)"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, color=COLOR_GRID)
    
    # Stats annotation
    max_depth = np.nanmax(np.abs(depth_profile))
    mean_depth = np.nanmean(np.abs(depth_profile))
    ax.text(0.02, 0.02, f'Max: {max_depth:.0f} m\nMean: {mean_depth:.0f} m\nLength: {x_km.max():.0f} km',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_salinity_density_along_gate(
    salinity_profile: np.ndarray,
    density_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    dpi: int = 300
) -> bytes:
    """
    🌡️ Salinity & Density Along Gate - Due pannelli.
    Colors match Streamlit/Plotly charts.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=dpi, sharex=True)
    fig.patch.set_facecolor('white')
    
    # Salinity (using DTU green)
    ax1.set_facecolor('white')
    ax1.plot(x_km, salinity_profile, color=COLOR_DTU, linewidth=2)
    ax1.fill_between(x_km, salinity_profile.min() * 0.99, salinity_profile, 
                    alpha=0.3, color=COLOR_DTU)
    ax1.set_ylabel('Salinity (PSU)', fontsize=10)
    ax1.set_title(f'{gate_name} - Salinity Along Gate', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, color=COLOR_GRID)
    
    mean_sal = np.nanmean(salinity_profile)
    ax1.axhline(y=mean_sal, color='darkgreen', linestyle='--', 
               label=f'Mean: {mean_sal:.2f} PSU')
    ax1.legend(loc='best')
    
    # Density (using CMEMS L4 color)
    ax2.set_facecolor('white')
    ax2.plot(x_km, density_profile, color=COLOR_CMEMS_L4, linewidth=2)
    ax2.fill_between(x_km, density_profile.min() * 0.999, density_profile,
                    alpha=0.3, color=COLOR_CMEMS_L4)
    ax2.set_xlabel('Distance along gate (km)', fontsize=10)
    ax2.set_ylabel('Density (kg/m³)', fontsize=10)
    ax2.set_title(f'{gate_name} - Density Along Gate', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, color=COLOR_GRID)
    
    mean_dens = np.nanmean(density_profile)
    ax2.axhline(y=mean_dens, color='darkviolet', linestyle='--',
               label=f'Mean: {mean_dens:.2f} kg/m³')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ==============================================================================
# ZIP ARCHIVE CREATION
# ==============================================================================

def create_export_zip(
    files: Dict[str, bytes],
    base_folder: str = "export"
) -> bytes:
    """
    Create a ZIP archive from a dictionary of files.
    
    Args:
        files: Dict mapping file paths (relative) to file contents (bytes)
               e.g., {"volume_transport/davis_strait_timeseries.png": bytes_data}
        base_folder: Base folder name in the ZIP
        
    Returns:
        ZIP file as bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path, file_content in files.items():
            full_path = f"{base_folder}/{file_path}"
            
            # Handle both bytes and string content
            if isinstance(file_content, str):
                zf.writestr(full_path, file_content.encode('utf-8'))
            else:
                zf.writestr(full_path, file_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def generate_full_export(
    gate_data: Dict[str, Any],
    include_images: bool = True,
    include_csv: bool = True,
    dpi: int = 300,
    # Checkbox options per selezionare quali export includere
    export_options: Dict[str, bool] = None
) -> bytes:
    """
    Generate complete export ZIP with all images and CSV files.
    
    Args:
        gate_data: Dictionary containing all gate data
        include_images: Whether to include PNG images
        include_csv: Whether to include CSV files
        dpi: Image resolution (default 300)
        export_options: Dict of export options (checkboxes)
            - slope_timeline: 📈 Slope Timeline
            - dot_profile: 📊 DOT Profile Along Gate
            - spatial_map: 🗺️ Spatial Map (geography)
            - monthly_dot: 📊 Monthly DOT Analysis (3×4)
            - monthly_velocity: 📊 Monthly Velocity Profiles (3×4)
            - velocity_comparison: 📈 v_perp vs v_geo
            - total_transport: 📈 Total Transport Time Series
            - bathymetry: 🏔️ Bathymetry Profile
            - salinity_density: 🌡️ Salinity & Density
            - volume_transport_stats: 📊 Volume Transport Statistics (monthly)
        
    Returns:
        ZIP file bytes
    """
    # Default: all exports enabled
    if export_options is None:
        export_options = {
            'slope_timeline': True,
            'dot_profile': True,
            'spatial_map': True,
            'monthly_dot': True,
            'monthly_velocity': True,
            'velocity_comparison': True,
            'total_transport': True,
            'bathymetry': True,
            'salinity_density': True,
            'volume_transport_stats': True,
        }
    
    files = {}
    
    gate_name = gate_data['gate_name']
    gate_name_safe = gate_name.lower().replace(' ', '_').replace('-', '_')
    dataset = gate_data.get('dataset', 'cmems_l4')
    time_array = gate_data['time_array']
    
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    
    logger.info(f"Generating export for {gate_name} ({start_year}-{end_year})")
    
    # Extract data
    v_perp = gate_data.get('v_perp')
    v_geo = gate_data.get('v_geo')
    x_km = gate_data.get('x_km')
    dot_matrix = gate_data.get('dot_matrix')
    transport_sv = gate_data.get('transport_sv')
    depth_profile = gate_data.get('depth_profile')
    gate_lon = gate_data.get('gate_lon')
    gate_lat = gate_data.get('gate_lat')
    salinity_profile = gate_data.get('salinity_profile')
    density_profile = gate_data.get('density_profile')
    slope_values = gate_data.get('slope_values')
    
    # =========================================================================
    # CSV FILES
    # =========================================================================
    if include_csv:
        if transport_sv is not None:
            # Raw monthly data
            df_raw = generate_volume_transport_raw_csv(
                transport_sv, time_array, gate_name, dataset
            )
            files[f"csv/{gate_name_safe}_volume_transport_raw.csv"] = df_raw.to_csv(index=False)
            
            # Monthly climatology
            df_clim = generate_volume_transport_climatology_csv(
                transport_sv, time_array, gate_name, dataset
            )
            files[f"csv/{gate_name_safe}_volume_transport_climatology.csv"] = df_clim.to_csv(index=False)
            
            # Annual statistics
            df_annual = generate_volume_transport_annual_csv(
                transport_sv, time_array, gate_name, dataset
            )
            files[f"csv/{gate_name_safe}_volume_transport_annual.csv"] = df_annual.to_csv(index=False)
        
        # Salt flux CSV
        salt_flux = gate_data.get('salt_flux_kg_s')
        if salt_flux is not None:
            df_salt = generate_salt_flux_raw_csv(
                salt_flux, time_array, gate_name
            )
            files[f"csv/{gate_name_safe}_salt_flux_raw.csv"] = df_salt.to_csv(index=False)
    
    # =========================================================================
    # IMAGE FILES (based on export_options)
    # =========================================================================
    if include_images:
        
        # 📈 Slope Timeline
        if export_options.get('slope_timeline', True) and slope_values is not None:
            try:
                img = export_slope_timeline(slope_values, time_array, gate_name, dataset, dpi)
                files[f"dot_analysis/{gate_name_safe}_slope_timeline.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export slope timeline: {e}")
        
        # 📊 DOT Profile Along Gate
        if export_options.get('dot_profile', True) and dot_matrix is not None and x_km is not None:
            try:
                img = export_dot_profile_along_gate(
                    dot_matrix, x_km, gate_name, dataset, start_year, end_year, dpi
                )
                files[f"dot_analysis/{gate_name_safe}_dot_profile.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export DOT profile: {e}")
        
        # 🗺️ Spatial Map (with geography)
        if export_options.get('spatial_map', True) and gate_lon is not None and gate_lat is not None:
            try:
                img = export_spatial_map(gate_lon, gate_lat, gate_name, dpi)
                files[f"spatial/{gate_name_safe}_geographic_map.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export spatial map: {e}")
        
        # 📊 Monthly DOT Analysis (3×4) with slope mm/km and R²
        if export_options.get('monthly_dot', True) and dot_matrix is not None and x_km is not None:
            try:
                img = export_monthly_dot_profiles_grid(
                    dot_matrix, x_km, time_array, gate_name, dataset, dpi
                )
                files[f"monthly_analysis/{gate_name_safe}_monthly_dot_grid.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export monthly DOT grid: {e}")
        
        # 📊 Monthly Velocity Profiles (3×4) with slope and R²
        if export_options.get('monthly_velocity', True) and v_perp is not None and x_km is not None:
            try:
                img = export_monthly_velocity_profiles_grid(
                    v_perp, x_km, time_array, gate_name, dataset, dpi
                )
                files[f"velocity/{gate_name_safe}_monthly_velocity_grid.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export monthly velocity grid: {e}")
        
        # 📈 v_perp vs v_geo Time Series
        if export_options.get('velocity_comparison', True) and v_perp is not None:
            try:
                img = export_velocity_comparison_timeseries(
                    v_perp, v_geo, time_array, gate_name, dataset, dpi
                )
                files[f"velocity/{gate_name_safe}_velocity_comparison.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export velocity comparison: {e}")
        
        # 📈 Total Transport Time Series
        if export_options.get('total_transport', True) and transport_sv is not None:
            try:
                img = export_total_transport_timeseries(
                    transport_sv, time_array, gate_name, dataset, dpi
                )
                files[f"volume_transport/{gate_name_safe}_total_transport_timeseries.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export transport timeseries: {e}")
        
        # 🏔️ Bathymetry Profile (clean, zero at top)
        if export_options.get('bathymetry', True) and depth_profile is not None and x_km is not None:
            try:
                img = export_bathymetry_profile_clean(
                    depth_profile, x_km, gate_name, gate_lon, gate_lat, dpi
                )
                files[f"bathymetry/{gate_name_safe}_bathymetry.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export bathymetry: {e}")
        
        # 🌡️ Salinity & Density Along Gate
        if export_options.get('salinity_density', True):
            if salinity_profile is not None and density_profile is not None and x_km is not None:
                try:
                    img = export_salinity_density_along_gate(
                        salinity_profile, density_profile, x_km, gate_name, dpi
                    )
                    files[f"salt_flux/{gate_name_safe}_salinity_density.png"] = img
                except Exception as e:
                    logger.warning(f"Failed to export salinity/density: {e}")
        
        # 📊 Volume Transport Monthly Statistics (boxplot)
        if export_options.get('volume_transport_stats', True) and transport_sv is not None:
            try:
                img = export_volume_transport_statistics(
                    transport_sv, time_array, gate_name, dataset, dpi
                )
                files[f"volume_transport/{gate_name_safe}_monthly_statistics.png"] = img
            except Exception as e:
                logger.warning(f"Failed to export transport statistics: {e}")
    
    # Create timestamp for folder name
    timestamp = datetime.now().strftime("%Y-%m-%d")
    base_folder = f"export_{gate_name_safe}_{start_year}-{end_year}_{timestamp}"
    
    return create_export_zip(files, base_folder)


def generate_multi_gate_export(
    gates_data: List[Dict[str, Any]],
    include_images: bool = True,
    include_csv: bool = True,
    dpi: int = 300,
    export_options: Dict[str, bool] = None
) -> bytes:
    """
    Generate export ZIP for multiple gates.
    Uses same export options as generate_full_export.
    """
    all_files = {}
    
    for gate_data in gates_data:
        gate_name = gate_data['gate_name']
        gate_name_safe = gate_name.lower().replace(' ', '_').replace('-', '_')
        
        dataset = gate_data.get('dataset', 'cmems_l4')
        time_array = gate_data['time_array']
        time_pd = pd.to_datetime(time_array)
        start_year = time_pd.min().year
        end_year = time_pd.max().year
        
        # Extract data
        v_perp = gate_data.get('v_perp')
        v_geo = gate_data.get('v_geo')
        x_km = gate_data.get('x_km')
        dot_matrix = gate_data.get('dot_matrix')
        transport_sv = gate_data.get('transport_sv')
        depth_profile = gate_data.get('depth_profile')
        gate_lon = gate_data.get('gate_lon')
        gate_lat = gate_data.get('gate_lat')
        salinity_profile = gate_data.get('salinity_profile')
        density_profile = gate_data.get('density_profile')
        slope_values = gate_data.get('slope_values')
        
        # Default options
        if export_options is None:
            export_options = {
                'slope_timeline': True, 'dot_profile': True, 'spatial_map': True,
                'monthly_dot': True, 'monthly_velocity': True, 'velocity_comparison': True,
                'total_transport': True, 'bathymetry': True, 'salinity_density': True,
                'volume_transport_stats': True,
            }
        
        # CSV
        if include_csv:
            if transport_sv is not None:
                df_raw = generate_volume_transport_raw_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_raw.csv"] = df_raw.to_csv(index=False)
                
                df_clim = generate_volume_transport_climatology_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_climatology.csv"] = df_clim.to_csv(index=False)
                
                df_annual = generate_volume_transport_annual_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_annual.csv"] = df_annual.to_csv(index=False)
        
        # Images
        if include_images:
            # Apply same export logic as generate_full_export
            if export_options.get('spatial_map', True) and gate_lon is not None and gate_lat is not None:
                try:
                    img = export_spatial_map(gate_lon, gate_lat, gate_name, dpi)
                    all_files[f"spatial/{gate_name_safe}_geographic_map.png"] = img
                except Exception as e:
                    logger.warning(f"Failed spatial map for {gate_name}: {e}")
            
            if export_options.get('monthly_velocity', True) and v_perp is not None and x_km is not None:
                try:
                    img = export_monthly_velocity_profiles_grid(v_perp, x_km, time_array, gate_name, dataset, dpi)
                    all_files[f"velocity/{gate_name_safe}_monthly_velocity_grid.png"] = img
                except Exception as e:
                    logger.warning(f"Failed monthly velocity for {gate_name}: {e}")
            
            if export_options.get('velocity_comparison', True) and v_perp is not None:
                try:
                    img = export_velocity_comparison_timeseries(v_perp, v_geo, time_array, gate_name, dataset, dpi)
                    all_files[f"velocity/{gate_name_safe}_velocity_comparison.png"] = img
                except Exception as e:
                    logger.warning(f"Failed velocity comparison for {gate_name}: {e}")
            
            if export_options.get('total_transport', True) and transport_sv is not None:
                try:
                    img = export_total_transport_timeseries(transport_sv, time_array, gate_name, dataset, dpi)
                    all_files[f"volume_transport/{gate_name_safe}_total_transport_timeseries.png"] = img
                except Exception as e:
                    logger.warning(f"Failed transport timeseries for {gate_name}: {e}")
            
            if export_options.get('bathymetry', True) and depth_profile is not None and x_km is not None:
                try:
                    img = export_bathymetry_profile_clean(depth_profile, x_km, gate_name, gate_lon, gate_lat, dpi)
                    all_files[f"bathymetry/{gate_name_safe}_bathymetry.png"] = img
                except Exception as e:
                    logger.warning(f"Failed bathymetry for {gate_name}: {e}")
            
            if export_options.get('monthly_dot', True) and dot_matrix is not None and x_km is not None:
                try:
                    img = export_monthly_dot_profiles_grid(dot_matrix, x_km, time_array, gate_name, dataset, dpi)
                    all_files[f"monthly_analysis/{gate_name_safe}_monthly_dot_grid.png"] = img
                except Exception as e:
                    logger.warning(f"Failed monthly DOT for {gate_name}: {e}")
            
            if export_options.get('dot_profile', True) and dot_matrix is not None and x_km is not None:
                try:
                    img = export_dot_profile_along_gate(dot_matrix, x_km, gate_name, dataset, start_year, end_year, dpi)
                    all_files[f"dot_analysis/{gate_name_safe}_dot_profile.png"] = img
                except Exception as e:
                    logger.warning(f"Failed DOT profile for {gate_name}: {e}")
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    n_gates = len(gates_data)
    base_folder = f"arctic_gates_export_{n_gates}gates_{timestamp}"
    
    return create_export_zip(all_files, base_folder)
