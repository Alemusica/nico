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
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(transport_sv[~np.isnan(transport_sv)])
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    # Plot time series
    ax.plot(time_pd, transport_sv, 'b-', alpha=0.7, linewidth=0.8)
    
    # Add rolling mean
    df = pd.DataFrame({'time': time_pd, 'transport': transport_sv})
    df = df.set_index('time')
    rolling_mean = df['transport'].rolling(window=30, center=True).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label='30-day mean')
    
    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Labels and title
    title = create_figure_title(gate_name, "Volume Transport Time Series", dataset, 
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
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
    
    # Boxplot per month
    monthly_data = [df[df['month'] == m]['transport'].dropna().values for m in range(1, 13)]
    bp = ax.boxplot(monthly_data, labels=MONTH_ABBREV, patch_artist=True)
    
    # Color boxes
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.6)
    
    # Zero line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Monthly means as line
    monthly_means = [np.nanmean(d) if len(d) > 0 else np.nan for d in monthly_data]
    ax.plot(range(1, 13), monthly_means, 'ro-', markersize=8, linewidth=2, label='Monthly Mean')
    
    title = create_figure_title(gate_name, "Volume Transport Monthly Statistics", dataset,
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
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
        
        # Bar chart with error bars
        colors = ['mediumpurple' if v >= 0 else 'indianred' for v in y_values]
        ax.bar(bin_centers, y_values, width=bin_centers[1]-bin_centers[0] if len(bin_centers)>1 else 5,
               color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.errorbar(bin_centers, y_values, yerr=y_errors, fmt='none', color='black', 
                    capsize=2, alpha=0.5)
        
        # Zero line
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        
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
        ax.grid(True, alpha=0.3)
        
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


def export_velocity_hovmoller(
    v_perp: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    Generate Hovmöller diagram (time vs distance) as PNG bytes.
    
    Args:
        v_perp: Perpendicular velocity (m/s), shape (n_pts, n_time)
        x_km: Distance along gate (km)
        time_array: Time values
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = v_perp.shape[1]
    
    fig, ax = plt.subplots(figsize=(14, 8), dpi=dpi)
    
    # Convert to cm/s for display
    v_cm_s = v_perp * 100
    
    # Create mesh grid
    time_num = mdates.date2num(time_pd)
    X, Y = np.meshgrid(time_num, x_km)
    
    # Plot
    vmax = np.nanpercentile(np.abs(v_cm_s), 98)
    pcm = ax.pcolormesh(X, Y, v_cm_s, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
    
    # Colorbar
    cbar = plt.colorbar(pcm, ax=ax, label='Perpendicular Velocity (cm/s)')
    
    # Labels
    title = create_figure_title(gate_name, "Velocity Hovmöller Diagram", dataset,
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Distance along gate (km)', fontsize=10)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
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
    dpi: int = 300
) -> bytes:
    """
    Generate bathymetry profile plot as PNG bytes.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    
    # Plot depth (positive downward)
    ax.fill_between(x_km, 0, -depth_profile, color='saddlebrown', alpha=0.6)
    ax.plot(x_km, -depth_profile, 'k-', linewidth=1.5)
    
    # Sea surface
    ax.axhline(y=0, color='steelblue', linewidth=2, label='Sea Surface')
    ax.fill_between(x_km, 0, 50, color='lightblue', alpha=0.3)
    
    # Labels
    ax.set_title(f"{gate_name} - Bathymetry Profile\n"
                 f"Dataset: {DATASET_FULL_NAMES['gebco']}\n"
                 f"Gate Length: {x_km.max():.0f} km | Max Depth: {np.nanmax(depth_profile):.0f} m",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
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
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(salt_flux_kg_s[~np.isnan(salt_flux_kg_s)])
    
    # Convert to Gt/yr for display
    seconds_per_year = 365.25 * 24 * 3600
    salt_flux_gt_yr = salt_flux_kg_s * seconds_per_year / 1e12
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    ax.plot(time_pd, salt_flux_gt_yr, 'purple', alpha=0.7, linewidth=0.8)
    
    # Rolling mean
    df = pd.DataFrame({'time': time_pd, 'flux': salt_flux_gt_yr})
    df = df.set_index('time')
    rolling_mean = df['flux'].rolling(window=30, center=True).mean()
    ax.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label='30-day mean')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    title = create_figure_title(gate_name, "Salt Flux Time Series", dataset,
                                start_year, end_year, n_obs)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Salt Flux (Gt/yr equivalent)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
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
# NEW EXPORT FUNCTIONS (2026-01-26)
# ==============================================================================

def export_mean_dot_profile(
    profile_mean: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    profile_std: Optional[np.ndarray] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    n_obs: Optional[int] = None,
    dpi: int = 300
) -> bytes:
    """Generate Mean DOT Profile plot as PNG bytes."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    y_cm = profile_mean * 100  # Convert m to cm
    ax.plot(x_km, y_cm, color='#9B59B6', linewidth=2, label='Mean DOT')
    if profile_std is not None:
        std_cm = profile_std * 100
        ax.fill_between(x_km, y_cm - std_cm, y_cm + std_cm,
                        color='#9B59B6', alpha=0.2, label='±1 Std Dev')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    valid = np.isfinite(y_cm)
    if np.any(valid):
        y_range = np.nanmax(y_cm[valid]) - np.nanmin(y_cm[valid])
        y_text = np.nanmax(y_cm[valid]) - 0.05 * y_range
        ax.text(x_km[valid].min() + 5, y_text, "WEST", fontsize=12, fontweight='bold', ha='left')
        ax.text(x_km[valid].max() - 5, y_text, "EAST", fontsize=12, fontweight='bold', ha='right')
    title = f"{gate_name} - Mean DOT Profile\nDataset: {DATASET_FULL_NAMES.get(dataset, dataset)}"
    if start_year and end_year:
        title += f"\nPeriod: {start_year}-{end_year}"
    if n_obs:
        title += f" | Observations: {n_obs:,}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('DOT (cm)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_gate_spatial_map(
    gate_lon: np.ndarray,
    gate_lat: np.ndarray,
    gate_name: str,
    dpi: int = 300
) -> bytes:
    """Generate spatial map with gate location."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.plot(gate_lon, gate_lat, 'r-', linewidth=3, marker='o', markersize=4, 
            label='Gate transect', zorder=5)
    ax.plot(gate_lon[0], gate_lat[0], 'go', markersize=12, label='Start (West)', zorder=6)
    ax.plot(gate_lon[-1], gate_lat[-1], 'bs', markersize=12, label='End (East)', zorder=6)
    lon_buffer = (gate_lon.max() - gate_lon.min()) * 0.3 + 2
    lat_buffer = (gate_lat.max() - gate_lat.min()) * 0.3 + 1
    ax.set_xlim(gate_lon.min() - lon_buffer, gate_lon.max() + lon_buffer)
    ax.set_ylim(gate_lat.min() - lat_buffer, gate_lat.max() + lat_buffer)
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)
    ax.set_title(f"{gate_name} - Gate Location\n"
                 f"Start: ({gate_lon[0]:.2f}°E, {gate_lat[0]:.2f}°N) → "
                 f"End: ({gate_lon[-1]:.2f}°E, {gate_lat[-1]:.2f}°N)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_velocity_comparison_timeseries(
    v_perp_mean_ts: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    v_geo_ts: Optional[np.ndarray] = None,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """Generate v_perp vs v_geo comparison plot."""
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(time_array)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    ax.plot(time_pd, v_perp_mean_ts, color='#1E3A5F', linewidth=1.5, 
            alpha=0.8, label='v_perp (from ugos/vgos)')
    if v_geo_ts is not None:
        v_geo_cm = v_geo_ts * 100 if np.nanmax(np.abs(v_geo_ts)) < 1 else v_geo_ts
        ax.plot(time_pd, v_geo_cm, color='#E07B53', linewidth=1.5, 
                alpha=0.8, label='v_geo (from DOT slope)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    title_str = "v_perp vs v_geo Comparison" if v_geo_ts is not None else "v_perp Time Series"
    title = f"{gate_name} - {title_str}\nDataset: {DATASET_FULL_NAMES.get(dataset, dataset)}"
    title += f"\nPeriod: {start_year}-{end_year} | Observations: {n_obs:,}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Velocity (cm/s)', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    stats_text = f"v_perp: mean={np.nanmean(v_perp_mean_ts):.2f}, std={np.nanstd(v_perp_mean_ts):.2f} cm/s"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_monthly_velocity_grid(
    v_perp: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """Generate 3x4 grid of monthly mean velocity profiles."""
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = v_perp.shape[1] if len(v_perp.shape) > 1 else len(time_array)
    months = time_pd.month
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    axes = axes.flatten()
    all_y = []
    for month in range(1, 13):
        month_mask = months == month
        if np.any(month_mask):
            v_month = v_perp[:, month_mask] if len(v_perp.shape) > 1 else v_perp[month_mask]
            v_mean = np.nanmean(v_month, axis=1 if len(v_month.shape) > 1 else 0) * 100
            valid = np.isfinite(v_mean)
            if np.any(valid):
                if hasattr(v_mean, '__iter__'):
                    all_y.extend(v_mean[valid])
                else:
                    all_y.append(v_mean)
    if all_y:
        y_min = min(np.percentile(all_y, 2) * 1.2, -5)
        y_max = max(np.percentile(all_y, 98) * 1.2, 5)
    else:
        y_min, y_max = -20, 20
    for month in range(1, 13):
        ax = axes[month - 1]
        month_mask = months == month
        if not np.any(month_mask):
            ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(x_km.min(), x_km.max())
            ax.set_ylim(y_min, y_max)
            continue
        v_month = v_perp[:, month_mask] if len(v_perp.shape) > 1 else v_perp[month_mask]
        v_mean = np.nanmean(v_month, axis=1 if len(v_month.shape) > 1 else 0) * 100
        v_std = np.nanstd(v_month, axis=1 if len(v_month.shape) > 1 else 0) * 100
        valid = np.isfinite(v_mean) if hasattr(v_mean, '__iter__') else True
        if not np.any(valid):
            ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        x_valid = x_km[valid] if hasattr(v_mean, '__iter__') else x_km
        v_valid = v_mean[valid] if hasattr(v_mean, '__iter__') else [v_mean]
        colors = ['#5B8DBE' if v >= 0 else '#CD6155' for v in v_valid]
        bar_width = (x_km[1] - x_km[0]) * 0.8 if len(x_km) > 1 else 5
        ax.bar(x_valid, v_valid, width=bar_width, color=colors, alpha=0.7, edgecolor='black', linewidth=0.3)
        if hasattr(v_std, '__iter__'):
            v_std_valid = v_std[valid]
            ax.errorbar(x_valid, v_valid, yerr=v_std_valid, fmt='none', color='black', capsize=2, alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        if hasattr(v_mean, '__iter__') and np.sum(valid) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(x_valid, v_valid)
            x_fit = np.array([x_valid.min(), x_valid.max()])
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, 'k--', linewidth=1.5, alpha=0.8)
            ax.text(0.05, 0.95, f"slope={slope:.2e}\nR²={r_value**2:.3f}",
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        if month > 8:
            ax.set_xlabel('Distance (km)', fontsize=9)
        if month in [1, 5, 9]:
            ax.set_ylabel('Velocity (cm/s)', fontsize=9)
    fig.suptitle(f"{gate_name} - Monthly Mean Velocity Profiles\n"
                 f"Dataset: {DATASET_FULL_NAMES.get(dataset, dataset)}\n"
                 f"Period: {start_year}-{end_year} | Observations: {n_obs:,}",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_monthly_dot_analysis(
    dot_matrix: np.ndarray,
    x_km: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """Generate 3x4 grid of monthly DOT analysis (scatter + regression)."""
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = dot_matrix.shape[1] if len(dot_matrix.shape) > 1 else len(time_array)
    months = time_pd.month
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    axes = axes.flatten()
    all_y = []
    for month in range(1, 13):
        month_mask = months == month
        if np.any(month_mask):
            dot_month = dot_matrix[:, month_mask] if len(dot_matrix.shape) > 1 else dot_matrix
            dot_mean = np.nanmean(dot_month, axis=1 if len(dot_month.shape) > 1 else 0) * 100
            valid = np.isfinite(dot_mean)
            if np.any(valid):
                if hasattr(dot_mean, '__iter__'):
                    all_y.extend(dot_mean[valid])
                else:
                    all_y.append(dot_mean)
    if all_y:
        y_min = np.percentile(all_y, 2) * 1.1
        y_max = np.percentile(all_y, 98) * 1.1
    else:
        y_min, y_max = -20, 20
    for month in range(1, 13):
        ax = axes[month - 1]
        month_mask = months == month
        if not np.any(month_mask):
            ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        dot_month = dot_matrix[:, month_mask] if len(dot_matrix.shape) > 1 else dot_matrix
        dot_mean = np.nanmean(dot_month, axis=1 if len(dot_month.shape) > 1 else 0) * 100
        valid = np.isfinite(dot_mean) & np.isfinite(x_km)
        if np.sum(valid) < 2:
            ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        ax.scatter(x_km[valid], dot_mean[valid], s=15, color='#9B59B6', alpha=0.6)
        slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], dot_mean[valid])
        r_squared = r_value ** 2
        x_line = np.linspace(x_km[valid].min(), x_km[valid].max(), 50)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=2)
        slope_m_100km = slope / 100
        ax.text(0.95, 0.95, f"R²={r_squared:.3f}\nslope={slope_m_100km:.4f} m/100km",
                transform=ax.transAxes, fontsize=8, verticalalignment='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title(MONTH_ABBREV[month-1], fontsize=10, fontweight='bold')
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        if month > 8:
            ax.set_xlabel('Distance (km)', fontsize=9)
        if month in [1, 5, 9]:
            ax.set_ylabel('DOT (cm)', fontsize=9)
    fig.suptitle(f"{gate_name} - Monthly DOT Analysis (DOT vs Distance)\n"
                 f"Dataset: {DATASET_FULL_NAMES.get(dataset, dataset)}\n"
                 f"Period: {start_year}-{end_year} | Observations: {n_obs:,}",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_salinity_density_profile(
    sos_matrix: np.ndarray,
    dos_matrix: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    time_array: Optional[np.ndarray] = None,
    dpi: int = 300
) -> bytes:
    """Generate Salinity & Density profiles along gate (side by side)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi)
    sos_mean = np.nanmean(sos_matrix, axis=1) if len(sos_matrix.shape) > 1 else sos_matrix
    sos_std = np.nanstd(sos_matrix, axis=1) if len(sos_matrix.shape) > 1 else np.zeros_like(sos_mean)
    ax1.plot(x_km, sos_mean, color='#3498DB', linewidth=2, label='Mean Salinity')
    ax1.fill_between(x_km, sos_mean - sos_std, sos_mean + sos_std, color='#3498DB', alpha=0.2, label='±1 Std')
    ax1.set_xlabel('Distance along gate (km)', fontsize=10)
    ax1.set_ylabel('Salinity (PSU)', fontsize=10)
    ax1.set_title('Salinity Profile', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    dos_mean = np.nanmean(dos_matrix, axis=1) if len(dos_matrix.shape) > 1 else dos_matrix
    dos_std = np.nanstd(dos_matrix, axis=1) if len(dos_matrix.shape) > 1 else np.zeros_like(dos_mean)
    ax2.plot(x_km, dos_mean, color='#E74C3C', linewidth=2, label='Mean Density')
    ax2.fill_between(x_km, dos_mean - dos_std, dos_mean + dos_std, color='#E74C3C', alpha=0.2, label='±1 Std')
    ax2.set_xlabel('Distance along gate (km)', fontsize=10)
    ax2.set_ylabel('Density (kg/m³)', fontsize=10)
    ax2.set_title('Density Profile', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    period_str = ""
    if time_array is not None:
        time_pd = pd.to_datetime(time_array)
        period_str = f" | Period: {time_pd.min().year}-{time_pd.max().year}"
    fig.suptitle(f"{gate_name} - Salinity & Density Along Gate\n"
                 f"Dataset: CMEMS SSS (MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013){period_str}",
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_bathymetry_profile_fixed(
    depth_profile: np.ndarray,
    x_km: np.ndarray,
    gate_name: str,
    gate_lon: Optional[np.ndarray] = None,
    gate_lat: Optional[np.ndarray] = None,
    dpi: int = 300
) -> bytes:
    """Generate bathymetry profile with Y-axis 0 at top (depth increases downward)."""
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    depth_positive = np.abs(depth_profile)
    ax.plot(x_km, depth_positive, color='#2C3E50', linewidth=2)
    max_depth = np.nanmax(depth_positive)
    ax.fill_between(x_km, depth_positive, max_depth * 1.1, color='#AED6F1', alpha=0.3)
    ax.set_ylim(max_depth * 1.1, 0)  # max at bottom, 0 at top
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    title = f"{gate_name} - Bathymetry Profile (GEBCO 2023)"
    if gate_lon is not None and gate_lat is not None:
        title += f"\n({gate_lon[0]:.2f}°E, {gate_lat[0]:.2f}°N) → ({gate_lon[-1]:.2f}°E, {gate_lat[-1]:.2f}°N)"
    ax.set_title(title, fontsize=11, fontweight='bold')
    stats_text = f"Max depth: {max_depth:.0f} m | Mean: {np.nanmean(depth_positive):.0f} m"
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
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
    dpi: int = 300
) -> bytes:
    """
    Generate complete export ZIP with all images and CSV files.
    
    Args:
        gate_data: Dictionary containing all gate data:
            - gate_name: str
            - dataset: str
            - transport_sv: np.ndarray
            - salt_flux_kg_s: np.ndarray (optional)
            - time_array: np.ndarray
            - v_perp: np.ndarray
            - x_km: np.ndarray
            - depth_profile: np.ndarray
            - gate_lon: np.ndarray
            - gate_lat: np.ndarray
            - monthly_v_perp: Dict (optional)
            - monthly_salt_flux: Dict (optional)
        include_images: Whether to include PNG images
        include_csv: Whether to include CSV files
        dpi: Image resolution
        
    Returns:
        ZIP file bytes
    """
    files = {}
    
    gate_name = gate_data['gate_name']
    gate_name_safe = gate_name.lower().replace(' ', '_').replace('-', '_')
    dataset = gate_data.get('dataset', 'cmems_l4')
    time_array = gate_data['time_array']
    
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    
    logger.info(f"Generating export for {gate_name} ({start_year}-{end_year})")
    
    # =========================================================================
    # CSV FILES
    # =========================================================================
    if include_csv:
        transport_sv = gate_data.get('transport_sv')
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
    # IMAGE FILES
    # =========================================================================
    if include_images:
        n_obs = len(time_array)
        
        # Volume Transport
        transport_sv = gate_data.get('transport_sv')
        if transport_sv is not None:
            # Time series
            img = export_volume_transport_timeseries(
                transport_sv, time_array, gate_name, dataset, dpi
            )
            files[f"volume_transport/{gate_name_safe}_timeseries.png"] = img
            
            # Statistics boxplot
            img = export_volume_transport_statistics(
                transport_sv, time_array, gate_name, dataset, dpi
            )
            files[f"volume_transport/{gate_name_safe}_statistics.png"] = img
        
        # Monthly velocity profiles (3x4 grid)
        monthly_v_perp = gate_data.get('monthly_v_perp')
        if monthly_v_perp is not None:
            img = export_monthly_profiles_grid(
                monthly_v_perp, gate_name, "Volume Transport",
                dataset, start_year, end_year, n_obs,
                y_label="Velocity (cm/s)", y_scale=100.0,
                show_regression=True, dpi=dpi
            )
            files[f"volume_transport/{gate_name_safe}_monthly_profiles_grid.png"] = img
        
        # Velocity Hovmöller
        v_perp = gate_data.get('v_perp')
        x_km = gate_data.get('x_km')
        if v_perp is not None and x_km is not None:
            img = export_velocity_hovmoller(
                v_perp, x_km, time_array, gate_name, dataset, dpi
            )
            files[f"velocity/{gate_name_safe}_hovmoller.png"] = img
        
        # Bathymetry
        depth_profile = gate_data.get('depth_profile')
        gate_lon = gate_data.get('gate_lon')
        gate_lat = gate_data.get('gate_lat')
        if depth_profile is not None and x_km is not None:
            img = export_bathymetry_profile(
                depth_profile, x_km, gate_name, gate_lon, gate_lat, dpi
            )
            files[f"bathymetry/{gate_name_safe}_depth_profile.png"] = img
        
        # Salt Flux
        salt_flux = gate_data.get('salt_flux_kg_s')
        if salt_flux is not None:
            img = export_salt_flux_timeseries(
                salt_flux, time_array, gate_name, dataset, dpi
            )
            files[f"salt_flux/{gate_name_safe}_timeseries.png"] = img
        
        # Salt flux monthly profiles (3x4 grid)
        monthly_salt_flux = gate_data.get('monthly_salt_flux')
        if monthly_salt_flux is not None:
            img = export_monthly_profiles_grid(
                monthly_salt_flux, gate_name, "Salt Flux Along Gate",
                dataset, start_year, end_year, n_obs,
                y_label="Salt Flux (kg/m·s)", y_scale=1.0,
                show_regression=True, dpi=dpi
            )
            files[f"salt_flux/{gate_name_safe}_along_gate_grid.png"] = img
    
    # Create timestamp for folder name
    timestamp = datetime.now().strftime("%Y-%m-%d")
    base_folder = f"export_{gate_name_safe}_{start_year}-{end_year}_{timestamp}"
    
    return create_export_zip(files, base_folder)


def generate_multi_gate_export(
    gates_data: List[Dict[str, Any]],
    include_images: bool = True,
    include_csv: bool = True,
    dpi: int = 300
) -> bytes:
    """
    Generate export ZIP for multiple gates.
    
    Args:
        gates_data: List of gate data dictionaries
        include_images: Whether to include PNG images
        include_csv: Whether to include CSV files
        dpi: Image resolution
        
    Returns:
        ZIP file bytes
    """
    all_files = {}
    
    for gate_data in gates_data:
        gate_name = gate_data['gate_name']
        gate_name_safe = gate_name.lower().replace(' ', '_').replace('-', '_')
        
        # Generate files for this gate
        single_gate_files = {}
        
        # ... (same logic as generate_full_export but without creating ZIP)
        # Add to all_files with gate subfolder
        
        dataset = gate_data.get('dataset', 'cmems_l4')
        time_array = gate_data['time_array']
        time_pd = pd.to_datetime(time_array)
        start_year = time_pd.min().year
        end_year = time_pd.max().year
        n_obs = len(time_array)
        
        # CSV
        if include_csv:
            transport_sv = gate_data.get('transport_sv')
            if transport_sv is not None:
                df_raw = generate_volume_transport_raw_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_raw.csv"] = df_raw.to_csv(index=False)
                
                df_clim = generate_volume_transport_climatology_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_climatology.csv"] = df_clim.to_csv(index=False)
                
                df_annual = generate_volume_transport_annual_csv(transport_sv, time_array, gate_name, dataset)
                all_files[f"csv/{gate_name_safe}_volume_transport_annual.csv"] = df_annual.to_csv(index=False)
        
        # Images
        if include_images:
            transport_sv = gate_data.get('transport_sv')
            if transport_sv is not None:
                img = export_volume_transport_timeseries(transport_sv, time_array, gate_name, dataset, dpi)
                all_files[f"volume_transport/{gate_name_safe}_timeseries.png"] = img
                
                img = export_volume_transport_statistics(transport_sv, time_array, gate_name, dataset, dpi)
                all_files[f"volume_transport/{gate_name_safe}_statistics.png"] = img
            
            monthly_v_perp = gate_data.get('monthly_v_perp')
            if monthly_v_perp is not None:
                img = export_monthly_profiles_grid(
                    monthly_v_perp, gate_name, "Volume Transport", dataset,
                    start_year, end_year, n_obs, "Velocity (cm/s)", 100.0, True, dpi
                )
                all_files[f"volume_transport/{gate_name_safe}_monthly_profiles_grid.png"] = img
            
            v_perp = gate_data.get('v_perp')
            x_km = gate_data.get('x_km')
            if v_perp is not None and x_km is not None:
                img = export_velocity_hovmoller(v_perp, x_km, time_array, gate_name, dataset, dpi)
                all_files[f"velocity/{gate_name_safe}_hovmoller.png"] = img
            
            depth_profile = gate_data.get('depth_profile')
            gate_lon = gate_data.get('gate_lon')
            gate_lat = gate_data.get('gate_lat')
            if depth_profile is not None and x_km is not None:
                img = export_bathymetry_profile(depth_profile, x_km, gate_name, gate_lon, gate_lat, dpi)
                all_files[f"bathymetry/{gate_name_safe}_depth_profile.png"] = img
            
            salt_flux = gate_data.get('salt_flux_kg_s')
            if salt_flux is not None:
                img = export_salt_flux_timeseries(salt_flux, time_array, gate_name, dataset, dpi)
                all_files[f"salt_flux/{gate_name_safe}_timeseries.png"] = img
            
            monthly_salt_flux = gate_data.get('monthly_salt_flux')
            if monthly_salt_flux is not None:
                img = export_monthly_profiles_grid(
                    monthly_salt_flux, gate_name, "Salt Flux Along Gate", dataset,
                    start_year, end_year, n_obs, "Salt Flux (kg/m·s)", 1.0, True, dpi
                )
                all_files[f"salt_flux/{gate_name_safe}_along_gate_grid.png"] = img
            
            # NEW EXPORTS (2026-01-26)
            dot_matrix = gate_data.get('dot_matrix')
            if dot_matrix is not None and x_km is not None:
                dot_mean = np.nanmean(dot_matrix, axis=1) if len(dot_matrix.shape) > 1 else dot_matrix
                dot_std = np.nanstd(dot_matrix, axis=1) if len(dot_matrix.shape) > 1 else None
                img = export_mean_dot_profile(dot_mean, x_km, gate_name, dataset, dot_std, start_year, end_year, n_obs, dpi)
                all_files[f"dot_profile/{gate_name_safe}_mean_dot_profile.png"] = img
            
            if gate_lon is not None and gate_lat is not None:
                img = export_gate_spatial_map(gate_lon, gate_lat, gate_name, dpi)
                all_files[f"spatial/{gate_name_safe}_gate_map.png"] = img
            
            v_perp_mean_ts = gate_data.get('v_perp_mean_ts')
            v_geo_ts = gate_data.get('v_geo_ts')
            if v_perp_mean_ts is not None:
                img = export_velocity_comparison_timeseries(v_perp_mean_ts, time_array, gate_name, v_geo_ts, dataset, dpi)
                all_files[f"velocity/{gate_name_safe}_v_perp_vs_v_geo.png"] = img
            
            if v_perp is not None and x_km is not None:
                img = export_monthly_velocity_grid(v_perp, x_km, time_array, gate_name, dataset, dpi)
                all_files[f"velocity/{gate_name_safe}_monthly_velocity_grid.png"] = img
            
            if dot_matrix is not None and x_km is not None:
                img = export_monthly_dot_analysis(dot_matrix, x_km, time_array, gate_name, dataset, dpi)
                all_files[f"monthly_analysis/{gate_name_safe}_monthly_dot_analysis.png"] = img
            
            sos_matrix = gate_data.get('sos_matrix')
            dos_matrix = gate_data.get('dos_matrix')
            if sos_matrix is not None and dos_matrix is not None and x_km is not None:
                img = export_salinity_density_profile(sos_matrix, dos_matrix, x_km, gate_name, time_array, dpi)
                all_files[f"salt_flux/{gate_name_safe}_salinity_density.png"] = img
            
            if depth_profile is not None and x_km is not None:
                img = export_bathymetry_profile_fixed(depth_profile, x_km, gate_name, gate_lon, gate_lat, dpi)
                all_files[f"bathymetry/{gate_name_safe}_depth_profile_v2.png"] = img
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    n_gates = len(gates_data)
    base_folder = f"arctic_gates_export_{n_gates}gates_{timestamp}"
    
    return create_export_zip(all_files, base_folder)
