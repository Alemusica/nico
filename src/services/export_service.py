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
# EXPORT FUNCTIONS - Richieste specifiche utente
# ==============================================================================

def export_slope_timeline(
    slope_values: np.ndarray,
    time_array: np.ndarray,
    gate_name: str,
    dataset: str = "cmems_l4",
    dpi: int = 300
) -> bytes:
    """
    📈 Slope Timeline - Pendenza DOT nel tempo.
    Mostra l'evoluzione della slope (mm/km) nel tempo.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    
    # Slope in mm/km
    slope_mm_km = slope_values * 1000  # m/km -> mm/km
    
    ax.plot(time_pd, slope_mm_km, 'b-', linewidth=0.8, alpha=0.7)
    
    # Rolling mean (30 days)
    df = pd.DataFrame({'time': time_pd, 'slope': slope_mm_km}).set_index('time')
    rolling = df['slope'].rolling(window=30, center=True).mean()
    ax.plot(rolling.index, rolling.values, 'r-', linewidth=2, label='30-day mean')
    
    # Mean line
    mean_slope = np.nanmean(slope_mm_km)
    ax.axhline(y=mean_slope, color='green', linestyle='--', linewidth=1.5, 
               label=f'Mean: {mean_slope:.2f} mm/km')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    
    title = f"{gate_name} - DOT Slope Timeline\n{DATASET_FULL_NAMES.get(dataset, dataset)}\n{start_year}-{end_year}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Slope (mm/km)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
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
    📊 DOT Profile Along Gate - Profilo medio DOT con ±std.
    """
    dot_mean = np.nanmean(dot_matrix, axis=1)
    dot_std = np.nanstd(dot_matrix, axis=1)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    ax.plot(x_km, dot_mean, 'darkblue', linewidth=2, label='Mean DOT')
    ax.fill_between(x_km, dot_mean - dot_std, dot_mean + dot_std,
                   alpha=0.3, color='blue', label='±1 std')
    
    # Linear regression
    valid = ~np.isnan(dot_mean)
    if valid.sum() > 2:
        slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], dot_mean[valid])
        ax.plot(x_km, slope * x_km + intercept, 'r--', linewidth=1.5, alpha=0.8,
               label=f'Fit: slope={slope*1000:.2f} mm/km, R²={r_value**2:.3f}')
    
    title = f"{gate_name} - DOT Profile Along Gate\n{DATASET_FULL_NAMES.get(dataset, dataset)}"
    if start_year and end_year:
        title += f"\n{start_year}-{end_year}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('DOT (m)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
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
    dpi: int = 300
) -> bytes:
    """
    🗺️ Spatial Map - Mappa geografica con coastlines, confini e griglia lat/lon.
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        
        # Calculate bounds
        lon_min, lon_max = gate_lon.min() - 10, gate_lon.max() + 10
        lat_min, lat_max = gate_lat.min() - 5, gate_lat.max() + 5
        
        # Projection
        central_lon = (lon_min + lon_max) / 2
        central_lat = (lat_min + lat_max) / 2
        
        if central_lat > 60:
            proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
        else:
            proj = ccrs.LambertConformal(central_longitude=central_lon, central_latitude=central_lat)
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=dpi, subplot_kw={'projection': proj})
        
        # Extent
        if central_lat > 60:
            ax.set_extent([lon_min, lon_max, max(55, lat_min), 90], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Features
        ax.add_feature(cfeature.LAND, facecolor='#f0e68c', edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, facecolor='#add8e6')
        ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, edgecolor='gray')
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
        ax.add_feature(cfeature.LAKES, facecolor='lightblue', edgecolor='blue', linewidth=0.3)
        
        # Gridlines with labels
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 9}
        gl.ylabel_style = {'size': 9}
        
        # Plot gate line
        ax.plot(gate_lon, gate_lat, 'r-', linewidth=4, transform=ccrs.PlateCarree(),
               label=f'{gate_name}', zorder=10)
        
        # Start/End markers
        ax.scatter(gate_lon[0], gate_lat[0], c='lime', s=150, marker='o', edgecolor='black',
                  transform=ccrs.PlateCarree(), label=f'Start ({gate_lon[0]:.1f}°, {gate_lat[0]:.1f}°)', zorder=11)
        ax.scatter(gate_lon[-1], gate_lat[-1], c='red', s=150, marker='s', edgecolor='black',
                  transform=ccrs.PlateCarree(), label=f'End ({gate_lon[-1]:.1f}°, {gate_lat[-1]:.1f}°)', zorder=11)
        
        ax.set_title(f'{gate_name} - Geographic Location', fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=9)
        
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
    """
    time_pd = pd.to_datetime(time_array)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    axes = axes.flatten()
    
    for month_idx in range(12):
        ax = axes[month_idx]
        month_mask = time_pd.month == (month_idx + 1)
        
        if month_mask.sum() > 0:
            dot_month = dot_matrix[:, month_mask]
            dot_mean = np.nanmean(dot_month, axis=1)
            dot_std = np.nanstd(dot_month, axis=1)
            
            # Plot mean with std
            ax.plot(x_km, dot_mean, 'darkblue', linewidth=1.5)
            ax.fill_between(x_km, dot_mean - dot_std, dot_mean + dot_std, alpha=0.3, color='blue')
            
            # Linear regression
            valid = ~np.isnan(dot_mean)
            if valid.sum() > 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], dot_mean[valid])
                slope_mm_km = slope * 1000  # mm/km
                ax.plot(x_km, slope * x_km + intercept, 'r--', linewidth=1, alpha=0.7)
                ax.text(0.05, 0.95, f'slope={slope_mm_km:.2f} mm/km\nR²={r_value**2:.3f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(MONTH_ABBREV[month_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (km)' if month_idx >= 8 else '', fontsize=8)
        ax.set_ylabel('DOT (m)' if month_idx % 4 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3)
    
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
    Colori consistenti con Streamlit.
    """
    time_pd = pd.to_datetime(time_array)
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), dpi=dpi)
    axes = axes.flatten()
    
    # Color scheme matching Streamlit
    positive_color = '#1f77b4'  # Blue for northward
    negative_color = '#d62728'  # Red for southward
    
    for month_idx in range(12):
        ax = axes[month_idx]
        month_mask = time_pd.month == (month_idx + 1)
        
        if month_mask.sum() > 0:
            v_month = v_perp[:, month_mask]
            v_mean = np.nanmean(v_month, axis=1) * 100  # cm/s
            v_std = np.nanstd(v_month, axis=1) * 100
            
            # Color based on mean direction
            color = positive_color if np.nanmean(v_mean) >= 0 else negative_color
            
            ax.plot(x_km, v_mean, color=color, linewidth=1.5)
            ax.fill_between(x_km, v_mean - v_std, v_mean + v_std, alpha=0.3, color=color)
            
            # Linear regression
            valid = ~np.isnan(v_mean)
            if valid.sum() > 2:
                slope, intercept, r_value, _, _ = stats.linregress(x_km[valid], v_mean[valid])
                ax.plot(x_km, slope * x_km + intercept, 'k--', linewidth=1, alpha=0.7)
                ax.text(0.05, 0.95, f'slope={slope:.4f}\nR²={r_value**2:.3f}',
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_title(MONTH_ABBREV[month_idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('Distance (km)' if month_idx >= 8 else '', fontsize=8)
        ax.set_ylabel('v (cm/s)' if month_idx % 4 == 0 else '', fontsize=8)
        ax.grid(True, alpha=0.3)
    
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
    dpi: int = 300
) -> bytes:
    """
    📈 Time Series: v_perp vs v_geo - Entrambe le velocità nello stesso plot.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    
    # Mean along gate for each time step
    v_perp_mean = np.nanmean(v_perp, axis=0) * 100  # cm/s
    v_geo_mean = np.nanmean(v_geo, axis=0) * 100 if v_geo is not None else None
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    
    ax.plot(time_pd, v_perp_mean, 'b-', linewidth=0.8, alpha=0.7, label='v_perp')
    if v_geo_mean is not None:
        ax.plot(time_pd, v_geo_mean, 'r-', linewidth=0.8, alpha=0.7, label='v_geo')
    
    # Rolling means
    df_perp = pd.DataFrame({'time': time_pd, 'v': v_perp_mean}).set_index('time')
    rolling_perp = df_perp['v'].rolling(window=30, center=True).mean()
    ax.plot(rolling_perp.index, rolling_perp.values, 'b-', linewidth=2, label='v_perp (30-day)')
    
    if v_geo_mean is not None:
        df_geo = pd.DataFrame({'time': time_pd, 'v': v_geo_mean}).set_index('time')
        rolling_geo = df_geo['v'].rolling(window=30, center=True).mean()
        ax.plot(rolling_geo.index, rolling_geo.values, 'r-', linewidth=2, label='v_geo (30-day)')
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    title = f"{gate_name} - Cross-Gate Velocity Comparison\n{DATASET_FULL_NAMES.get(dataset, dataset)}\n{start_year}-{end_year}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Velocity (cm/s)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
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
    dpi: int = 300
) -> bytes:
    """
    📈 Total Transport Time Series - Volume transport nel tempo.
    Colorato come su Streamlit.
    """
    time_pd = pd.to_datetime(time_array)
    start_year = time_pd.min().year
    end_year = time_pd.max().year
    n_obs = len(transport_sv[~np.isnan(transport_sv)])
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    
    # Color by sign (matching Streamlit)
    positive_mask = transport_sv >= 0
    
    ax.fill_between(time_pd, 0, transport_sv, where=positive_mask, 
                   color='#1f77b4', alpha=0.3, label='Northward')
    ax.fill_between(time_pd, 0, transport_sv, where=~positive_mask, 
                   color='#d62728', alpha=0.3, label='Southward')
    ax.plot(time_pd, transport_sv, 'k-', linewidth=0.5, alpha=0.5)
    
    # Rolling mean
    df = pd.DataFrame({'time': time_pd, 'transport': transport_sv}).set_index('time')
    rolling = df['transport'].rolling(window=30, center=True).mean()
    ax.plot(rolling.index, rolling.values, 'purple', linewidth=2, label='30-day mean')
    
    # Statistics
    mean_val = np.nanmean(transport_sv)
    std_val = np.nanstd(transport_sv)
    ax.axhline(y=mean_val, color='green', linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_val:.2f} ± {std_val:.2f} Sv')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
    
    title = f"{gate_name} - Total Volume Transport\n{DATASET_FULL_NAMES.get(dataset, dataset)}\n{start_year}-{end_year} | N={n_obs}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Volume Transport (Sv)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    
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
    dpi: int = 300
) -> bytes:
    """
    🏔️ Bathymetry Profile - Senza marrone, zero in alto.
    """
    fig, ax = plt.subplots(figsize=(12, 5), dpi=dpi)
    
    # Depth as positive values, zero at top
    depth_positive = np.abs(depth_profile)
    
    # Plot seafloor line (blue/navy)
    ax.plot(x_km, depth_positive, 'navy', linewidth=2)
    ax.fill_between(x_km, depth_positive, depth_positive.max() * 1.1, 
                   color='lightblue', alpha=0.3)
    
    # Sea level at top (y=0)
    ax.axhline(y=0, color='steelblue', linewidth=2, label='Sea Surface')
    
    # Invert y-axis so 0 is at top
    ax.invert_yaxis()
    ax.set_ylim(depth_positive.max() * 1.1, -50)  # Small buffer above sea level
    
    ax.set_xlabel('Distance along gate (km)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    
    title = f"{gate_name} - Bathymetry Profile\n{DATASET_FULL_NAMES['gebco']}"
    if gate_lon is not None and gate_lat is not None:
        title += f"\nStart: ({gate_lon[0]:.2f}°, {gate_lat[0]:.2f}°) → End: ({gate_lon[-1]:.2f}°, {gate_lat[-1]:.2f}°)"
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Stats annotation
    max_depth = np.nanmax(depth_positive)
    mean_depth = np.nanmean(depth_positive)
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
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=dpi, sharex=True)
    
    # Salinity
    ax1.plot(x_km, salinity_profile, 'g-', linewidth=2)
    ax1.fill_between(x_km, salinity_profile.min() * 0.99, salinity_profile, 
                    alpha=0.3, color='green')
    ax1.set_ylabel('Salinity (PSU)', fontsize=10)
    ax1.set_title(f'{gate_name} - Salinity Along Gate', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    mean_sal = np.nanmean(salinity_profile)
    ax1.axhline(y=mean_sal, color='darkgreen', linestyle='--', 
               label=f'Mean: {mean_sal:.2f} PSU')
    ax1.legend(loc='best')
    
    # Density
    ax2.plot(x_km, density_profile, 'purple', linewidth=2)
    ax2.fill_between(x_km, density_profile.min() * 0.999, density_profile,
                    alpha=0.3, color='purple')
    ax2.set_xlabel('Distance along gate (km)', fontsize=10)
    ax2.set_ylabel('Density (kg/m³)', fontsize=10)
    ax2.set_title(f'{gate_name} - Density Along Gate', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
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
