"""
Matplotlib Charts
=================
Static visualization functions using Matplotlib (for publication-quality figures).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

# Optional cartopy for map projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


def create_three_panel_plot(
    df: pd.DataFrame,
    gate_geometry=None,
    title: str = "DOT Analysis",
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Create 3-panel DOT analysis figure.
    
    Panels:
    1. Slope time series
    2. Mean DOT profile
    3. Spatial map
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: lon, lat, dot, time, year_month
    gate_geometry : GeoDataFrame, optional
        Gate geometry for map overlay
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    dpi : int
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # Calculate slope time series
    time_periods = sorted(df["year_month"].unique())
    slopes = []
    times = []
    
    mean_lat = df["lat"].mean()
    r_earth = 6371.0
    lat_rad = np.deg2rad(mean_lat)
    
    for period in time_periods:
        month_data = df[df["year_month"] == period]
        if len(month_data) < 10:
            continue
        
        # Bin by longitude
        lon_bins = np.arange(
            month_data["lon"].min(),
            month_data["lon"].max() + 0.01,
            0.01
        )
        if len(lon_bins) < 2:
            continue
        
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        
        # Calculate distance in km
        lon_rad = np.deg2rad(lon_centers)
        x_km = r_earth * np.abs(lon_rad - lon_rad[0]) * np.cos(lat_rad)
        
        # Bin DOT values
        month_data = month_data.copy()
        month_data["lon_bin"] = pd.cut(
            month_data["lon"], bins=lon_bins, labels=False
        )
        binned = month_data.groupby("lon_bin")["dot"].mean()
        
        y = np.full(len(lon_centers), np.nan)
        for idx in binned.index:
            if idx < len(y):
                y[int(idx)] = binned[idx]
        
        # Fit slope
        mask = np.isfinite(x_km) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        
        slope, _ = np.polyfit(x_km[mask], y[mask], 1)
        slopes.append(slope * 100)  # m/100km
        times.append(pd.Timestamp(str(period)))
    
    # Panel 1: Slope time series
    ax0 = axes[0]
    ax0.plot(times, slopes, "-o", markersize=3, linewidth=1.5)
    ax0.axhline(0, color="k", linewidth=0.8)
    ax0.grid(True, linestyle=":", alpha=0.6)
    ax0.set_ylabel("Slope (m / 100 km)")
    ax0.set_title("Monthly DOT Slope")
    ax0.tick_params(axis="x", rotation=45)
    
    # Panel 2: Mean DOT profile
    ax1 = axes[1]
    mean_profile = df.groupby("lon")["dot"].mean()
    ax1.plot(mean_profile.index, mean_profile.values, linewidth=2, color="darkblue")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlabel("Longitude (°)")
    ax1.set_ylabel("DOT (m)")
    ax1.set_title("Mean DOT Profile")
    
    # Panel 3: Spatial map
    ax2 = axes[2]
    mean_dot = df.groupby(["lon", "lat"])["dot"].mean().reset_index()
    sc = ax2.scatter(
        mean_dot["lon"], mean_dot["lat"],
        c=mean_dot["dot"], s=10, cmap="viridis", alpha=0.8
    )
    plt.colorbar(sc, ax=ax2, label="DOT (m)")
    ax2.set_xlabel("Longitude (°)")
    ax2.set_ylabel("Latitude (°)")
    ax2.set_title("Mean DOT Map")
    ax2.grid(True, linestyle=":", alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def create_monthly_analysis_figure(
    df: pd.DataFrame,
    bin_size: float = 0.01,
    title: str = "Monthly DOT Analysis",
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Create 12-subplot monthly analysis figure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: lon, lat, dot, month
    bin_size : float
        Longitude bin size
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    dpi : int
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    
    # Conversion factor
    mean_lat = df["lat"].mean()
    meters_per_deg = 111320 * np.cos(np.radians(mean_lat))
    
    # DOT range for consistent y-limits
    dot_min, dot_max = df["dot"].min(), df["dot"].max()
    dot_margin = 0.05 * (dot_max - dot_min)
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    axes = axes.flatten()
    
    for month in range(1, 13):
        ax = axes[month - 1]
        month_data = df[df["month"] == month]
        
        if len(month_data) < 10:
            ax.text(0.5, 0.5, f"{month_names[month]}\nNo data",
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        # Bin by longitude
        lon_min = month_data["lon"].min()
        lon_max = month_data["lon"].max()
        bins = np.arange(lon_min, lon_max + bin_size, bin_size)
        
        if len(bins) < 2:
            continue
        
        centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate binned means
        month_data = month_data.copy()
        month_data["lon_bin"] = pd.cut(month_data["lon"], bins=bins, labels=False)
        binned = month_data.groupby("lon_bin")["dot"].mean()
        
        lon_valid = []
        dot_valid = []
        for idx in binned.index:
            if idx < len(centers):
                lon_valid.append(centers[int(idx)])
                dot_valid.append(binned[idx])
        
        if len(lon_valid) < 2:
            continue
        
        lon_valid = np.array(lon_valid)
        dot_valid = np.array(dot_valid)
        
        # Scatter plot
        ax.scatter(lon_valid, dot_valid, s=30, alpha=0.7, color="steelblue")
        
        # Linear fit
        slope_m_deg, intercept = np.polyfit(lon_valid, dot_valid, 1)
        fit_line = slope_m_deg * lon_valid + intercept
        
        ss_res = np.sum((dot_valid - fit_line) ** 2)
        ss_tot = np.sum((dot_valid - np.mean(dot_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        slope_mm_m = (slope_m_deg / meters_per_deg) * 1000
        
        ax.plot(lon_valid, fit_line, "--", color="darkred", linewidth=2)
        
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("DOT (m)")
        ax.set_ylim(dot_min - dot_margin, dot_max + dot_margin)
        ax.grid(True, alpha=0.3)
        
        n_obs = len(month_data)
        ax.set_title(f"{month_names[month]}\n{slope_mm_m:.4f} mm/m | R²={r2:.3f} | n={n_obs}")
    
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def create_dot_map(
    df: pd.DataFrame,
    gate_gdf=None,
    title: str = "DOT Map",
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Create DOT spatial map with optional gate overlay.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lat, lon, dot columns
    gate_gdf : GeoDataFrame, optional
        Gate geometry
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    dpi : int
        Resolution
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection=proj))
        
        ax.coastlines(linewidth=1.0)
        ax.add_feature(cfeature.LAND, facecolor="beige", alpha=0.7)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.3)
        
        # Set extent
        lon_margin = 2
        lat_margin = 2
        ax.set_extent([
            df["lon"].min() - lon_margin,
            df["lon"].max() + lon_margin,
            df["lat"].min() - lat_margin,
            df["lat"].max() + lat_margin,
        ], crs=proj)
        
        # Gridlines
        gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        transform = proj
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        transform = None
    
    # Plot DOT
    kwargs = {"transform": transform} if transform else {}
    sc = ax.scatter(
        df["lon"], df["lat"], c=df["dot"],
        s=15, cmap="viridis", alpha=0.7, **kwargs
    )
    plt.colorbar(sc, ax=ax, label="DOT (m)")
    
    # Plot gate
    if gate_gdf is not None:
        gate_gdf.plot(ax=ax, color="red", linewidth=2, **kwargs)
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig


def create_slope_timeline_figure(
    timeline_df: pd.DataFrame,
    title: str = "DOT Slope Timeline",
    save_path: str | None = None,
    dpi: int = 300,
):
    """
    Create slope timeline figure with error bars.
    
    Parameters
    ----------
    timeline_df : pd.DataFrame
        DataFrame with date, slope_mm_per_m, slope_err_mm_per_m columns
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    dpi : int
        Resolution
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Light line connecting points
    ax.plot(
        timeline_df["date"],
        timeline_df["slope_mm_per_m"],
        color="steelblue",
        linewidth=1.5,
        alpha=0.5,
    )
    
    # Error bar points
    ax.errorbar(
        timeline_df["date"],
        timeline_df["slope_mm_per_m"],
        yerr=timeline_df["slope_err_mm_per_m"],
        fmt="o",
        markersize=5,
        color="steelblue",
        ecolor="steelblue",
        elinewidth=1,
        capsize=2,
        alpha=0.8,
    )
    
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.4, linestyle="--")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("DOT Slope (mm/m)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    return fig
