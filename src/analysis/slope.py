"""
Slope Analysis
==============
Functions for computing DOT slopes using longitude binning and linear regression.
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


# Earth radius in meters
R_EARTH_M = 6_371_000
R_EARTH_KM = 6_371.0

# Meters per degree at equator
METERS_PER_DEG_EQUATOR = 111_320


@dataclass
class SlopeResult:
    """Container for slope analysis results."""
    slope_m_per_deg: float
    slope_mm_per_m: float
    slope_err_mm_per_m: float
    r_squared: float
    p_value: float
    n_points: int
    intercept: float


def bin_by_longitude(
    lon: np.ndarray,
    values: np.ndarray,
    bin_size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin data by longitude.
    
    Parameters
    ----------
    lon : np.ndarray
        Longitude values
    values : np.ndarray
        Values to bin (e.g., DOT)
    bin_size : float
        Bin size in degrees (default: 0.01°)
        
    Returns
    -------
    tuple
        (bin_centers, bin_means, bin_stds, bin_counts)
    """
    lon = np.asarray(lon).flatten()
    values = np.asarray(values).flatten()
    
    # Remove NaN values
    mask = ~np.isnan(lon) & ~np.isnan(values)
    lon = lon[mask]
    values = values[mask]
    
    if len(lon) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Create bins
    lon_min, lon_max = np.min(lon), np.max(lon)
    bins = np.arange(lon_min, lon_max + bin_size, bin_size)
    
    if len(bins) < 2:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Bin indices
    bin_indices = np.digitize(lon, bins) - 1
    
    # Compute statistics per bin
    bin_centers = []
    bin_means = []
    bin_stds = []
    bin_counts = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        count = np.sum(mask)
        if count > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(np.nanmean(values[mask]))
            bin_stds.append(np.nanstd(values[mask]) if count > 1 else 0.0)
            bin_counts.append(count)
    
    return (
        np.array(bin_centers),
        np.array(bin_means),
        np.array(bin_stds),
        np.array(bin_counts),
    )


def convert_slope_to_mm_per_m(
    slope_m_per_deg: float,
    std_err_m_per_deg: float,
    mean_latitude: float,
) -> tuple[float, float]:
    """
    Convert slope from m/deg to mm/m.
    
    Parameters
    ----------
    slope_m_per_deg : float
        Slope in meters per degree longitude
    std_err_m_per_deg : float
        Standard error in m/deg
    mean_latitude : float
        Mean latitude for projection correction
        
    Returns
    -------
    tuple[float, float]
        (slope_mm_per_m, std_err_mm_per_m)
    """
    # Meters per degree at given latitude
    meters_per_deg = METERS_PER_DEG_EQUATOR * np.cos(np.radians(mean_latitude))
    
    # Convert: m/deg -> m/m -> mm/m
    slope_mm_per_m = (slope_m_per_deg / meters_per_deg) * 1000
    std_err_mm_per_m = (std_err_m_per_deg / meters_per_deg) * 1000
    
    return slope_mm_per_m, std_err_mm_per_m


def compute_slope(
    lon: np.ndarray,
    dot: np.ndarray,
    mean_latitude: float,
    bin_size: float = 0.01,
) -> SlopeResult | None:
    """
    Compute DOT slope using longitude binning and linear regression.
    
    Parameters
    ----------
    lon : np.ndarray
        Longitude values
    dot : np.ndarray
        DOT values
    mean_latitude : float
        Mean latitude for unit conversion
    bin_size : float
        Longitude bin size in degrees
        
    Returns
    -------
    SlopeResult or None
        Slope analysis results, or None if insufficient data
    """
    # Bin by longitude
    bin_centers, bin_means, _, bin_counts = bin_by_longitude(lon, dot, bin_size)
    
    if len(bin_centers) < 3:
        return None
    
    # Linear regression
    try:
        slope_m_deg, intercept, r_value, p_value, std_err = stats.linregress(
            bin_centers, bin_means
        )
    except Exception:
        return None
    
    # Convert units
    slope_mm_m, err_mm_m = convert_slope_to_mm_per_m(
        slope_m_deg, std_err, mean_latitude
    )
    
    return SlopeResult(
        slope_m_per_deg=slope_m_deg,
        slope_mm_per_m=slope_mm_m,
        slope_err_mm_per_m=err_mm_m,
        r_squared=r_value ** 2,
        p_value=p_value,
        n_points=int(np.sum(bin_counts)),
        intercept=intercept,
    )


def compute_slope_timeline(
    df: pd.DataFrame,
    bin_size: float = 0.01,
    group_by: str = "month",
) -> pd.DataFrame:
    """
    Compute slope timeline grouped by time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: lon, lat, dot, time, year, month
    bin_size : float
        Longitude bin size
    group_by : str
        Grouping method: 'month' for monthly, 'cycle' for per-cycle
        
    Returns
    -------
    pd.DataFrame
        Timeline with slope, error, date, etc.
    """
    mean_lat = df["lat"].mean()
    results = []
    
    if group_by == "month":
        # Group by year-month
        for (year, month), group in df.groupby(["year", "month"]):
            if len(group) < 10:
                continue
            
            result = compute_slope(
                group["lon"].values,
                group["dot"].values,
                mean_lat,
                bin_size,
            )
            
            if result is not None:
                results.append({
                    "year": year,
                    "month": month,
                    "date": pd.Timestamp(year=int(year), month=int(month), day=1),
                    "slope_mm_per_m": result.slope_mm_per_m,
                    "slope_err_mm_per_m": result.slope_err_mm_per_m,
                    "r_squared": result.r_squared,
                    "p_value": result.p_value,
                    "n_points": result.n_points,
                })
    
    elif group_by == "cycle":
        # Group by cycle
        for cycle, group in df.groupby("cycle"):
            if len(group) < 10:
                continue
            
            result = compute_slope(
                group["lon"].values,
                group["dot"].values,
                mean_lat,
                bin_size,
            )
            
            if result is not None:
                results.append({
                    "cycle": cycle,
                    "date": group["time"].mean(),
                    "slope_mm_per_m": result.slope_mm_per_m,
                    "slope_err_mm_per_m": result.slope_err_mm_per_m,
                    "r_squared": result.r_squared,
                    "p_value": result.p_value,
                    "n_points": result.n_points,
                })
    
    return pd.DataFrame(results)


def compute_distance_km(
    lon: np.ndarray,
    lat_mean: float,
) -> np.ndarray:
    """
    Compute distance in km along longitude at given latitude.
    
    Parameters
    ----------
    lon : np.ndarray
        Longitude values
    lat_mean : float
        Mean latitude for projection
        
    Returns
    -------
    np.ndarray
        Distance in km from first point
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat_mean)
    
    dlon = lon_rad - lon_rad[0]
    x_km = R_EARTH_KM * np.abs(dlon) * np.cos(lat_rad)
    
    return x_km
