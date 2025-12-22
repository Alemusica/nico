"""
Geoid Interpolation
===================
Functions for loading and interpolating geoid data.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

from ..core.coordinates import wrap_longitudes


def load_geoid_interpolator(geoid_path: str | Path) -> RegularGridInterpolator:
    """
    Load geoid data and create an interpolator.
    
    Parameters
    ----------
    geoid_path : str or Path
        Path to geoid NetCDF file (e.g., TUM_ogmoc.nc)
        
    Returns
    -------
    RegularGridInterpolator
        Interpolator for geoid values
    """
    ds_geoid = xr.open_dataset(geoid_path)
    lat_geoid = ds_geoid["lat"].values
    lon_geoid = ds_geoid["lon"].values
    geoid_values = ds_geoid["value"].values
    
    # Wrap longitudes to [-180, 180] and sort
    lon_wrapped = wrap_longitudes(lon_geoid)
    sort_idx = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[sort_idx]
    
    # Remove duplicates
    unique_idx = np.concatenate(([True], np.diff(lon_sorted) != 0))
    lon_sorted = lon_sorted[unique_idx]
    geoid_sorted = geoid_values[:, sort_idx][:, unique_idx]
    
    return RegularGridInterpolator(
        (lat_geoid, lon_sorted),
        geoid_sorted,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )


def interpolate_geoid(
    geoid_path: str | Path,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
) -> np.ndarray:
    """
    Interpolate geoid values at target lat/lon positions.
    
    Parameters
    ----------
    geoid_path : str or Path
        Path to geoid NetCDF file
    target_lats : np.ndarray
        Target latitudes
    target_lons : np.ndarray
        Target longitudes
        
    Returns
    -------
    np.ndarray
        Interpolated geoid values
    """
    interp = load_geoid_interpolator(geoid_path)
    
    # Wrap target longitudes
    target_lons_wrapped = wrap_longitudes(target_lons)
    points = np.column_stack([target_lats, target_lons_wrapped])
    
    return interp(points)


def add_geoid_to_dataset(
    ds: xr.Dataset,
    geoid_path: str | Path,
) -> xr.Dataset:
    """
    Add interpolated geoid values to a dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with latitude/longitude coordinates
    geoid_path : str or Path
        Path to geoid NetCDF file
        
    Returns
    -------
    xr.Dataset
        Dataset with 'geoid' variable added
    """
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    
    geoid_vals = interpolate_geoid(geoid_path, lat, lon)
    
    return ds.assign(geoid=(("time",), geoid_vals))
