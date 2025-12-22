"""
Coordinate Utilities
====================
Functions for handling geographic coordinates, wrapping, and spatial filtering.
"""

import numpy as np
import xarray as xr


def wrap_longitudes(values: np.ndarray | list) -> np.ndarray:
    """
    Wrap longitude values to [-180, 180] range.
    
    Parameters
    ----------
    values : array-like
        Longitude values in degrees
        
    Returns
    -------
    np.ndarray
        Wrapped longitude values in [-180, 180]
        
    Examples
    --------
    >>> wrap_longitudes([190, -200, 0])
    array([-170.,  160.,    0.])
    """
    arr = np.asarray(values, dtype=float)
    return ((arr + 180) % 360) - 180


def lon_in_bounds(lon_wrapped: np.ndarray, lon_min: float, lon_max: float) -> np.ndarray:
    """
    Dateline-aware longitude window check.
    
    If lon_min > lon_max, assumes the interval crosses the dateline
    and uses an OR mask instead of AND.
    
    Parameters
    ----------
    lon_wrapped : np.ndarray
        Wrapped longitude values in [-180, 180]
    lon_min : float
        Minimum longitude bound
    lon_max : float
        Maximum longitude bound
        
    Returns
    -------
    np.ndarray
        Boolean mask for points within bounds
        
    Examples
    --------
    >>> lon_in_bounds(np.array([170, -170, 0]), 160, -160)  # crosses dateline
    array([ True,  True, False])
    """
    if lon_min <= lon_max:
        return (lon_wrapped >= lon_min) & (lon_wrapped <= lon_max)
    # Crosses dateline
    return (lon_wrapped >= lon_min) | (lon_wrapped <= lon_max)


def get_lon_lat_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract longitude and latitude arrays from dataset.
    
    Handles multiple naming conventions: longitude/latitude, lon/lat.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing coordinate data
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (longitude, latitude) arrays
        
    Raises
    ------
    ValueError
        If coordinates cannot be found
    """
    if "longitude" in ds and "latitude" in ds:
        return ds["longitude"].values, ds["latitude"].values
    elif "lon" in ds and "lat" in ds:
        return ds["lon"].values, ds["lat"].values
    else:
        raise ValueError("Could not find longitude/latitude variables in dataset")


def create_spatial_mask(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Create a spatial mask for filtering data by coordinates.
    
    Parameters
    ----------
    lon : np.ndarray
        Longitude array
    lat : np.ndarray
        Latitude array
    lon_range : tuple, optional
        (lon_min, lon_max) filter range
    lat_range : tuple, optional
        (lat_min, lat_max) filter range
        
    Returns
    -------
    np.ndarray
        Boolean mask
    """
    mask = np.ones(len(lat), dtype=bool)
    
    if lat_range is not None:
        mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])
    
    if lon_range is not None:
        lon_wrapped = wrap_longitudes(lon)
        lon_min, lon_max = lon_range
        mask &= lon_in_bounds(lon_wrapped, lon_min, lon_max)
    
    return mask
