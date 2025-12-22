"""
Data Filters
=============
Functions for filtering satellite altimetry data.
"""

import numpy as np
import xarray as xr


def apply_quality_filter(
    ds: xr.Dataset,
    flag_var: str = "validation_flag",
    valid_value: int = 0,
) -> xr.Dataset:
    """
    Filter dataset by quality flag.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    flag_var : str
        Name of quality flag variable
    valid_value : int
        Value indicating valid data
        
    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    if flag_var not in ds:
        return ds
    
    valid_mask = ds[flag_var] == valid_value
    return ds.isel(time=valid_mask)


def filter_by_pass(
    ds: xr.Dataset,
    pass_number: int,
    pass_var: str = "pass",
) -> xr.Dataset:
    """
    Filter dataset to keep only data from a specific pass.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    pass_number : int
        Pass number to keep
    pass_var : str
        Name of pass variable
        
    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    if pass_var not in ds:
        # Try alternative names
        for alt in ["track", "pass_number", "track_number"]:
            if alt in ds:
                pass_var = alt
                break
        else:
            return ds
    
    pass_vals = np.round(ds[pass_var].values).astype(int)
    mask = pass_vals == int(pass_number)
    
    return ds.isel(time=mask)


def filter_by_time_range(
    ds: xr.Dataset,
    start_time=None,
    end_time=None,
) -> xr.Dataset:
    """
    Filter dataset by time range.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with time coordinate
    start_time : datetime-like, optional
        Start of time range
    end_time : datetime-like, optional
        End of time range
        
    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    mask = np.ones(ds.sizes.get("time", 0), dtype=bool)
    
    time_vals = ds["time"].values
    
    if start_time is not None:
        mask &= time_vals >= np.datetime64(start_time)
    
    if end_time is not None:
        mask &= time_vals <= np.datetime64(end_time)
    
    return ds.isel(time=mask)


def remove_outliers(
    data: np.ndarray,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Remove outliers beyond n standard deviations.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    n_sigma : float
        Number of standard deviations for outlier threshold
        
    Returns
    -------
    np.ndarray
        Boolean mask (True = keep, False = outlier)
    """
    mean = np.nanmean(data)
    std = np.nanstd(data)
    
    return np.abs(data - mean) <= n_sigma * std
