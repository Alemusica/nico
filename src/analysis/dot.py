"""
DOT (Dynamic Ocean Topography) Computation
==========================================
Functions for computing DOT from SSH and reference surfaces.
"""

import numpy as np
import pandas as pd
import xarray as xr


def compute_dot(
    ds: xr.Dataset,
    ssh_var: str = "corssh",
    reference_var: str = "mean_sea_surface",
) -> xr.DataArray:
    """
    Compute Dynamic Ocean Topography (DOT).
    
    DOT = Sea Surface Height - Reference Surface (MSS or Geoid)
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing SSH and reference surface
    ssh_var : str
        Name of sea surface height variable
    reference_var : str
        Name of reference surface variable ('mean_sea_surface' or 'geoid')
        
    Returns
    -------
    xr.DataArray
        DOT values in meters
        
    Raises
    ------
    ValueError
        If required variables are missing
    """
    if ssh_var not in ds.data_vars:
        raise ValueError(f"SSH variable '{ssh_var}' not found in dataset")
    
    if reference_var not in ds.data_vars:
        raise ValueError(f"Reference variable '{reference_var}' not found in dataset")
    
    dot = ds[ssh_var] - ds[reference_var]
    dot.name = "DOT"
    dot.attrs["long_name"] = "Dynamic Ocean Topography"
    dot.attrs["units"] = "m"
    dot.attrs["ssh_variable"] = ssh_var
    dot.attrs["reference_variable"] = reference_var
    
    return dot


def compute_dot_dataframe(
    ds: xr.Dataset,
    ssh_var: str = "corssh",
    reference_var: str = "geoid",
) -> pd.DataFrame:
    """
    Compute DOT and return as DataFrame with all relevant columns.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing SSH, reference, and coordinates
    ssh_var : str
        Name of SSH variable
    reference_var : str
        Name of reference variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: lat, lon, time, corssh, reference, dot, cycle, month, year
    """
    dot = ds[ssh_var].values - ds[reference_var].values
    
    df = pd.DataFrame({
        "lat": ds["latitude"].values,
        "lon": ds["longitude"].values,
        "time": pd.to_datetime(ds["time"].values),
        ssh_var: ds[ssh_var].values,
        reference_var: ds[reference_var].values,
        "dot": dot,
    })
    
    if "cycle" in ds.coords:
        df["cycle"] = ds["cycle"].values
    
    if "pass" in ds:
        df["pass"] = ds["pass"].values
    
    # Add time components
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df["year_month"] = df["time"].dt.to_period("M")
    
    return df


def get_dot_statistics(dot: np.ndarray | xr.DataArray) -> dict:
    """
    Compute basic statistics for DOT values.
    
    Parameters
    ----------
    dot : array-like
        DOT values
        
    Returns
    -------
    dict
        Statistics including mean, std, min, max, etc.
    """
    values = np.asarray(dot).flatten()
    valid = values[np.isfinite(values)]
    
    if len(valid) == 0:
        return {"n_valid": 0}
    
    return {
        "n_valid": len(valid),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "median": float(np.median(valid)),
        "range": float(np.max(valid) - np.min(valid)),
    }
