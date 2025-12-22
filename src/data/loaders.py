"""
Data Loaders
============
Functions for loading NetCDF satellite altimetry data.
"""

import os
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr

from ..core.satellite import detect_satellite_type
from ..core.coordinates import wrap_longitudes, lon_in_bounds, get_lon_lat_arrays
from ..core.helpers import extract_cycle_number


def load_cycle(
    filepath: str | Path,
    decode_times: bool = False,
) -> xr.Dataset | None:
    """
    Load a single cycle NetCDF file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to NetCDF file
    decode_times : bool
        Whether to decode time values
        
    Returns
    -------
    xr.Dataset or None
        Loaded dataset, or None if loading fails
    """
    try:
        return xr.open_dataset(filepath, decode_times=decode_times)
    except Exception:
        return None


def load_multiple_cycles(
    base_dir: str | Path,
    cycles: list[int] | range,
    satellite_type: str | None = None,
    verbose: bool = True,
) -> Iterator[tuple[int, xr.Dataset]]:
    """
    Generator that yields (cycle_number, dataset) tuples.
    
    Parameters
    ----------
    base_dir : str or Path
        Directory containing SLCCI NetCDF files
    cycles : list or range
        Cycle numbers to load
    satellite_type : str, optional
        'J1' or 'J2', auto-detected if None
    verbose : bool
        Print loading progress
        
    Yields
    ------
    tuple[int, xr.Dataset]
        (cycle_number, dataset) for each successfully loaded cycle
    """
    base_dir = Path(base_dir)
    
    if satellite_type is None:
        satellite_type = detect_satellite_type(base_dir)
    
    if verbose:
        print(f"🛰️ Loading {satellite_type} cycles from {base_dir}")
    
    for cycle in cycles:
        cycle_str = str(cycle).zfill(3)
        filename = f"SLCCI_ALTDB_{satellite_type}_Cycle{cycle_str}_V2.nc"
        filepath = base_dir / filename
        
        if not filepath.exists():
            continue
        
        ds = load_cycle(filepath)
        if ds is not None:
            if verbose:
                n_points = ds.sizes.get("time", 0)
                print(f"  ✅ Cycle {cycle}: {n_points:,} points")
            yield cycle, ds


def load_from_upload(file_obj) -> xr.Dataset | None:
    """
    Load NetCDF from Streamlit uploaded file object.
    
    Parameters
    ----------
    file_obj : UploadedFile
        Streamlit file uploader object
        
    Returns
    -------
    xr.Dataset or None
        Loaded dataset
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp:
            tmp.write(file_obj.getvalue())
            tmp_path = tmp.name
        
        ds = xr.open_dataset(tmp_path)
        os.unlink(tmp_path)
        return ds
    except Exception:
        return None


def load_filtered_cycles(
    cycles: list[int] | range,
    base_dir: str | Path,
    lat_range: tuple[float, float] | None = None,
    lon_range: tuple[float, float] | None = None,
    use_flag: bool = True,
    pass_number: int | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Load and filter satellite altimetry cycles.
    
    This is the main data loading function that:
    - Loads cycles from NetCDF files
    - Applies spatial filtering
    - Applies quality flag filtering
    - Optionally filters by pass number
    - Combines into single dataset
    
    Parameters
    ----------
    cycles : list or range
        Cycle numbers to load
    base_dir : str or Path
        Directory with SLCCI files
    lat_range : tuple, optional
        (lat_min, lat_max) spatial filter
    lon_range : tuple, optional
        (lon_min, lon_max) spatial filter
    use_flag : bool
        Apply validation_flag == 0 filter
    pass_number : int, optional
        Filter to specific pass
    verbose : bool
        Print progress
        
    Returns
    -------
    xr.Dataset
        Combined, filtered dataset
    """
    satellite_type = detect_satellite_type(base_dir)
    base_dir = Path(base_dir)
    
    cycle_datasets = []
    loaded_cycles = []
    
    for cycle in cycles:
        cycle_str = str(cycle).zfill(3)
        filename = f"SLCCI_ALTDB_{satellite_type}_Cycle{cycle_str}_V2.nc"
        filepath = base_dir / filename
        
        if not filepath.exists():
            continue
        
        try:
            ds = xr.open_dataset(filepath, decode_times=False)
            
            # Get coordinates
            lon, lat = get_lon_lat_arrays(ds)
            lon_wrapped = wrap_longitudes(lon)
            
            # Spatial mask
            mask = np.ones(len(lat), dtype=bool)
            
            if lat_range:
                mask &= (lat >= lat_range[0]) & (lat <= lat_range[1])
            
            if lon_range:
                mask &= lon_in_bounds(lon_wrapped, lon_range[0], lon_range[1])
            
            if mask.sum() == 0:
                continue
            
            ds_filtered = ds.isel(time=mask)
            
            # Update longitude to wrapped values
            ds_filtered = ds_filtered.assign_coords(
                longitude=(("time",), lon_wrapped[mask])
            )
            
            # Decode time
            time_vals = pd.to_datetime(
                ds_filtered["time"].values, origin="1950-01-01", unit="D"
            )
            ds_filtered = ds_filtered.assign_coords(time=time_vals)
            
            # Quality filter
            if use_flag and "validation_flag" in ds_filtered:
                valid_mask = ds_filtered["validation_flag"] == 0
                ds_filtered = ds_filtered.isel(time=valid_mask)
            
            # Pass filter
            if pass_number is not None and "pass" in ds_filtered:
                pass_vals = np.round(ds_filtered["pass"].values).astype(int)
                pass_mask = pass_vals == int(pass_number)
                if pass_mask.sum() == 0:
                    continue
                ds_filtered = ds_filtered.isel(time=pass_mask)
            
            if ds_filtered.sizes.get("time", 0) == 0:
                continue
            
            # Add cycle coordinate
            ds_filtered = ds_filtered.assign_coords(
                cycle=("time", np.full(ds_filtered.sizes["time"], cycle))
            )
            
            cycle_datasets.append(ds_filtered)
            loaded_cycles.append(cycle)
            
            if verbose:
                print(f"  ✅ Cycle {cycle}: {ds_filtered.sizes['time']:,} points")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ Cycle {cycle}: {e}")
            continue
    
    if not cycle_datasets:
        raise ValueError("No cycles could be loaded with the given filters")
    
    combined = xr.concat(cycle_datasets, dim="time")
    combined.attrs["satellite_type"] = satellite_type
    combined.attrs["loaded_cycles"] = loaded_cycles
    
    return combined
