"""
Utility functions for loading and processing Jason-1 and Jason-2 satellite altimetry data.

This module provides functions to:
- Load filtered cycles from both J1 and J2 datasets
- Interpolate geoid values
- Filter passes from cycles
- Add geoid data to cycles

The functions are designed to work with both J1 and J2 folder structures,
automatically detecting the satellite type from the base directory name.
"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import os


# -----------------------------------------------------------------------------
# BASIC HELPERS
# -----------------------------------------------------------------------------

def detect_satellite_type(base_dir):
    """
    Detect satellite type (J1 or J2) from the base directory name.
    
    Parameters
    ----------
    base_dir : str or Path
        Base directory path containing the satellite data
        
    Returns
    -------
    str
        'J1' or 'J2' based on directory name, defaults to 'J1' if unclear
    """
    base_dir_str = os.fspath(base_dir)
    dir_name = os.path.basename(base_dir_str.rstrip("/"))
    if "J2" in dir_name.upper():
        return "J2"
    else:
        return "J1"  # Default to J1


def _detect_pass_var(ds):
    """
    Detect which variable contains pass/track information.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to search for pass variable
        
    Returns
    -------
    str or None
        Name of the pass variable if found, None otherwise
    """
    candidates = ["pass", "track", "pass_number", "track_number"]
    for var in candidates:
        if var in ds.variables:
            return var
    return None


def _get_lon_lat_arrays(ds):
    """
    Extract longitude and latitude arrays from dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing longitude and latitude data
        
    Returns
    -------
    tuple
        (longitude, latitude) numpy arrays
    """
    if "longitude" in ds and "latitude" in ds:
        return ds["longitude"].values, ds["latitude"].values
    elif "lon" in ds and "lat" in ds:
        return ds["lon"].values, ds["lat"].values
    else:
        raise ValueError("Could not find longitude/latitude variables in dataset")


def _wrap_longitudes(values):
    """Wrap longitudes to [-180, 180]."""
    arr = np.asarray(values, dtype=float)
    return ((arr + 180) % 360) - 180


def _lon_in_bounds(lon_wrapped, lon_min, lon_max):
    """
    Dateline-aware longitude window check.
    If lon_min > lon_max we assume the interval crosses the dateline and
    use an OR mask instead of AND.
    """
    if lon_min <= lon_max:
        return (lon_wrapped >= lon_min) & (lon_wrapped <= lon_max)
    return (lon_wrapped >= lon_min) | (lon_wrapped <= lon_max)


# -----------------------------------------------------------------------------
# MAIN LOADING FUNCTION
# -----------------------------------------------------------------------------

def load_filtered_cycles_serial_J2(
    cycles,
    gate_path,
    base_dir,
    use_flag=True,
    pass_number=None,
    lat_override=None,
    lon_override=None,
    verbose=True,
    lat_buffer_deg=2.0,
    lon_buffer_deg=5.0,
):
    """
    Load and filter satellite altimetry cycles for a specific region.

    Features:
    - Detects J1/J2 from base_dir.
    - Spatial filter from gate bounds (with buffer) or lat/lon override.
    - Optionally filters by pass number using a robustly detected pass/track variable.
    - Standardizes the pass variable name to 'pass' in the output dataset.

    Parameters
    ----------
    cycles : iterable of int
        Cycle numbers to attempt loading (e.g. range(1, 200)).
    gate_path : str or Path
        Path to a vector file (e.g. shapefile/geojson) defining the gate area.
        Used to infer a spatial bounding box if lat_override/lon_override are not given.
    base_dir : str or Path
        Folder containing SLCCI_ALTDB_*_CycleXXX_V2.nc files.
    use_flag : bool, optional
        If True, filter using 'validation_flag == 0' when available.
    pass_number : int or None, optional
        If provided, only keep points for this pass/track.
    lat_override : tuple(float, float) or None
        If provided, override latitude filtering window as (lat_min, lat_max).
    lon_override : tuple(float, float) or None
        If provided, override longitude filtering window as (lon_min, lon_max)
        in degrees [-180, 180].
    verbose : bool, optional
        If True, print detailed loading information for each cycle.
    lat_buffer_deg : float, optional
        Extra latitude margin (degrees) added to gate bounds when no override.
    lon_buffer_deg : float, optional
        Extra longitude margin (degrees) added to gate bounds when no override.

    Returns
    -------
    xarray.Dataset
        Combined dataset of all loaded cycles (dimension 'time'), with:
        - attribute 'satellite_type'
        - attribute 'loaded_cycles'
        - attribute 'spatial_bounds'
        - attribute 'quality_filtered'
        - attribute 'pass_number'
        - variable 'pass' always present when pass information existed in source files.
    """

    satellite_type = detect_satellite_type(base_dir)
    if verbose:
        print(f"🛰️ Detected satellite type: {satellite_type}")

    gate = gpd.read_file(gate_path).to_crs("EPSG:4326")
    lon_min_g, lat_min_g, lon_max_g, lat_max_g = gate.total_bounds

    # Use manual override if provided, else gate bounds + buffer
    lat_min = float(lat_override[0]) if lat_override else float(lat_min_g) - float(lat_buffer_deg)
    lat_max = float(lat_override[1]) if lat_override else float(lat_max_g) + float(lat_buffer_deg)
    lon_min = float(lon_override[0]) if lon_override else float(lon_min_g) - float(lon_buffer_deg)
    lon_max = float(lon_override[1]) if lon_override else float(lon_max_g) + float(lon_buffer_deg)

    # Wrap longitudes to [-180, 180] and detect dateline crossing
    lon_min = float(_wrap_longitudes([lon_min])[0])
    lon_max = float(_wrap_longitudes([lon_max])[0])
    crosses_dateline = lon_min > lon_max

    if verbose:
        print(
            f"🌊 Search window: "
            f"lon=[{lon_min:.2f}, {lon_max:.2f}]"
            f"{' (dateline wrap)' if crosses_dateline else ''}, "
            f"lat=[{lat_min:.2f}, {lat_max:.2f}]"
        )

    base_dir_str = os.fspath(base_dir)

    cycle_datasets = []
    loaded_cycles = []

    for cycle in cycles:
        cycle_str = str(cycle).zfill(3)
        filename = f"SLCCI_ALTDB_{satellite_type}_Cycle{cycle_str}_V2.nc"
        filepath = os.path.join(base_dir_str, filename)

        if not os.path.exists(filepath):
            if verbose:
                print(f"⚠️ Cycle {cycle}: File not found - {filename}")
            continue

        try:
            with xr.open_dataset(filepath, decode_times=False) as ds:
                if verbose:
                    print(f"\n📁 Cycle {cycle}: opening {filename}")
                    print(f"   Variables in file: {list(ds.variables.keys())}")

                # Lon/lat + wrap to [-180, 180]
                lon, lat = _get_lon_lat_arrays(ds)
                lon_wrapped = _wrap_longitudes(lon)

                # Spatial mask
                lon_mask = _lon_in_bounds(lon_wrapped, lon_min, lon_max)
                mask_spatial = (
                    (lat >= lat_min)
                    & (lat <= lat_max)
                    & lon_mask
                )
                if mask_spatial.sum() == 0:
                    if verbose:
                        print(f"   ⛔ Cycle {cycle}: no points in the spatial window")
                    continue

                # Apply spatial mask on 'time' dimension
                ds_filtered = ds.isel(time=mask_spatial)

                # Fix longitude coord to wrapped one
                ds_filtered = ds_filtered.assign_coords(
                    longitude=(("time",), lon_wrapped[mask_spatial])
                )

                # Decode time from days since 1950-01-01
                time_vals = pd.to_datetime(
                    ds_filtered["time"].values, origin="1950-01-01", unit="D"
                )
                ds_filtered = ds_filtered.assign_coords(time=time_vals)

                # Quality flag
                if use_flag and "validation_flag" in ds_filtered:
                    valid_mask = ds_filtered["validation_flag"] == 0
                    n_before = ds_filtered.sizes.get("time", 0)
                    ds_filtered = ds_filtered.isel(time=valid_mask)
                    n_after = ds_filtered.sizes.get("time", 0)
                    if verbose:
                        print(
                            f"   ✅ Quality filter: {n_after}/{n_before} points "
                            f"(validation_flag == 0)"
                        )

                # Detect pass variable
                pass_var = _detect_pass_var(ds_filtered)
                if pass_var:
                    if verbose:
                        print(f"   🔎 Detected pass variable: '{pass_var}'")
                    # Standardize name to 'pass'
                    if pass_var != "pass":
                        ds_filtered = ds_filtered.rename({pass_var: "pass"})
                        pass_var = "pass"
                else:
                    if verbose:
                        print("   ⚠️ No pass/track variable found in this file.")

                # Filter by pass_number if requested
                if pass_number is not None:
                    if not pass_var:
                        if verbose:
                            print(
                                f"   ⚠️ pass_number={pass_number} requested but "
                                f"no pass variable in cycle {cycle}. Skipping this cycle."
                            )
                        continue

                    pass_vals = ds_filtered[pass_var].values
                    pass_int = np.round(pass_vals).astype(int)
                    mask_pass = pass_int == int(pass_number)

                    if mask_pass.sum() == 0:
                        if verbose:
                            print(
                                f"   ⚠️ Cycle {cycle}: pass {pass_number} "
                                f"not found (variable '{pass_var}')"
                            )
                        continue

                    ds_filtered = ds_filtered.isel(time=mask_pass)
                    if verbose:
                        print(
                            f"   🎯 Cycle {cycle}: {ds_filtered.sizes.get('time', 0)} "
                            f"points for pass {pass_number}"
                        )
                else:
                    if verbose:
                        print(
                            f"   📌 Cycle {cycle}: keeping all passes "
                            f"({ds_filtered.sizes.get('time', 0)} points)"
                        )

                if ds_filtered.sizes.get("time", 0) == 0:
                    if verbose:
                        print(f"   ⚠️ Cycle {cycle}: no data after all filters.")
                    continue

                # Add cycle as coordinate
                ds_filtered = ds_filtered.assign_coords(
                    cycle=("time", np.full(ds_filtered.sizes["time"], cycle))
                )

                cycle_datasets.append(ds_filtered)
                loaded_cycles.append(cycle)

        except Exception as e:
            if verbose:
                print(f"❌ Error loading cycle {cycle}: {str(e)}")
            continue

    if not cycle_datasets:
        raise ValueError("No cycles could be loaded successfully with the given filters.")

    if verbose:
        print(f"\n🎯 Successfully loaded {len(loaded_cycles)} cycles: {loaded_cycles}")
    
    combined_ds = xr.concat(cycle_datasets, dim="time")

    # Add metadata
    combined_ds.attrs["satellite_type"] = satellite_type
    combined_ds.attrs["loaded_cycles"] = loaded_cycles
    combined_ds.attrs["spatial_bounds"] = {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max,
    }
    combined_ds.attrs["quality_filtered"] = use_flag
    if pass_number is not None:
        combined_ds.attrs["pass_number"] = pass_number

    return combined_ds


# -----------------------------------------------------------------------------
# GEOID INTERPOLATION
# -----------------------------------------------------------------------------

def interpolate_geoid(geoid_path, target_lats, target_lons):
    """
    Interpolate geoid values at target lat/lon positions.
    
    Parameters
    ----------
    geoid_path : str
        Path to geoid NetCDF file
    target_lats : array-like
        Target latitudes
    target_lons : array-like
        Target longitudes (will be wrapped to [-180, 180])
        
    Returns
    -------
    numpy.ndarray
        Interpolated geoid values
    """
    ds_geoid = xr.open_dataset(geoid_path)
    lat_geoid = ds_geoid["lat"].values
    lon_geoid = ds_geoid["lon"].values
    geoid_values = ds_geoid["value"].values

    # Wrap longitudes to [-180, 180] and sort
    lon_wrapped = ((lon_geoid + 180) % 360) - 180
    sort_idx = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[sort_idx]
    
    # Remove duplicates
    unique_idx = np.concatenate(([True], np.diff(lon_sorted) != 0))
    lon_sorted = lon_sorted[unique_idx]
    geoid_sorted = geoid_values[:, sort_idx][:, unique_idx]

    # Create interpolator
    interp = RegularGridInterpolator(
        (lat_geoid, lon_sorted),
        geoid_sorted,
        method="nearest",
        bounds_error=False,
        fill_value=np.nan,
    )

    # Wrap target longitudes
    target_lons_wrapped = ((target_lons + 180) % 360) - 180
    points = np.column_stack([target_lats, target_lons_wrapped])
    
    return interp(points)


# -----------------------------------------------------------------------------
# GEOID APPLICATION TO CYCLES
# -----------------------------------------------------------------------------

def add_geoid_to_cycles(ds_combined, geoid_interp):
    """
    Add geoid values to each cycle in the combined dataset.
    
    Parameters
    ----------
    ds_combined : xarray.Dataset
        Combined dataset from load_filtered_cycles_serial_J2
    geoid_interp : numpy.ndarray
        Interpolated geoid values (same length as time dimension)
        
    Returns
    -------
    dict
        Dictionary mapping cycle_num -> xarray.Dataset with geoid added
    """
    sub_cycles = {}
    
    unique_cycles = np.unique(ds_combined["cycle"].values)
    
    for cycle_num in unique_cycles:
        cycle_num = int(cycle_num)
        ds_cycle = ds_combined.where(ds_combined["cycle"] == cycle_num, drop=True)
        
        if ds_cycle.sizes.get("time", 0) == 0:
            continue
        
        # Get indices for this cycle in the original combined dataset
        cycle_mask = ds_combined["cycle"].values == cycle_num
        cycle_geoid = geoid_interp[cycle_mask]
        
        # Add geoid as a variable
        ds_cycle = ds_cycle.assign(geoid=(("time",), cycle_geoid))
        
        sub_cycles[cycle_num] = ds_cycle
    
    return sub_cycles


# -----------------------------------------------------------------------------
# PASS FILTERING
# -----------------------------------------------------------------------------

def filter_pass_from_cycles(sub_cycles, pass_number):
    """
    Filter cycles to keep only data from a specific pass.
    
    Parameters
    ----------
    sub_cycles : dict
        Dictionary of cycle datasets from add_geoid_to_cycles
    pass_number : int
        Pass number to filter
        
    Returns
    -------
    dict
        Filtered dictionary with only the specified pass
    """
    filtered = {}
    
    for cycle_num, ds in sub_cycles.items():
        if "pass" not in ds:
            continue
        
        pass_vals = ds["pass"].values
        pass_int = np.round(pass_vals).astype(int)
        mask = pass_int == int(pass_number)
        
        if np.any(mask):
            filtered[cycle_num] = ds.isel(time=mask)
    
    return filtered


# -----------------------------------------------------------------------------
# DOT CALCULATION
# -----------------------------------------------------------------------------

def calculate_dot(ds_cycle):
    """
    Calculate Dynamic Ocean Topography (DOT) from SSH and geoid.
    
    Parameters
    ----------
    ds_cycle : xarray.Dataset
        Dataset with 'corssh' and 'geoid' variables
        
    Returns
    -------
    xarray.DataArray
        DOT values (corssh - geoid)
    """
    if "corssh" not in ds_cycle or "geoid" not in ds_cycle:
        raise ValueError("Dataset must contain 'corssh' and 'geoid' variables")
    
    return ds_cycle["corssh"] - ds_cycle["geoid"]


# -----------------------------------------------------------------------------
# PASS STATISTICS
# -----------------------------------------------------------------------------

def get_pass_statistics(sub_cycles_pass, pass_number):
    """
    Calculate statistics for a specific pass across all cycles.
    
    Parameters
    ----------
    sub_cycles_pass : dict
        Dictionary of filtered cycle datasets for a specific pass
    pass_number : int
        Pass number
        
    Returns
    -------
    dict
        Statistics including mean, std, min, max DOT values and time range
    """
    all_dots = []
    all_times = []
    cycles = []

    for cycle_num, ds in sub_cycles_pass.items():
        if "corssh" in ds.variables and "geoid" in ds.variables:
            dot = calculate_dot(ds)
            all_dots.extend(dot.values)
            all_times.extend(pd.to_datetime(ds["time"].values))
            cycles.append(cycle_num)

    stats = {
        "pass_number": pass_number,
        "n_cycles": len(cycles),
        "cycles": sorted(cycles),
        "total_points": len(all_dots),
    }

    if all_dots:
        stats.update(
            {
                "dot_mean": np.nanmean(all_dots),
                "dot_std": np.nanstd(all_dots),
                "dot_min": np.nanmin(all_dots),
                "dot_max": np.nanmax(all_dots),
                "time_range": (min(all_times), max(all_times)),
            }
        )

    return stats


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR MULTI-STRAIT PROCESSING
# -----------------------------------------------------------------------------

def extract_strait_info(path):
    """Extract strait name and pass number from a gate shapefile path."""
    import re
    filename = Path(path).stem
    strait_name = filename.replace("_", " ").replace("-", " ").title()
    match = re.search(r'pass[_\s]*(\d+)', filename, re.IGNORECASE)
    pass_from_filename = int(match.group(1)) if match else None
    return strait_name, pass_from_filename


def find_closest_pass_to_gate(gate_gdf, base_dir, geoid_interp, cycles_to_load, use_flag=True):
    """
    Find the satellite pass that passes closest to the gate centroid.
    Uses same logic as find_closest_passes_to_gate but returns only the best pass.
    """
    import tempfile
    from shapely.geometry import box
    
    gate_centroid = gate_gdf.geometry.centroid.iloc[0]
    gate_lon, gate_lat = gate_centroid.x, gate_centroid.y
    print(f"   📍 Gate centroid: lon={gate_lon:.4f}°, lat={gate_lat:.4f}°")
    
    # Try progressively wider windows around the gate to find any data
    gate_bounds = gate_gdf.total_bounds  # [minx, miny, maxx, maxy]
    expand_candidates = [5.0, 10.0, 20.0]
    ds_all = None
    used_expand = None
    
    for expand_deg in expand_candidates:
        expanded_bounds = box(
            gate_bounds[0] - expand_deg,
            gate_bounds[1] - expand_deg,
            gate_bounds[2] + expand_deg,
            gate_bounds[3] + expand_deg,
        )
        temp_gdf = gpd.GeoDataFrame(geometry=[expanded_bounds], crs="EPSG:4326")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_shp_path = Path(tmpdir) / f"temp_expanded_gate_{int(expand_deg)}.shp"
            temp_gdf.to_file(temp_shp_path)
            
            try:
                ds_all = load_filtered_cycles_serial_J2(
                    cycles=cycles_to_load,
                    gate_path=str(temp_shp_path),
                    base_dir=str(base_dir),
                    use_flag=use_flag,
                    verbose=False,
                    lat_override=(expanded_bounds.bounds[1], expanded_bounds.bounds[3]),
                    lon_override=(expanded_bounds.bounds[0], expanded_bounds.bounds[2]),
                    lat_buffer_deg=0.0,
                    lon_buffer_deg=0.0,
                )
            except Exception as e:
                print(f"   ⚠️ Error loading data (±{expand_deg}° window): {e}")
                ds_all = None
            
        if ds_all is not None and ds_all.sizes.get("time", 0) > 0:
            used_expand = expand_deg
            break
    
    if ds_all is None or ds_all.sizes.get("time", 0) == 0:
        print("   ⚠️ No data loaded after expanded searches, returning pass 1 as fallback")
        return 1
    else:
        print(
            f"   ✅ Data found with ±{used_expand:.0f}° search window "
            f"({ds_all.sizes.get('time', 0)} points)"
        )
    
    # Add geoid using pre-computed interpolator
    lats, lons = ds_all["latitude"].values, ds_all["longitude"].values
    lons_wrapped = _wrap_longitudes(lons)
    points = np.column_stack([lats, lons_wrapped])
    geoid_values = geoid_interp(points)
    ds_all = ds_all.assign(geoid=(("time",), geoid_values))
    
    # Get cycles with geoid
    sub_cycles = {}
    for cycle_num in cycles_to_load:
        ds_cycle = ds_all.where(ds_all["cycle"] == cycle_num, drop=True)
        if ds_cycle.sizes.get("time", 0) > 0:
            sub_cycles[cycle_num] = ds_cycle
    
    if not sub_cycles:
        print("   ⚠️ No valid cycles found, returning pass 1 as fallback")
        return 1
    
    # Get all unique passes
    all_passes = set()
    for cycle_num, ds in sub_cycles.items():
        if "pass" in ds:
            passes_in_cycle = np.unique(ds["pass"].values)
            all_passes.update(int(p) for p in passes_in_cycle if not np.isnan(p))
    
    if not all_passes:
        print("   ⚠️ No passes found in data, returning pass 1 as fallback")
        return 1
    
    print(f"   🔍 Analyzing {len(all_passes)} unique passes...")
    
    # Calculate minimum distance for each pass
    pass_min_distances = {}
    for pass_num in sorted(all_passes):
        all_lons, all_lats = [], []
        for cycle_num, ds in sub_cycles.items():
            if "pass" not in ds:
                continue
            mask = ds["pass"].values == pass_num
            if np.any(mask):
                all_lons.extend(ds["longitude"].values[mask])
                all_lats.extend(ds["latitude"].values[mask])
        
        if not all_lons:
            continue
        
        distances = []
        for lon, lat in zip(all_lons, all_lats):
            dlat = (lat - gate_lat) * 111000
            dlon_deg = ((lon - gate_lon + 180) % 360) - 180  # shortest path across dateline
            dlon = dlon_deg * 111000 * np.cos(np.radians((lat + gate_lat) / 2))
            distances.append(np.sqrt(dlat**2 + dlon**2))
        pass_min_distances[pass_num] = np.min(distances)
    
    if not pass_min_distances:
        print("   ⚠️ Could not calculate distances, returning pass 1 as fallback")
        return 1
    
    best_pass = min(pass_min_distances.keys(), key=lambda p: pass_min_distances[p])
    best_distance_km = pass_min_distances[best_pass] / 1000.0
    print(f"   ✅ Best pass: {best_pass} (min distance: {best_distance_km:.2f} km)")
    return best_pass


def load_pass_dataframe(gate_path, pass_num, cycles, base_dir, geoid_path, use_flag=True):
    """Load satellite data for a specific pass and return as DataFrame."""
    ds_pass = load_filtered_cycles_serial_J2(
        cycles=cycles, gate_path=gate_path, base_dir=str(base_dir),
        use_flag=use_flag, pass_number=pass_num, verbose=False,
    )
    
    n_points = ds_pass.sizes.get("time", 0)
    if n_points == 0:
        return None, 0
    
    geoid_interp_local = interpolate_geoid(
        geoid_path=geoid_path,
        target_lats=ds_pass["latitude"].values,
        target_lons=ds_pass["longitude"].values,
    )
    
    sub_cycles = add_geoid_to_cycles(ds_pass, geoid_interp_local)
    sub_cycles_pass = filter_pass_from_cycles(sub_cycles, pass_num)
    
    data_list = []
    for cycle_num, ds_cycle in sub_cycles_pass.items():
        dot = ds_cycle["corssh"].values - ds_cycle["geoid"].values
        df_cycle = pd.DataFrame({
            "cycle": cycle_num, "pass": pass_num,
            "lat": ds_cycle["latitude"].values, "lon": ds_cycle["longitude"].values,
            "corssh": ds_cycle["corssh"].values, "geoid": ds_cycle["geoid"].values,
            "dot": dot, "time": pd.to_datetime(ds_cycle["time"].values),
        })
        data_list.append(df_cycle)
    
    if data_list:
        pass_df = pd.concat(data_list, ignore_index=True)
        pass_df["month"] = pass_df["time"].dt.month
        pass_df["year"] = pass_df["time"].dt.year
        return pass_df, len(pass_df)
    return None, 0


def get_gate_profile_points(gate_gdf, gate_points_cache=None, n_gate_pts=200):
    """Sample points along the gate line for profile analysis."""
    from shapely.ops import linemerge
    from shapely.geometry import MultiLineString, LineString
    
    gate_bounds = tuple(gate_gdf.total_bounds)
    if gate_points_cache is not None and gate_bounds in gate_points_cache:
        return gate_points_cache[gate_bounds]
    
    geom = gate_gdf.geometry.unary_union
    if isinstance(geom, MultiLineString):
        geom = linemerge(geom)
    
    if isinstance(geom, LineString):
        total_length = geom.length
        distances = np.linspace(0, total_length, n_gate_pts)
        points = [geom.interpolate(d) for d in distances]
        gate_lon_pts = np.array([p.x for p in points])
        gate_lat_pts = np.array([p.y for p in points])
    else:
        bounds = gate_gdf.total_bounds
        gate_lon_pts = np.linspace(bounds[0], bounds[2], n_gate_pts)
        gate_lat_pts = np.linspace(bounds[1], bounds[3], n_gate_pts)
    
    R_earth = 6371.0
    lat0_rad = np.deg2rad(np.mean(gate_lat_pts))
    lon_rad = np.deg2rad(gate_lon_pts)
    x_km = (lon_rad - lon_rad[0]) * np.cos(lat0_rad) * R_earth
    
    result = (gate_lon_pts, gate_lat_pts, x_km)
    if gate_points_cache is not None:
        gate_points_cache[gate_bounds] = result
    return result
