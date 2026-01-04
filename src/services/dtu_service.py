"""
DTUSpace Service
================
Service for loading and processing DTUSpace v4 gridded DOT products.

Dataset Info:
    - Name: DTUSpace Arctic Ocean DOT v4.0
    - Source: DTU Space (Technical University of Denmark)
    - Type: GRIDDED (lat × lon × time)
    - Local file only (no API)

This is a GRIDDED dataset (lat × lon × time), NOT along-track like SLCCI/CMEMS L3.
There are no real satellite passes - the "pass" is synthetically defined by the gate.

Data Flow:
    UI → DTUService → xr.open_dataset → NetCDF file (local only)
                    → KD-tree gate matching → DOT along gate
                    → slope computation → time series

Comparison with other datasets:
    | Dataset      | Type        | Filter Variable | Source    |
    |--------------|-------------|-----------------|-----------|
    | SLCCI        | Along-track | pass            | Local     |
    | CMEMS L3     | Along-track | track           | Local     |
    | CMEMS L4     | Gridded     | (none)          | API       |
    | DTUSpace     | Gridded     | (none)          | Local     |

Key Differences from Along-Track (SLCCI, CMEMS L3):
    - Gridded data (not along-track)
    - No pass/track numbers - gate defines the "synthetic track"
    - No API access (local files only)
    - Uses pcolormesh for spatial maps (not scatter)
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from scipy.spatial import cKDTree

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class DTUConfig:
    """Configuration for DTUSpace data loading."""
    
    # NetCDF file path (required)
    nc_path: str = ""
    
    # Gate shapefile path
    gate_path: str = ""
    
    # Time range filter
    start_year: int = 2006
    end_year: int = 2017
    
    # Processing options
    n_gate_pts: int = 400  # Number of points to interpolate along gate


@dataclass
class DTUPassData:
    """
    Container for DTUSpace processed data.
    
    Mirrors PassData from SLCCI/CMEMS but with DTU-specific attributes.
    Note: "pass" is synthetic - defined by the gate geometry, not satellite orbit.
    """
    
    # Metadata
    strait_name: str = ""
    dataset_name: str = "DTUSpace v4"
    start_year: int = 2006
    end_year: int = 2017
    
    # Gate geometry
    gate_lon_pts: np.ndarray = field(default_factory=lambda: np.array([]))
    gate_lat_pts: np.ndarray = field(default_factory=lambda: np.array([]))
    x_km: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Time dimension
    time_array: np.ndarray = field(default_factory=lambda: np.array([]))
    n_time: int = 0
    
    # DOT data
    dot_matrix: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_gate_pts, n_time)
    profile_mean: np.ndarray = field(default_factory=lambda: np.array([]))  # Mean across time
    slope_series: np.ndarray = field(default_factory=lambda: np.array([]))  # (n_time,)
    
    # Geostrophic velocity
    v_geostrophic_series: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_latitude: float = 0.0
    coriolis_f: float = 0.0
    
    # Gridded data for spatial map (DTU-specific)
    # Store as numpy arrays for Streamlit session_state compatibility
    dot_mean_grid: np.ndarray = field(default_factory=lambda: np.array([]))  # Mean DOT grid (lat, lon)
    lat_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    lon_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Map extent (auto-computed from gate)
    map_extent: dict = field(default_factory=dict)  # {lon_min, lon_max, lat_min, lat_max}
    
    # No DataFrame for DTU (gridded, not point observations)
    # But we can create a synthetic one for compatibility
    df: pd.DataFrame = field(default_factory=pd.DataFrame)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """Load gate shapefile and ensure EPSG:4326."""
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    
    gate_gdf = gpd.read_file(gate_path)
    
    if gate_gdf.crs is None:
        logger.warning(f"Gate has no CRS, assuming EPSG:3413")
        gate_gdf = gate_gdf.set_crs("EPSG:3413")
    
    if not gate_gdf.crs.is_geographic:
        gate_gdf = gate_gdf.to_crs("EPSG:4326")
    
    return gate_gdf


def _extract_strait_name(gate_path: str) -> str:
    """Extract strait name from gate filename."""
    import re
    
    filename = Path(gate_path).stem
    
    # Remove pass suffixes
    name = re.sub(r"_TPJ_pass_\d+", "", filename)
    name = re.sub(r"_S3_pass_\d+", "", name)
    
    # Clean up
    name = name.replace("_", " ").replace("-", " ").title()
    
    return name


def _build_gate_points(gate_gdf: gpd.GeoDataFrame, n_pts: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate n_pts points along the gate geometry.
    Returns sorted by longitude (West to East).
    """
    gate_geom = gate_gdf.geometry.unary_union
    
    gate_points = np.array([
        gate_geom.interpolate(t, normalized=True).coords[0]
        for t in np.linspace(0, 1, n_pts)
    ])
    
    gate_lon = gate_points[:, 0]
    gate_lat = gate_points[:, 1]
    
    # Sort West to East
    sort_idx = np.argsort(gate_lon)
    gate_lon = gate_lon[sort_idx]
    gate_lat = gate_lat[sort_idx]
    
    return gate_lon, gate_lat


def _compute_x_km(gate_lon: np.ndarray, gate_lat: np.ndarray) -> np.ndarray:
    """Compute distance along gate in km (from westernmost point)."""
    R_earth = 6371.0  # km
    lat0_rad = np.deg2rad(np.mean(gate_lat))
    lon_rad = np.deg2rad(gate_lon)
    
    x_km = (lon_rad - lon_rad[0]) * np.cos(lat0_rad) * R_earth
    
    return x_km


def _compute_slope_series(dot_matrix: np.ndarray, x_km: np.ndarray) -> np.ndarray:
    """
    Compute slope time series from DOT matrix.
    
    Args:
        dot_matrix: (n_gate_pts, n_time) array of DOT values
        x_km: (n_gate_pts,) distance along gate
        
    Returns:
        slope_series: (n_time,) slopes in m/100km
    """
    n_time = dot_matrix.shape[1]
    slope_series = np.full(n_time, np.nan, dtype=float)
    
    for it in range(n_time):
        y = dot_matrix[:, it]
        mask = np.isfinite(x_km) & np.isfinite(y)
        
        if np.sum(mask) < 2:
            continue
        
        a, _ = np.polyfit(x_km[mask], y[mask], 1)
        slope_series[it] = a * 100.0  # m/100km
    
    return slope_series


def _compute_geostrophic_velocity(
    slope_series: np.ndarray,
    mean_lat: float
) -> Tuple[np.ndarray, float]:
    """
    Compute geostrophic velocity from slope.
    
    v = -g/f * (dη/dx)
    
    Args:
        slope_series: (n_time,) slopes in m/100km
        mean_lat: mean latitude for Coriolis parameter
        
    Returns:
        v_geo: (n_time,) velocities in m/s
        coriolis_f: Coriolis parameter used
    """
    g = 9.81  # m/s²
    OMEGA = 7.2921e-5  # rad/s
    
    lat_rad = np.deg2rad(mean_lat)
    f = 2 * OMEGA * np.sin(lat_rad)
    
    # Convert slope from m/100km to m/m
    slope_m_m = slope_series / 100000.0
    
    # Geostrophic velocity
    v_geo = -g / f * slope_m_m
    
    return v_geo, f


# ==============================================================================
# DTU SERVICE CLASS
# ==============================================================================

class DTUService:
    """
    Service for loading and processing DTUSpace v4 gridded DOT data.
    
    Usage:
        service = DTUService()
        pass_data = service.load_gate_data(
            nc_path="/path/to/arctic_ocean_prod_DTUSpace_v4.0.nc",
            gate_path="/path/to/gate.shp",
            start_year=2006,
            end_year=2017
        )
    """
    
    def __init__(self):
        """Initialize DTU service."""
        self._ds: Optional[xr.Dataset] = None
        self._tree: Optional[cKDTree] = None
        self._grid_xy: Optional[np.ndarray] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        
        logger.info("DTUService initialized")
    
    @log_call(logger)
    def load_gate_data(
        self,
        nc_path: str,
        gate_path: str,
        start_year: int = 2006,
        end_year: int = 2017,
        n_gate_pts: int = 400
    ) -> DTUPassData:
        """
        Load DTUSpace data and extract DOT along gate.
        
        This is the main entry point for DTUSpace processing.
        
        Args:
            nc_path: Path to DTUSpace NetCDF file
            gate_path: Path to gate shapefile
            start_year: Start year for filtering
            end_year: End year for filtering
            n_gate_pts: Number of points to interpolate along gate
            
        Returns:
            DTUPassData with all computed fields
        """
        logger.info(f"Loading DTUSpace data from {nc_path}")
        logger.info(f"Gate: {gate_path}")
        logger.info(f"Period: {start_year}-{end_year}")
        
        # 1. Load NetCDF
        ds = xr.open_dataset(nc_path, decode_times=True)
        
        # 2. Filter by time
        time_var = ds["date"].values
        start_date = np.datetime64(f"{start_year}-01-01")
        end_date = np.datetime64(f"{end_year}-12-31")
        mask_t = (time_var >= start_date) & (time_var <= end_date)
        ds_sel = ds.isel(date=mask_t)
        
        # 3. Get DOT variable
        dot = ds_sel["dot"]
        
        # Ensure correct dimension order (lat, lon, date)
        if dot.dims != ("lat", "lon", "date"):
            dot = dot.transpose("lat", "lon", "date")
        
        lats = ds_sel["lat"].values
        lons = ds_sel["lon"].values
        time_array = ds_sel["date"].values
        n_time = len(time_array)
        
        logger.info(f"DOT shape: {dot.shape}, {n_time} time steps")
        
        # 4. Build KD-tree for grid matching
        lon2d, lat2d = np.meshgrid(lons, lats)
        grid_xy = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        tree = cKDTree(grid_xy)
        
        # Store for reuse
        self._ds = ds_sel
        self._tree = tree
        self._grid_xy = grid_xy
        self._lats = lats
        self._lons = lons
        
        # 5. Load gate
        gate_gdf = _load_gate_gdf(gate_path)
        strait_name = _extract_strait_name(gate_path)
        
        # 6. Build gate points
        gate_lon, gate_lat = _build_gate_points(gate_gdf, n_gate_pts)
        x_km = _compute_x_km(gate_lon, gate_lat)
        
        logger.info(f"Gate: {strait_name}, length: {x_km[-1]:.1f} km")
        
        # 7. Match gate points to grid cells
        dist_deg, idx_flat = tree.query(
            np.column_stack([gate_lon, gate_lat]),
            k=1
        )
        
        lat_idx = idx_flat // len(lons)
        lon_idx = idx_flat % len(lons)
        
        # 8. Extract DOT along gate for all times
        dot_matrix = dot.values[lat_idx, lon_idx, :]  # (n_gate_pts, n_time)
        
        # 9. Compute derived quantities
        profile_mean = np.nanmean(dot_matrix, axis=1)
        slope_series = _compute_slope_series(dot_matrix, x_km)
        
        mean_lat = np.mean(gate_lat)
        v_geo_series, coriolis_f = _compute_geostrophic_velocity(slope_series, mean_lat)
        
        logger.info(f"Computed slope for {np.isfinite(slope_series).sum()}/{n_time} time steps")
        
        # 10. Compute mean DOT grid for spatial map
        dot_mean = dot.mean(dim="date", skipna=True)
        
        # Compute map extent from gate
        gate_bounds = gate_gdf.total_bounds
        lon_buffer = 10.0
        lat_buffer = 5.0
        map_extent = {
            "lon_min": gate_bounds[0] - lon_buffer,
            "lon_max": gate_bounds[2] + lon_buffer,
            "lat_min": gate_bounds[1] - lat_buffer,
            "lat_max": gate_bounds[3] + lat_buffer
        }
        
        # Subset grid to map extent
        dot_mean_sub = dot_mean.sel(
            lat=slice(map_extent["lat_min"], map_extent["lat_max"]),
            lon=slice(map_extent["lon_min"], map_extent["lon_max"])
        )
        
        # 11. Create synthetic DataFrame for compatibility
        # This creates a "fake" observation-style DataFrame
        df_rows = []
        for ig in range(n_gate_pts):
            for it in range(n_time):
                df_rows.append({
                    "lon": gate_lon[ig],
                    "lat": gate_lat[ig],
                    "time": pd.Timestamp(time_array[it]),
                    "dot": dot_matrix[ig, it],
                    "x_km": x_km[ig]
                })
        
        df = pd.DataFrame(df_rows)
        df["month"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        
        # 12. Build result
        dataset_name = Path(nc_path).stem.replace("_", " ")
        
        result = DTUPassData(
            strait_name=strait_name,
            dataset_name=dataset_name,
            start_year=start_year,
            end_year=end_year,
            gate_lon_pts=gate_lon,
            gate_lat_pts=gate_lat,
            x_km=x_km,
            time_array=time_array,
            n_time=n_time,
            dot_matrix=dot_matrix,
            profile_mean=profile_mean,
            slope_series=slope_series,
            v_geostrophic_series=v_geo_series,
            mean_latitude=mean_lat,
            coriolis_f=coriolis_f,
            # Convert xarray to numpy for Streamlit session_state compatibility
            dot_mean_grid=dot_mean_sub.values,
            lat_grid=dot_mean_sub["lat"].values,
            lon_grid=dot_mean_sub["lon"].values,
            map_extent=map_extent,
            df=df
        )
        
        logger.info(f"DTUSpace data loaded successfully: {len(df)} synthetic observations")
        
        return result
    
    def get_available_years(self, nc_path: str) -> Tuple[int, int]:
        """Get available year range from NetCDF file."""
        ds = xr.open_dataset(nc_path, decode_times=True)
        times = ds["date"].values
        
        min_year = pd.Timestamp(times.min()).year
        max_year = pd.Timestamp(times.max()).year
        
        ds.close()
        
        return min_year, max_year
