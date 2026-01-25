"""
CMEMS L4 Service
================
Service for loading and processing Copernicus Marine (CMEMS) **L4 Gridded** data via API.

Dataset Info:
    - Product: SEALEVEL_GLO_PHY_L4_MY_008_047
    - Name: Global Ocean Gridded L4 Sea Surface Heights And Derived Variables Reprocessed 1993 Ongoing
    - DOI: https://doi.org/10.48670/moi-00148
    - Web: https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description
    - Type: GRIDDED (lat × lon × time)
    - Resolution: 0.125° (~14km) daily
    - Variables: adt, sla, ugos, vgos, ugosa, vgosa, err_sla, flag_ice

Description:
    Altimeter satellite gridded Sea Level Anomalies (SLA) computed with respect to a 
    twenty-year [1993, 2012] mean. The SLA is estimated by Optimal Interpolation, merging 
    the L3 along-track measurement from the different altimeter missions available.
    Processed by the DUACS multimission altimeter data processing system.

Features:
    - Intelligent two-level caching (L1: raw xarray, L2: processed CMEMSL4PassData)
    - 4500x+ speedup on cache hits
    - API download only on cache miss
    - Automatic invalidation on parameter changes

Data Flow:
    UI → CMEMSL4Service → copernicusmarine.subset() → API download
                        → xr.open_dataset → NetCDF in memory
                        → KD-tree gate matching → DOT along gate
                        → slope computation → time series

Key Differences from Along-Track (SLCCI, CMEMS L3):
    - GRIDDED data (not along-track)
    - NO track/pass numbers - gate defines the "synthetic track"
    - Data comes from API (not local files)
    - Uses pcolormesh for spatial maps (not scatter)
    - Similar workflow to DTUSpace

Comparison with other datasets:
    | Dataset      | Type        | Filter Variable | Source    |
    |--------------|-------------|-----------------|-----------|
    | SLCCI        | Along-track | pass            | Local     |
    | CMEMS L3     | Along-track | track           | Local     |
    | CMEMS L4     | Gridded     | (none)          | API       |
    | DTUSpace     | Gridded     | (none)          | Local     |

API Usage:
    import copernicusmarine
    
    copernicusmarine.subset(
        dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
        variables=["adt", "sla"],
        minimum_longitude=bbox[0],
        maximum_longitude=bbox[2],
        minimum_latitude=bbox[1],
        maximum_latitude=bbox[3],
        start_datetime="2010-01-01",
        end_datetime="2020-12-31",
    )
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, date
from scipy.spatial import cKDTree
from scipy import stats

from src.core.logging_config import get_logger, log_call
from src.services.intelligent_cache import IntelligentCache, CacheConfig, get_intelligent_cache

logger = get_logger(__name__)

# Try to import copernicusmarine
try:
    import copernicusmarine
    COPERNICUSMARINE_AVAILABLE = True
except ImportError:
    COPERNICUSMARINE_AVAILABLE = False
    logger.warning("copernicusmarine not installed. Use: pip install copernicusmarine")


# ==============================================================================
# CONSTANTS
# ==============================================================================

# CMEMS L4 Dataset IDs
CMEMS_L4_DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
CMEMS_L4_DATASET_VERSION = ""  # Empty = use latest version (auto-detected by API)

# Default variables to download
DEFAULT_VARIABLES = ["adt", "sla", "ugos", "vgos"]

# Geostrophic constants
G = 9.81  # m/s² - gravitational acceleration
OMEGA = 7.2921e-5  # rad/s - Earth's angular velocity
R_EARTH = 6371.0  # km - Earth's radius


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CMEMSL4Config:
    """Configuration for CMEMS L4 gridded data loading via API."""
    
    # Gate path (required)
    gate_path: str = ""
    
    # Time range
    time_start: str = "2010-01-01"
    time_end: str = "2020-12-31"
    
    # Spatial buffer around gate (degrees)
    buffer_deg: float = 2.0
    
    # Variables to download
    variables: List[str] = field(default_factory=lambda: ["adt", "sla"])
    
    # Processing parameters
    n_gate_pts: int = 400  # Points to sample along gate
    
    # API options
    dataset_id: str = CMEMS_L4_DATASET_ID
    dataset_version: str = CMEMS_L4_DATASET_VERSION
    disable_progress_bar: bool = False
    
    # Cache options
    use_cache: bool = True
    cache_dir: str = ""  # Auto-set if empty


@dataclass
class CMEMSL4PassData:
    """
    Standard interface for CMEMS L4 gridded data.
    Compatible with tabs.py visualization (same structure as DTUPassData).
    """
    # Metadata (required fields first)
    strait_name: str
    
    # Tab 1: Slope Timeline
    slope_series: np.ndarray  # Shape: (n_time,) - m/100km
    time_array: np.ndarray  # Shape: (n_time,) - datetime objects
    
    # Tab 2: DOT Profile
    profile_mean: np.ndarray  # Shape: (n_gate_pts,)
    x_km: np.ndarray  # Shape: (n_gate_pts,) - distance along gate
    dot_matrix: np.ndarray  # Shape: (n_gate_pts, n_time)
    
    # Tab 3: Spatial Map
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    
    # Tab 4: Geostrophic Velocities (from Copernicus, not computed)
    ugos_matrix: Optional[np.ndarray] = None  # Shape: (n_gate_pts, n_time) - m/s
    vgos_matrix: Optional[np.ndarray] = None  # Shape: (n_gate_pts, n_time) - m/s
    
    # Computed geostrophic velocity from slope (for tabs.py compatibility)
    v_geostrophic_series: np.ndarray = field(default_factory=lambda: np.array([]))  # m/s
    mean_latitude: float = 70.0  # For Coriolis calculation
    coriolis_f: float = 1.38e-4  # s⁻¹ at ~70°N
    
    # Fields with defaults (must come after required fields)
    data_source: str = "CMEMS L4"
    dataset_name: str = "CMEMS L4"  # For tabs.py compatibility
    
    # Raw data for advanced analysis
    ds: Optional[xr.Dataset] = None  # Original xarray dataset
    
    # Statistics
    n_observations: int = 0
    time_range: Tuple[str, str] = ("", "")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """Load gate shapefile and ensure it's in EPSG:4326."""
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    gate_gdf = gpd.read_file(gate_path)
    
    if gate_gdf.crs is None:
        logger.warning(f"Gate {gate_path} has no CRS, assuming EPSG:3413")
        gate_gdf = gate_gdf.set_crs("EPSG:3413")
    
    return gate_gdf.to_crs("EPSG:4326")


def _extract_strait_name(gate_path: str) -> str:
    """
    Extract clean strait name from gate filename.
    
    Returns lowercase with underscores for cache key consistency.
    Example: "fram_strait_S3_pass_481.shp" -> "fram_strait"
    """
    import re
    filename = Path(gate_path).stem
    
    # Remove satellite/pass markers
    name = re.sub(r'_TPJ_pass_\d+', '', filename, flags=re.IGNORECASE)
    name = re.sub(r'_S\d_pass_\d+', '', name)
    name = re.sub(r'_pass_\d+', '', name)
    
    # Normalize: lowercase, underscores (for cache key consistency)
    name = name.lower().replace(' ', '_').replace('-', '_')
    
    return name


def _build_gate_points(gate_gdf: gpd.GeoDataFrame, n_pts: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Sample N points along gate geometry."""
    gate_geom = gate_gdf.geometry.unary_union
    
    gate_points = np.array([
        gate_geom.interpolate(t, normalized=True).coords[0]
        for t in np.linspace(0, 1, n_pts)
    ])
    
    gate_lon = gate_points[:, 0]
    gate_lat = gate_points[:, 1]
    
    # Sort by longitude
    sort_idx = np.argsort(gate_lon)
    gate_lon = gate_lon[sort_idx]
    gate_lat = gate_lat[sort_idx]
    
    return gate_lon, gate_lat


def _compute_x_km(gate_lon: np.ndarray, gate_lat: np.ndarray) -> np.ndarray:
    """Compute cumulative distance along gate in km."""
    x_km = np.zeros(len(gate_lon))
    
    for i in range(1, len(gate_lon)):
        dlat = (gate_lat[i] - gate_lat[i-1]) * 111.0  # km per degree lat
        mean_lat = (gate_lat[i] + gate_lat[i-1]) / 2
        dlon = (gate_lon[i] - gate_lon[i-1]) * 111.0 * np.cos(np.radians(mean_lat))
        x_km[i] = x_km[i-1] + np.sqrt(dlat**2 + dlon**2)
    
    return x_km


def _compute_slope_series(dot_matrix: np.ndarray, x_km: np.ndarray) -> np.ndarray:
    """Compute slope time series from DOT matrix."""
    n_time = dot_matrix.shape[1]
    slope_series = np.full(n_time, np.nan)
    
    for it in range(n_time):
        y = dot_matrix[:, it]
        mask = np.isfinite(x_km) & np.isfinite(y)
        
        if np.sum(mask) < 2:
            continue
        
        try:
            slope, intercept, r, p, se = stats.linregress(x_km[mask], y[mask])
            # Convert to m / 100km
            slope_series[it] = slope * 100.0
        except Exception:
            continue
    
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
    lat_rad = np.deg2rad(mean_lat)
    f = 2 * OMEGA * np.sin(lat_rad)
    
    # Convert slope from m/100km to m/m
    slope_m_m = slope_series / 100000.0
    
    # Geostrophic velocity
    v_geo = -G / f * slope_m_m
    
    return v_geo, f


# ==============================================================================
# CMEMS L4 SERVICE CLASS
# ==============================================================================

class CMEMSL4Service:
    """
    Service for loading and processing CMEMS L4 gridded SSH data via API.
    
    Features intelligent two-level caching:
    - L1 (raw): xarray Dataset after API download
    - L2 (processed): CMEMSL4PassData after gate extraction
    
    Cache speedup: ~4500x on L2 hit, API download only on full cache miss
    
    Usage:
        service = CMEMSL4Service()
        
        config = CMEMSL4Config(
            gate_path="/path/to/gate.shp",
            time_start="2010-01-01",
            time_end="2020-12-31",
        )
        
        pass_data = service.load_gate_data(config)
        
        # Check cache stats
        print(service.get_cache_stats())
        
        # Clear cache  
        service.clear_cache()
    """
    
    SERVICE_NAME = "cmems_l4"  # Cache service identifier
    
    def __init__(self, cache: Optional[IntelligentCache] = None):
        """
        Initialize CMEMS L4 service.
        
        Parameters
        ----------
        cache : IntelligentCache, optional
            Cache instance. If None, uses global singleton.
        """
        if not COPERNICUSMARINE_AVAILABLE:
            logger.warning("copernicusmarine not available - API downloads will fail")
        
        # Use provided cache or global singleton
        self._cache = cache or get_intelligent_cache()
        
        logger.info("CMEMSL4Service initialized with intelligent caching")
    
    def clear_cache(self, strait_name: Optional[str] = None):
        """Clear cache (all CMEMS L4 entries or specific strait)."""
        if strait_name:
            self._cache.invalidate(service=self.SERVICE_NAME, entity_key=strait_name)
        else:
            self._cache.invalidate(service=self.SERVICE_NAME)
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.get_stats()
    
    @log_call(logger)
    def load_gate_data(
        self,
        config: CMEMSL4Config,
        progress_callback: Optional[callable] = None,
        force_reload: bool = False,
        use_cache: bool = True,
    ) -> Optional[CMEMSL4PassData]:
        """
        Load CMEMS L4 gridded data for a gate via API.
        
        Uses two-level caching for maximum performance:
        - L2 hit (processed): ~0.01 sec (instant!)
        - L1 hit (raw): ~1-2 sec (reprocess from cached xarray)
        - Miss: ~30-120 sec (API download + process)
        
        Parameters
        ----------
        config : CMEMSL4Config
            Configuration with gate path and parameters
        progress_callback : callable, optional
            Callback(progress, message) for UI updates
        force_reload : bool
            If True, bypass cache and reload from API
        use_cache : bool
            If False, bypass cache entirely (sidebar manages its own cache)
        
        Returns
        -------
        CMEMSL4PassData or None
            Processed data object or None if loading fails
        """
        if not COPERNICUSMARINE_AVAILABLE:
            logger.error("copernicusmarine not installed")
            return None
        
        if not config.gate_path:
            logger.error("No gate path provided")
            return None
        
        # Extract gate name for cache key
        strait_name = _extract_strait_name(config.gate_path)
        cache_key = f"{strait_name}_{config.time_start}_{config.time_end}"
        
        logger.info(f"Loading CMEMS L4 data for {strait_name}")
        
        # Skip cache if use_cache=False (sidebar manages its own cache)
        skip_cache = not use_cache or force_reload
        
        # --- CHECK L2 CACHE (processed) ---
        if not skip_cache:
            cached_result = self._cache.get_processed(
                self.SERVICE_NAME, cache_key, n_gate_pts=config.n_gate_pts
            )
            if cached_result is not None:
                logger.info(f"✅ Cache HIT (L2 processed) for {strait_name}")
                if progress_callback:
                    progress_callback(1.0, "Loaded from cache (instant!)")
                return cached_result
        
        # --- CHECK L1 CACHE (raw xarray) ---
        cached_ds = None if force_reload else self._cache.get_raw(self.SERVICE_NAME, cache_key)
        
        if cached_ds is not None:
            logger.info(f"✅ Cache HIT (L1 raw) for {strait_name}, processing...")
            ds = cached_ds
            if progress_callback:
                progress_callback(0.5, "Processing cached data...")
        else:
            # --- DOWNLOAD FROM API ---
            if progress_callback:
                progress_callback(0.1, "Loading gate...")
            
            gate_gdf = _load_gate_gdf(config.gate_path)
            
            # Get gate bounds with buffer
            bounds = gate_gdf.total_bounds  # [minx, miny, maxx, maxy]
            lon_min = bounds[0] - config.buffer_deg
            lon_max = bounds[2] + config.buffer_deg
            lat_min = bounds[1] - config.buffer_deg
            lat_max = bounds[3] + config.buffer_deg
            
            logger.info(f"Bbox: lon[{lon_min:.2f}, {lon_max:.2f}], lat[{lat_min:.2f}, {lat_max:.2f}]")
            
            # Download data via API
            if progress_callback:
                progress_callback(0.2, "Downloading from Copernicus Marine API...")
            
            try:
                ds = self._download_subset(
                    lon_min=lon_min,
                    lon_max=lon_max,
                    lat_min=lat_min,
                    lat_max=lat_max,
                    time_start=config.time_start,
                    time_end=config.time_end,
                    variables=config.variables,
                    dataset_id=config.dataset_id,
                    dataset_version=config.dataset_version,
                    disable_progress_bar=config.disable_progress_bar,
                )
            except Exception as e:
                logger.error(f"API download failed: {e}")
                return None
            
            if ds is None:
                logger.error("No data returned from API")
                return None
            
            # --- STORE IN L1 CACHE ---
            self._cache.set_raw(self.SERVICE_NAME, cache_key, ds)
            # Force save to disk for persistence
            self._cache.save_to_disk()
            logger.info(f"💾 Cached raw xarray Dataset from API")
        
        # --- PROCESS (from cached or fresh ds) ---
        if progress_callback:
            progress_callback(0.5, "Processing downloaded data...")
        
        # Load gate
        gate_gdf = _load_gate_gdf(config.gate_path)
        
        # Build gate points
        gate_lon_pts, gate_lat_pts = _build_gate_points(gate_gdf, config.n_gate_pts)
        x_km = _compute_x_km(gate_lon_pts, gate_lat_pts)
        
        # Get grid coordinates
        lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        time_vals = pd.to_datetime(ds["time"].values)
        
        # Get ADT variable
        if "adt" in ds.data_vars:
            adt_var = ds["adt"]
        elif "sla" in ds.data_vars:
            logger.warning("ADT not found, using SLA instead")
            adt_var = ds["sla"]
        else:
            logger.error(f"No ADT or SLA variable found. Variables: {list(ds.data_vars)}")
            return None
        
        if progress_callback:
            progress_callback(0.6, "Building KD-tree for spatial matching...")
        
        # Build KD-tree for nearest neighbor
        lon2d, lat2d = np.meshgrid(lons, lats)
        grid_xy = np.column_stack([lon2d.ravel(), lat2d.ravel()])
        tree = cKDTree(grid_xy)
        
        # Find nearest grid points to gate
        gate_coords = np.column_stack([gate_lon_pts, gate_lat_pts])
        dist, idx_flat = tree.query(gate_coords, k=1)
        
        lat_idx = idx_flat // len(lons)
        lon_idx = idx_flat % len(lons)
        
        if progress_callback:
            progress_callback(0.7, "Extracting DOT along gate...")
        
        # Extract DOT along gate for all times
        n_time = len(time_vals)
        n_pts = len(gate_lon_pts)
        
        # Handle different dimension orders
        adt_data = adt_var.values
        if adt_var.dims == ("time", "latitude", "longitude") or adt_var.dims == ("time", "lat", "lon"):
            # Shape: (time, lat, lon)
            dot_matrix = np.zeros((n_pts, n_time))
            for it in range(n_time):
                dot_matrix[:, it] = adt_data[it, lat_idx, lon_idx]
        elif adt_var.dims == ("latitude", "longitude", "time") or adt_var.dims == ("lat", "lon", "time"):
            # Shape: (lat, lon, time)
            dot_matrix = np.zeros((n_pts, n_time))
            for it in range(n_time):
                dot_matrix[:, it] = adt_data[lat_idx, lon_idx, it]
        else:
            logger.error(f"Unexpected ADT dimensions: {adt_var.dims}")
            return None
        
        if progress_callback:
            progress_callback(0.75, "Extracting geostrophic velocities...")
        
        # Extract UGOS and VGOS if available (Copernicus geostrophic velocities)
        ugos_matrix = None
        vgos_matrix = None
        
        for vel_name, matrix_name in [("ugos", "ugos_matrix"), ("vgos", "vgos_matrix")]:
            if vel_name in ds.data_vars:
                vel_var = ds[vel_name]
                vel_data = vel_var.values
                vel_matrix = np.zeros((n_pts, n_time))
                
                if vel_var.dims == ("time", "latitude", "longitude") or vel_var.dims == ("time", "lat", "lon"):
                    for it in range(n_time):
                        vel_matrix[:, it] = vel_data[it, lat_idx, lon_idx]
                elif vel_var.dims == ("latitude", "longitude", "time") or vel_var.dims == ("lat", "lon", "time"):
                    for it in range(n_time):
                        vel_matrix[:, it] = vel_data[lat_idx, lon_idx, it]
                else:
                    logger.warning(f"Unexpected {vel_name} dimensions: {vel_var.dims}")
                    vel_matrix = None
                
                if vel_name == "ugos":
                    ugos_matrix = vel_matrix
                else:
                    vgos_matrix = vel_matrix
                
                if vel_matrix is not None:
                    logger.info(f"Extracted {vel_name}: shape {vel_matrix.shape}")
            else:
                logger.warning(f"{vel_name} not found in dataset")
        
        if progress_callback:
            progress_callback(0.8, "Computing slope time series...")
        
        # Compute slope series
        slope_series = _compute_slope_series(dot_matrix, x_km)
        
        # Compute mean profile
        profile_mean = np.nanmean(dot_matrix, axis=1)
        
        # Compute geostrophic velocity from slope (same as DTUService)
        mean_lat = np.mean(gate_lat_pts)
        v_geo_series, coriolis_f = _compute_geostrophic_velocity(slope_series, mean_lat)
        
        # Count valid observations
        n_obs = np.sum(np.isfinite(dot_matrix))
        
        if progress_callback:
            progress_callback(1.0, "Done!")
        
        logger.info(f"Loaded CMEMS L4: {n_obs} observations, {n_time} time steps")
        
        pass_data = CMEMSL4PassData(
            strait_name=strait_name,
            data_source="CMEMS L4 (Gridded)",
            slope_series=slope_series,
            time_array=time_vals,
            profile_mean=profile_mean,
            x_km=x_km,
            dot_matrix=dot_matrix,
            gate_lon_pts=gate_lon_pts,
            gate_lat_pts=gate_lat_pts,
            ugos_matrix=ugos_matrix,
            vgos_matrix=vgos_matrix,
            v_geostrophic_series=v_geo_series,
            mean_latitude=mean_lat,
            coriolis_f=coriolis_f,
            ds=ds,
            n_observations=int(n_obs),
            time_range=(str(time_vals.min()), str(time_vals.max())),
        )
        
        # --- STORE IN L2 CACHE ---
        self._cache.set_processed(
            self.SERVICE_NAME, cache_key, pass_data, n_gate_pts=config.n_gate_pts
        )
        # Force save to disk for persistence across sessions
        self._cache.save_to_disk()
        logger.info(f"💾 Cached processed CMEMSL4PassData (n_gate_pts={config.n_gate_pts})")
        
        return pass_data
    
    def _download_subset(
        self,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        time_start: str,
        time_end: str,
        variables: List[str],
        dataset_id: str,
        dataset_version: str,
        disable_progress_bar: bool = False,
    ) -> Optional[xr.Dataset]:
        """
        Download subset from Copernicus Marine API.
        
        Uses copernicusmarine.subset() to download data directly to memory.
        """
        if not COPERNICUSMARINE_AVAILABLE:
            return None
        
        logger.info(f"Downloading CMEMS L4 subset:")
        logger.info(f"  Dataset: {dataset_id}")
        logger.info(f"  Bbox: [{lon_min:.2f}, {lon_max:.2f}] x [{lat_min:.2f}, {lat_max:.2f}]")
        logger.info(f"  Time: {time_start} to {time_end}")
        logger.info(f"  Variables: {variables}")
        
        try:
            # Build kwargs for open_dataset
            kwargs = {
                "dataset_id": dataset_id,
                "variables": variables,
                "minimum_longitude": lon_min,
                "maximum_longitude": lon_max,
                "minimum_latitude": lat_min,
                "maximum_latitude": lat_max,
                "start_datetime": time_start,
                "end_datetime": time_end,
            }
            
            # Only add version if specified (empty = auto-detect latest)
            if dataset_version:
                kwargs["dataset_version"] = dataset_version
            
            # Download to xarray dataset in memory
            ds = copernicusmarine.open_dataset(**kwargs)
            
            logger.info(f"Downloaded: {ds.dims}")
            return ds
            
        except Exception as e:
            logger.error(f"copernicusmarine download failed: {e}")
            raise
    
    def check_api_credentials(self) -> bool:
        """Check if Copernicus Marine API credentials are configured."""
        if not COPERNICUSMARINE_AVAILABLE:
            return False
        
        # Try to access the API
        try:
            # This will fail if credentials are not set
            copernicusmarine.describe(dataset_id=CMEMS_L4_DATASET_ID)
            return True
        except Exception as e:
            logger.warning(f"API credentials check failed: {e}")
            return False
    
    @staticmethod
    def get_dataset_info() -> dict:
        """Return dataset metadata."""
        return {
            "name": "CMEMS L4 Gridded SSH",
            "product_id": "SEALEVEL_GLO_PHY_L4_MY_008_047",
            "dataset_id": CMEMS_L4_DATASET_ID,
            "doi": "https://doi.org/10.48670/moi-00148",
            "url": "https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_L4_MY_008_047/description",
            "type": "Gridded",
            "resolution": "0.125° (~14km) daily",
            "variables": ["adt", "sla", "ugos", "vgos", "ugosa", "vgosa", "err_sla", "flag_ice"],
            "description": (
                "Altimeter satellite gridded Sea Level Anomalies (SLA) computed with respect to a "
                "twenty-year [1993, 2012] mean. The SLA is estimated by Optimal Interpolation, "
                "merging the L3 along-track measurement from the different altimeter missions. "
                "Processed by DUACS multimission altimeter data processing system."
            ),
        }
