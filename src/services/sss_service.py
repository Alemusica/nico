"""
SSS Service - Sea Surface Salinity
==================================
Service for loading Sea Surface Salinity (SSS) and Density (DOS) from Copernicus Marine API.

Dataset Info:
    - Product: MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013
    - Name: Multi Observation Global Ocean Sea Surface Salinity and Sea Surface Density
    - DOI: https://doi.org/10.48670/moi-00051
    - Type: GRIDDED (lat × lon × time)
    - Resolution: 0.125° (~14km) daily
    - Variables: sos (salinity), dos (density), sea_ice_fraction

Data Flow:
    UI → SSSService → copernicusmarine.open_dataset() → API download
                    → KD-tree gate matching → SSS/DOS along gate
                    → SSSData object for flux calculation

Usage:
    service = SSSService()
    config = SSSConfig(gate_path="path/to/gate.shp", time_start="2010-01-01", time_end="2020-12-31")
    sss_data = service.load_gate_data(config)
    
    # Access data
    salinity = sss_data.sos_matrix  # Shape: (n_gate_pts, n_time)
    density = sss_data.dos_matrix   # Shape: (n_gate_pts, n_time)
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
from src.services.intelligent_cache import IntelligentCache, get_intelligent_cache

logger = get_logger(__name__)

# Try to import copernicusmarine
try:
    import copernicusmarine
    COPERNICUSMARINE_AVAILABLE = True
except ImportError:
    COPERNICUSMARINE_AVAILABLE = False
    logger.warning("copernicusmarine not installed")


# ==============================================================================
# CONSTANTS
# ==============================================================================

SSS_DATASET_ID = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
SSS_DATASET_VERSION = "202311"
SSS_DEFAULT_VARIABLES = ["sos", "dos", "sea_ice_fraction"]


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SSSConfig:
    """Configuration for SSS data loading."""
    gate_path: str = ""
    time_start: str = "2010-01-01"
    time_end: str = "2020-12-31"
    buffer_deg: float = 2.0
    variables: List[str] = field(default_factory=lambda: ["sos", "dos", "sea_ice_fraction"])
    n_gate_pts: int = 400
    dataset_id: str = SSS_DATASET_ID
    dataset_version: str = SSS_DATASET_VERSION


@dataclass
class SSSData:
    """
    Sea Surface Salinity and Density data along a gate.
    
    Attributes:
        sos_matrix: Salinity [PSU], shape (n_gate_pts, n_time)
        dos_matrix: Density [kg/m³], shape (n_gate_pts, n_time)
        ice_matrix: Sea ice fraction [0-1], shape (n_gate_pts, n_time)
        time_array: Datetime array, shape (n_time,)
        gate_lon_pts: Longitude of gate points
        gate_lat_pts: Latitude of gate points
        x_km: Distance along gate [km]
    """
    strait_name: str
    sos_matrix: np.ndarray  # Salinity [PSU]
    dos_matrix: np.ndarray  # Density [kg/m³]
    time_array: np.ndarray
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    x_km: np.ndarray
    
    # Optional
    ice_matrix: Optional[np.ndarray] = None
    ds: Optional[xr.Dataset] = None
    
    # Stats
    n_observations: int = 0
    time_range: Tuple[str, str] = ("", "")
    data_source: str = "CMEMS SSS"


# ==============================================================================
# HELPER FUNCTIONS (reuse from cmems_l4_service pattern)
# ==============================================================================

def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """Load gate shapefile in EPSG:4326."""
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    gdf = gpd.read_file(gate_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3413")
    return gdf.to_crs("EPSG:4326")


def _extract_strait_name(gate_path: str) -> str:
    """Extract clean strait name from filename."""
    import re
    name = Path(gate_path).stem
    name = re.sub(r'_TPJ_pass_\d+|_S\d_pass_\d+|_pass_\d+', '', name, flags=re.IGNORECASE)
    return name.lower().replace(' ', '_').replace('-', '_')


def _build_gate_points(gdf: gpd.GeoDataFrame, n_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample N points along gate, sorted by longitude."""
    geom = gdf.geometry.unary_union
    pts = np.array([geom.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n_pts)])
    lon, lat = pts[:, 0], pts[:, 1]
    idx = np.argsort(lon)
    return lon[idx], lat[idx]


def _compute_x_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Cumulative distance along gate in km."""
    x = np.zeros(len(lon))
    for i in range(1, len(lon)):
        dlat = (lat[i] - lat[i-1]) * 111.0
        dlon = (lon[i] - lon[i-1]) * 111.0 * np.cos(np.radians((lat[i] + lat[i-1]) / 2))
        x[i] = x[i-1] + np.sqrt(dlat**2 + dlon**2)
    return x


def _extract_along_gate(
    ds: xr.Dataset,
    var_name: str,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    n_pts: int,
    n_time: int
) -> Optional[np.ndarray]:
    """Extract variable along gate points for all times."""
    if var_name not in ds.data_vars:
        logger.warning(f"{var_name} not in dataset")
        return None
    
    var = ds[var_name]
    data = var.values
    
    # Handle depth dimension if present (squeeze to surface)
    if "depth" in var.dims:
        depth_idx = list(var.dims).index("depth")
        data = np.take(data, 0, axis=depth_idx)
    
    matrix = np.full((n_pts, n_time), np.nan)
    
    # Determine dimension order
    dims = [d for d in var.dims if d != "depth"]
    if dims[0] == "time":
        # Shape: (time, lat, lon)
        for it in range(n_time):
            matrix[:, it] = data[it, lat_idx, lon_idx]
    else:
        # Shape: (lat, lon, time)
        for it in range(n_time):
            matrix[:, it] = data[lat_idx, lon_idx, it]
    
    return matrix


# ==============================================================================
# SSS SERVICE CLASS
# ==============================================================================

class SSSService:
    """
    Service for loading Sea Surface Salinity and Density data.
    
    Uses intelligent caching (L1: raw xarray, L2: processed SSSData).
    Same pattern as CMEMSL4Service for consistency.
    """
    
    SERVICE_NAME = "sss"
    
    def __init__(self, cache: Optional[IntelligentCache] = None):
        if not COPERNICUSMARINE_AVAILABLE:
            logger.warning("copernicusmarine not available")
        self._cache = cache or get_intelligent_cache()
        logger.info("SSSService initialized")
    
    def clear_cache(self, strait_name: Optional[str] = None):
        """Clear SSS cache entries."""
        if strait_name:
            self._cache.invalidate(service=self.SERVICE_NAME, entity_key=strait_name)
        else:
            self._cache.invalidate(service=self.SERVICE_NAME)
    
    def get_cache_stats(self) -> dict:
        return self._cache.get_stats()
    
    @log_call(logger)
    def load_gate_data(
        self,
        config: SSSConfig,
        progress_callback: Optional[callable] = None,
        force_reload: bool = False,
        use_cache: bool = True,
    ) -> Optional[SSSData]:
        """
        Load SSS/DOS data for a gate.
        
        Returns SSSData with:
        - sos_matrix: Salinity [PSU], shape (n_pts, n_time)
        - dos_matrix: Density [kg/m³], shape (n_pts, n_time)
        - ice_matrix: Ice fraction [0-1], shape (n_pts, n_time)
        """
        if not COPERNICUSMARINE_AVAILABLE:
            logger.error("copernicusmarine not installed")
            return None
        
        if not config.gate_path:
            logger.error("No gate path provided")
            return None
        
        strait_name = _extract_strait_name(config.gate_path)
        cache_key = f"{strait_name}_{config.time_start}_{config.time_end}"
        
        logger.info(f"Loading SSS data for {strait_name}")
        
        skip_cache = not use_cache or force_reload
        
        # --- L2 CACHE CHECK ---
        if not skip_cache:
            cached = self._cache.get_processed(self.SERVICE_NAME, cache_key, n_gate_pts=config.n_gate_pts)
            if cached is not None:
                logger.info(f"✅ Cache HIT (L2) for SSS {strait_name}")
                if progress_callback:
                    progress_callback(1.0, "Loaded from cache")
                return cached
        
        # --- L1 CACHE CHECK ---
        ds = None if force_reload else self._cache.get_raw(self.SERVICE_NAME, cache_key)
        
        if ds is not None:
            logger.info(f"✅ Cache HIT (L1 raw) for SSS {strait_name}")
            if progress_callback:
                progress_callback(0.5, "Processing cached data...")
        else:
            # --- DOWNLOAD ---
            if progress_callback:
                progress_callback(0.1, "Loading gate...")
            
            gdf = _load_gate_gdf(config.gate_path)
            bounds = gdf.total_bounds
            
            lon_min, lon_max = bounds[0] - config.buffer_deg, bounds[2] + config.buffer_deg
            lat_min, lat_max = bounds[1] - config.buffer_deg, bounds[3] + config.buffer_deg
            
            if progress_callback:
                progress_callback(0.2, "Downloading SSS from Copernicus API...")
            
            try:
                kwargs = {
                    "dataset_id": config.dataset_id,
                    "dataset_version": config.dataset_version,
                    "variables": config.variables,
                    "minimum_longitude": lon_min,
                    "maximum_longitude": lon_max,
                    "minimum_latitude": lat_min,
                    "maximum_latitude": lat_max,
                    "start_datetime": config.time_start,
                    "end_datetime": config.time_end,
                    "minimum_depth": 0,
                    "maximum_depth": 0,
                }
                ds = copernicusmarine.open_dataset(**kwargs)
                logger.info(f"Downloaded SSS: {ds.dims}")
            except Exception as e:
                logger.error(f"SSS download failed: {e}")
                return None
            
            # Cache raw
            self._cache.set_raw(self.SERVICE_NAME, cache_key, ds)
            self._cache.save_to_disk()
            logger.info("💾 Cached raw SSS dataset")
        
        # --- PROCESS ---
        if progress_callback:
            progress_callback(0.5, "Processing SSS data...")
        
        gdf = _load_gate_gdf(config.gate_path)
        gate_lon, gate_lat = _build_gate_points(gdf, config.n_gate_pts)
        x_km = _compute_x_km(gate_lon, gate_lat)
        
        # Grid coords
        lats = ds["latitude"].values if "latitude" in ds.coords else ds["lat"].values
        lons = ds["longitude"].values if "longitude" in ds.coords else ds["lon"].values
        time_vals = pd.to_datetime(ds["time"].values)
        
        # KD-tree for nearest neighbor
        lon2d, lat2d = np.meshgrid(lons, lats)
        tree = cKDTree(np.column_stack([lon2d.ravel(), lat2d.ravel()]))
        _, idx_flat = tree.query(np.column_stack([gate_lon, gate_lat]), k=1)
        
        lat_idx = idx_flat // len(lons)
        lon_idx = idx_flat % len(lons)
        
        n_pts, n_time = len(gate_lon), len(time_vals)
        
        if progress_callback:
            progress_callback(0.7, "Extracting SSS/DOS along gate...")
        
        # Extract variables
        sos_matrix = _extract_along_gate(ds, "sos", lat_idx, lon_idx, n_pts, n_time)
        dos_matrix = _extract_along_gate(ds, "dos", lat_idx, lon_idx, n_pts, n_time)
        ice_matrix = _extract_along_gate(ds, "sea_ice_fraction", lat_idx, lon_idx, n_pts, n_time)
        
        if sos_matrix is None or dos_matrix is None:
            logger.error("Failed to extract SSS or DOS")
            return None
        
        n_obs = int(np.sum(np.isfinite(sos_matrix)))
        
        if progress_callback:
            progress_callback(1.0, "Done!")
        
        logger.info(f"Loaded SSS: {n_obs} obs, mean S={np.nanmean(sos_matrix):.2f} PSU, mean ρ={np.nanmean(dos_matrix):.1f} kg/m³")
        
        sss_data = SSSData(
            strait_name=strait_name,
            sos_matrix=sos_matrix,
            dos_matrix=dos_matrix,
            ice_matrix=ice_matrix,
            time_array=time_vals,
            gate_lon_pts=gate_lon,
            gate_lat_pts=gate_lat,
            x_km=x_km,
            ds=ds,
            n_observations=n_obs,
            time_range=(str(time_vals.min()), str(time_vals.max())),
        )
        
        # Cache processed
        self._cache.set_processed(self.SERVICE_NAME, cache_key, sss_data, n_gate_pts=config.n_gate_pts)
        self._cache.save_to_disk()
        logger.info(f"💾 Cached processed SSSData")
        
        return sss_data
    
    @staticmethod
    def get_dataset_info() -> dict:
        """Return dataset metadata."""
        return {
            "name": "CMEMS SSS/DOS",
            "product_id": "MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013",
            "dataset_id": SSS_DATASET_ID,
            "doi": "https://doi.org/10.48670/moi-00051",
            "type": "Gridded",
            "resolution": "0.125° daily",
            "variables": ["sos (salinity)", "dos (density)", "sea_ice_fraction"],
            "description": "Multi-observation Sea Surface Salinity and Density from SMAP/SMOS satellites.",
        }
