"""
Bathymetry Service - Load and extract GEBCO bathymetry data.

Supports:
- Loading GEBCO NetCDF files (global or regional)
- Extracting depth profile along a gate line
- Interpolating to gate points
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator
import logging

logger = logging.getLogger(__name__)

# Default GEBCO file location
DEFAULT_GEBCO_PATH = Path("data/bathymetry/gebco_2024.nc")


@dataclass
class BathymetryProfile:
    """Bathymetry profile along a gate."""
    depth: np.ndarray  # Depth values (positive = depth below sea level)
    lon: np.ndarray  # Longitude points
    lat: np.ndarray  # Latitude points
    x_km: np.ndarray  # Distance along gate (km)
    sill_depth: float  # Minimum depth (sill)
    mean_depth: float  # Mean depth
    max_depth: float  # Maximum depth
    source: str = "GEBCO"


class BathymetryService:
    """
    Service for loading and extracting GEBCO bathymetry data.
    
    Usage:
        service = BathymetryService("data/bathymetry/gebco_north.nc")
        profile = service.extract_profile(gate_lon, gate_lat, x_km)
    """
    
    def __init__(self, gebco_path: Optional[str] = None):
        """
        Initialize bathymetry service.
        
        Args:
            gebco_path: Path to GEBCO NetCDF file. If None, uses default.
        """
        self.gebco_path = Path(gebco_path) if gebco_path else DEFAULT_GEBCO_PATH
        self._ds: Optional[xr.Dataset] = None
        self._interpolator: Optional[RegularGridInterpolator] = None
        
    def _load_gebco(self) -> bool:
        """Load GEBCO dataset (lazy loading)."""
        if self._ds is not None:
            return True
            
        if not self.gebco_path.exists():
            logger.error(f"GEBCO file not found: {self.gebco_path}")
            return False
        
        try:
            logger.info(f"Loading GEBCO from {self.gebco_path}...")
            self._ds = xr.open_dataset(self.gebco_path)
            logger.info(f"GEBCO loaded: {self._ds.dims}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GEBCO: {e}")
            return False
    
    def _get_interpolator(self, lon_min: float, lon_max: float, 
                          lat_min: float, lat_max: float) -> Optional[RegularGridInterpolator]:
        """
        Create interpolator for a bounding box.
        
        Subsets GEBCO to reduce memory usage.
        """
        if not self._load_gebco():
            return None
        
        ds = self._ds
        
        # Find coordinate names (GEBCO uses 'lon'/'lat' or 'longitude'/'latitude')
        lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
        lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
        
        # Add buffer
        buffer = 0.5  # degrees
        lon_min -= buffer
        lon_max += buffer
        lat_min -= buffer
        lat_max += buffer
        
        # Subset to bounding box
        lon_vals = ds[lon_name].values
        lat_vals = ds[lat_name].values
        
        lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
        lat_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
        
        if not np.any(lon_mask) or not np.any(lat_mask):
            logger.error(f"No GEBCO data in bbox: lon[{lon_min}, {lon_max}], lat[{lat_min}, {lat_max}]")
            return None
        
        # Get elevation variable (GEBCO uses 'elevation')
        elev_name = 'elevation' if 'elevation' in ds.data_vars else list(ds.data_vars)[0]
        
        # Extract subset
        lon_subset = lon_vals[lon_mask]
        lat_subset = lat_vals[lat_mask]
        elev_subset = ds[elev_name].values[np.ix_(lat_mask, lon_mask)]
        
        logger.info(f"GEBCO subset: {len(lon_subset)}x{len(lat_subset)} points")
        
        # Create interpolator
        # Note: elevation is negative for ocean, positive for land
        # We want depth (positive below sea level)
        depth_values = -elev_subset  # Convert to depth
        
        # RegularGridInterpolator expects (lat, lon) order
        interpolator = RegularGridInterpolator(
            (lat_subset, lon_subset),
            depth_values,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        return interpolator
    
    def extract_profile(
        self,
        gate_lon: np.ndarray,
        gate_lat: np.ndarray,
        x_km: np.ndarray
    ) -> Optional[BathymetryProfile]:
        """
        Extract bathymetry profile along a gate line.
        
        Args:
            gate_lon: Longitude points along gate
            gate_lat: Latitude points along gate
            x_km: Distance along gate (km)
            
        Returns:
            BathymetryProfile with interpolated depths
        """
        if len(gate_lon) != len(gate_lat):
            logger.error("gate_lon and gate_lat must have same length")
            return None
        
        # Get bounding box
        lon_min, lon_max = np.min(gate_lon), np.max(gate_lon)
        lat_min, lat_max = np.min(gate_lat), np.max(gate_lat)
        
        # Create interpolator for this region
        interpolator = self._get_interpolator(lon_min, lon_max, lat_min, lat_max)
        if interpolator is None:
            return None
        
        # Interpolate depths at gate points
        points = np.column_stack([gate_lat, gate_lon])  # (lat, lon) order
        depth_values = interpolator(points)
        
        # Handle land points (negative depth = above sea level)
        depth_values = np.maximum(depth_values, 0)  # Set land to 0
        
        # Calculate statistics
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            logger.warning("No valid ocean depths found along gate")
            sill_depth = 0
            mean_depth = 0
            max_depth = 0
        else:
            sill_depth = float(np.min(valid_depths))
            mean_depth = float(np.mean(valid_depths))
            max_depth = float(np.max(valid_depths))
        
        logger.info(f"Bathymetry profile: sill={sill_depth:.0f}m, mean={mean_depth:.0f}m, max={max_depth:.0f}m")
        
        return BathymetryProfile(
            depth=depth_values,
            lon=gate_lon,
            lat=gate_lat,
            x_km=x_km,
            sill_depth=sill_depth,
            mean_depth=mean_depth,
            max_depth=max_depth,
            source=f"GEBCO ({self.gebco_path.name})"
        )
    
    def get_available_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get the available lon/lat bounds of the loaded GEBCO file."""
        if not self._load_gebco():
            return None
        
        ds = self._ds
        lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
        lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
        
        lon_vals = ds[lon_name].values
        lat_vals = ds[lat_name].values
        
        return (
            float(np.min(lon_vals)),
            float(np.max(lon_vals)),
            float(np.min(lat_vals)),
            float(np.max(lat_vals))
        )


# Singleton instance
_bathymetry_service: Optional[BathymetryService] = None


def get_bathymetry_service(gebco_path: Optional[str] = None) -> BathymetryService:
    """Get or create bathymetry service singleton."""
    global _bathymetry_service
    
    if _bathymetry_service is None or (gebco_path and str(_bathymetry_service.gebco_path) != gebco_path):
        _bathymetry_service = BathymetryService(gebco_path)
    
    return _bathymetry_service
