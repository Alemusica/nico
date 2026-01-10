"""
GEBCO Bathymetry Service
========================
Service for loading GEBCO bathymetry data along gates.

Dataset Info:
    - Name: GEBCO 2025 Grid
    - Source: General Bathymetric Chart of the Oceans
    - Type: GRIDDED (lat × lon)
    - Resolution: ~15 arc-seconds (~460m)
    - Variable: elevation (height_above_mean_sea_level)
    - Ocean depths are NEGATIVE values

Features:
    - Efficient interpolation using RegularGridInterpolator
    - Caching of bathymetry profiles per gate (avoids reloading 830MB file)
    - Support for fixed depth cap or full bathymetry

Usage:
    service = GEBCOService("/path/to/gebco_2025.nc")
    depths = service.get_depths_along_gate(gate_lons, gate_lats)
    # depths is positive (depth below sea level)
    
    # With caching:
    cache = BathymetryCache()
    depths = cache.get_or_compute(
        gate_name="fram_strait",
        gate_lons=lons, 
        gate_lats=lats,
        gebco_path="/path/to/gebco.nc"
    )
"""

import numpy as np
import xarray as xr
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)


class GEBCOService:
    """
    Service for extracting bathymetry data from GEBCO NetCDF files.
    
    GEBCO stores elevation (positive = land, negative = ocean depth).
    This service converts to depth (positive values for ocean depth).
    """
    
    def __init__(self, nc_path: Optional[str] = None):
        """
        Initialize GEBCO service.
        
        Args:
            nc_path: Path to GEBCO NetCDF file. Can be set later via load().
        """
        self._nc_path: Optional[str] = nc_path
        self._ds: Optional[xr.Dataset] = None
        self._interpolator: Optional[RegularGridInterpolator] = None
        self._lats: Optional[np.ndarray] = None
        self._lons: Optional[np.ndarray] = None
        
        if nc_path:
            self.load(nc_path)
    
    @log_call(logger)
    def load(self, nc_path: str) -> None:
        """
        Load GEBCO NetCDF file.
        
        Args:
            nc_path: Path to GEBCO NetCDF file
        """
        logger.info(f"Loading GEBCO bathymetry from {nc_path}")
        
        if not Path(nc_path).exists():
            raise FileNotFoundError(f"GEBCO file not found: {nc_path}")
        
        self._nc_path = nc_path
        self._ds = xr.open_dataset(nc_path)
        
        # Get coordinates
        self._lats = self._ds["lat"].values
        self._lons = self._ds["lon"].values
        
        # Get elevation data (lazy loading)
        elevation = self._ds["elevation"].values
        
        # Build interpolator for fast queries
        # Note: RegularGridInterpolator expects (lat, lon) order
        self._interpolator = RegularGridInterpolator(
            (self._lats, self._lons),
            elevation,
            method="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        
        logger.info(f"GEBCO loaded: {len(self._lats)}x{len(self._lons)} grid")
        logger.info(f"Lat range: [{self._lats.min():.2f}, {self._lats.max():.2f}]")
        logger.info(f"Lon range: [{self._lons.min():.2f}, {self._lons.max():.2f}]")
    
    def is_loaded(self) -> bool:
        """Check if GEBCO data is loaded."""
        return self._interpolator is not None
    
    @log_call(logger)
    def get_depths_along_gate(
        self,
        gate_lons: np.ndarray,
        gate_lats: np.ndarray,
        depth_cap: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract bathymetry depths along a gate.
        
        Args:
            gate_lons: Longitudes of gate points
            gate_lats: Latitudes of gate points
            depth_cap: Optional maximum depth cap (e.g., 250m).
                      If provided, returns min(actual_depth, depth_cap)
        
        Returns:
            depths: Positive depth values in meters (0 for land/above sea level)
        """
        if not self.is_loaded():
            raise RuntimeError("GEBCO data not loaded. Call load() first.")
        
        gate_lons = np.asarray(gate_lons)
        gate_lats = np.asarray(gate_lats)
        
        # Query points (lat, lon order for interpolator)
        points = np.column_stack([gate_lats, gate_lons])
        
        # Get elevation (negative for ocean)
        elevation = self._interpolator(points)
        
        # Convert to depth (positive for ocean, 0 for land)
        # GEBCO: negative = below sea level (ocean)
        #        positive = above sea level (land)
        depths = np.maximum(-elevation, 0.0)
        
        # Apply depth cap if specified
        if depth_cap is not None and depth_cap > 0:
            depths = np.minimum(depths, depth_cap)
        
        logger.info(f"Extracted depths for {len(depths)} points: "
                   f"min={depths.min():.1f}m, max={depths.max():.1f}m, "
                   f"mean={depths.mean():.1f}m")
        
        return depths
    
    def get_depth_at_point(self, lon: float, lat: float) -> float:
        """Get depth at a single point."""
        depths = self.get_depths_along_gate(
            np.array([lon]), 
            np.array([lat])
        )
        return float(depths[0])
    
    def get_cross_section_area(
        self,
        gate_lons: np.ndarray,
        gate_lats: np.ndarray,
        x_km: np.ndarray,
        depth_cap: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute cross-sectional areas for each gate segment.
        
        Args:
            gate_lons: Longitudes of gate points
            gate_lats: Latitudes of gate points  
            x_km: Distance along gate in km
            depth_cap: Optional maximum depth cap
            
        Returns:
            segment_areas: Area of each segment in m² (n_points - 1)
            total_area: Total cross-sectional area in m²
        """
        depths = self.get_depths_along_gate(gate_lons, gate_lats, depth_cap)
        
        # Segment widths in meters
        dx_km = np.diff(x_km)
        dx_m = dx_km * 1000.0
        
        # Average depth for each segment
        depth_avg = (depths[:-1] + depths[1:]) / 2.0
        
        # Area = width × depth
        segment_areas = dx_m * depth_avg
        total_area = np.sum(segment_areas)
        
        logger.info(f"Cross-section: {len(segment_areas)} segments, "
                   f"total area = {total_area/1e6:.2f} km²")
        
        return segment_areas, total_area
    
    def close(self) -> None:
        """Close the dataset."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None
            self._interpolator = None
            logger.info("GEBCO dataset closed")


# Default GEBCO file path (can be overridden)
DEFAULT_GEBCO_PATH = "/Users/nicolocaron/Desktop/ARCFRESH/GEBCO_05_Jan_2026_a8956c607108/gebco_2025_n80.0_s60.0_w-180.0_e180.0.nc"


def get_effective_depths(
    gate_lons: np.ndarray,
    gate_lats: np.ndarray,
    method: str = "fixed",
    fixed_depth: float = 250.0,
    gebco_path: Optional[str] = None,
    gebco_service: Optional[GEBCOService] = None
) -> np.ndarray:
    """
    Get effective depths along gate using specified method.
    
    This is a convenience function that handles both depth methods.
    
    Args:
        gate_lons: Gate longitudes
        gate_lats: Gate latitudes
        method: "fixed" or "gebco"
        fixed_depth: Depth to use for "fixed" method (default 250m)
        gebco_path: Path to GEBCO file (for "gebco" method)
        gebco_service: Pre-loaded GEBCOService instance (optional)
        
    Returns:
        depths: Effective depths in meters
    """
    n_points = len(gate_lons)
    
    if method == "fixed":
        logger.info(f"Using fixed depth cap: {fixed_depth}m")
        return np.full(n_points, fixed_depth)
    
    elif method == "gebco":
        if gebco_service is None:
            path = gebco_path or DEFAULT_GEBCO_PATH
            gebco_service = GEBCOService(path)
        
        # Use GEBCO bathymetry with optional cap
        # Here we use min(gebco_depth, fixed_depth) as a reasonable hybrid
        depths = gebco_service.get_depths_along_gate(
            gate_lons, gate_lats,
            depth_cap=fixed_depth  # Still cap at 250m for shallow areas
        )
        return depths
    
    else:
        raise ValueError(f"Unknown depth method: {method}. Use 'fixed' or 'gebco'.")


# =============================================================================
# BATHYMETRY CACHE
# =============================================================================

# Cache directory for bathymetry data
BATHYMETRY_CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "cache" / "bathymetry"
BATHYMETRY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class BathymetryCacheEntry:
    """Cached bathymetry data for a gate."""
    gate_name: str
    n_points: int
    depths: np.ndarray           # Depth values in meters
    lons: np.ndarray             # Gate longitudes
    lats: np.ndarray             # Gate latitudes
    depth_min: float
    depth_max: float
    depth_mean: float
    created_at: str
    gebco_source: str            # GEBCO file used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "gate_name": self.gate_name,
            "n_points": self.n_points,
            "depths": self.depths.tolist(),
            "lons": self.lons.tolist(),
            "lats": self.lats.tolist(),
            "depth_min": self.depth_min,
            "depth_max": self.depth_max,
            "depth_mean": self.depth_mean,
            "created_at": self.created_at,
            "gebco_source": self.gebco_source,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BathymetryCacheEntry":
        """Load from dict."""
        return cls(
            gate_name=d["gate_name"],
            n_points=d["n_points"],
            depths=np.array(d["depths"]),
            lons=np.array(d["lons"]),
            lats=np.array(d["lats"]),
            depth_min=d["depth_min"],
            depth_max=d["depth_max"],
            depth_mean=d["depth_mean"],
            created_at=d["created_at"],
            gebco_source=d["gebco_source"],
        )


class BathymetryCache:
    """
    Cache for GEBCO bathymetry profiles along gates.
    
    Avoids reloading the 830MB GEBCO file every time.
    Stores small extracted profiles per gate (~KB instead of 830MB).
    
    Usage:
        cache = BathymetryCache()
        
        # Get or compute (loads from cache if available)
        depths = cache.get_or_compute(
            gate_name="fram_strait",
            gate_lons=lons,
            gate_lats=lats,
            gebco_path="/path/to/gebco.nc"
        )
        
        # Force refresh
        depths = cache.get_or_compute(..., force_refresh=True)
        
        # Clear cache
        cache.clear("fram_strait")
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache."""
        self.cache_dir = cache_dir or BATHYMETRY_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._gebco_service: Optional[GEBCOService] = None
    
    def _cache_path(self, gate_name: str) -> Path:
        """Get cache file path for a gate."""
        safe_name = gate_name.replace(" ", "_").lower()
        return self.cache_dir / f"{safe_name}_bathymetry.pkl"
    
    def _hash_gate(self, lons: np.ndarray, lats: np.ndarray) -> str:
        """Generate hash of gate coordinates for validation."""
        data = np.concatenate([lons, lats]).tobytes()
        return hashlib.md5(data).hexdigest()[:12]
    
    def exists(self, gate_name: str) -> bool:
        """Check if bathymetry is cached for this gate."""
        return self._cache_path(gate_name).exists()
    
    def load(self, gate_name: str) -> Optional[BathymetryCacheEntry]:
        """Load cached bathymetry for a gate."""
        path = self._cache_path(gate_name)
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            entry = BathymetryCacheEntry.from_dict(data)
            logger.info(f"📦 Loaded bathymetry from cache: {gate_name} "
                       f"({entry.n_points} points, {entry.depth_mean:.1f}m mean)")
            return entry
        except Exception as e:
            logger.warning(f"Failed to load bathymetry cache: {e}")
            return None
    
    def save(
        self, 
        gate_name: str,
        depths: np.ndarray,
        lons: np.ndarray,
        lats: np.ndarray,
        gebco_source: str
    ) -> bool:
        """Save bathymetry to cache."""
        try:
            entry = BathymetryCacheEntry(
                gate_name=gate_name,
                n_points=len(depths),
                depths=depths,
                lons=lons,
                lats=lats,
                depth_min=float(depths.min()),
                depth_max=float(depths.max()),
                depth_mean=float(depths.mean()),
                created_at=datetime.now().isoformat(),
                gebco_source=gebco_source,
            )
            
            path = self._cache_path(gate_name)
            with open(path, 'wb') as f:
                pickle.dump(entry.to_dict(), f)
            
            size_kb = path.stat().st_size / 1024
            logger.info(f"💾 Saved bathymetry cache: {gate_name} "
                       f"({entry.n_points} points, {size_kb:.1f} KB)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save bathymetry cache: {e}")
            return False
    
    def get_or_compute(
        self,
        gate_name: str,
        gate_lons: np.ndarray,
        gate_lats: np.ndarray,
        gebco_path: Optional[str] = None,
        depth_cap: Optional[float] = None,
        force_refresh: bool = False
    ) -> np.ndarray:
        """
        Get bathymetry from cache or compute from GEBCO.
        
        Args:
            gate_name: Name of the gate (used as cache key)
            gate_lons: Gate longitude coordinates
            gate_lats: Gate latitude coordinates
            gebco_path: Path to GEBCO file (only needed if not cached)
            depth_cap: Optional maximum depth cap
            force_refresh: If True, recompute even if cached
            
        Returns:
            depths: Bathymetry depths in meters (positive values)
        """
        # Try loading from cache first
        if not force_refresh:
            cached = self.load(gate_name)
            if cached is not None:
                # Validate coordinates match
                if len(cached.depths) == len(gate_lons):
                    depths = cached.depths
                    if depth_cap is not None:
                        depths = np.minimum(depths, depth_cap)
                    return depths
                else:
                    logger.warning(f"Cache mismatch for {gate_name}: "
                                  f"{len(cached.depths)} vs {len(gate_lons)} points")
        
        # Compute from GEBCO
        logger.info(f"Computing bathymetry for {gate_name} from GEBCO...")
        
        path = gebco_path or DEFAULT_GEBCO_PATH
        if self._gebco_service is None or self._gebco_service._nc_path != path:
            self._gebco_service = GEBCOService(path)
        
        # Get depths without cap (for caching the full bathymetry)
        depths = self._gebco_service.get_depths_along_gate(
            gate_lons, gate_lats, depth_cap=None
        )
        
        # Save to cache (full bathymetry without cap)
        self.save(gate_name, depths, gate_lons, gate_lats, path)
        
        # Apply cap if requested
        if depth_cap is not None:
            depths = np.minimum(depths, depth_cap)
        
        return depths
    
    def clear(self, gate_name: str) -> bool:
        """Clear cached bathymetry for a gate."""
        path = self._cache_path(gate_name)
        if path.exists():
            path.unlink()
            logger.info(f"🗑️ Cleared bathymetry cache: {gate_name}")
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all cached bathymetry data."""
        count = 0
        for path in self.cache_dir.glob("*_bathymetry.pkl"):
            path.unlink()
            count += 1
        logger.info(f"🗑️ Cleared {count} bathymetry cache files")
        return count
    
    def list_cached(self) -> list:
        """List all cached gates."""
        cached = []
        for path in self.cache_dir.glob("*_bathymetry.pkl"):
            name = path.stem.replace("_bathymetry", "")
            cached.append(name)
        return cached


# Global cache instance for convenience
_bathymetry_cache: Optional[BathymetryCache] = None


def get_bathymetry_cache() -> BathymetryCache:
    """Get the global bathymetry cache instance."""
    global _bathymetry_cache
    if _bathymetry_cache is None:
        _bathymetry_cache = BathymetryCache()
    return _bathymetry_cache
