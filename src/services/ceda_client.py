"""
CEDA OPeNDAP Client for SLCCI Data
==================================

Client for accessing ESA Sea Level CCI data from CEDA archive via OPeNDAP.

CEDA SLCCI data is publicly accessible - NO AUTHENTICATION REQUIRED!

Features:
- Spatial subsetting via bbox (reduces download size)
- Cycle discovery (list available cycles)
- Local caching to avoid re-downloads

Usage:
    client = CEDAClient()
    
    # Discover available cycles
    cycles = client.discover_cycles(satellite="J2")
    
    # Fetch data for a bbox
    ds = client.fetch_cycle(
        satellite="J2",
        cycle=10,
        bbox=(-65, 67, -53, 69),  # (lon_min, lat_min, lon_max, lat_max)
    )
"""

import os
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import requests

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class CEDAConfig:
    """Configuration for CEDA client."""
    base_url: str = "https://dap.ceda.ac.uk/neodc/esacci/sea_level/data/FCDR/v2.0"
    cache_dir: str = "data/cache/slcci"
    
    # Variables to fetch (minimum needed for DOT analysis)
    variables: Tuple[str, ...] = (
        "corssh",      # Corrected Sea Surface Height
        "latitude",
        "longitude", 
        "time",
        "flag",        # Quality flag
        "pass",        # Pass number
    )


def load_ceda_config() -> CEDAConfig:
    """Load CEDA configuration (no auth needed - public data)."""
    return CEDAConfig()


# ==============================================================================
# CEDA CLIENT
# ==============================================================================

class CEDAClient:
    """
    Client for accessing SLCCI data from CEDA via OPeNDAP.
    
    CEDA SLCCI data is publicly accessible - no authentication required!
    
    Example:
        client = CEDAClient()
        
        # Check connection
        if client.test_connection():
            cycles = client.discover_cycles("J2")
            ds = client.fetch_cycle("J2", 10, bbox=(-65, 67, -53, 69))
    """
    
    def __init__(self, config: Optional[CEDAConfig] = None):
        """Initialize CEDA client."""
        self.config = config or load_ceda_config()
        self._session: Optional[requests.Session] = None
        
        # Create cache directory
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        
    @property
    def session(self) -> requests.Session:
        """Get HTTP session (no auth needed)."""
        if self._session is None:
            self._session = requests.Session()
        return self._session
    
    # ==========================================================================
    # CONNECTION & DISCOVERY
    # ==========================================================================
    
    @log_call(logger)
    def test_connection(self) -> bool:
        """Test connection to CEDA server."""
        try:
            url = f"{self.config.base_url}/J2/"
            response = self.session.head(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    @log_call(logger)
    def discover_cycles(
        self, 
        satellite: str = "J2"
    ) -> List[int]:
        """
        Discover available cycles for a satellite.
        
        Parameters
        ----------
        satellite : str
            Satellite identifier ("J1" or "J2")
            
        Returns
        -------
        List[int]
            List of available cycle numbers
        """
        # Known cycles for each satellite (from CEDA archive structure)
        known_cycles = {
            "J1": list(range(1, 260)),   # Jason-1: cycles 1-259
            "J2": list(range(1, 304)),   # Jason-2: cycles 1-303
        }
        
        satellite = satellite.upper()
        if satellite not in known_cycles:
            logger.warning(f"Unknown satellite: {satellite}")
            return []
        
        # For efficiency, return known range
        # In production, could probe server for actual availability
        cycles = known_cycles[satellite]
        logger.info(f"Available cycles for {satellite}: {min(cycles)}-{max(cycles)}")
        
        return cycles
    
    def get_cycle_url(self, satellite: str, cycle: int) -> str:
        """Get URL for a specific cycle file."""
        satellite = satellite.upper()
        filename = f"SLCCI_ALTDB_{satellite}_Cycle{cycle:03d}_V2.nc"
        return f"{self.config.base_url}/{satellite}/{filename}"
    
    # ==========================================================================
    # DATA FETCHING
    # ==========================================================================
    
    @log_call(logger)
    def fetch_cycle(
        self,
        satellite: str,
        cycle: int,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        use_cache: bool = True,
    ) -> Optional[xr.Dataset]:
        """
        Fetch data for a single cycle with optional spatial filtering.
        
        Parameters
        ----------
        satellite : str
            Satellite identifier ("J1" or "J2")
        cycle : int
            Cycle number
        bbox : tuple, optional
            Bounding box (lon_min, lat_min, lon_max, lat_max)
        use_cache : bool
            Whether to use local cache
            
        Returns
        -------
        xr.Dataset or None
            Dataset with requested variables, or None if failed
        """
        satellite = satellite.upper()
        
        # Check cache first
        cache_path = self._get_cache_path(satellite, cycle, bbox)
        if use_cache and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path.name}")
            try:
                return xr.open_dataset(cache_path)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
        
        # Build OPeNDAP URL
        base_url = self.get_cycle_url(satellite, cycle)
        
        try:
            # Open with xarray via OPeNDAP
            logger.info(f"Fetching {satellite} cycle {cycle} from CEDA...")
            
            # Set up authentication for xarray
            # Note: xarray uses netCDF4 which reads .dodsrc for auth
            ds = self._open_opendap(base_url)
            
            if ds is None:
                return None
            
            # Apply bbox filter if provided
            if bbox:
                ds = self._apply_bbox_filter(ds, bbox)
            
            # Select only needed variables
            available_vars = [v for v in self.config.variables if v in ds.data_vars or v in ds.coords]
            ds = ds[available_vars]
            
            # Cache the result
            if use_cache and ds is not None:
                self._save_to_cache(ds, cache_path)
            
            return ds
            
        except Exception as e:
            logger.error(f"Failed to fetch cycle {cycle}: {e}")
            return None
    
    @log_call(logger)
    def fetch_cycles(
        self,
        satellite: str,
        cycles: List[int],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        use_cache: bool = True,
    ) -> Optional[xr.Dataset]:
        """
        Fetch and concatenate multiple cycles.
        
        Parameters
        ----------
        satellite : str
            Satellite identifier
        cycles : List[int]
            List of cycle numbers
        bbox : tuple, optional
            Bounding box
        use_cache : bool
            Whether to use cache
            
        Returns
        -------
        xr.Dataset or None
            Combined dataset
        """
        datasets = []
        
        for cycle in cycles:
            ds = self.fetch_cycle(satellite, cycle, bbox, use_cache)
            if ds is not None and ds.sizes.get('time', 0) > 0:
                # Add cycle coordinate
                ds = ds.assign_coords(cycle=cycle)
                datasets.append(ds)
        
        if not datasets:
            logger.warning("No data fetched for any cycle")
            return None
        
        # Concatenate along time dimension
        combined = xr.concat(datasets, dim='time')
        logger.info(f"Combined {len(datasets)} cycles, {combined.sizes.get('time', 0)} total points")
        
        return combined
    
    # ==========================================================================
    # PRIVATE METHODS
    # ==========================================================================
    
    def _open_opendap(self, url: str) -> Optional[xr.Dataset]:
        """Open OPeNDAP URL with authentication."""
        try:
            # Method 1: Try with session auth
            # xarray/netCDF4 may not use requests session, so we need .netrc or .dodsrc
            
            # For now, try direct open (works if .netrc is configured)
            ds = xr.open_dataset(url, engine='netcdf4')
            return ds
            
        except Exception as e:
            logger.warning(f"Direct OPeNDAP open failed: {e}")
            
            # Method 2: Download to temp file and open
            try:
                return self._download_and_open(url)
            except Exception as e2:
                logger.error(f"Download fallback failed: {e2}")
                return None
    
    def _download_and_open(self, url: str) -> Optional[xr.Dataset]:
        """Download file and open locally."""
        import tempfile
        
        # Convert OPeNDAP URL to direct download URL
        # CEDA URLs: .nc for direct, add .dods for OPeNDAP binary
        download_url = url
        
        response = self.session.get(download_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
            temp_path = f.name
        
        try:
            ds = xr.open_dataset(temp_path)
            return ds
        finally:
            # Cleanup
            os.unlink(temp_path)
    
    def _apply_bbox_filter(
        self, 
        ds: xr.Dataset, 
        bbox: Tuple[float, float, float, float]
    ) -> xr.Dataset:
        """Apply bounding box filter to dataset.
        
        Handles longitude wrapping (CEDA uses 0-360, user may provide -180 to 180).
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        
        # Find lat/lon variable names
        lat_var = 'latitude' if 'latitude' in ds else 'lat'
        lon_var = 'longitude' if 'longitude' in ds else 'lon'
        
        if lat_var not in ds and lon_var not in ds:
            logger.warning("No lat/lon variables found for bbox filtering")
            return ds
        
        # Convert user bbox from -180/180 to 0-360 if needed
        def to_360(lon):
            """Convert longitude to 0-360 range."""
            return lon % 360
        
        lon_min_360 = to_360(lon_min)
        lon_max_360 = to_360(lon_max)
        
        # Get data longitude range
        data_lon = ds[lon_var].values
        data_lon_min, data_lon_max = float(np.nanmin(data_lon)), float(np.nanmax(data_lon))
        
        logger.debug(f"Data lon range: {data_lon_min:.1f}-{data_lon_max:.1f}, bbox lon: {lon_min_360:.1f}-{lon_max_360:.1f}")
        
        # Create lat mask
        lat_mask = (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max)
        
        # Create lon mask (handle wraparound)
        if lon_min_360 <= lon_max_360:
            # Normal case: e.g., 290-320 or 10-50
            lon_mask = (ds[lon_var] >= lon_min_360) & (ds[lon_var] <= lon_max_360)
        else:
            # Wraparound case: e.g., 350-10 means 350-360 OR 0-10
            lon_mask = (ds[lon_var] >= lon_min_360) | (ds[lon_var] <= lon_max_360)
        
        mask = lat_mask & lon_mask
        
        # Apply mask
        ds_filtered = ds.where(mask, drop=True)
        
        n_before = ds.sizes.get('time', len(ds[lat_var]))
        n_after = ds_filtered.sizes.get('time', len(ds_filtered[lat_var]) if lat_var in ds_filtered else 0)
        logger.info(f"Bbox filter: {n_before} -> {n_after} points")
        
        return ds_filtered
    
    def _get_cache_path(
        self, 
        satellite: str, 
        cycle: int, 
        bbox: Optional[Tuple[float, float, float, float]]
    ) -> Path:
        """Generate cache file path."""
        if bbox:
            bbox_str = f"bbox_{bbox[0]:.0f}_{bbox[1]:.0f}_{bbox[2]:.0f}_{bbox[3]:.0f}"
        else:
            bbox_str = "global"
        
        filename = f"{satellite}_cycle{cycle:03d}_{bbox_str}.nc"
        return Path(self.config.cache_dir) / filename
    
    def _save_to_cache(self, ds: xr.Dataset, path: Path):
        """Save dataset to cache."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(path)
            logger.info(f"Cached: {path.name}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_ceda_client() -> CEDAClient:
    """Get configured CEDA client."""
    return CEDAClient()


def fetch_slcci_data(
    satellite: str = "J2",
    cycles: List[int] = None,
    bbox: Tuple[float, float, float, float] = None,
) -> Optional[xr.Dataset]:
    """
    Convenience function to fetch SLCCI data.
    
    Parameters
    ----------
    satellite : str
        "J1" or "J2"
    cycles : List[int]
        Cycles to fetch
    bbox : tuple
        (lon_min, lat_min, lon_max, lat_max)
        
    Returns
    -------
    xr.Dataset or None
    """
    client = get_ceda_client()
    
    if cycles is None:
        cycles = list(range(1, 11))  # Default: first 10 cycles
    
    return client.fetch_cycles(satellite, cycles, bbox)
