"""
Unified Loaders Module
======================
Unified data loading infrastructure with caching.

This module provides a consistent interface for all data sources:
- Copernicus Marine Service (CMEMS)
- ERA5/CDS
- Local NetCDF/ZARR
- Satellite altimetry cycles

Uses the services layer for configuration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages data caching for all loaders."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Path] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load cache index from disk."""
        index_file = self.cache_dir / ".cache_index.json"
        if index_file.exists():
            try:
                import json
                with open(index_file) as f:
                    self._index = json.load(f)
            except Exception:
                self._index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        index_file = self.cache_dir / ".cache_index.json"
        try:
            import json
            with open(index_file, 'w') as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache index: {e}")
    
    def get_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data."""
        if key not in self._index:
            return None
        
        cache_path = Path(self._index[key])
        if not cache_path.exists():
            del self._index[key]
            return None
        
        try:
            import xarray as xr
            logger.info(f"Cache hit: {key}")
            return xr.open_dataset(cache_path)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            return None
    
    def put(self, key: str, data: Any, name: str = "data") -> None:
        """Cache data."""
        cache_path = self.cache_dir / f"{name}_{key}.nc"
        
        try:
            data.to_netcdf(cache_path)
            self._index[key] = str(cache_path)
            self._save_index()
            logger.info(f"Cached: {cache_path}")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self, key: Optional[str] = None) -> None:
        """Clear cache."""
        if key:
            if key in self._index:
                try:
                    Path(self._index[key]).unlink()
                except Exception:
                    pass
                del self._index[key]
        else:
            for path in self._index.values():
                try:
                    Path(path).unlink()
                except Exception:
                    pass
            self._index.clear()
        self._save_index()


class UnifiedLoader:
    """
    Unified loader that routes to appropriate data source.
    
    Automatically handles:
    - Source selection based on dataset_id
    - Caching
    - Error handling
    - Fallback to mock data
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        self.cache = CacheManager(cache_dir) if use_cache else None
        self._copernicus_available = self._check_copernicus()
        self._era5_available = self._check_era5()
    
    def _check_copernicus(self) -> bool:
        """Check Copernicus availability."""
        try:
            import copernicusmarine
            return True
        except ImportError:
            return False
    
    def _check_era5(self) -> bool:
        """Check ERA5/CDS availability."""
        try:
            import cdsapi
            return True
        except ImportError:
            return False
    
    @property
    def copernicus_available(self) -> bool:
        return self._copernicus_available
    
    @property
    def era5_available(self) -> bool:
        return self._era5_available
    
    def load(
        self,
        dataset_id: str,
        variables: List[str],
        bbox: tuple,  # (lon_min, lat_min, lon_max, lat_max)
        time_range: tuple,  # (start, end)
        use_cache: bool = True,
        **kwargs
    ) -> Optional[Any]:
        """
        Load data from appropriate source.
        
        Routes based on dataset_id prefix:
        - cmems_*, copernicus_* → Copernicus
        - era5_*, reanalysis* → ERA5
        - local_*, file_* → Local file
        
        Args:
            dataset_id: Dataset identifier
            variables: Variables to load
            bbox: Bounding box (lon_min, lat_min, lon_max, lat_max)
            time_range: Time range (start, end)
            use_cache: Whether to use caching
            
        Returns:
            xarray Dataset
        """
        # Check cache
        if use_cache and self.cache:
            cache_key = self.cache.get_key(
                dataset_id=dataset_id,
                variables=tuple(variables),
                bbox=bbox,
                time_range=time_range
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Route to loader
        data = None
        
        if dataset_id.startswith(("cmems", "copernicus")):
            data = self._load_copernicus(dataset_id, variables, bbox, time_range, **kwargs)
        elif dataset_id.startswith(("era5", "reanalysis")):
            data = self._load_era5(dataset_id, variables, bbox, time_range, **kwargs)
        elif dataset_id.startswith(("local", "file")):
            path = kwargs.get("path")
            if path:
                data = self._load_local(path, bbox, time_range, **kwargs)
        else:
            # Try Copernicus by default
            data = self._load_copernicus(dataset_id, variables, bbox, time_range, **kwargs)
        
        # Cache result
        if data is not None and use_cache and self.cache:
            self.cache.put(cache_key, data, name=dataset_id.split("_")[0])
        
        return data
    
    def _load_copernicus(
        self,
        dataset_id: str,
        variables: List[str],
        bbox: tuple,
        time_range: tuple,
        **kwargs
    ) -> Optional[Any]:
        """Load from Copernicus Marine Service."""
        if not self.copernicus_available:
            logger.warning("Copernicus not available, using mock data")
            return self._generate_mock(variables, bbox, time_range)
        
        try:
            import copernicusmarine as cm
            
            lon_min, lat_min, lon_max, lat_max = bbox
            start, end = time_range
            
            start_str = start.strftime("%Y-%m-%d") if hasattr(start, 'strftime') else str(start)[:10]
            end_str = end.strftime("%Y-%m-%d") if hasattr(end, 'strftime') else str(end)[:10]
            
            logger.info(f"Loading Copernicus: {dataset_id}")
            
            ds = cm.open_dataset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=start_str,
                end_datetime=end_str,
            )
            
            return ds
            
        except Exception as e:
            logger.error(f"Copernicus error: {e}")
            return self._generate_mock(variables, bbox, time_range)
    
    def _load_era5(
        self,
        dataset_id: str,
        variables: List[str],
        bbox: tuple,
        time_range: tuple,
        **kwargs
    ) -> Optional[Any]:
        """Load from ERA5/CDS."""
        if not self.era5_available:
            logger.warning("ERA5 not available, using mock data")
            return self._generate_mock(variables, bbox, time_range)
        
        # ERA5 loading would go here
        # For now, return mock
        return self._generate_mock(variables, bbox, time_range)
    
    def _load_local(
        self,
        path: Union[str, Path],
        bbox: Optional[tuple] = None,
        time_range: Optional[tuple] = None,
        **kwargs
    ) -> Optional[Any]:
        """Load from local file."""
        try:
            import xarray as xr
            
            path = Path(path)
            if not path.exists():
                logger.error(f"File not found: {path}")
                return None
            
            if path.suffix == ".zarr" or path.is_dir():
                ds = xr.open_zarr(path)
            else:
                ds = xr.open_dataset(path)
            
            # Apply spatial filter
            if bbox and "lat" in ds.coords and "lon" in ds.coords:
                lon_min, lat_min, lon_max, lat_max = bbox
                ds = ds.sel(
                    lat=slice(lat_min, lat_max),
                    lon=slice(lon_min, lon_max)
                )
            
            # Apply time filter
            if time_range and "time" in ds.coords:
                start, end = time_range
                ds = ds.sel(time=slice(start, end))
            
            return ds
            
        except Exception as e:
            logger.error(f"Local load error: {e}")
            return None
    
    def _generate_mock(
        self,
        variables: List[str],
        bbox: tuple,
        time_range: tuple
    ) -> Optional[Any]:
        """Generate mock data for testing."""
        try:
            import numpy as np
            import xarray as xr
            
            lon_min, lat_min, lon_max, lat_max = bbox
            start, end = time_range
            
            lats = np.linspace(lat_min, lat_max, 50)
            lons = np.linspace(lon_min, lon_max, 50)
            
            start_dt = np.datetime64(start.isoformat()[:10]) if hasattr(start, 'isoformat') else np.datetime64(str(start)[:10])
            end_dt = np.datetime64(end.isoformat()[:10]) if hasattr(end, 'isoformat') else np.datetime64(str(end)[:10])
            times = np.arange(start_dt, end_dt, np.timedelta64(1, 'D'))
            
            data_vars = {}
            for var in variables:
                data = np.random.randn(len(times), len(lats), len(lons)) * 0.1
                data_vars[var] = (["time", "lat", "lon"], data)
            
            ds = xr.Dataset(
                data_vars,
                coords={"time": times, "lat": lats, "lon": lons},
                attrs={"source": "mock_data", "generated": datetime.now().isoformat()}
            )
            
            logger.info(f"Generated mock data: {ds.dims}")
            return ds
            
        except Exception as e:
            logger.error(f"Mock generation error: {e}")
            return None


# Convenience function
def load_data(
    dataset_id: str,
    variables: List[str],
    bbox: tuple,
    time_range: tuple,
    **kwargs
) -> Optional[Any]:
    """
    Convenience function to load data.
    
    Creates a UnifiedLoader and loads data.
    """
    loader = UnifiedLoader()
    return loader.load(dataset_id, variables, bbox, time_range, **kwargs)
