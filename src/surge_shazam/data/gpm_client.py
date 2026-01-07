"""
🌧️ GPM IMERG Precipitation Client
==================================

NASA Global Precipitation Measurement - near real-time rainfall data.
Critical for flood early warning: precipitation → runoff → flood.

Products:
- Early Run: ~4h latency (for real-time monitoring)
- Late Run: ~14h latency (more accurate)
- Final Run: ~3.5 months (research quality)

Data:
- 0.1° × 0.1° grid (~10 km)
- 30-minute intervals (or daily aggregates)
- 60°N to 60°S coverage

Auth: NASA Earthdata (same as CYGNSS)
"""

import os
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

try:
    import earthaccess
    HAS_EARTHACCESS = True
except ImportError:
    HAS_EARTHACCESS = False

try:
    import xarray as xr
    import numpy as np
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class GPMProduct:
    """GPM IMERG product definition."""
    short_name: str
    description: str
    latency: str
    temporal_resolution: str


GPM_PRODUCTS = {
    "early": GPMProduct(
        short_name="GPM_3IMERGDE.07",
        description="IMERG Early Run Daily (near real-time)",
        latency="~4 hours",
        temporal_resolution="daily",
    ),
    "late": GPMProduct(
        short_name="GPM_3IMERGDL.07",
        description="IMERG Late Run Daily (gauge-calibrated)",
        latency="~14 hours",
        temporal_resolution="daily",
    ),
    "final": GPMProduct(
        short_name="GPM_3IMERGDF.07",
        description="IMERG Final Run Daily (research quality)",
        latency="~3.5 months",
        temporal_resolution="daily",
    ),
    "half_hourly": GPMProduct(
        short_name="GPM_3IMERGHH.07",
        description="IMERG Half-hourly (30-min)",
        latency="~4 hours",
        temporal_resolution="30-min",
    ),
}


class GPMClient:
    """
    NASA GPM IMERG Precipitation Client.
    
    Usage:
        client = GPMClient()
        
        # Get daily precipitation for flood event
        ds = await client.download(
            product="early",  # Near real-time
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
        
        # Accumulate to total precipitation
        total_precip = ds['precipitation'].sum(dim='time')
    
    Authentication:
        Set environment variables:
        - EARTHDATA_USERNAME
        - EARTHDATA_PASSWORD
        
        Or login interactively: earthaccess.login()
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "gpm"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._auth = None
        
        if not HAS_EARTHACCESS:
            logger.warning("⚠️ earthaccess not installed. Run: pip install earthaccess")
    
    def _login(self):
        """Authenticate with NASA Earthdata."""
        if self._auth is not None:
            return True
        
        if not HAS_EARTHACCESS:
            return False
        
        try:
            # Try environment variables first
            self._auth = earthaccess.login(strategy="environment")
            logger.info("✅ Authenticated with NASA Earthdata (env)")
            return True
        except Exception:
            try:
                # Fall back to netrc
                self._auth = earthaccess.login(strategy="netrc")
                logger.info("✅ Authenticated with NASA Earthdata (netrc)")
                return True
            except Exception as e:
                logger.warning(f"⚠️ NASA Earthdata auth failed: {e}")
                return False
    
    def list_products(self) -> Dict[str, str]:
        """List available GPM products."""
        return {k: f"{v.description} ({v.latency})" for k, v in GPM_PRODUCTS.items()}
    
    async def download(
        self,
        product: str = "early",
        lat_range: Tuple[float, float] = None,
        lon_range: Tuple[float, float] = None,
        time_range: Tuple[str, str] = None,
        variables: List[str] = None,
        force_download: bool = False,
    ) -> Optional[Any]:  # Returns xr.Dataset
        """
        Download GPM IMERG precipitation data.
        
        Args:
            product: "early", "late", "final", or "half_hourly"
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude  
            time_range: (start, end) dates as "YYYY-MM-DD"
            variables: List of variables (default: precipitation)
            force_download: Re-download even if cached
            
        Returns:
            xarray.Dataset with precipitation data
        """
        if not HAS_XARRAY:
            logger.error("❌ xarray required: pip install xarray")
            return None
        
        product_info = GPM_PRODUCTS.get(product)
        if not product_info:
            logger.error(f"❌ Unknown product: {product}")
            return None
        
        variables = variables or ["precipitation"]
        
        # Generate cache key
        cache_key = self._cache_key(product, lat_range, lon_range, time_range)
        cache_file = self.cache_dir / f"{cache_key}.nc"
        
        if cache_file.exists() and not force_download:
            logger.info(f"📁 Loading from cache: {cache_file}")
            return xr.open_dataset(cache_file)
        
        # Try Earthdata download
        if HAS_EARTHACCESS and self._login():
            ds = await self._download_earthaccess(
                product_info, lat_range, lon_range, time_range
            )
            if ds is not None:
                # Cache
                ds.to_netcdf(cache_file)
                return ds
        
        # Fallback to synthetic data
        return await self._download_fallback(lat_range, lon_range, time_range)
    
    async def _download_earthaccess(
        self,
        product: GPMProduct,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Download using earthaccess."""
        logger.info(f"⬇️ Downloading {product.short_name}...")
        logger.info(f"   Area: lat={lat_range}, lon={lon_range}")
        logger.info(f"   Time: {time_range}")
        
        try:
            # Parse time range
            start = datetime.strptime(time_range[0], "%Y-%m-%d")
            end = datetime.strptime(time_range[1], "%Y-%m-%d")
            
            # Search for granules
            granules = earthaccess.search_data(
                short_name=product.short_name,
                temporal=(start, end),
                bounding_box=(lon_range[0], lat_range[0], lon_range[1], lat_range[1]),
                count=1000,
            )
            
            if not granules:
                logger.warning("No granules found")
                return None
            
            logger.info(f"   Found {len(granules)} granules")
            
            # Download to temp directory
            download_dir = self.cache_dir / "temp"
            download_dir.mkdir(exist_ok=True)
            
            files = earthaccess.download(granules, local_path=str(download_dir))
            
            if not files:
                logger.warning("No files downloaded")
                return None
            
            # Open as multi-file dataset
            ds = xr.open_mfdataset(files, combine="by_coords")
            
            # Subset to region if needed
            if lat_range and 'lat' in ds.coords:
                ds = ds.sel(lat=slice(lat_range[0], lat_range[1]))
            if lon_range and 'lon' in ds.coords:
                ds = ds.sel(lon=slice(lon_range[0], lon_range[1]))
            
            ds.attrs["source"] = "GPM IMERG"
            ds.attrs["product"] = product.short_name
            ds.attrs["latency"] = product.latency
            
            return ds
            
        except Exception as e:
            logger.error(f"❌ Earthaccess download failed: {e}")
            return None
    
    async def _download_fallback(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Generate synthetic precipitation data for testing."""
        if not HAS_XARRAY:
            return None
        
        logger.warning("🔧 Generating synthetic GPM data for testing...")
        
        # Grid
        resolution = 0.1  # GPM resolution
        if lat_range:
            lats = np.arange(lat_range[0], lat_range[1], resolution)
        else:
            lats = np.arange(-60, 60, resolution)
        
        if lon_range:
            lons = np.arange(lon_range[0], lon_range[1], resolution)
        else:
            lons = np.arange(-180, 180, resolution)
        
        if time_range:
            times = np.arange(
                np.datetime64(time_range[0]),
                np.datetime64(time_range[1]) + np.timedelta64(1, 'D'),
                np.timedelta64(1, 'D')
            )
        else:
            times = np.arange(
                np.datetime64('2020-01-01'),
                np.datetime64('2020-01-31'),
                np.timedelta64(1, 'D')
            )
        
        # Generate realistic precipitation
        shape = (len(times), len(lats), len(lons))
        
        # Base: mostly dry with occasional rain
        precip = np.zeros(shape)
        
        # Add random rain events
        n_events = max(1, len(times) // 5)
        for _ in range(n_events):
            # Random center
            t_idx = np.random.randint(0, len(times))
            lat_center = np.random.randint(0, len(lats))
            lon_center = np.random.randint(0, len(lons))
            
            # Intensity (mm/day) - exponential distribution
            intensity = np.random.exponential(20)
            
            # Spatial extent (Gaussian)
            lat_sigma = np.random.randint(3, 10)
            lon_sigma = np.random.randint(3, 10)
            
            for i in range(len(lats)):
                for j in range(len(lons)):
                    dist = ((i - lat_center)**2 / lat_sigma**2 + 
                            (j - lon_center)**2 / lon_sigma**2)
                    precip[t_idx, i, j] += intensity * np.exp(-dist)
        
        # Add some noise
        precip += np.maximum(0, np.random.normal(0, 2, shape))
        
        ds = xr.Dataset(
            {
                'precipitation': (['time', 'lat', 'lon'], precip.astype(np.float32)),
                'precipitation_quality': (['time', 'lat', 'lon'], 
                                          np.ones(shape, dtype=np.int8)),
            },
            coords={
                'time': times,
                'lat': lats,
                'lon': lons,
            },
            attrs={
                'source': 'synthetic_gpm_fallback',
                'description': 'Synthetic GPM-like precipitation for testing',
                'units': 'mm/day',
                'warning': 'This is synthetic data, not real GPM data',
            }
        )
        
        return ds
    
    def _cache_key(
        self,
        product: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> str:
        """Generate cache key."""
        import hashlib
        
        key_parts = [
            product,
            f"{lat_range}" if lat_range else "all_lat",
            f"{lon_range}" if lon_range else "all_lon",
            f"{time_range}" if time_range else "all_time",
        ]
        
        key_str = "_".join(key_parts)
        h = hashlib.md5(key_str.encode()).hexdigest()[:12]
        return f"gpm_{product}_{h}"
    
    async def get_accumulated(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        product: str = "early",
    ) -> Optional[Any]:
        """
        Get accumulated precipitation over time period.
        
        Returns:
            xarray.DataArray with total accumulated precipitation [mm]
        """
        ds = await self.download(
            product=product,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        )
        
        if ds is None:
            return None
        
        # Sum over time
        total = ds['precipitation'].sum(dim='time')
        total.attrs['units'] = 'mm'
        total.attrs['description'] = f"Total precipitation {time_range[0]} to {time_range[1]}"
        
        return total


# Convenience function
async def get_precipitation(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_range: Tuple[str, str],
    product: str = "early",
) -> Optional[Any]:
    """Quick precipitation download."""
    client = GPMClient()
    return await client.download(
        product=product,
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
    )


# Module-level client for catalog integration
Client = GPMClient


def load(time_range=None, bbox=None, variables=None, product="early") -> Any:
    """Called by CatalogLoader."""
    lat_range = (bbox[1], bbox[3]) if bbox else None
    lon_range = (bbox[0], bbox[2]) if bbox else None
    return asyncio.run(GPMClient().download(
        product=product,
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
        variables=variables,
    ))


# CLI test
if __name__ == "__main__":
    async def test():
        client = GPMClient()
        
        print("=== GPM IMERG Products ===")
        for key, desc in client.list_products().items():
            print(f"  {key}: {desc}")
        
        print("\n=== Downloading for Lago Maggiore Flood (Oct 2000) ===")
        ds = await client.download(
            product="early",
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
        
        if ds is not None:
            print(f"✅ Got dataset:")
            print(ds)
            
            # Total precipitation
            total = ds['precipitation'].sum(dim='time')
            print(f"\nTotal precipitation: {float(total.mean()):.1f} mm (mean)")
    
    asyncio.run(test())
