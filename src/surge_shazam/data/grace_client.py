"""
🌍 GRACE-FO Mass Change Client
==============================

GRACE Follow-On measures Earth's gravity field variations.
These reveal mass redistribution: groundwater, ice, ocean mass.

For Early Warning:
- Terrestrial Water Storage (TWS) = soil moisture + groundwater + snow
- Pre-conditioning for floods: saturated soil = higher runoff
- Drought monitoring: declining TWS

Data:
- Monthly grids (MASCON solutions)
- ~300 km effective resolution
- ~2 months latency (research quality)

Sources:
- NASA JPL: https://grace.jpl.nasa.gov/
- GFZ Potsdam: https://www.gfz-potsdam.de/grace/
- CSR UT Austin

Access: NASA Earthdata (same as CYGNSS, GPM)
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
    import xarray as xr
    import numpy as np
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import earthaccess
    HAS_EARTHACCESS = True
except ImportError:
    HAS_EARTHACCESS = False


@dataclass
class GRACEProduct:
    """GRACE-FO product definition."""
    short_name: str
    name: str
    provider: str
    resolution: str
    description: str


GRACE_PRODUCTS = {
    "jpl_mascon": GRACEProduct(
        short_name="TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3",
        name="JPL MASCON RL06.1",
        provider="JPL",
        resolution="0.5°",
        description="Mass concentration (MASCON) solution - recommended",
    ),
    "csr_mascon": GRACEProduct(
        short_name="TELLUS_GRAC-GRFO_MASCON_GRID_RL06.1_V3",
        name="CSR MASCON RL06.1",
        provider="CSR",
        resolution="0.5°",
        description="UT CSR MASCON solution",
    ),
    "gfz_mascon": GRACEProduct(
        short_name="GFZ_GRACE-FO_MASCON",
        name="GFZ MASCON",
        provider="GFZ",
        resolution="0.5°",
        description="GFZ Potsdam MASCON solution",
    ),
}


class GRACEClient:
    """
    GRACE-FO Terrestrial Water Storage Client.
    
    Usage:
        client = GRACEClient()
        
        # Get water storage anomaly
        tws = await client.download(
            product="jpl_mascon",
            bbox=(7.0, 44.0, 12.0, 47.0),
            time_range=("2020-01-01", "2020-12-31"),
        )
        
        # Compute regional mean
        regional_mean = tws['lwe_thickness'].mean(dim=['lat', 'lon'])
        
        # Check for drought/wet conditions
        if regional_mean[-1] < -10:  # cm
            print("Drought conditions")
        elif regional_mean[-1] > 10:
            print("Wet conditions - flood risk if rainfall")
    """
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "grace"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._auth = None
    
    def _login(self) -> bool:
        """Authenticate with NASA Earthdata."""
        if not HAS_EARTHACCESS:
            return False
        
        if self._auth is not None:
            return True
        
        try:
            self._auth = earthaccess.login(strategy="environment")
            logger.info("✅ Authenticated with NASA Earthdata")
            return True
        except Exception:
            try:
                self._auth = earthaccess.login(strategy="netrc")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Earthdata auth failed: {e}")
                return False
    
    def list_products(self) -> Dict[str, str]:
        """List available GRACE products."""
        return {k: f"{v.name} ({v.provider}): {v.description}" for k, v in GRACE_PRODUCTS.items()}
    
    async def download(
        self,
        product: str = "jpl_mascon",
        bbox: Tuple[float, float, float, float] = None,
        time_range: Tuple[str, str] = None,
        variables: List[str] = None,
    ) -> Optional[Any]:
        """
        Download GRACE-FO water storage data.
        
        Args:
            product: "jpl_mascon", "csr_mascon", or "gfz_mascon"
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time_range: (start, end) as "YYYY-MM-DD"
            variables: Variables to extract (default: lwe_thickness)
            
        Returns:
            xarray.Dataset with:
            - lwe_thickness: Liquid Water Equivalent thickness anomaly [cm]
            - uncertainty: Measurement uncertainty [cm]
        """
        if not HAS_XARRAY:
            logger.error("xarray required")
            return None
        
        product_info = GRACE_PRODUCTS.get(product)
        if not product_info:
            logger.error(f"Unknown product: {product}")
            return None
        
        variables = variables or ["lwe_thickness"]
        
        # Try Earthdata
        if HAS_EARTHACCESS and self._login():
            ds = await self._download_earthaccess(product_info, bbox, time_range)
            if ds is not None:
                return ds
        
        # Fallback to synthetic
        return await self._download_fallback(bbox, time_range)
    
    async def _download_earthaccess(
        self,
        product: GRACEProduct,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Download via Earthdata."""
        logger.info(f"⬇️ Downloading {product.name}...")
        
        try:
            start = datetime.strptime(time_range[0], "%Y-%m-%d")
            end = datetime.strptime(time_range[1], "%Y-%m-%d")
            
            granules = earthaccess.search_data(
                short_name=product.short_name,
                temporal=(start, end),
                bounding_box=bbox if bbox else None,
                count=100,
            )
            
            if not granules:
                logger.warning("No granules found")
                return None
            
            download_dir = self.cache_dir / "temp"
            download_dir.mkdir(exist_ok=True)
            
            files = earthaccess.download(granules, local_path=str(download_dir))
            
            if not files:
                return None
            
            ds = xr.open_mfdataset(files, combine="by_coords")
            
            if bbox:
                if 'lat' in ds.coords and 'lon' in ds.coords:
                    ds = ds.sel(
                        lat=slice(bbox[1], bbox[3]),
                        lon=slice(bbox[0], bbox[2])
                    )
            
            ds.attrs["source"] = "GRACE-FO"
            ds.attrs["product"] = product.name
            
            return ds
            
        except Exception as e:
            logger.error(f"Earthdata download failed: {e}")
            return None
    
    async def _download_fallback(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Generate synthetic GRACE data."""
        if not HAS_XARRAY:
            return None
        
        logger.warning("🔧 Generating synthetic GRACE data")
        
        # GRACE grid (0.5° MASCON)
        resolution = 0.5
        
        if bbox:
            lons = np.arange(bbox[0], bbox[2], resolution)
            lats = np.arange(bbox[1], bbox[3], resolution)
        else:
            lons = np.arange(-180, 180, resolution)
            lats = np.arange(-90, 90, resolution)
        
        # Monthly time steps
        if time_range:
            times = np.arange(
                np.datetime64(time_range[0][:7]),  # First of month
                np.datetime64(time_range[1][:7]) + np.timedelta64(1, 'M'),
                np.timedelta64(1, 'M')
            )
        else:
            times = np.arange(
                np.datetime64('2020-01'),
                np.datetime64('2021-01'),
                np.timedelta64(1, 'M')
            )
        
        shape = (len(times), len(lats), len(lons))
        
        # Generate realistic TWS anomaly
        # - Seasonal cycle (wet winter, dry summer in Europe)
        # - Interannual variability
        # - Spatial correlation
        
        # Seasonal signal (cm)
        month_idx = np.array([t.astype('datetime64[M]').astype(int) % 12 for t in times])
        seasonal = 10 * np.sin(2 * np.pi * month_idx / 12)  # Peak in spring
        
        # Add spatial pattern and noise
        lwe = np.zeros(shape, dtype=np.float32)
        for t in range(len(times)):
            base = seasonal[t]
            spatial = np.random.normal(0, 3, (len(lats), len(lons)))
            # Smooth spatially
            from scipy.ndimage import gaussian_filter
            try:
                spatial = gaussian_filter(spatial, sigma=2)
            except ImportError:
                pass
            lwe[t] = base + spatial
        
        # Uncertainty (increases with latitude due to orbit)
        uncertainty = np.ones(shape, dtype=np.float32) * 2.0
        for i, lat in enumerate(lats):
            uncertainty[:, i, :] *= (1 + 0.01 * abs(lat))
        
        ds = xr.Dataset(
            {
                "lwe_thickness": (["time", "lat", "lon"], lwe),
                "uncertainty": (["time", "lat", "lon"], uncertainty),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "source": "synthetic_grace_fallback",
                "units": "cm",
                "description": "Liquid Water Equivalent thickness anomaly",
                "reference": "2004-2009 mean",
                "warning": "This is synthetic data for testing",
            }
        )
        
        return ds
    
    async def get_regional_timeseries(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
        product: str = "jpl_mascon",
    ) -> Optional[Any]:
        """
        Get regional mean TWS time series.
        
        Returns:
            pandas.DataFrame with date and lwe_thickness columns
        """
        ds = await self.download(product, bbox, time_range)
        if ds is None:
            return None
        
        # Compute area-weighted mean
        weights = np.cos(np.deg2rad(ds.lat))
        regional = ds['lwe_thickness'].weighted(weights).mean(dim=['lat', 'lon'])
        
        try:
            import pandas as pd
            df = regional.to_dataframe().reset_index()
            df.columns = ['date', 'lwe_cm']
            return df
        except ImportError:
            return regional
    
    def interpret_tws(self, lwe_cm: float) -> str:
        """
        Interpret TWS anomaly value.
        
        Args:
            lwe_cm: Liquid Water Equivalent in cm
            
        Returns:
            Interpretation string
        """
        if lwe_cm < -15:
            return "Severe drought - very low soil moisture"
        elif lwe_cm < -5:
            return "Moderate drought - below normal water storage"
        elif lwe_cm < 5:
            return "Normal conditions"
        elif lwe_cm < 15:
            return "Wet conditions - elevated flood risk if rainfall"
        else:
            return "Very wet - saturated soils, high flood risk"


# Module interface
Client = GRACEClient


async def get_water_storage(
    bbox: Tuple[float, float, float, float],
    time_range: Tuple[str, str],
) -> Optional[Any]:
    """Quick GRACE data access."""
    client = GRACEClient()
    return await client.download(bbox=bbox, time_range=time_range)


# CLI test
if __name__ == "__main__":
    async def test():
        print("=== GRACE-FO Client Test ===\n")
        
        client = GRACEClient()
        
        print("Available products:")
        for key, desc in client.list_products().items():
            print(f"  {key}: {desc}")
        
        # Northern Italy
        bbox = (7.0, 44.0, 12.0, 47.0)
        time_range = ("2020-01-01", "2020-12-31")
        
        print(f"\n1. Downloading TWS for Northern Italy...")
        ds = await client.download(bbox=bbox, time_range=time_range)
        
        if ds:
            print(f"   Got dataset: {ds}")
            print(f"   Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            print(f"   Mean TWS: {float(ds['lwe_thickness'].mean()):.1f} cm")
        
        print(f"\n2. Getting regional time series...")
        ts = await client.get_regional_timeseries(bbox, time_range)
        if ts is not None:
            print(f"   Got {len(ts)} monthly values")
        
        print(f"\n3. Interpretation examples:")
        for val in [-20, -10, 0, 10, 20]:
            print(f"   {val:+3d} cm: {client.interpret_tws(val)}")
        
        print("\n✅ Test complete")
    
    asyncio.run(test())
