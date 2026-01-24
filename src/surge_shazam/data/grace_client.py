"""
🌍 GRACE-FO Mass Change Client
==============================

GRACE Follow-On measures Earth's gravity field variations.
These reveal mass redistribution: groundwater, ice, ocean mass.

Implements the DataClient interface ("parking spot" contract).

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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import logging

# Import base client interface
from .base_client import (
    DataClient,
    DataFormat,
    ClientStatus,
    BoundingBox,
    TimeRange,
    HealthCheckResult,
    DataClientError,
    XArrayClientMixin,
)

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    import numpy as np
    import pandas as pd
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


class GRACEClient(XArrayClientMixin, DataClient):
    """
    GRACE-FO Terrestrial Water Storage Client.

    Implements the DataClient interface for unified data access.

    Usage (new interface):
        client = GRACEClient()

        ds = await client.download(
            variables=["lwe_thickness"],
            bbox=BoundingBox(lon_min=7.0, lat_min=44.0, lon_max=12.0, lat_max=47.0),
            time_range=TimeRange.from_strings("2020-01-01", "2020-12-31"),
            product="jpl_mascon",
        )
    """

    # =========================================================================
    # DataClient REQUIRED PROPERTIES
    # =========================================================================

    @property
    def source_id(self) -> str:
        """Unique identifier matching api_registry.py."""
        return "grace_fo"

    # output_format is provided by XArrayClientMixin

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

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
            logger.info("Authenticated with NASA Earthdata")
            return True
        except Exception:
            try:
                self._auth = earthaccess.login(strategy="netrc")
                return True
            except Exception as e:
                logger.warning(f"Earthdata auth failed: {e}")
                return False

    # =========================================================================
    # DataClient REQUIRED METHODS
    # =========================================================================

    def list_products(self) -> Dict[str, str]:
        """List available GRACE products."""
        return {k: f"{v.name} ({v.provider}): {v.description}" for k, v in GRACE_PRODUCTS.items()}

    async def health_check(self) -> HealthCheckResult:
        """Check if GRACE/Earthdata API is available."""
        start_time = time.time()

        if not HAS_EARTHACCESS:
            return HealthCheckResult(
                status=ClientStatus.UNHEALTHY,
                message="earthaccess library not installed",
                details={"install": "pip install earthaccess"}
            )

        if not self._login():
            return HealthCheckResult(
                status=ClientStatus.DEGRADED,
                message="NASA Earthdata authentication failed",
                details={
                    "env_vars": ["EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"],
                    "signup": "https://urs.earthdata.nasa.gov"
                }
            )

        latency_ms = (time.time() - start_time) * 1000
        return HealthCheckResult(
            status=ClientStatus.HEALTHY,
            message="GRACE/Earthdata client ready",
            latency_ms=latency_ms,
            details={"products_available": len(GRACE_PRODUCTS)}
        )

    async def download(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        *,
        product: str = "jpl_mascon",
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Download GRACE-FO water storage data (DataClient interface).

        Args:
            variables: List of variable names (e.g., ["lwe_thickness"])
            bbox: Geographic bounding box
            time_range: Start and end time
            product: "jpl_mascon", "csr_mascon", or "gfz_mascon"

        Returns:
            xr.Dataset with lwe_thickness and uncertainty

        Raises:
            DataClientError: If download fails
        """
        try:
            result = await self._download_impl(
                product=product,
                bbox_tuple=(bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max),
                time_range=(
                    time_range.start.strftime("%Y-%m-%d"),
                    time_range.end.strftime("%Y-%m-%d")
                ),
                variables=variables,
            )

            if result is None:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    message="Failed to download GRACE data",
                    fallback_available=True
                )

            return self.standardize_output(result, variables, bbox, time_range)

        except DataClientError:
            raise
        except Exception as e:
            logger.warning(f"[{self.source_id}] Download failed, trying synthetic: {e}")
            try:
                return await self.generate_synthetic(variables, bbox, time_range)
            except Exception:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    original_error=e,
                    message="Both real and synthetic download failed",
                    fallback_available=False
                )

    async def generate_synthetic(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> xr.Dataset:
        """Generate synthetic GRACE data."""
        return await self._download_fallback(
            bbox_tuple=(bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max),
            time_range=(
                time_range.start.strftime("%Y-%m-%d"),
                time_range.end.strftime("%Y-%m-%d")
            ),
        )

    # =========================================================================
    # LEGACY METHODS
    # =========================================================================

    async def download_legacy(
        self,
        product: str = "jpl_mascon",
        bbox: Tuple[float, float, float, float] = None,
        time_range: Tuple[str, str] = None,
        variables: List[str] = None,
    ) -> Optional[xr.Dataset]:
        """Legacy download method (backward compatibility)."""
        return await self._download_impl(
            product=product,
            bbox_tuple=bbox,
            time_range=time_range,
            variables=variables,
        )

    async def _download_impl(
        self,
        product: str = "jpl_mascon",
        bbox_tuple: Tuple[float, float, float, float] = None,
        time_range: Tuple[str, str] = None,
        variables: List[str] = None,
    ) -> Optional[xr.Dataset]:
        """Internal download implementation."""
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
            ds = await self._download_earthaccess(product_info, bbox_tuple, time_range)
            if ds is not None:
                return ds

        # Fallback to synthetic
        return await self._download_fallback(bbox_tuple, time_range)
    
    async def _download_earthaccess(
        self,
        product: GRACEProduct,
        bbox_tuple: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[xr.Dataset]:
        """Download via Earthdata."""
        logger.info(f"Downloading {product.name}...")

        try:
            start = datetime.strptime(time_range[0], "%Y-%m-%d")
            end = datetime.strptime(time_range[1], "%Y-%m-%d")

            granules = earthaccess.search_data(
                short_name=product.short_name,
                temporal=(start, end),
                bounding_box=bbox_tuple if bbox_tuple else None,
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

            if bbox_tuple:
                if 'lat' in ds.coords and 'lon' in ds.coords:
                    ds = ds.sel(
                        lat=slice(bbox_tuple[1], bbox_tuple[3]),
                        lon=slice(bbox_tuple[0], bbox_tuple[2])
                    )

            ds.attrs["source"] = self.source_id
            ds.attrs["product"] = product.name

            return ds

        except Exception as e:
            logger.error(f"Earthdata download failed: {e}")
            return None

    async def _download_fallback(
        self,
        bbox_tuple: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[xr.Dataset]:
        """Generate synthetic GRACE data."""
        if not HAS_XARRAY:
            return None

        logger.info("Generating synthetic GRACE data")

        # GRACE grid (0.5° MASCON)
        resolution = 0.5

        if bbox_tuple:
            lons = np.arange(bbox_tuple[0], bbox_tuple[2], resolution)
            lats = np.arange(bbox_tuple[1], bbox_tuple[3], resolution)
        else:
            lons = np.arange(-180, 180, resolution)
            lats = np.arange(-90, 90, resolution)

        # Ensure at least 1 point
        if len(lons) == 0:
            lons = np.array([bbox_tuple[0] if bbox_tuple else 0])
        if len(lats) == 0:
            lats = np.array([bbox_tuple[1] if bbox_tuple else 0])

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

        # Seasonal signal (cm)
        month_idx = np.array([t.astype('datetime64[M]').astype(int) % 12 for t in times])
        seasonal = 10 * np.sin(2 * np.pi * month_idx / 12)  # Peak in spring

        # Add spatial pattern and noise
        lwe = np.zeros(shape, dtype=np.float32)
        for t in range(len(times)):
            base = seasonal[t]
            spatial = np.random.normal(0, 3, (len(lats), len(lons)))
            # Smooth spatially
            try:
                from scipy.ndimage import gaussian_filter
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
                "source": self.source_id,
                "synthetic": True,
                "units": "cm",
                "description": "Liquid Water Equivalent thickness anomaly",
                "reference": "2004-2009 mean",
                "warning": "This is synthetic data for testing",
                "created": datetime.now().isoformat(),
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


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("=== GRACE-FO Client Test ===\n")

        client = GRACEClient()

        print("=== Health Check ===")
        health = await client.health_check()
        print(f"Status: {health.status.value}")
        print(f"Message: {health.message}")

        print("\n=== Available Products ===")
        for key, desc in client.list_products().items():
            print(f"  {key}: {desc}")

        print("\n=== Download Test (Northern Italy) ===")

        # New interface
        bbox = BoundingBox(lon_min=7.0, lat_min=44.0, lon_max=12.0, lat_max=47.0)
        time_range = TimeRange.from_strings("2020-01-01", "2020-12-31")

        try:
            ds = await client.download(
                variables=["lwe_thickness"],
                bbox=bbox,
                time_range=time_range,
                product="jpl_mascon",
            )
            print(f"Got dataset with shape: {dict(ds.dims)}")
            print(f"Synthetic: {ds.attrs.get('synthetic', False)}")
            print(f"Mean TWS: {float(ds['lwe_thickness'].mean()):.1f} cm")
        except DataClientError as e:
            print(f"Error: {e}")

        print("\n=== Interpretation Examples ===")
        for val in [-20, -10, 0, 10, 20]:
            print(f"   {val:+3d} cm: {client.interpret_tws(val)}")

        print("\nTest complete")

    asyncio.run(test())
