"""
🌊 CYGNSS Wind Speed Client
===========================

NASA CYGNSS constellation for ocean surface wind speeds.
Near real-time via NASA PO.DAAC (2-24h latency).

Implements the DataClient interface ("parking spot" contract).

Data:
- 0.2° x 0.2° grid
- Daily composites
- Ocean surface wind speed from GPS reflectometry

Auth: NASA Earthdata (same as GPM, GRACE)
"""

import os
import asyncio
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any, Union

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

CYGNSS_L3 = "CYGNSS_L3_GLOBAL_DAILY_V3.1"


class CYGNSSClient(XArrayClientMixin, DataClient):
    """
    NASA CYGNSS Ocean Wind Speed Client.

    Implements the DataClient interface for unified data access.

    Usage (new interface):
        client = CYGNSSClient()

        ds = await client.download(
            variables=["wind_speed"],
            bbox=BoundingBox(lon_min=-90, lat_min=20, lon_max=-60, lat_max=35),
            time_range=TimeRange.from_strings("2023-01-01", "2023-01-31"),
        )
    """

    # =========================================================================
    # DataClient REQUIRED PROPERTIES
    # =========================================================================

    @property
    def source_id(self) -> str:
        """Unique identifier matching api_registry.py."""
        return "cygnss"

    # output_format is provided by XArrayClientMixin

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "cygnss"
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
        """List available CYGNSS products."""
        return {
            "wind_speed": "Ocean surface wind speed from GPS reflectometry",
            "wind_speed_uncertainty": "Wind speed measurement uncertainty",
        }

    async def health_check(self) -> HealthCheckResult:
        """Check if CYGNSS/Earthdata API is available."""
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
            message="CYGNSS/Earthdata client ready",
            latency_ms=latency_ms,
            details={"product": CYGNSS_L3}
        )

    async def download(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame]:
        """
        Download CYGNSS wind speed data (DataClient interface).

        Args:
            variables: List of variable names (e.g., ["wind_speed"])
            bbox: Geographic bounding box
            time_range: Start and end time

        Returns:
            xr.Dataset with wind speed data

        Raises:
            DataClientError: If download fails
        """
        try:
            result = await self._download_impl(
                time_range=(time_range.start, time_range.end),
                bbox_tuple=(bbox.lon_min, bbox.lat_min, bbox.lon_max, bbox.lat_max),
                variables=variables,
            )

            if result is None or (hasattr(result, 'attrs') and 'note' in result.attrs):
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    message="Failed to download CYGNSS data",
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
        """Generate synthetic CYGNSS data."""
        logger.info("Generating synthetic CYGNSS data")

        if not HAS_XARRAY:
            raise DataClientError(
                source_id=self.source_id,
                operation="generate_synthetic",
                message="xarray not installed"
            )

        # CYGNSS grid (0.2°)
        resolution = 0.2
        lons = np.arange(bbox.lon_min, bbox.lon_max, resolution)
        lats = np.arange(bbox.lat_min, bbox.lat_max, resolution)

        if len(lons) == 0:
            lons = np.array([bbox.lon_min])
        if len(lats) == 0:
            lats = np.array([bbox.lat_min])

        times = np.arange(
            np.datetime64(time_range.start.strftime("%Y-%m-%d")),
            np.datetime64(time_range.end.strftime("%Y-%m-%d")) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D')
        )

        shape = (len(times), len(lats), len(lons))

        # Generate realistic wind speeds (5-20 m/s typical)
        wind_speed = 10 + 5 * np.random.randn(*shape)
        wind_speed = np.clip(wind_speed, 0, 50).astype(np.float32)

        # Add some storm events
        storm_mask = np.random.random(shape) > 0.95
        wind_speed[storm_mask] = np.random.uniform(25, 50, storm_mask.sum())

        ds = xr.Dataset(
            {
                "wind_speed": (["time", "lat", "lon"], wind_speed),
                "wind_speed_uncertainty": (["time", "lat", "lon"], np.ones(shape, dtype=np.float32) * 2.0),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "source": self.source_id,
                "synthetic": True,
                "units": "m/s",
                "description": "Ocean surface wind speed",
                "warning": "This is synthetic data for testing",
                "created": datetime.now().isoformat(),
            }
        )

        return ds

    # =========================================================================
    # LEGACY METHODS
    # =========================================================================

    def search_granules(
        self,
        time_range: Tuple[datetime, datetime],
        bbox: Tuple[float, float, float, float] = None,
        max_results: int = 100,
    ) -> List:
        """Search for CYGNSS granules."""
        if not HAS_EARTHACCESS or not self._login():
            return []

        return earthaccess.search_data(
            short_name=CYGNSS_L3,
            temporal=time_range,
            bounding_box=bbox,
            count=max_results,
        )

    async def _download_impl(
        self,
        time_range: Tuple[datetime, datetime] = None,
        bbox_tuple: Tuple[float, float, float, float] = None,
        variables: List[str] = None,
    ) -> Optional[xr.Dataset]:
        """Internal download implementation."""
        if not HAS_EARTHACCESS:
            return None

        if not self._login():
            return None

        granules = self.search_granules(time_range, bbox_tuple)

        if not granules:
            return None

        download_dir = self.cache_dir / "temp"
        download_dir.mkdir(exist_ok=True)

        files = earthaccess.download(granules, local_path=str(download_dir))

        if not files:
            return None

        ds = xr.open_mfdataset(files, combine="by_coords")

        if variables:
            # Filter to requested variables if they exist
            available_vars = [v for v in variables if v in ds.data_vars]
            if available_vars:
                ds = ds[available_vars]

        ds.attrs["source"] = self.source_id
        ds.attrs["latency"] = "2-24h"

        return ds


# =============================================================================
# MODULE INTERFACE
# =============================================================================

Client = CYGNSSClient


def load(time_range=None, bbox=None, variables=None) -> xr.Dataset:
    """Called by CatalogLoader (legacy interface)."""
    client = CYGNSSClient()
    return asyncio.run(client._download_impl(time_range, bbox, variables))


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    async def test():
        print("=== CYGNSS Client Test ===\n")

        client = CYGNSSClient()

        print("=== Health Check ===")
        health = await client.health_check()
        print(f"Status: {health.status.value}")
        print(f"Message: {health.message}")

        print("\n=== Available Products ===")
        for key, desc in client.list_products().items():
            print(f"  {key}: {desc}")

        print("\n=== Download Test (Gulf of Mexico) ===")

        bbox = BoundingBox(lon_min=-95, lat_min=20, lon_max=-80, lat_max=30)
        time_range = TimeRange.from_strings("2023-01-01", "2023-01-07")

        try:
            ds = await client.download(
                variables=["wind_speed"],
                bbox=bbox,
                time_range=time_range,
            )
            print(f"Got dataset with shape: {dict(ds.dims)}")
            print(f"Synthetic: {ds.attrs.get('synthetic', False)}")
            print(f"Mean wind speed: {float(ds['wind_speed'].mean()):.1f} m/s")
        except DataClientError as e:
            print(f"Error: {e}")

        print("\nTest complete")

    asyncio.run(test())
