"""
🛰️ Sentinel Client - ESA Copernicus Satellites
===============================================

Sentinel-1: SAR (Synthetic Aperture Radar)
- Flood mapping, sea state, wind fields
- All-weather, day/night imaging
- 6-12 day revisit

Sentinel-3: OLCI/SLSTR
- Ocean color, SST, altimetry
- Sediment plumes (river discharge proxy)
- Chlorophyll (ecosystem response)

Data Access: Copernicus Data Space Ecosystem
https://dataspace.copernicus.eu/
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import logging
import json

logger = logging.getLogger(__name__)

try:
    import xarray as xr
    import numpy as np
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class SentinelProduct:
    """Sentinel product definition."""
    id: str
    name: str
    satellite: str
    instrument: str
    processing_level: str
    variables: List[str]
    resolution: str
    description: str


SENTINEL_PRODUCTS = {
    # Sentinel-1 SAR
    "s1_grd": SentinelProduct(
        id="SENTINEL-1-GRD",
        name="Sentinel-1 GRD",
        satellite="Sentinel-1",
        instrument="SAR",
        processing_level="L1",
        variables=["sigma0_vv", "sigma0_vh"],
        resolution="10m",
        description="Ground Range Detected - calibrated backscatter",
    ),
    "s1_ocn": SentinelProduct(
        id="SENTINEL-1-OCN",
        name="Sentinel-1 Ocean",
        satellite="Sentinel-1",
        instrument="SAR",
        processing_level="L2",
        variables=["owi_wind_speed", "owi_wind_direction", "osw_wave_spectra", "rvl_radial_velocity"],
        resolution="1km",
        description="Ocean products: wind, waves, currents",
    ),
    
    # Sentinel-3 OLCI
    "s3_olci_l2": SentinelProduct(
        id="SENTINEL-3-OLCI-L2",
        name="Sentinel-3 OLCI L2",
        satellite="Sentinel-3",
        instrument="OLCI",
        processing_level="L2",
        variables=["chlorophyll", "tsm", "cdom", "water_reflectance"],
        resolution="300m",
        description="Ocean color: chlorophyll, sediments, CDOM",
    ),
    
    # Sentinel-3 SLSTR
    "s3_slstr_l2": SentinelProduct(
        id="SENTINEL-3-SLSTR-L2",
        name="Sentinel-3 SLSTR L2",
        satellite="Sentinel-3",
        instrument="SLSTR",
        processing_level="L2",
        variables=["sst", "lst"],
        resolution="1km",
        description="Sea/Land Surface Temperature",
    ),
    
    # Sentinel-3 Altimetry
    "s3_sral": SentinelProduct(
        id="SENTINEL-3-SRAL",
        name="Sentinel-3 SRAL",
        satellite="Sentinel-3",
        instrument="SRAL",
        processing_level="L2",
        variables=["ssha", "swh", "wind_speed", "sigma0"],
        resolution="along-track 300m",
        description="Radar altimetry: sea surface height, waves",
    ),
}


class CopernicusDataSpaceClient:
    """
    Copernicus Data Space Ecosystem Client.
    
    Replaces old Copernicus Open Access Hub (scihub).
    New unified access to all Copernicus data.
    
    Auth: OAuth2 via https://identity.dataspace.copernicus.eu/
    API: OData + STAC
    
    Usage:
        client = CopernicusDataSpaceClient()
        
        # Search Sentinel-1 products
        products = await client.search(
            collection="SENTINEL-1",
            bbox=(7.0, 44.0, 12.0, 47.0),
            time_range=("2024-01-01", "2024-01-31"),
            product_type="GRD",
        )
        
        # Download product
        await client.download(products[0], output_dir="/data/sentinel")
    """
    
    # Copernicus Data Space endpoints
    IDENTITY_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    DOWNLOAD_URL = "https://zipper.dataspace.copernicus.eu/odata/v1"
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        cache_dir: Path = None,
    ):
        self.username = username or os.getenv("COPERNICUS_USERNAME")
        self.password = password or os.getenv("COPERNICUS_PASSWORD")
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "sentinel"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._token = None
        self._token_expiry = None
    
    async def _get_token(self) -> Optional[str]:
        """Get OAuth2 access token."""
        if not self.username or not self.password:
            logger.warning("Copernicus credentials not set")
            return None
        
        # Check if token still valid
        if self._token and self._token_expiry and datetime.now() < self._token_expiry:
            return self._token
        
        if not HAS_AIOHTTP:
            return None
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "grant_type": "password",
                    "username": self.username,
                    "password": self.password,
                    "client_id": "cdse-public",
                }
                
                async with session.post(self.IDENTITY_URL, data=data) as resp:
                    if resp.status != 200:
                        logger.error(f"Auth failed: {resp.status}")
                        return None
                    
                    result = await resp.json()
                    self._token = result.get("access_token")
                    expires_in = result.get("expires_in", 300)
                    self._token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                    
                    return self._token
                    
        except Exception as e:
            logger.error(f"Token request failed: {e}")
            return None
    
    async def search(
        self,
        collection: str = "SENTINEL-1",
        bbox: Tuple[float, float, float, float] = None,
        time_range: Tuple[str, str] = None,
        product_type: str = None,
        max_results: int = 100,
    ) -> List[Dict]:
        """
        Search for Sentinel products.
        
        Args:
            collection: SENTINEL-1, SENTINEL-2, SENTINEL-3, etc.
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time_range: (start, end) as "YYYY-MM-DD"
            product_type: GRD, SLC, OCN, etc.
            max_results: Maximum products to return
            
        Returns:
            List of product metadata dicts
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required")
            return []
        
        # Build OData filter
        filters = [f"Collection/Name eq '{collection}'"]
        
        if time_range:
            filters.append(f"ContentDate/Start ge {time_range[0]}T00:00:00.000Z")
            filters.append(f"ContentDate/Start le {time_range[1]}T23:59:59.999Z")
        
        if product_type:
            filters.append(f"contains(Name, '{product_type}')")
        
        if bbox:
            # OData geography filter
            polygon = f"POLYGON(({bbox[0]} {bbox[1]},{bbox[2]} {bbox[1]},{bbox[2]} {bbox[3]},{bbox[0]} {bbox[3]},{bbox[0]} {bbox[1]}))"
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")
        
        filter_str = " and ".join(filters)
        
        url = f"{self.CATALOG_URL}/Products?$filter={filter_str}&$top={max_results}&$orderby=ContentDate/Start desc"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        logger.error(f"Search failed: {resp.status}")
                        return []
                    
                    data = await resp.json()
                    products = data.get("value", [])
                    
                    logger.info(f"Found {len(products)} products")
                    return products
                    
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    async def download(
        self,
        product: Dict,
        output_dir: Path = None,
    ) -> Optional[Path]:
        """
        Download a Sentinel product.
        
        Args:
            product: Product metadata from search()
            output_dir: Output directory
            
        Returns:
            Path to downloaded file
        """
        token = await self._get_token()
        if not token:
            logger.error("Cannot download without authentication")
            return None
        
        output_dir = output_dir or self.cache_dir
        product_id = product.get("Id")
        product_name = product.get("Name", "unknown")
        
        url = f"{self.DOWNLOAD_URL}/Products({product_id})/$value"
        output_file = output_dir / f"{product_name}.zip"
        
        if output_file.exists():
            logger.info(f"Already downloaded: {output_file}")
            return output_file
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {token}"}
                
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error(f"Download failed: {resp.status}")
                        return None
                    
                    with open(output_file, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                    
                    logger.info(f"Downloaded: {output_file}")
                    return output_file
                    
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None


class SentinelClient:
    """
    Unified Sentinel data client.
    
    Provides high-level access to Sentinel-1 (SAR) and Sentinel-3 (Ocean).
    
    Usage:
        client = SentinelClient()
        
        # Get SAR-derived wind field
        wind = await client.get_sar_wind(
            bbox=(7.0, 44.0, 12.0, 47.0),
            time_range=("2024-01-01", "2024-01-15"),
        )
        
        # Get ocean color (sediment plume)
        chlor = await client.get_ocean_color(
            bbox=(7.0, 44.0, 12.0, 47.0),
            date="2024-01-10",
        )
    """
    
    def __init__(self):
        self.cdse = CopernicusDataSpaceClient()
    
    def list_products(self) -> Dict[str, str]:
        """List available Sentinel products."""
        return {k: f"{v.name}: {v.description}" for k, v in SENTINEL_PRODUCTS.items()}
    
    async def get_sar_wind(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """
        Get SAR-derived ocean wind from Sentinel-1 OCN products.
        
        Returns:
            xarray.Dataset with wind_speed and wind_direction
        """
        products = await self.cdse.search(
            collection="SENTINEL-1",
            bbox=bbox,
            time_range=time_range,
            product_type="OCN",
        )
        
        if not products:
            logger.warning("No S1 OCN products found, generating synthetic")
            return await self._synthetic_sar_wind(bbox, time_range)
        
        # Would download and process products
        # For now return synthetic
        return await self._synthetic_sar_wind(bbox, time_range)
    
    async def get_ocean_color(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
        variable: str = "chlorophyll",
    ) -> Optional[Any]:
        """
        Get ocean color from Sentinel-3 OLCI.
        
        Args:
            variable: chlorophyll, tsm (sediment), cdom
            
        Returns:
            xarray.Dataset
        """
        products = await self.cdse.search(
            collection="SENTINEL-3",
            bbox=bbox,
            time_range=time_range,
            product_type="OL_2_WFR",
        )
        
        if not products:
            logger.warning("No S3 OLCI products found, generating synthetic")
            return await self._synthetic_ocean_color(bbox, time_range, variable)
        
        return await self._synthetic_ocean_color(bbox, time_range, variable)
    
    async def get_flood_mask(
        self,
        bbox: Tuple[float, float, float, float],
        date: str,
    ) -> Optional[Any]:
        """
        Get flood extent mask from Sentinel-1 SAR.
        
        Uses change detection between pre/post event images.
        Water has low backscatter in SAR.
        
        Returns:
            xarray.DataArray with flood probability (0-1)
        """
        # Would need pre-event reference image and post-event image
        # Then apply thresholding on sigma0 change
        logger.warning("Flood mask from SAR not yet implemented")
        return None
    
    async def _synthetic_sar_wind(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Generate synthetic SAR wind data."""
        if not HAS_XARRAY:
            return None
        
        logger.warning("🔧 Generating synthetic SAR wind data")
        
        # Grid
        lons = np.arange(bbox[0], bbox[2], 0.01)  # ~1km
        lats = np.arange(bbox[1], bbox[3], 0.01)
        
        # Time (satellite passes)
        times = np.arange(
            np.datetime64(time_range[0]),
            np.datetime64(time_range[1]),
            np.timedelta64(6, 'D')  # ~6 day revisit
        )
        
        shape = (len(times), len(lats), len(lons))
        
        # Wind speed (realistic range 0-25 m/s)
        wind_speed = np.abs(np.random.normal(8, 4, shape)).astype(np.float32)
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Wind direction (0-360)
        wind_direction = np.random.uniform(0, 360, shape).astype(np.float32)
        
        ds = xr.Dataset(
            {
                "wind_speed": (["time", "lat", "lon"], wind_speed),
                "wind_direction": (["time", "lat", "lon"], wind_direction),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "source": "synthetic_sentinel1_sar",
                "units_wind_speed": "m/s",
                "units_wind_direction": "degrees",
            }
        )
        
        return ds
    
    async def _synthetic_ocean_color(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
        variable: str,
    ) -> Optional[Any]:
        """Generate synthetic ocean color data."""
        if not HAS_XARRAY:
            return None
        
        logger.warning(f"🔧 Generating synthetic {variable} data")
        
        # Grid (300m resolution)
        lons = np.arange(bbox[0], bbox[2], 0.003)
        lats = np.arange(bbox[1], bbox[3], 0.003)
        
        times = np.arange(
            np.datetime64(time_range[0]),
            np.datetime64(time_range[1]),
            np.timedelta64(1, 'D')
        )
        
        shape = (len(times), len(lats), len(lons))
        
        if variable == "chlorophyll":
            # mg/m³, typical range 0.1-10
            data = np.abs(np.random.lognormal(0, 0.5, shape)).astype(np.float32)
            units = "mg/m³"
        elif variable == "tsm":
            # Total Suspended Matter, g/m³
            data = np.abs(np.random.lognormal(1, 0.8, shape)).astype(np.float32)
            units = "g/m³"
        else:  # cdom
            # CDOM absorption, 1/m
            data = np.abs(np.random.exponential(0.5, shape)).astype(np.float32)
            units = "1/m"
        
        ds = xr.Dataset(
            {
                variable: (["time", "lat", "lon"], data),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "source": "synthetic_sentinel3_olci",
                "units": units,
            }
        )
        
        return ds


# Module interface
Client = SentinelClient


async def get_sentinel_data(
    product_type: str,
    bbox: Tuple[float, float, float, float],
    time_range: Tuple[str, str],
) -> Optional[Any]:
    """Quick Sentinel data access."""
    client = SentinelClient()
    
    if product_type in ["sar_wind", "s1_ocn"]:
        return await client.get_sar_wind(bbox, time_range)
    elif product_type in ["ocean_color", "chlorophyll", "s3_olci"]:
        return await client.get_ocean_color(bbox, time_range)
    else:
        logger.error(f"Unknown product type: {product_type}")
        return None


# CLI test
if __name__ == "__main__":
    async def test():
        print("=== Sentinel Client Test ===\n")
        
        client = SentinelClient()
        
        print("Available products:")
        for key, desc in client.list_products().items():
            print(f"  {key}: {desc}")
        
        bbox = (7.0, 44.0, 12.0, 47.0)
        time_range = ("2024-01-01", "2024-01-15")
        
        print(f"\n1. SAR Wind (bbox={bbox})...")
        wind = await client.get_sar_wind(bbox, time_range)
        if wind:
            print(f"   Got {wind}")
        
        print(f"\n2. Ocean Color...")
        chlor = await client.get_ocean_color(bbox, time_range)
        if chlor:
            print(f"   Got {chlor}")
        
        print("\n✅ Test complete")
    
    asyncio.run(test())
