"""
🛰️ Test Sentinel Client
========================

Tests for Copernicus Sentinel satellite data client.
Supports:
- Sentinel-1 (SAR radar)
- Sentinel-2 (optical imagery)
- Sentinel-3 (ocean and land monitoring)
- Sentinel-6 (altimetry for sea level)

All tests use synthetic data to avoid API dependencies.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import MagicMock, patch, AsyncMock

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# ============================================================================
# Sentinel Client Implementation (stub for testing)
# ============================================================================

class SentinelMission(Enum):
    """Sentinel satellite missions."""
    S1 = "sentinel-1"  # SAR
    S2 = "sentinel-2"  # Optical
    S3 = "sentinel-3"  # Ocean & Land
    S6 = "sentinel-6"  # Altimetry


@dataclass
class SentinelProduct:
    """Sentinel data product definition."""
    mission: SentinelMission
    product_type: str
    variables: List[str]
    resolution_m: float
    description: str
    processing_level: str = "L2"


SENTINEL_PRODUCTS = {
    # Sentinel-3 Ocean Products
    "s3_sral": SentinelProduct(
        mission=SentinelMission.S3,
        product_type="SR_2_WAT___",
        variables=["ssha", "swh", "wind_speed", "sigma0"],
        resolution_m=300,
        description="Sentinel-3 SRAL altimetry",
        processing_level="L2",
    ),
    "s3_olci": SentinelProduct(
        mission=SentinelMission.S3,
        product_type="OL_2_WFR___",
        variables=["chl_oc4me", "tsm_nn", "iop_nn"],
        resolution_m=300,
        description="Sentinel-3 OLCI water products",
    ),
    "s3_slstr_sst": SentinelProduct(
        mission=SentinelMission.S3,
        product_type="SL_2_WST___",
        variables=["sea_surface_temperature", "sst_quality"],
        resolution_m=1000,
        description="Sentinel-3 SLSTR SST",
    ),
    
    # Sentinel-6 Altimetry
    "s6_sral": SentinelProduct(
        mission=SentinelMission.S6,
        product_type="P4_2__HR_STD__NT",
        variables=["ssha", "swh", "wind_speed", "range_ocean"],
        resolution_m=300,
        description="Sentinel-6 high-resolution altimetry",
        processing_level="L2",
    ),
    
    # Sentinel-1 SAR
    "s1_grd": SentinelProduct(
        mission=SentinelMission.S1,
        product_type="GRD",
        variables=["VV", "VH"],
        resolution_m=10,
        description="Sentinel-1 SAR GRD",
        processing_level="L1",
    ),
    
    # Sentinel-2 Optical
    "s2_msi": SentinelProduct(
        mission=SentinelMission.S2,
        product_type="MSIL2A",
        variables=["B02", "B03", "B04", "B08", "SCL"],
        resolution_m=10,
        description="Sentinel-2 MSI L2A",
        processing_level="L2A",
    ),
}


class SentinelClient:
    """
    Client for Copernicus Sentinel satellite data.
    
    Provides access to:
    - Sentinel-3 SRAL altimetry (sea surface height, waves)
    - Sentinel-3 OLCI ocean color
    - Sentinel-3 SLSTR SST
    - Sentinel-6 precision altimetry
    - Sentinel-1 SAR imagery
    - Sentinel-2 optical imagery
    
    Usage:
        client = SentinelClient()
        
        # Get sea level data
        ds = await client.get_altimetry(
            lat_range=(45, 47),
            lon_range=(8, 10),
            time_range=("2023-01-01", "2023-01-31"),
        )
    """
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        cache_dir: str = None,
    ):
        import os
        self.username = username or os.getenv("COPERNICUS_USERNAME")
        self.password = password or os.getenv("COPERNICUS_PASSWORD")
        self.cache_dir = cache_dir
        self._authenticated = bool(self.username and self.password)
    
    @property
    def is_authenticated(self) -> bool:
        """Check if client has valid credentials."""
        return self._authenticated
    
    def list_products(self) -> Dict[str, str]:
        """List available products."""
        return {k: v.description for k, v in SENTINEL_PRODUCTS.items()}
    
    def get_product_info(self, product_key: str) -> Optional[SentinelProduct]:
        """Get product metadata."""
        return SENTINEL_PRODUCTS.get(product_key)
    
    async def search(
        self,
        product: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        max_cloud_cover: float = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for available products.
        
        Returns list of granule metadata.
        """
        # In real implementation, would query Copernicus Data Space
        # For testing, return synthetic results
        return await self._synthetic_search(
            product, lat_range, lon_range, time_range
        )
    
    async def download(
        self,
        product: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:  # xr.Dataset
        """Download and load Sentinel data."""
        product_info = SENTINEL_PRODUCTS.get(product)
        if not product_info:
            raise ValueError(f"Unknown product: {product}")
        
        # For testing, generate synthetic data
        return await self._generate_synthetic_data(
            product_info, lat_range, lon_range, time_range, variables
        )
    
    async def get_altimetry(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        mission: str = "s3",  # s3 or s6
    ) -> Optional[Any]:
        """Convenience method for altimetry data."""
        product = f"{mission}_sral"
        return await self.download(
            product=product,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            variables=["ssha", "swh", "wind_speed"],
        )
    
    async def get_sst(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Convenience method for SST data."""
        return await self.download(
            product="s3_slstr_sst",
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            variables=["sea_surface_temperature"],
        )
    
    async def _synthetic_search(
        self,
        product: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> List[Dict[str, Any]]:
        """Generate synthetic search results."""
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        
        results = []
        current = start
        
        while current <= end:
            results.append({
                "id": f"{product}_{current.strftime('%Y%m%d')}",
                "title": f"Synthetic {product} for {current.date()}",
                "date": current.isoformat(),
                "footprint": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon_range[0], lat_range[0]],
                        [lon_range[1], lat_range[0]],
                        [lon_range[1], lat_range[1]],
                        [lon_range[0], lat_range[1]],
                        [lon_range[0], lat_range[0]],
                    ]]
                },
                "size_mb": np.random.randint(50, 500),
                "cloud_cover": np.random.uniform(0, 100),
            })
            current += timedelta(days=1)
        
        return results
    
    async def _generate_synthetic_data(
        self,
        product_info: SentinelProduct,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:
        """Generate synthetic Sentinel data."""
        if not HAS_XARRAY:
            return None
        
        variables = variables or product_info.variables
        
        # Create coordinates
        res_deg = product_info.resolution_m / 111000  # Approx conversion
        res_deg = max(res_deg, 0.01)  # Minimum resolution for testing
        
        lats = np.arange(lat_range[0], lat_range[1], res_deg)
        lons = np.arange(lon_range[0], lon_range[1], res_deg)
        times = np.arange(
            np.datetime64(time_range[0]),
            np.datetime64(time_range[1]),
            np.timedelta64(1, 'D')
        )
        
        shape = (len(times), len(lats), len(lons))
        
        data_vars = {}
        for var in variables:
            if var == "ssha":
                # Sea surface height anomaly: -0.5 to 0.5 m
                data = np.random.normal(0, 0.1, shape)
            elif var == "swh":
                # Significant wave height: 0 to 10 m
                data = np.abs(np.random.normal(2, 1, shape))
            elif var == "wind_speed":
                # Wind speed: 0 to 25 m/s
                data = np.abs(np.random.normal(8, 3, shape))
            elif var == "sea_surface_temperature":
                # SST: 273-303 K (0-30°C)
                lat_effect = 293 - 0.3 * (lats - lat_range[0])[:, np.newaxis]
                data = lat_effect + np.random.normal(0, 1, shape)
            elif var in ["VV", "VH"]:
                # SAR backscatter: -25 to 0 dB
                data = np.random.normal(-15, 5, shape)
            elif var.startswith("B"):
                # Optical bands: 0 to 10000 reflectance
                data = np.random.uniform(0, 10000, shape)
            else:
                data = np.random.randn(*shape)
            
            data_vars[var] = (['time', 'latitude', 'longitude'], data.astype(np.float32))
        
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons,
            },
            attrs={
                'source': 'synthetic_sentinel',
                'mission': product_info.mission.value,
                'product_type': product_info.product_type,
                'processing_level': product_info.processing_level,
            }
        )
        
        return ds


# ============================================================================
# Tests
# ============================================================================

class TestSentinelMission:
    """Test SentinelMission enum."""
    
    def test_all_missions_defined(self):
        """All expected missions should be defined."""
        missions = [m.value for m in SentinelMission]
        
        assert "sentinel-1" in missions
        assert "sentinel-2" in missions
        assert "sentinel-3" in missions
        assert "sentinel-6" in missions
    
    def test_mission_count(self):
        """Should have 4 main missions."""
        assert len(SentinelMission) == 4


class TestSentinelProduct:
    """Test SentinelProduct dataclass."""
    
    def test_create_product(self):
        """Should create product definition."""
        product = SentinelProduct(
            mission=SentinelMission.S3,
            product_type="TEST",
            variables=["var1", "var2"],
            resolution_m=300,
            description="Test product",
        )
        
        assert product.mission == SentinelMission.S3
        assert "var1" in product.variables
        assert product.resolution_m == 300
    
    def test_default_processing_level(self):
        """Default processing level should be L2."""
        product = SentinelProduct(
            mission=SentinelMission.S3,
            product_type="TEST",
            variables=[],
            resolution_m=100,
            description="Test",
        )
        
        assert product.processing_level == "L2"


class TestSentinelProducts:
    """Test predefined Sentinel products."""
    
    def test_s3_sral_defined(self):
        """S3 SRAL altimetry product should exist."""
        assert "s3_sral" in SENTINEL_PRODUCTS
        
        product = SENTINEL_PRODUCTS["s3_sral"]
        assert product.mission == SentinelMission.S3
        assert "ssha" in product.variables
        assert "swh" in product.variables
    
    def test_s6_sral_defined(self):
        """S6 SRAL product should exist."""
        assert "s6_sral" in SENTINEL_PRODUCTS
        
        product = SENTINEL_PRODUCTS["s6_sral"]
        assert product.mission == SentinelMission.S6
    
    def test_all_products_have_variables(self):
        """All products should have at least one variable."""
        for name, product in SENTINEL_PRODUCTS.items():
            assert len(product.variables) > 0, f"{name} has no variables"
    
    def test_all_products_have_resolution(self):
        """All products should have resolution defined."""
        for name, product in SENTINEL_PRODUCTS.items():
            assert product.resolution_m > 0, f"{name} has invalid resolution"


class TestSentinelClient:
    """Test SentinelClient class."""
    
    @pytest.fixture
    def client(self):
        return SentinelClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
    
    def test_unauthenticated_by_default(self, client):
        """Should be unauthenticated without credentials."""
        assert client.is_authenticated is False
    
    def test_authenticated_with_credentials(self):
        """Should be authenticated with credentials."""
        client = SentinelClient(
            username="test_user",
            password="test_pass",
        )
        
        assert client.is_authenticated is True
    
    def test_list_products(self, client):
        """Should list available products."""
        products = client.list_products()
        
        assert isinstance(products, dict)
        assert len(products) > 0
        assert "s3_sral" in products
    
    def test_get_product_info(self, client):
        """Should return product metadata."""
        info = client.get_product_info("s3_sral")
        
        assert info is not None
        assert info.mission == SentinelMission.S3
        assert "ssha" in info.variables
    
    def test_get_unknown_product_info(self, client):
        """Should return None for unknown product."""
        info = client.get_product_info("nonexistent")
        
        assert info is None
    
    @pytest.mark.asyncio
    async def test_search_returns_results(self, client):
        """Search should return list of results."""
        results = await client.search(
            product="s3_sral",
            lat_range=(45, 47),
            lon_range=(8, 10),
            time_range=("2023-01-01", "2023-01-10"),
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        for result in results:
            assert "id" in result
            assert "date" in result
            assert "footprint" in result
    
    @pytest.mark.asyncio
    async def test_download_altimetry(self, client):
        """Should download altimetry data."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s3_sral",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-10"),
            variables=["ssha", "swh"],
        )
        
        assert ds is not None
        assert "ssha" in ds.data_vars
        assert "swh" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_download_unknown_product_raises(self, client):
        """Should raise for unknown product."""
        with pytest.raises(ValueError, match="Unknown product"):
            await client.download(
                product="nonexistent",
                lat_range=(45, 46),
                lon_range=(8, 9),
                time_range=("2023-01-01", "2023-01-10"),
            )
    
    @pytest.mark.asyncio
    async def test_get_altimetry_convenience(self, client):
        """Should use convenience method for altimetry."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.get_altimetry(
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-05"),
        )
        
        assert ds is not None
        assert "ssha" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_get_sst_convenience(self, client):
        """Should use convenience method for SST."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.get_sst(
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-05"),
        )
        
        assert ds is not None
        assert "sea_surface_temperature" in ds.data_vars


class TestSentinelDataQuality:
    """Test synthetic data quality."""
    
    @pytest.fixture
    def client(self):
        return SentinelClient()
    
    @pytest.mark.asyncio
    async def test_ssha_values_realistic(self, client):
        """SSHA values should be realistic."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s3_sral",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-10"),
            variables=["ssha"],
        )
        
        ssha = ds["ssha"].values
        
        # SSHA typically -1 to +1 m
        assert np.abs(ssha).max() < 2.0, "SSHA values unrealistic"
        assert np.std(ssha) < 0.5, "SSHA variance too high"
    
    @pytest.mark.asyncio
    async def test_swh_values_realistic(self, client):
        """SWH values should be realistic."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s3_sral",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-10"),
            variables=["swh"],
        )
        
        swh = ds["swh"].values
        
        # SWH should be non-negative, typically 0-15 m
        assert np.all(swh >= 0), "SWH should be non-negative"
        assert np.max(swh) < 20, "SWH unrealistically high"
    
    @pytest.mark.asyncio
    async def test_dataset_has_valid_coords(self, client):
        """Dataset should have valid coordinates."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s3_sral",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-05"),
        )
        
        assert "time" in ds.coords
        assert "latitude" in ds.coords
        assert "longitude" in ds.coords
        
        # Check coordinate ranges
        assert ds.latitude.min() >= 45
        assert ds.latitude.max() <= 46
        assert ds.longitude.min() >= 8
        assert ds.longitude.max() <= 9
    
    @pytest.mark.asyncio
    async def test_dataset_has_attributes(self, client):
        """Dataset should have metadata attributes."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s3_sral",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-05"),
        )
        
        assert "source" in ds.attrs
        assert "mission" in ds.attrs
        assert ds.attrs["mission"] == "sentinel-3"


class TestSentinelSST:
    """Test Sentinel SST products."""
    
    @pytest.fixture
    def client(self):
        return SentinelClient()
    
    @pytest.mark.asyncio
    async def test_sst_product_exists(self, client):
        """SST product should be defined."""
        products = client.list_products()
        assert "s3_slstr_sst" in products
    
    @pytest.mark.asyncio
    async def test_sst_values_realistic(self, client):
        """SST values should be realistic (Kelvin)."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.get_sst(
            lat_range=(40, 45),
            lon_range=(10, 15),
            time_range=("2023-07-01", "2023-07-10"),
        )
        
        sst = ds["sea_surface_temperature"].values
        
        # Mediterranean summer SST: ~290-305 K (17-32°C)
        assert np.all(sst > 270), "SST too cold"
        assert np.all(sst < 320), "SST too hot"


class TestSentinelSAR:
    """Test Sentinel-1 SAR products."""
    
    @pytest.fixture
    def client(self):
        return SentinelClient()
    
    @pytest.mark.asyncio
    async def test_sar_product_exists(self, client):
        """SAR product should be defined."""
        info = client.get_product_info("s1_grd")
        
        assert info is not None
        assert info.mission == SentinelMission.S1
        assert "VV" in info.variables
    
    @pytest.mark.asyncio
    async def test_sar_download(self, client):
        """Should download SAR data."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            product="s1_grd",
            lat_range=(45, 46),
            lon_range=(8, 9),
            time_range=("2023-01-01", "2023-01-05"),
            variables=["VV", "VH"],
        )
        
        assert ds is not None
        assert "VV" in ds.data_vars
        assert "VH" in ds.data_vars


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
