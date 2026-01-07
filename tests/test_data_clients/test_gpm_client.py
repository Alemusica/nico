"""
🌧️ Test GPM Client
===================

Tests for NASA Global Precipitation Measurement (GPM) data client.
GPM provides:
- IMERG (Integrated Multi-satellitE Retrievals for GPM)
- Near-real-time precipitation (30 min latency)
- Research-quality precipitation (3-4 month latency)

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
# GPM Client Implementation (stub for testing)
# ============================================================================

class GPMProduct(Enum):
    """GPM data products."""
    IMERG_EARLY = "imerg_early"  # ~4 hour latency
    IMERG_LATE = "imerg_late"  # ~14 hour latency
    IMERG_FINAL = "imerg_final"  # ~3.5 month latency (research quality)


class GPMResolution(Enum):
    """Temporal resolution options."""
    HALF_HOURLY = "30min"  # 30-minute data
    DAILY = "daily"  # Daily accumulated


@dataclass
class GPMDataset:
    """GPM dataset definition."""
    product: GPMProduct
    resolution: GPMResolution
    spatial_res_deg: float
    variables: List[str]
    description: str
    latency_hours: float
    time_start: str


GPM_DATASETS = {
    "imerg_early_30min": GPMDataset(
        product=GPMProduct.IMERG_EARLY,
        resolution=GPMResolution.HALF_HOURLY,
        spatial_res_deg=0.1,
        variables=["precipitation", "precipitationQualityIndex", "randomError"],
        description="IMERG Early Run (Near Real-Time, 30-min)",
        latency_hours=4,
        time_start="2000-06-01",
    ),
    "imerg_late_30min": GPMDataset(
        product=GPMProduct.IMERG_LATE,
        resolution=GPMResolution.HALF_HOURLY,
        spatial_res_deg=0.1,
        variables=["precipitation", "precipitationQualityIndex", "randomError"],
        description="IMERG Late Run (Near Real-Time, 30-min)",
        latency_hours=14,
        time_start="2000-06-01",
    ),
    "imerg_final_30min": GPMDataset(
        product=GPMProduct.IMERG_FINAL,
        resolution=GPMResolution.HALF_HOURLY,
        spatial_res_deg=0.1,
        variables=["precipitation", "precipitationQualityIndex", "randomError", "gaugeRelativeWeighting"],
        description="IMERG Final Run (Research Quality, 30-min)",
        latency_hours=3.5 * 30 * 24,  # ~3.5 months
        time_start="2000-06-01",
    ),
    "imerg_final_daily": GPMDataset(
        product=GPMProduct.IMERG_FINAL,
        resolution=GPMResolution.DAILY,
        spatial_res_deg=0.1,
        variables=["precipitation", "precipitationQualityIndex"],
        description="IMERG Final Run (Research Quality, Daily)",
        latency_hours=3.5 * 30 * 24,
        time_start="2000-06-01",
    ),
}


class GPMClient:
    """
    Client for NASA GPM (Global Precipitation Measurement) data.
    
    Provides access to IMERG precipitation products:
    - Early Run: ~4 hour latency, near real-time
    - Late Run: ~14 hour latency, improved accuracy
    - Final Run: ~3.5 month latency, research quality
    
    Usage:
        client = GPMClient()
        
        # Get precipitation data
        ds = await client.get_precipitation(
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
            product="imerg_final",
            resolution="daily",
        )
    """
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        cache_dir: str = None,
    ):
        import os
        self.username = username or os.getenv("EARTHDATA_USERNAME")
        self.password = password or os.getenv("EARTHDATA_PASSWORD")
        self.cache_dir = cache_dir
        self._authenticated = bool(self.username and self.password)
    
    @property
    def is_authenticated(self) -> bool:
        """Check if NASA Earthdata credentials are configured."""
        return self._authenticated
    
    def list_datasets(self) -> Dict[str, str]:
        """List available GPM datasets."""
        return {k: v.description for k, v in GPM_DATASETS.items()}
    
    def get_dataset_info(self, dataset: str) -> Optional[GPMDataset]:
        """Get dataset metadata."""
        return GPM_DATASETS.get(dataset)
    
    async def download(
        self,
        dataset: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:  # xr.Dataset
        """
        Download GPM data.
        
        Args:
            dataset: Dataset key (e.g., "imerg_final_daily")
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude
            time_range: (start, end) as "YYYY-MM-DD"
            variables: Variables to include
            
        Returns:
            xarray Dataset with precipitation data
        """
        ds_info = GPM_DATASETS.get(dataset)
        if not ds_info:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return await self._generate_synthetic_data(
            ds_info, lat_range, lon_range, time_range, variables
        )
    
    async def get_precipitation(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        product: str = "imerg_final",  # imerg_early, imerg_late, imerg_final
        resolution: str = "daily",  # 30min, daily
    ) -> Optional[Any]:
        """
        Convenience method for precipitation data.
        
        Args:
            lat_range: Latitude range
            lon_range: Longitude range  
            time_range: Time range
            product: IMERG product type
            resolution: Temporal resolution
            
        Returns:
            xarray Dataset
        """
        dataset_key = f"{product}_{resolution}"
        if dataset_key not in GPM_DATASETS:
            # Try 30min variant
            dataset_key = f"{product}_30min"
        
        return await self.download(
            dataset=dataset_key,
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            variables=["precipitation"],
        )
    
    async def get_accumulation(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Dict[str, float]:
        """
        Compute total precipitation accumulation.
        
        Returns:
            Dict with accumulation statistics
        """
        ds = await self.get_precipitation(
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
            resolution="daily",
        )
        
        if ds is None:
            return {"total_mm": 0, "max_daily_mm": 0, "rainy_days": 0}
        
        precip = ds["precipitation"].values
        
        # Compute stats
        total = float(np.nansum(precip.mean(axis=(1, 2))))  # Spatial mean, sum over time
        max_daily = float(np.nanmax(precip.mean(axis=(1, 2))))
        rainy_days = int(np.sum(precip.mean(axis=(1, 2)) > 1))  # Days with >1mm
        
        return {
            "total_mm": total,
            "max_daily_mm": max_daily,
            "rainy_days": rainy_days,
        }
    
    async def _generate_synthetic_data(
        self,
        ds_info: GPMDataset,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
    ) -> Optional[Any]:
        """Generate synthetic GPM data."""
        if not HAS_XARRAY:
            return None
        
        variables = variables or ds_info.variables
        
        # Create coordinates
        res = ds_info.spatial_res_deg
        lats = np.arange(lat_range[0], lat_range[1], res)
        lons = np.arange(lon_range[0], lon_range[1], res)
        
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        
        if ds_info.resolution == GPMResolution.DAILY:
            times = np.arange(
                np.datetime64(time_range[0]),
                np.datetime64(time_range[1]),
                np.timedelta64(1, 'D')
            )
        else:
            # 30-minute data
            times = np.arange(
                np.datetime64(time_range[0]),
                np.datetime64(time_range[1]),
                np.timedelta64(30, 'm')
            )
        
        shape = (len(times), len(lats), len(lons))
        
        data_vars = {}
        
        for var in variables:
            if var == "precipitation":
                # Precipitation rate mm/hr (or mm/day for daily)
                # Exponential distribution with some storms
                base = np.random.exponential(0.5, shape)
                
                # Add storm events (10% of time steps)
                storm_mask = np.random.random(shape) > 0.9
                base[storm_mask] = np.random.exponential(5, storm_mask.sum())
                
                # Scale for daily vs 30-min
                if ds_info.resolution == GPMResolution.DAILY:
                    data = base * 24  # mm/day
                else:
                    data = base  # mm/hr
                
                # Cap at realistic maximum (300mm/day is extreme)
                data = np.clip(data, 0, 300)
                
            elif var == "precipitationQualityIndex":
                # Quality index 0-1
                data = np.random.uniform(0.7, 1.0, shape)
                
            elif var == "randomError":
                # Random error estimate
                data = np.random.uniform(0.1, 2.0, shape)
                
            elif var == "gaugeRelativeWeighting":
                # Gauge contribution 0-1
                data = np.random.uniform(0.2, 0.8, shape)
                
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
                'source': 'synthetic_gpm',
                'product': ds_info.product.value,
                'resolution': ds_info.resolution.value,
                'spatial_resolution': f"{ds_info.spatial_res_deg}°",
            }
        )
        
        return ds


# ============================================================================
# Tests
# ============================================================================

class TestGPMProduct:
    """Test GPMProduct enum."""
    
    def test_products_defined(self):
        """All GPM products should be defined."""
        products = [p.value for p in GPMProduct]
        
        assert "imerg_early" in products
        assert "imerg_late" in products
        assert "imerg_final" in products


class TestGPMResolution:
    """Test GPMResolution enum."""
    
    def test_resolutions_defined(self):
        """All resolutions should be defined."""
        resolutions = [r.value for r in GPMResolution]
        
        assert "30min" in resolutions
        assert "daily" in resolutions


class TestGPMDatasets:
    """Test predefined GPM datasets."""
    
    def test_final_daily_defined(self):
        """IMERG Final Daily should be defined."""
        assert "imerg_final_daily" in GPM_DATASETS
        
        ds = GPM_DATASETS["imerg_final_daily"]
        assert ds.product == GPMProduct.IMERG_FINAL
        assert ds.resolution == GPMResolution.DAILY
    
    def test_all_have_precipitation(self):
        """All datasets should have precipitation variable."""
        for name, ds in GPM_DATASETS.items():
            assert "precipitation" in ds.variables, f"{name} missing precipitation"
    
    def test_spatial_resolution(self):
        """All IMERG products should be 0.1°."""
        for name, ds in GPM_DATASETS.items():
            assert ds.spatial_res_deg == 0.1


class TestGPMClient:
    """Test GPMClient class."""
    
    @pytest.fixture
    def client(self):
        return GPMClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
    
    def test_unauthenticated_by_default(self, client):
        """Should be unauthenticated without credentials."""
        assert client.is_authenticated is False
    
    def test_list_datasets(self, client):
        """Should list available datasets."""
        datasets = client.list_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        assert "imerg_final_daily" in datasets
    
    def test_get_dataset_info(self, client):
        """Should return dataset info."""
        info = client.get_dataset_info("imerg_final_daily")
        
        assert info is not None
        assert info.product == GPMProduct.IMERG_FINAL
    
    @pytest.mark.asyncio
    async def test_download_returns_dataset(self, client):
        """Download should return xarray Dataset."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            dataset="imerg_final_daily",
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        assert ds is not None
        assert "precipitation" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_download_unknown_raises(self, client):
        """Should raise for unknown dataset."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            await client.download(
                dataset="nonexistent",
                lat_range=(44, 48),
                lon_range=(6, 12),
                time_range=("2023-10-01", "2023-10-10"),
            )
    
    @pytest.mark.asyncio
    async def test_get_precipitation(self, client):
        """Should use convenience method."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.get_precipitation(
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
            product="imerg_final",
            resolution="daily",
        )
        
        assert ds is not None
        assert "precipitation" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_get_accumulation(self, client):
        """Should compute accumulation stats."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        stats = await client.get_accumulation(
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        assert "total_mm" in stats
        assert "max_daily_mm" in stats
        assert "rainy_days" in stats
        assert stats["total_mm"] >= 0


class TestGPMDataQuality:
    """Test synthetic GPM data quality."""
    
    @pytest.fixture
    def client(self):
        return GPMClient()
    
    @pytest.mark.asyncio
    async def test_precipitation_non_negative(self, client):
        """Precipitation should be non-negative."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            dataset="imerg_final_daily",
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        precip = ds["precipitation"].values
        assert np.all(precip >= 0), "Precipitation should be non-negative"
    
    @pytest.mark.asyncio
    async def test_precipitation_realistic_range(self, client):
        """Daily precipitation should be realistic."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            dataset="imerg_final_daily",
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        precip = ds["precipitation"].values
        
        # Daily precip rarely exceeds 300 mm
        assert np.max(precip) < 500, "Daily precipitation unrealistically high"
    
    @pytest.mark.asyncio
    async def test_quality_index_range(self, client):
        """Quality index should be 0-1."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        ds = await client.download(
            dataset="imerg_final_daily",
            lat_range=(44, 48),
            lon_range=(6, 12),
            time_range=("2023-10-01", "2023-10-10"),
            variables=["precipitationQualityIndex"],
        )
        
        qi = ds["precipitationQualityIndex"].values
        assert np.all(qi >= 0) and np.all(qi <= 1), "Quality index out of range"


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
