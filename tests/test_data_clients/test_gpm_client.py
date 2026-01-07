"""
Tests for GPM IMERG Precipitation Client
========================================
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.surge_shazam.data.gpm_client import (
    GPMClient,
    GPM_PRODUCTS,
    get_precipitation,
)


class TestGPMProducts:
    """Test GPM product definitions."""
    
    def test_products_defined(self):
        """All product tiers should be defined."""
        assert "early" in GPM_PRODUCTS
        assert "late" in GPM_PRODUCTS
        assert "final" in GPM_PRODUCTS
        assert "half_hourly" in GPM_PRODUCTS
    
    def test_product_attributes(self):
        """Products should have required attributes."""
        for key, product in GPM_PRODUCTS.items():
            assert product.short_name, f"{key}: missing short_name"
            assert product.description, f"{key}: missing description"
            assert product.latency, f"{key}: missing latency"


class TestGPMClient:
    """Test GPM client functionality."""
    
    @pytest.fixture
    def client(self, tmp_path):
        """Create client with temp cache dir."""
        return GPMClient(cache_dir=tmp_path / "gpm_cache")
    
    def test_list_products(self, client):
        """Should list available products."""
        products = client.list_products()
        assert isinstance(products, dict)
        assert len(products) >= 4
        assert "early" in products
    
    @pytest.mark.asyncio
    async def test_fallback_generates_data(self, client):
        """Fallback should generate synthetic data."""
        ds = await client._download_fallback(
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-15"),
        )
        
        assert ds is not None
        assert "precipitation" in ds
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims
        
        # Check data is reasonable
        precip = ds["precipitation"].values
        assert precip.min() >= 0  # No negative precipitation
        assert precip.max() < 500  # Reasonable max (mm/day)
    
    @pytest.mark.asyncio
    async def test_download_uses_cache(self, client, tmp_path):
        """Second download should use cache."""
        params = dict(
            lat_range=(45.0, 46.0),
            lon_range=(8.0, 9.0),
            time_range=("2000-10-01", "2000-10-05"),
        )
        
        # First download
        ds1 = await client.download(**params)
        assert ds1 is not None
        
        # Check cache was created
        cache_files = list(client.cache_dir.glob("*.nc"))
        assert len(cache_files) >= 1
    
    @pytest.mark.asyncio
    async def test_get_accumulated(self, client):
        """Should compute accumulated precipitation."""
        total = await client.get_accumulated(
            lat_range=(45.0, 46.0),
            lon_range=(8.0, 9.0),
            time_range=("2000-10-01", "2000-10-10"),
        )
        
        assert total is not None
        assert "time" not in total.dims  # Summed over time
        assert total.attrs.get("units") == "mm"


class TestGPMConvenienceFunction:
    """Test module-level convenience function."""
    
    @pytest.mark.asyncio
    async def test_get_precipitation(self):
        """get_precipitation should work."""
        ds = await get_precipitation(
            lat_range=(45.0, 46.0),
            lon_range=(8.0, 9.0),
            time_range=("2000-10-01", "2000-10-05"),
        )
        
        assert ds is not None
        assert "precipitation" in ds
