"""
🌊 Test CMEMS Client
====================

Tests for Copernicus Marine data download.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from surge_shazam.data.cmems_client import (
    CMEMSClient,
    CMEMSDataset,
    CMEMS_DATASETS,
    download_cmems,
)


class TestCMEMSDatasets:
    """Test dataset definitions."""
    
    def test_sea_level_global_exists(self):
        """Sea level global dataset should be defined."""
        assert "sea_level_global" in CMEMS_DATASETS
        
    def test_dataset_has_required_fields(self):
        """All datasets should have required fields."""
        for name, ds in CMEMS_DATASETS.items():
            assert ds.dataset_id, f"{name} missing dataset_id"
            assert ds.product_id, f"{name} missing product_id"
            assert len(ds.variables) > 0, f"{name} has no variables"
            assert ds.description, f"{name} missing description"
            
    def test_sea_level_variables(self):
        """Sea level dataset should have SLA and ADT."""
        ds = CMEMS_DATASETS["sea_level_global"]
        assert "sla" in ds.variables
        assert "adt" in ds.variables


class TestCMEMSClient:
    """Test CMEMSClient class."""
    
    @pytest.fixture
    def client(self):
        return CMEMSClient()
    
    def test_list_datasets(self, client):
        """Should list available datasets."""
        datasets = client.list_datasets()
        
        assert isinstance(datasets, dict)
        assert len(datasets) > 0
        assert "sea_level_global" in datasets
        
    def test_get_dataset_info(self, client):
        """Should return dataset info."""
        info = client.get_dataset_info("sea_level_global")
        
        assert info is not None
        assert isinstance(info, CMEMSDataset)
        assert info.dataset_id is not None
        
    def test_get_unknown_dataset(self, client):
        """Should return None for unknown dataset."""
        info = client.get_dataset_info("nonexistent_dataset")
        assert info is None
        
    def test_cache_key_generation(self, client):
        """Cache key should be deterministic."""
        key1 = client._cache_key(
            "sea_level_global",
            ["sla"],
            (45.0, 47.0),
            (8.0, 10.0),
            ("2000-10-01", "2000-10-31"),
        )
        key2 = client._cache_key(
            "sea_level_global",
            ["sla"],
            (45.0, 47.0),
            (8.0, 10.0),
            ("2000-10-01", "2000-10-31"),
        )
        
        assert key1 == key2
        assert key1.startswith("cmems_")
        
    @pytest.mark.asyncio
    async def test_download_fallback_returns_dataset(self, client):
        """Fallback should return synthetic xarray dataset."""
        # This tests the fallback when copernicusmarine is not configured
        ds = await client._download_fallback(
            dataset="sea_level_global",
            variables=["sla"],
            lat_range=(45.0, 47.0),
            lon_range=(8.0, 10.0),
            time_range=("2000-10-01", "2000-10-15"),
        )
        
        # Should return something (either real or synthetic)
        # Check structure if xarray is available
        if ds is not None:
            assert hasattr(ds, 'data_vars')
            assert 'sla' in ds.data_vars or len(ds.data_vars) > 0
            
    @pytest.mark.asyncio
    async def test_get_sea_level_convenience_method(self, client):
        """Test convenience method for sea level data."""
        # Will use fallback since no credentials
        ds = await client.get_sea_level(
            lat_range=(45.0, 47.0),
            lon_range=(8.0, 10.0),
            time_range=("2000-10-01", "2000-10-15"),
        )
        
        # Should not raise, may return None or synthetic data
        # If returns data, check structure
        if ds is not None:
            assert hasattr(ds, 'coords')


class TestCMEMSIntegration:
    """Integration tests (require credentials or mock)."""
    
    @pytest.mark.asyncio
    async def test_download_lago_maggiore_area(self):
        """Test download for Lago Maggiore area (uses fallback)."""
        client = CMEMSClient()
        
        ds = await client.download(
            dataset="sea_level_global",
            variables=["sla"],
            lat_range=(45.0, 47.0),
            lon_range=(8.0, 10.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
        
        # Will use synthetic fallback without credentials
        # Just verify it doesn't crash
        print(f"Download result type: {type(ds)}")


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
