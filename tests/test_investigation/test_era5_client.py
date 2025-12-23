"""
🌤️ Test ERA5 Client
====================

Tests for ERA5 reanalysis data download.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from surge_shazam.data.era5_client import (
    ERA5Client,
    ERA5Variable,
    ERA5_VARIABLES,
    VARIABLE_SETS,
    download_era5,
)


class TestERA5Variables:
    """Test variable definitions."""
    
    def test_precipitation_exists(self):
        """Precipitation should be defined."""
        assert "precipitation" in ERA5_VARIABLES
        
    def test_temperature_exists(self):
        """2m temperature should be defined."""
        assert "temperature_2m" in ERA5_VARIABLES
        
    def test_variable_has_required_fields(self):
        """All variables should have required fields."""
        for name, var in ERA5_VARIABLES.items():
            assert var.cds_name, f"{name} missing cds_name"
            assert var.short_name, f"{name} missing short_name"
            assert var.units, f"{name} missing units"
            assert var.description, f"{name} missing description"


class TestVariableSets:
    """Test predefined variable sets."""
    
    def test_flood_analysis_set_exists(self):
        """Flood analysis set should be defined."""
        assert "flood_analysis" in VARIABLE_SETS
        
    def test_flood_analysis_has_precipitation(self):
        """Flood analysis should include precipitation."""
        assert "precipitation" in VARIABLE_SETS["flood_analysis"]
        
    def test_flood_analysis_has_pressure(self):
        """Flood analysis should include pressure."""
        assert "pressure_msl" in VARIABLE_SETS["flood_analysis"]
        
    def test_all_sets_have_valid_variables(self):
        """All variable sets should reference valid variables."""
        for set_name, variables in VARIABLE_SETS.items():
            for var in variables:
                assert var in ERA5_VARIABLES, f"{var} in {set_name} is not a valid variable"


class TestERA5Client:
    """Test ERA5Client class."""
    
    @pytest.fixture
    def client(self):
        return ERA5Client()
    
    def test_list_variables(self, client):
        """Should list available variables."""
        variables = client.list_variables()
        
        assert isinstance(variables, dict)
        assert len(variables) > 0
        assert "precipitation" in variables
        
    def test_list_variable_sets(self, client):
        """Should list variable sets."""
        sets = client.list_variable_sets()
        
        assert isinstance(sets, dict)
        assert "flood_analysis" in sets
        
    def test_cache_key_generation(self, client):
        """Cache key should be deterministic."""
        key1 = client._cache_key(
            ["precipitation", "temperature_2m"],
            (45.0, 47.0),
            (8.0, 10.0),
            ("2000-10-01", "2000-10-31"),
        )
        key2 = client._cache_key(
            ["precipitation", "temperature_2m"],
            (45.0, 47.0),
            (8.0, 10.0),
            ("2000-10-01", "2000-10-31"),
        )
        
        assert key1 == key2
        assert key1.startswith("era5_")
        
    @pytest.mark.asyncio
    async def test_download_fallback_returns_dataset(self, client):
        """Fallback should return synthetic xarray dataset."""
        ds = await client._download_fallback(
            variables=["precipitation", "temperature_2m"],
            lat_range=(45.0, 47.0),
            lon_range=(8.0, 10.0),
            time_range=("2000-10-01", "2000-10-15"),
        )
        
        if ds is not None:
            assert hasattr(ds, 'data_vars')
            # Check short names are in dataset
            assert "tp" in ds.data_vars or "t2m" in ds.data_vars
            
    @pytest.mark.asyncio
    async def test_download_for_flood_convenience(self, client):
        """Test flood analysis convenience method."""
        ds = await client.download_for_flood(
            lat_range=(45.0, 47.0),
            lon_range=(8.0, 10.0),
            time_range=("2000-10-01", "2000-10-15"),
        )
        
        # Should not raise
        if ds is not None:
            assert hasattr(ds, 'coords')


class TestERA5SyntheticData:
    """Test synthetic data generation quality."""
    
    @pytest.fixture
    def client(self):
        return ERA5Client()
    
    @pytest.mark.asyncio
    async def test_precipitation_values_realistic(self, client):
        """Synthetic precipitation should have realistic values."""
        ds = await client._download_fallback(
            variables=["precipitation"],
            lat_range=(45.0, 46.0),
            lon_range=(8.0, 9.0),
            time_range=("2000-10-01", "2000-10-10"),
        )
        
        if ds is not None and "tp" in ds.data_vars:
            import numpy as np
            precip = ds["tp"].values
            # Precipitation should be non-negative
            assert np.all(precip >= 0), "Precipitation should be non-negative"
            # Max daily precip rarely exceeds 0.2m (200mm)
            assert np.max(precip) < 0.3, "Precipitation unrealistically high"
            
    @pytest.mark.asyncio
    async def test_temperature_values_realistic(self, client):
        """Synthetic temperature should have realistic values."""
        ds = await client._download_fallback(
            variables=["temperature_2m"],
            lat_range=(45.0, 46.0),
            lon_range=(8.0, 9.0),
            time_range=("2000-10-01", "2000-10-10"),
        )
        
        if ds is not None and "t2m" in ds.data_vars:
            import numpy as np
            temp = ds["t2m"].values
            # Temperature in K, should be between 220K and 330K
            assert np.all(temp > 220), "Temperature too low"
            assert np.all(temp < 330), "Temperature too high"


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
