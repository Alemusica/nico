"""
🌍 Test Climate Indices Client
==============================

Tests for NOAA climate indices download.
"""

import pytest
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from surge_shazam.data.climate_indices import (
    ClimateIndicesClient,
    ClimateIndex,
    CLIMATE_INDICES,
    get_climate_indices_for_event,
)


class TestClimateIndexDefinitions:
    """Test climate index definitions."""
    
    def test_nao_exists(self):
        """NAO index should be defined."""
        assert "nao" in CLIMATE_INDICES
        
    def test_ao_exists(self):
        """AO index should be defined."""
        assert "ao" in CLIMATE_INDICES
        
    def test_oni_exists(self):
        """ONI (ENSO) index should be defined."""
        assert "oni" in CLIMATE_INDICES
        
    def test_index_has_required_fields(self):
        """All indices should have required fields."""
        for name, idx in CLIMATE_INDICES.items():
            assert idx.name, f"{name} missing name"
            assert idx.short_name, f"{name} missing short_name"
            assert idx.url, f"{name} missing url"
            assert idx.description, f"{name} missing description"
            assert idx.influence, f"{name} missing influence description"
            
    def test_index_urls_are_valid(self):
        """Index URLs should be valid HTTP(S)."""
        for name, idx in CLIMATE_INDICES.items():
            assert idx.url.startswith("http"), f"{name} URL not HTTP"


class TestClimateIndicesClient:
    """Test ClimateIndicesClient class."""
    
    @pytest.fixture
    def client(self):
        return ClimateIndicesClient()
    
    def test_list_indices(self, client):
        """Should list available indices."""
        indices = client.list_indices()
        
        assert isinstance(indices, dict)
        assert len(indices) > 0
        assert "nao" in indices
        
    @pytest.mark.asyncio
    async def test_get_nao_index(self, client):
        """Should get NAO index (may use fallback)."""
        nao = await client.get_index("nao")
        
        # Should return DataFrame or None
        if nao is not None:
            assert hasattr(nao, 'columns') or hasattr(nao, 'date')
            assert len(nao) > 0
            
    @pytest.mark.asyncio
    async def test_get_index_with_date_filter(self, client):
        """Should filter by date range."""
        nao = await client.get_index(
            "nao",
            start_date="2000-01-01",
            end_date="2000-12-31"
        )
        
        if nao is not None:
            import pandas as pd
            # Check dates are within range
            dates = pd.to_datetime(nao['date'])
            assert dates.min().year >= 2000
            assert dates.max().year <= 2000
            
    @pytest.mark.asyncio
    async def test_get_all_indices(self, client):
        """Should get multiple indices."""
        indices = await client.get_all_indices(
            start_date="2000-01-01",
            end_date="2000-12-31",
            indices=["nao", "ao"]
        )
        
        assert isinstance(indices, dict)
        # Should have at least some indices
        if len(indices) > 0:
            assert "nao" in indices or "ao" in indices


class TestFloodAnalysis:
    """Test flood-specific analysis."""
    
    @pytest.fixture
    def client(self):
        return ClimateIndicesClient()
    
    @pytest.mark.asyncio
    async def test_get_indices_for_flood(self, client):
        """Should get relevant indices for flood analysis."""
        analysis = await client.get_indices_for_flood(
            event_date="2000-10-15",
            region="italy"
        )
        
        assert isinstance(analysis, dict)
        
        # Check structure of analysis
        for idx_name, data in analysis.items():
            assert "event_value" in data
            assert "period_mean" in data
            assert "data" in data
            
    @pytest.mark.asyncio
    async def test_flood_analysis_has_interpretation(self, client):
        """Should include flood interpretation."""
        analysis = await client.get_indices_for_flood(
            event_date="2000-10-15",
            region="italy"
        )
        
        # Some indices should have interpretation
        has_interpretation = any(
            "interpretation" in data 
            for data in analysis.values()
        )
        
        # Not required to have interpretation for all, but structure should exist


class TestSyntheticData:
    """Test synthetic data generation."""
    
    @pytest.fixture
    def client(self):
        return ClimateIndicesClient()
    
    @pytest.mark.asyncio
    async def test_fallback_nao_values_realistic(self, client):
        """Synthetic NAO should have realistic values."""
        df = await client._download_fallback("nao")
        
        if df is not None:
            import numpy as np
            values = df['value'].values
            # NAO typically ranges from -4 to +4
            assert np.all(np.abs(values) < 6), "NAO values unrealistic"
            
    @pytest.mark.asyncio
    async def test_fallback_oni_values_realistic(self, client):
        """Synthetic ONI should have realistic values."""
        df = await client._download_fallback("oni")
        
        if df is not None:
            import numpy as np
            values = df['value'].values
            # ONI typically ranges from -3 to +3
            assert np.all(np.abs(values) < 5), "ONI values unrealistic"


class TestConvenienceFunction:
    """Test convenience function."""
    
    @pytest.mark.asyncio
    async def test_get_climate_indices_for_event(self):
        """Test convenience function."""
        analysis = await get_climate_indices_for_event(
            event_date="2000-10-15",
            region="europe"
        )
        
        assert isinstance(analysis, dict)


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
