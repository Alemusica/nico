"""
🌍 Test Geo Resolver
====================

Tests for geographic resolution from natural language.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agent.tools.geo_resolver import GeoResolver, GeoLocation, KNOWN_LOCATIONS


class TestKnownLocations:
    """Test pre-defined known locations."""
    
    def test_lago_maggiore_in_known_locations(self):
        """Lago Maggiore should be in known locations."""
        assert "lago maggiore" in KNOWN_LOCATIONS
        
    def test_lago_maggiore_coordinates(self):
        """Verify Lago Maggiore coordinates are correct."""
        loc = KNOWN_LOCATIONS["lago maggiore"]
        assert loc.lat == pytest.approx(45.95, abs=0.1)
        assert loc.lon == pytest.approx(8.65, abs=0.1)
        
    def test_lago_maggiore_has_bbox(self):
        """Lago Maggiore should have bounding box."""
        loc = KNOWN_LOCATIONS["lago maggiore"]
        assert loc.bbox is not None
        assert len(loc.bbox) == 4
        # bbox format: (lat_min, lat_max, lon_min, lon_max)
        assert loc.bbox[0] < loc.bbox[1]  # lat_min < lat_max
        assert loc.bbox[2] < loc.bbox[3]  # lon_min < lon_max
        
    def test_all_known_locations_have_required_fields(self):
        """All known locations should have required fields."""
        for name, loc in KNOWN_LOCATIONS.items():
            assert loc.name, f"{name} missing name"
            assert -90 <= loc.lat <= 90, f"{name} invalid latitude"
            assert -180 <= loc.lon <= 180, f"{name} invalid longitude"


class TestGeoResolver:
    """Test GeoResolver class."""
    
    @pytest.fixture
    def resolver(self):
        return GeoResolver()
    
    @pytest.mark.asyncio
    async def test_resolve_known_location(self, resolver):
        """Should resolve known locations from cache."""
        location = await resolver.resolve("Lago Maggiore")
        
        assert location is not None
        assert location.name == "Lago Maggiore"
        assert location.lat == pytest.approx(45.95, abs=0.1)
        
    @pytest.mark.asyncio
    async def test_resolve_case_insensitive(self, resolver):
        """Resolution should be case insensitive."""
        loc1 = await resolver.resolve("lago maggiore")
        loc2 = await resolver.resolve("LAGO MAGGIORE")
        loc3 = await resolver.resolve("Lago Maggiore")
        
        assert loc1 is not None
        assert loc2 is not None
        assert loc3 is not None
        assert loc1.lat == loc2.lat == loc3.lat
        
    @pytest.mark.asyncio
    async def test_resolve_with_variations(self, resolver):
        """Should handle location name variations."""
        # Italian vs English
        loc_it = await resolver.resolve("Lago Maggiore")
        loc_en = await resolver.resolve("Lake Maggiore")
        
        # Both should resolve to same location
        assert loc_it is not None
        assert loc_en is not None
        
    @pytest.mark.asyncio
    async def test_resolve_from_text(self, resolver):
        """Should extract location from natural language text."""
        text = "analizza le alluvioni del Lago Maggiore nel 2000"
        locations = await resolver.resolve_from_text(text)
        
        assert locations is not None
        assert len(locations) > 0
        # First result should contain Maggiore
        assert any("maggiore" in loc.name.lower() for loc in locations)
        
    def test_expand_bbox(self, resolver):
        """Test bounding box expansion."""
        from agent.tools.geo_resolver import GeoLocation
        
        loc = GeoLocation(
            name="Test",
            lat=45.5,
            lon=8.5,
            bbox=(45.0, 46.0, 8.0, 9.0),
        )
        
        expanded = resolver.expand_bbox(loc, factor=2.0)
        
        # Expanded bbox should be larger
        # Format: (lat_min, lat_max, lon_min, lon_max)
        assert expanded[0] < loc.bbox[0]  # lat_min smaller
        assert expanded[1] > loc.bbox[1]  # lat_max larger
        assert expanded[2] < loc.bbox[2]  # lon_min smaller
        assert expanded[3] > loc.bbox[3]  # lon_max larger
        
    def test_get_temporal_context_flood(self, resolver):
        """Test temporal context for flood events."""
        from agent.tools.geo_resolver import GeoLocation
        from datetime import datetime
        
        loc = GeoLocation(name="Test", lat=45.5, lon=8.5)
        event_date = datetime(2000, 10, 15)
        
        context = resolver.get_temporal_context(
            location=loc,
            event_date=event_date,
            days_before=90,
            days_after=30,
        )
        
        assert "event_date" in context
        assert "precursor_window" in context
        assert "event_window" in context
        # Check precursor window has lookback
        assert context["precursor_window"]["days"] >= 30


class TestGeoLocation:
    """Test GeoLocation dataclass."""
    
    def test_create_geolocation(self):
        """Should create GeoLocation with required fields."""
        loc = GeoLocation(
            name="Test Location",
            lat=45.0,
            lon=8.0,
        )
        assert loc.name == "Test Location"
        assert loc.lat == 45.0
        assert loc.lon == 8.0
        
    def test_geolocation_with_bbox(self):
        """Should create GeoLocation with bbox."""
        loc = GeoLocation(
            name="Test",
            lat=45.0,
            lon=8.0,
            bbox=(7.0, 44.0, 9.0, 46.0),
        )
        assert loc.bbox is not None
        assert len(loc.bbox) == 4


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
