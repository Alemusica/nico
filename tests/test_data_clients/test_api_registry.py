"""
Tests for API Registry
======================

Verifies:
- All data sources are properly defined
- Registry queries work correctly
- Physics variable mapping is complete
"""

import pytest
from src.surge_shazam.data.api_registry import (
    API_REGISTRY,
    DataSource,
    DataCategory,
    Latency,
    Status,
    get_sources_by_category,
    get_sources_by_status,
    get_sources_by_latency,
    get_high_priority_sources,
    get_physics_variables_map,
)


class TestAPIRegistry:
    """Test API Registry structure and queries."""
    
    def test_registry_not_empty(self):
        """Registry should have data sources."""
        assert len(API_REGISTRY) > 0
        assert len(API_REGISTRY) >= 15  # We defined 18 sources
    
    def test_all_sources_have_required_fields(self):
        """Each source must have required fields."""
        for source_id, source in API_REGISTRY.items():
            assert source.id == source_id, f"{source_id}: id mismatch"
            assert source.name, f"{source_id}: missing name"
            assert isinstance(source.category, DataCategory), f"{source_id}: invalid category"
            assert source.provider, f"{source_id}: missing provider"
            assert isinstance(source.latency, Latency), f"{source_id}: invalid latency"
            assert isinstance(source.status, Status), f"{source_id}: invalid status"
            assert isinstance(source.variables, list), f"{source_id}: variables must be list"
    
    def test_sources_have_variables(self):
        """Each source should define at least one variable."""
        for source_id, source in API_REGISTRY.items():
            assert len(source.variables) > 0, f"{source_id}: no variables defined"
    
    def test_high_priority_sources_exist(self):
        """Should have HIGH priority sources."""
        high_priority = get_high_priority_sources()
        assert len(high_priority) >= 5, "Should have at least 5 HIGH priority sources"
        
        # Check expected high priority sources
        expected = ["cmems_sealevel", "era5_surface", "cygnss", "gpm_imerg", "amdar"]
        for expected_id in expected:
            if expected_id in API_REGISTRY:
                assert API_REGISTRY[expected_id].priority == "HIGH"


class TestCategoryQueries:
    """Test category-based queries."""
    
    def test_get_satellite_sources(self):
        """Should return satellite sources."""
        satellites = get_sources_by_category(DataCategory.SATELLITE)
        assert len(satellites) >= 5
        
        # Known satellite sources
        assert "cmems_sealevel" in satellites
        assert "cygnss" in satellites
    
    def test_get_aircraft_sources(self):
        """Should return aircraft sources."""
        aircraft = get_sources_by_category(DataCategory.AIRCRAFT)
        assert len(aircraft) >= 2
        assert "amdar" in aircraft
        assert "mode_s_ehs" in aircraft
    
    def test_get_in_situ_sources(self):
        """Should return in-situ sources."""
        in_situ = get_sources_by_category(DataCategory.IN_SITU)
        assert len(in_situ) >= 2
        assert "tide_gauges" in in_situ
    
    def test_all_categories_covered(self):
        """All categories should have at least one source."""
        for category in DataCategory:
            sources = get_sources_by_category(category)
            assert len(sources) >= 1, f"Category {category.value} has no sources"


class TestStatusQueries:
    """Test status-based queries."""
    
    def test_get_available_sources(self):
        """Should return available sources."""
        available = get_sources_by_status(Status.AVAILABLE)
        assert len(available) >= 3
        
        # These should be available
        assert "cmems_sealevel" in available
        assert "era5_surface" in available
    
    def test_get_todo_sources(self):
        """Should return TODO sources."""
        todo = get_sources_by_status(Status.TODO)
        # We have several TODO sources
        assert len(todo) >= 5


class TestLatencyQueries:
    """Test latency-based queries."""
    
    def test_get_realtime_sources(self):
        """Should return real-time/near-real-time sources."""
        fast = get_sources_by_latency(Latency.NEAR_RT)
        
        # Real-time sources
        realtime_ids = [s.id for s in fast.values() if s.latency == Latency.REALTIME]
        assert len(realtime_ids) >= 2  # AMDAR, Mode-S, tide gauges
    
    def test_latency_hours_correct(self):
        """Latency hours should be correctly set."""
        assert Latency.REALTIME.hours == 1
        assert Latency.NEAR_RT.hours == 6
        assert Latency.DELAYED.hours == 24
        assert Latency.ARCHIVE.hours == 168  # 7 days


class TestPhysicsVariables:
    """Test physics variable mapping."""
    
    def test_physics_map_not_empty(self):
        """Physics map should have entries."""
        physics_map = get_physics_variables_map()
        assert len(physics_map) >= 10
    
    def test_critical_physics_variables(self):
        """Critical SWE variables should be mapped."""
        physics_map = get_physics_variables_map()
        
        critical_vars = ["η", "τ_wind", "P_atm", "precip", "U_wind"]
        for var in critical_vars:
            assert var in physics_map, f"Missing critical variable: {var}"
            assert len(physics_map[var]) >= 1, f"No sources for {var}"
    
    def test_sea_level_has_multiple_sources(self):
        """Sea level should be available from multiple sources."""
        physics_map = get_physics_variables_map()
        
        # η from satellite, η_obs from tide gauges
        assert "η" in physics_map
        assert "η_obs" in physics_map
    
    def test_wind_from_multiple_sources(self):
        """Wind should be from multiple sources."""
        physics_map = get_physics_variables_map()
        
        assert "U_wind" in physics_map
        wind_sources = physics_map["U_wind"]
        assert len(wind_sources) >= 2  # CYGNSS + aircraft


class TestDataSourceMethods:
    """Test DataSource methods."""
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        source = API_REGISTRY["cmems_sealevel"]
        d = source.to_dict()
        
        assert isinstance(d, dict)
        assert d["id"] == "cmems_sealevel"
        assert "latency" in d
        assert "variables" in d
    
    def test_latency_badge(self):
        """Latency should have emoji badge."""
        assert Latency.REALTIME.badge == "🟢"
        assert Latency.NEAR_RT.badge == "🟡"
        assert Latency.DELAYED.badge == "🟠"
        assert Latency.ARCHIVE.badge == "🔴"
        assert Latency.HISTORICAL.badge == "⚫"


class TestAuthConfig:
    """Test authentication configuration."""
    
    def test_cmems_requires_auth(self):
        """CMEMS should require authentication."""
        source = API_REGISTRY["cmems_sealevel"]
        assert source.auth.required is True
        assert "CMEMS_USERNAME" in source.auth.env_vars
    
    def test_climate_indices_no_auth(self):
        """Climate indices should not require auth."""
        source = API_REGISTRY["noaa_indices"]
        assert source.auth.required is False
    
    def test_earthdata_sources_share_auth(self):
        """NASA sources should use same Earthdata auth."""
        cygnss = API_REGISTRY["cygnss"]
        gpm = API_REGISTRY["gpm_imerg"]
        
        assert "EARTHDATA_USERNAME" in cygnss.auth.env_vars
        assert "EARTHDATA_USERNAME" in gpm.auth.env_vars
