"""
Tests for Aircraft Meteorological Data Client
=============================================
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

from src.surge_shazam.data.aircraft_client import (
    AircraftClient,
    AircraftObservation,
    VerticalProfile,
    OpenSkyModeS,
    get_aircraft_data,
)


class TestAircraftObservation:
    """Test AircraftObservation dataclass."""
    
    def test_create_observation(self):
        """Should create observation with required fields."""
        obs = AircraftObservation(
            timestamp=datetime.now(),
            latitude=45.5,
            longitude=9.0,
            altitude_m=10000,
            temperature_k=220.0,
            wind_speed_ms=50.0,
            source="mode_s",
        )
        
        assert obs.latitude == 45.5
        assert obs.altitude_m == 10000
        assert obs.source == "mode_s"
    
    def test_to_dict(self):
        """to_dict should return serializable dict."""
        obs = AircraftObservation(
            timestamp=datetime(2000, 10, 15, 12, 0),
            latitude=45.5,
            longitude=9.0,
            altitude_m=10000,
        )
        
        d = obs.to_dict()
        assert isinstance(d, dict)
        assert d["lat"] == 45.5
        assert d["lon"] == 9.0
        assert d["alt_m"] == 10000


class TestVerticalProfile:
    """Test VerticalProfile dataclass."""
    
    def test_altitude_range(self):
        """Should compute altitude range from observations."""
        profile = VerticalProfile(
            flight_id="TEST123",
            timestamp_start=datetime.now(),
            timestamp_end=datetime.now(),
            latitude=45.5,
            longitude=9.0,
            profile_type="ascent",
            observations=[
                AircraftObservation(
                    timestamp=datetime.now(),
                    latitude=45.5,
                    longitude=9.0,
                    altitude_m=alt,
                ) for alt in [100, 1000, 5000, 10000]
            ],
        )
        
        assert profile.altitude_range == (100, 10000)
    
    def test_empty_profile_altitude_range(self):
        """Empty profile should return (0, 0)."""
        profile = VerticalProfile(
            flight_id="TEST",
            timestamp_start=datetime.now(),
            timestamp_end=datetime.now(),
            latitude=0,
            longitude=0,
            profile_type="cruise",
        )
        
        assert profile.altitude_range == (0, 0)


class TestAircraftClient:
    """Test AircraftClient functionality."""
    
    @pytest.fixture
    def client(self):
        return AircraftClient()
    
    @pytest.mark.asyncio
    async def test_generate_synthetic(self, client):
        """Should generate synthetic aircraft data."""
        bbox = (7.0, 44.0, 12.0, 47.0)
        time_range = (
            datetime.utcnow() - timedelta(hours=6),
            datetime.utcnow()
        )
        
        obs = await client.generate_synthetic(bbox, time_range, n_flights=5)
        
        assert len(obs) > 0
        
        # Check observations are valid
        for o in obs[:10]:
            assert bbox[0] <= o.longitude <= bbox[2]
            assert bbox[1] <= o.latitude <= bbox[3]
            assert o.altitude_m >= 0
            assert o.source == "synthetic"
    
    @pytest.mark.asyncio
    async def test_synthetic_has_meteorology(self, client):
        """Synthetic data should have meteorological values."""
        bbox = (7.0, 44.0, 12.0, 47.0)
        time_range = (datetime.utcnow() - timedelta(hours=1), datetime.utcnow())
        
        obs = await client.generate_synthetic(bbox, time_range, n_flights=3)
        
        # At least some observations should have temp and wind
        has_temp = any(o.temperature_k is not None for o in obs)
        has_wind = any(o.wind_speed_ms is not None for o in obs)
        
        assert has_temp
        assert has_wind
    
    def test_observations_to_dataframe(self, client):
        """Should convert observations to DataFrame."""
        obs = [
            AircraftObservation(
                timestamp=datetime.now(),
                latitude=45.0 + i * 0.1,
                longitude=9.0 + i * 0.1,
                altitude_m=10000,
                temperature_k=220.0,
            ) for i in range(5)
        ]
        
        df = client.observations_to_dataframe(obs)
        
        assert df is not None
        assert len(df) == 5
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "alt_m" in df.columns
    
    @pytest.mark.asyncio
    async def test_close(self, client):
        """Should close cleanly."""
        await client.close()  # Should not raise


class TestOpenSkyModeS:
    """Test OpenSky Mode-S client."""
    
    @pytest.fixture
    def opensky(self, tmp_path):
        return OpenSkyModeS(cache_dir=tmp_path / "opensky")
    
    def test_derive_meteorology_ground(self, opensky):
        """Ground aircraft should return None."""
        state = {
            'on_ground': True,
            'latitude': 45.0,
            'longitude': 9.0,
        }
        
        obs = opensky.derive_meteorology(state)
        assert obs is None
    
    def test_derive_meteorology_flying(self, opensky):
        """Flying aircraft should return observation."""
        state = {
            'on_ground': False,
            'latitude': 45.0,
            'longitude': 9.0,
            'baro_altitude': 10000,
            'geo_altitude': 10050,
            'time_position': datetime.now().timestamp(),
            'last_contact': datetime.now().timestamp(),
            'icao24': 'ABC123',
            'callsign': 'TEST123',
        }
        
        obs = opensky.derive_meteorology(state)
        
        assert obs is not None
        assert obs.latitude == 45.0
        assert obs.longitude == 9.0
        assert obs.altitude_m == 10000
        assert obs.source == "mode_s"


class TestConvenienceFunction:
    """Test module-level convenience function."""
    
    @pytest.mark.asyncio
    async def test_get_aircraft_data_current(self):
        """get_aircraft_data should work for current time."""
        # This will likely return empty list without real API
        bbox = (7.0, 44.0, 12.0, 47.0)
        
        # Should not raise
        result = await get_aircraft_data(bbox)
        assert isinstance(result, list)
