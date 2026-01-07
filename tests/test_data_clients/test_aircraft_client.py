"""
✈️ Test Aircraft Client
========================

Tests for AMDAR/ACARS meteorological aircraft data client.
Provides:
- Upper-air temperature profiles
- Wind data from aircraft
- Humidity observations
- Near real-time atmospheric soundings

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

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================================
# Aircraft Client Implementation (stub for testing)
# ============================================================================

class FlightPhase(Enum):
    """Aircraft flight phases."""
    ASCENT = "ascent"  # Takeoff climb
    CRUISE = "cruise"  # Level flight
    DESCENT = "descent"  # Landing approach
    LEVEL = "level"  # Short level segments


@dataclass
class AircraftObservation:
    """Single AMDAR observation."""
    time: datetime
    latitude: float
    longitude: float
    altitude_m: float
    pressure_hPa: float
    temperature_C: float
    wind_direction_deg: float
    wind_speed_ms: float
    humidity_percent: Optional[float] = None
    flight_phase: FlightPhase = FlightPhase.CRUISE
    aircraft_id: str = ""
    quality_flag: int = 0


@dataclass
class AircraftProfile:
    """Vertical profile from aircraft ascent/descent."""
    aircraft_id: str
    airport_icao: str
    time: datetime
    latitude: float
    longitude: float
    phase: FlightPhase
    observations: List[AircraftObservation] = field(default_factory=list)
    
    @property
    def max_altitude(self) -> float:
        """Maximum altitude in profile."""
        if not self.observations:
            return 0
        return max(o.altitude_m for o in self.observations)


class AircraftClient:
    """
    Client for AMDAR/ACARS meteorological aircraft data.
    
    AMDAR (Aircraft Meteorological Data Relay) provides:
    - Temperature profiles during takeoff/landing
    - Wind observations at cruise altitude
    - Humidity (on equipped aircraft)
    
    Data sources:
    - NOAA MADIS (Meteorological Assimilation Data Ingest System)
    - EUMETNET E-AMDAR
    - WMO GTS
    
    Usage:
        client = AircraftClient()
        
        # Get observations in region
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        # Get profiles near airport
        profiles = await client.get_airport_profiles(
            airport="LIMC",  # Milan Malpensa
            time_range=("2023-10-01", "2023-10-02"),
        )
    """
    
    def __init__(
        self,
        source: str = "madis",  # madis, eumetnet
        cache_dir: str = None,
    ):
        self.source = source
        self.cache_dir = cache_dir
    
    async def get_observations(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        altitude_range: Tuple[float, float] = None,  # meters
        flight_phase: FlightPhase = None,
    ) -> List[AircraftObservation]:
        """
        Get aircraft observations in region.
        
        Args:
            lat_range: Latitude range
            lon_range: Longitude range
            time_range: Time range (YYYY-MM-DD HH:MM or YYYY-MM-DD)
            altitude_range: Filter by altitude (meters)
            flight_phase: Filter by flight phase
            
        Returns:
            List of AircraftObservation
        """
        return await self._synthetic_observations(
            lat_range, lon_range, time_range, altitude_range, flight_phase
        )
    
    async def get_airport_profiles(
        self,
        airport: str,  # ICAO code (e.g., "EGLL")
        time_range: Tuple[str, str],
        max_distance_km: float = 50,
    ) -> List[AircraftProfile]:
        """
        Get vertical profiles near airport.
        
        Args:
            airport: ICAO airport code
            time_range: Time range
            max_distance_km: Maximum distance from airport
            
        Returns:
            List of AircraftProfile
        """
        return await self._synthetic_profiles(airport, time_range)
    
    async def to_xarray(
        self,
        observations: List[AircraftObservation],
    ) -> Optional[Any]:  # xr.Dataset
        """Convert observations to xarray Dataset."""
        if not HAS_XARRAY or not observations:
            return None
        
        n = len(observations)
        
        ds = xr.Dataset(
            data_vars={
                'temperature': (['obs'], np.array([o.temperature_C for o in observations])),
                'wind_speed': (['obs'], np.array([o.wind_speed_ms for o in observations])),
                'wind_direction': (['obs'], np.array([o.wind_direction_deg for o in observations])),
                'pressure': (['obs'], np.array([o.pressure_hPa for o in observations])),
                'latitude': (['obs'], np.array([o.latitude for o in observations])),
                'longitude': (['obs'], np.array([o.longitude for o in observations])),
                'altitude': (['obs'], np.array([o.altitude_m for o in observations])),
            },
            coords={
                'obs': np.arange(n),
                'time': (['obs'], np.array([np.datetime64(o.time) for o in observations])),
            },
            attrs={
                'source': 'amdar',
                'n_observations': n,
            }
        )
        
        return ds
    
    async def to_dataframe(
        self,
        observations: List[AircraftObservation],
    ) -> Optional[Any]:  # pd.DataFrame
        """Convert observations to pandas DataFrame."""
        if not HAS_PANDAS or not observations:
            return None
        
        data = []
        for obs in observations:
            data.append({
                'time': obs.time,
                'latitude': obs.latitude,
                'longitude': obs.longitude,
                'altitude_m': obs.altitude_m,
                'pressure_hPa': obs.pressure_hPa,
                'temperature_C': obs.temperature_C,
                'wind_speed_ms': obs.wind_speed_ms,
                'wind_direction_deg': obs.wind_direction_deg,
                'humidity_percent': obs.humidity_percent,
                'flight_phase': obs.flight_phase.value,
                'aircraft_id': obs.aircraft_id,
            })
        
        return pd.DataFrame(data)
    
    async def compute_lapse_rate(
        self,
        profile: AircraftProfile,
    ) -> float:
        """
        Compute temperature lapse rate from profile.
        
        Returns:
            Lapse rate in °C/km
        """
        if len(profile.observations) < 2:
            return 0.0
        
        obs = sorted(profile.observations, key=lambda o: o.altitude_m)
        
        # Linear fit
        altitudes = np.array([o.altitude_m for o in obs]) / 1000  # km
        temperatures = np.array([o.temperature_C for o in obs])
        
        if len(altitudes) < 2:
            return 0.0
        
        # Simple linear regression
        coeffs = np.polyfit(altitudes, temperatures, 1)
        lapse_rate = -coeffs[0]  # Positive = temperature decreases with altitude
        
        return float(lapse_rate)
    
    async def _synthetic_observations(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        altitude_range: Tuple[float, float] = None,
        flight_phase: FlightPhase = None,
    ) -> List[AircraftObservation]:
        """Generate synthetic AMDAR observations."""
        start = datetime.strptime(time_range[0][:10], "%Y-%m-%d")
        end = datetime.strptime(time_range[1][:10], "%Y-%m-%d")
        
        n_hours = max(1, (end - start).days * 24)
        
        # Aircraft density depends on region (busier near airports)
        observations = []
        
        for h in range(n_hours):
            # ~10-50 observations per hour in busy region
            n_obs = np.random.randint(10, 50)
            
            for _ in range(n_obs):
                time = start + timedelta(hours=h, minutes=np.random.randint(0, 60))
                
                lat = np.random.uniform(lat_range[0], lat_range[1])
                lon = np.random.uniform(lon_range[0], lon_range[1])
                
                # Altitude: bimodal (cruise ~10km, ascent/descent 0-10km)
                if np.random.random() > 0.3:
                    altitude = np.random.uniform(9000, 12000)  # Cruise
                    phase = FlightPhase.CRUISE
                else:
                    altitude = np.random.uniform(500, 8000)  # Ascent/descent
                    phase = np.random.choice([FlightPhase.ASCENT, FlightPhase.DESCENT])
                
                # Filter by altitude if specified
                if altitude_range:
                    if not (altitude_range[0] <= altitude <= altitude_range[1]):
                        continue
                
                # Filter by phase if specified
                if flight_phase and phase != flight_phase:
                    continue
                
                # Standard atmosphere: T = 15 - 6.5 * (altitude/1000)
                temperature = 15 - 6.5 * (altitude / 1000) + np.random.normal(0, 2)
                
                # Pressure from altitude (simplified)
                pressure = 1013.25 * (1 - altitude / 44330) ** 5.255
                
                # Wind
                wind_speed = np.random.uniform(5, 80)  # m/s at altitude
                wind_dir = np.random.uniform(0, 360)
                
                observations.append(AircraftObservation(
                    time=time,
                    latitude=lat,
                    longitude=lon,
                    altitude_m=altitude,
                    pressure_hPa=pressure,
                    temperature_C=temperature,
                    wind_direction_deg=wind_dir,
                    wind_speed_ms=wind_speed,
                    humidity_percent=np.random.uniform(20, 80) if np.random.random() > 0.7 else None,
                    flight_phase=phase,
                    aircraft_id=f"AC{np.random.randint(100, 999)}",
                ))
        
        return observations
    
    async def _synthetic_profiles(
        self,
        airport: str,
        time_range: Tuple[str, str],
    ) -> List[AircraftProfile]:
        """Generate synthetic airport profiles."""
        start = datetime.strptime(time_range[0][:10], "%Y-%m-%d")
        end = datetime.strptime(time_range[1][:10], "%Y-%m-%d")
        
        n_days = max(1, (end - start).days)
        
        # Airport coordinates (simplified)
        airport_coords = {
            "LIMC": (45.63, 8.72),  # Milan Malpensa
            "EGLL": (51.47, -0.46),  # London Heathrow
            "LFPG": (49.01, 2.55),  # Paris CDG
            "EDDF": (50.03, 8.57),  # Frankfurt
            "LEMD": (40.47, -3.57),  # Madrid
        }
        
        lat, lon = airport_coords.get(airport, (45.0, 10.0))
        
        profiles = []
        
        for d in range(n_days):
            # ~20-50 profiles per day at major airport
            n_profiles = np.random.randint(20, 50)
            
            for _ in range(n_profiles):
                time = start + timedelta(days=d, hours=np.random.randint(0, 24))
                phase = np.random.choice([FlightPhase.ASCENT, FlightPhase.DESCENT])
                
                # Generate profile observations
                n_levels = np.random.randint(15, 30)
                
                if phase == FlightPhase.ASCENT:
                    altitudes = np.linspace(100, 10000, n_levels)
                else:
                    altitudes = np.linspace(10000, 100, n_levels)
                
                observations = []
                for alt in altitudes:
                    obs_time = time + timedelta(minutes=int(alt / 500))  # ~500m/min vertical
                    
                    temp = 15 - 6.5 * (alt / 1000) + np.random.normal(0, 1)
                    pressure = 1013.25 * (1 - alt / 44330) ** 5.255
                    
                    observations.append(AircraftObservation(
                        time=obs_time,
                        latitude=lat + np.random.uniform(-0.2, 0.2),
                        longitude=lon + np.random.uniform(-0.2, 0.2),
                        altitude_m=alt,
                        pressure_hPa=pressure,
                        temperature_C=temp,
                        wind_direction_deg=np.random.uniform(200, 300),
                        wind_speed_ms=np.random.uniform(5, 30) * (alt / 5000),
                        flight_phase=phase,
                    ))
                
                profiles.append(AircraftProfile(
                    aircraft_id=f"AC{np.random.randint(100, 999)}",
                    airport_icao=airport,
                    time=time,
                    latitude=lat,
                    longitude=lon,
                    phase=phase,
                    observations=observations,
                ))
        
        return profiles


# ============================================================================
# Tests
# ============================================================================

class TestFlightPhase:
    """Test FlightPhase enum."""
    
    def test_phases_defined(self):
        """All flight phases should be defined."""
        phases = [p.value for p in FlightPhase]
        
        assert "ascent" in phases
        assert "cruise" in phases
        assert "descent" in phases


class TestAircraftObservation:
    """Test AircraftObservation dataclass."""
    
    def test_create_observation(self):
        """Should create observation."""
        obs = AircraftObservation(
            time=datetime(2023, 10, 1, 12, 0),
            latitude=45.0,
            longitude=10.0,
            altitude_m=10000,
            pressure_hPa=265,
            temperature_C=-50,
            wind_direction_deg=270,
            wind_speed_ms=50,
        )
        
        assert obs.altitude_m == 10000
        assert obs.temperature_C == -50


class TestAircraftProfile:
    """Test AircraftProfile dataclass."""
    
    def test_create_profile(self):
        """Should create profile."""
        profile = AircraftProfile(
            aircraft_id="AC123",
            airport_icao="LIMC",
            time=datetime(2023, 10, 1, 12, 0),
            latitude=45.63,
            longitude=8.72,
            phase=FlightPhase.ASCENT,
        )
        
        assert profile.aircraft_id == "AC123"
        assert profile.phase == FlightPhase.ASCENT
    
    def test_max_altitude_empty(self):
        """Max altitude should be 0 for empty profile."""
        profile = AircraftProfile(
            aircraft_id="AC123",
            airport_icao="LIMC",
            time=datetime(2023, 10, 1),
            latitude=45.0,
            longitude=10.0,
            phase=FlightPhase.ASCENT,
        )
        
        assert profile.max_altitude == 0


class TestAircraftClient:
    """Test AircraftClient class."""
    
    @pytest.fixture
    def client(self):
        return AircraftClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
        assert client.source == "madis"
    
    @pytest.mark.asyncio
    async def test_get_observations(self, client):
        """Should get observations."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        assert isinstance(obs, list)
        assert len(obs) > 0
        
        for o in obs:
            assert isinstance(o, AircraftObservation)
            assert 40 <= o.latitude <= 50
            assert 0 <= o.longitude <= 15
    
    @pytest.mark.asyncio
    async def test_filter_by_altitude(self, client):
        """Should filter by altitude range."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
            altitude_range=(8000, 12000),  # Cruise only
        )
        
        for o in obs:
            assert 8000 <= o.altitude_m <= 12000
    
    @pytest.mark.asyncio
    async def test_filter_by_phase(self, client):
        """Should filter by flight phase."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
            flight_phase=FlightPhase.ASCENT,
        )
        
        for o in obs:
            assert o.flight_phase == FlightPhase.ASCENT
    
    @pytest.mark.asyncio
    async def test_get_airport_profiles(self, client):
        """Should get airport profiles."""
        profiles = await client.get_airport_profiles(
            airport="LIMC",
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        
        for p in profiles:
            assert isinstance(p, AircraftProfile)
            assert p.airport_icao == "LIMC"
            assert len(p.observations) > 0
    
    @pytest.mark.asyncio
    async def test_to_xarray(self, client):
        """Should convert to xarray."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        ds = await client.to_xarray(obs)
        
        assert ds is not None
        assert "temperature" in ds.data_vars
        assert "wind_speed" in ds.data_vars
    
    @pytest.mark.asyncio
    async def test_to_dataframe(self, client):
        """Should convert to DataFrame."""
        if not HAS_PANDAS:
            pytest.skip("pandas not installed")
        
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        df = await client.to_dataframe(obs)
        
        assert df is not None
        assert "temperature_C" in df.columns
        assert "altitude_m" in df.columns
    
    @pytest.mark.asyncio
    async def test_compute_lapse_rate(self, client):
        """Should compute lapse rate."""
        profiles = await client.get_airport_profiles(
            airport="LIMC",
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        if profiles:
            lapse_rate = await client.compute_lapse_rate(profiles[0])
            
            # Standard lapse rate ~6.5°C/km
            assert 4 < lapse_rate < 10, "Lapse rate should be realistic"


class TestAircraftDataQuality:
    """Test synthetic aircraft data quality."""
    
    @pytest.fixture
    def client(self):
        return AircraftClient()
    
    @pytest.mark.asyncio
    async def test_temperature_realistic(self, client):
        """Temperature should follow standard atmosphere."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        for o in obs:
            # At cruise (10km), expect -50 to -60°C
            # At surface, expect -10 to +30°C
            expected_t = 15 - 6.5 * (o.altitude_m / 1000)
            assert abs(o.temperature_C - expected_t) < 20, "Temperature unrealistic"
    
    @pytest.mark.asyncio
    async def test_pressure_altitude_consistent(self, client):
        """Pressure should be consistent with altitude."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        for o in obs:
            # Rough check: higher altitude = lower pressure
            if o.altitude_m > 5000:
                assert o.pressure_hPa < 600, "Pressure too high for altitude"
            if o.altitude_m < 1000:
                assert o.pressure_hPa > 800, "Pressure too low for altitude"
    
    @pytest.mark.asyncio
    async def test_wind_speed_realistic(self, client):
        """Wind speed should be realistic."""
        obs = await client.get_observations(
            lat_range=(40, 50),
            lon_range=(0, 15),
            time_range=("2023-10-01", "2023-10-02"),
        )
        
        for o in obs:
            # Wind speed 0-100 m/s is reasonable
            assert 0 <= o.wind_speed_ms < 120, "Wind speed unrealistic"


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
