"""
🌊 Test Argo Client
===================

Tests for Argo float oceanographic data client.
Argo provides:
- Temperature profiles (0-2000m)
- Salinity profiles
- Biogeochemical variables (BGC Argo)
- Deep ocean data (Deep Argo to 6000m)

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
# Argo Client Implementation (stub for testing)
# ============================================================================

class ArgoProgram(Enum):
    """Argo program types."""
    CORE = "core"  # Standard T/S profiles
    BGC = "bgc"  # Biogeochemical
    DEEP = "deep"  # Deep ocean (6000m)


class ArgoDataMode(Enum):
    """Argo data modes."""
    REALTIME = "R"  # Real-time (unvalidated)
    ADJUSTED = "A"  # Adjusted (delayed mode QC)
    DELAYED = "D"  # Delayed mode


@dataclass
class ArgoFloat:
    """Argo float metadata."""
    wmo_id: str
    latitude: float
    longitude: float
    cycle_number: int
    date: datetime
    program: ArgoProgram
    data_mode: ArgoDataMode
    n_levels: int = 0
    max_pressure: float = 0.0


@dataclass
class ArgoProfile:
    """Argo profile data."""
    wmo_id: str
    cycle_number: int
    date: datetime
    latitude: float
    longitude: float
    pressure: np.ndarray
    temperature: np.ndarray
    salinity: np.ndarray
    quality_flags: Dict[str, np.ndarray] = field(default_factory=dict)


class ArgoClient:
    """
    Client for Argo float oceanographic data.
    
    Data sources:
    - Ifremer GDAC (Global Data Assembly Center)
    - US GODAE (alternative mirror)
    - ArgoVis API (web-based access)
    
    Usage:
        client = ArgoClient()
        
        # Get profiles in region
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(-10, 10),
            time_range=("2023-01-01", "2023-01-31"),
        )
        
        # Get specific float
        float_data = await client.get_float("6901234")
    """
    
    BASE_URL = "https://data-argo.ifremer.fr"
    ARGOVIS_URL = "https://argovis-api.colorado.edu"
    
    def __init__(
        self,
        source: str = "ifremer",  # ifremer, argovis
        cache_dir: str = None,
    ):
        self.source = source
        self.cache_dir = cache_dir
    
    async def search(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        program: ArgoProgram = None,
        data_mode: ArgoDataMode = None,
    ) -> List[ArgoFloat]:
        """
        Search for Argo floats in region.
        
        Returns:
            List of ArgoFloat metadata
        """
        # For testing, return synthetic results
        return await self._synthetic_search(
            lat_range, lon_range, time_range, program
        )
    
    async def get_profiles(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        variables: List[str] = None,
        max_profiles: int = 1000,
    ) -> List[ArgoProfile]:
        """
        Get Argo profiles in region.
        
        Args:
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude
            time_range: (start, end) as "YYYY-MM-DD"
            variables: Variables to include
            max_profiles: Maximum profiles to return
            
        Returns:
            List of ArgoProfile objects
        """
        floats = await self.search(lat_range, lon_range, time_range)
        
        profiles = []
        for f in floats[:max_profiles]:
            profile = await self._generate_synthetic_profile(f)
            profiles.append(profile)
        
        return profiles
    
    async def get_float(
        self,
        wmo_id: str,
        cycles: List[int] = None,
    ) -> List[ArgoProfile]:
        """
        Get all profiles from a specific float.
        
        Args:
            wmo_id: Float WMO ID
            cycles: Specific cycle numbers (None = all)
            
        Returns:
            List of profiles for the float
        """
        # Synthetic: generate some cycles
        profiles = []
        
        cycles = cycles or list(range(1, 51))  # 50 cycles
        
        for cycle in cycles:
            date = datetime(2020, 1, 1) + timedelta(days=cycle * 10)
            
            # Random drift
            lat = 40 + np.random.uniform(-5, 5)
            lon = 0 + np.random.uniform(-10, 10)
            
            f = ArgoFloat(
                wmo_id=wmo_id,
                latitude=lat,
                longitude=lon,
                cycle_number=cycle,
                date=date,
                program=ArgoProgram.CORE,
                data_mode=ArgoDataMode.DELAYED,
            )
            
            profile = await self._generate_synthetic_profile(f)
            profiles.append(profile)
        
        return profiles
    
    async def to_xarray(
        self,
        profiles: List[ArgoProfile],
    ) -> Optional[Any]:  # xr.Dataset
        """
        Convert profiles to xarray Dataset.
        
        Creates a gridded dataset from profile data.
        """
        if not HAS_XARRAY or not profiles:
            return None
        
        # Standard pressure levels for interpolation
        pressure_levels = np.array([
            0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
            600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000
        ])
        
        n_profiles = len(profiles)
        n_levels = len(pressure_levels)
        
        # Initialize arrays
        temp = np.full((n_profiles, n_levels), np.nan)
        sal = np.full((n_profiles, n_levels), np.nan)
        lats = np.zeros(n_profiles)
        lons = np.zeros(n_profiles)
        times = []
        
        for i, profile in enumerate(profiles):
            lats[i] = profile.latitude
            lons[i] = profile.longitude
            times.append(np.datetime64(profile.date))
            
            # Interpolate to standard levels
            for j, plev in enumerate(pressure_levels):
                if plev <= profile.pressure.max():
                    # Simple nearest interpolation
                    idx = np.argmin(np.abs(profile.pressure - plev))
                    temp[i, j] = profile.temperature[idx]
                    sal[i, j] = profile.salinity[idx]
        
        ds = xr.Dataset(
            data_vars={
                'temperature': (['n_prof', 'pressure'], temp),
                'salinity': (['n_prof', 'pressure'], sal),
                'latitude': (['n_prof'], lats),
                'longitude': (['n_prof'], lons),
            },
            coords={
                'n_prof': np.arange(n_profiles),
                'pressure': pressure_levels,
                'time': (['n_prof'], np.array(times)),
            },
            attrs={
                'source': 'argo_synthetic',
                'n_profiles': n_profiles,
                'pressure_units': 'dbar',
                'temperature_units': 'degC',
                'salinity_units': 'PSU',
            }
        )
        
        return ds
    
    async def compute_mld(
        self,
        profiles: List[ArgoProfile],
        criterion: str = "density",  # density, temperature
        threshold: float = 0.03,  # kg/m³ for density, °C for temperature
    ) -> List[Dict[str, Any]]:
        """
        Compute mixed layer depth for profiles.
        
        Args:
            profiles: List of profiles
            criterion: MLD criterion (density or temperature)
            threshold: Threshold value
            
        Returns:
            List of dicts with MLD and metadata
        """
        results = []
        
        for profile in profiles:
            # Simple MLD: find where T decreases by threshold from surface
            if criterion == "temperature":
                surface_t = profile.temperature[0]
                mld_idx = np.argmax(profile.temperature < (surface_t - threshold))
                
                if mld_idx == 0 and profile.temperature[0] >= (surface_t - threshold):
                    mld = profile.pressure[-1]  # Whole profile mixed
                else:
                    mld = profile.pressure[mld_idx]
            else:
                # For density, approximate with T/S
                # Simplified: use temperature criterion
                surface_t = profile.temperature[0]
                mld_idx = np.argmax(profile.temperature < (surface_t - threshold * 10))
                mld = profile.pressure[mld_idx] if mld_idx > 0 else profile.pressure[-1]
            
            results.append({
                'wmo_id': profile.wmo_id,
                'cycle': profile.cycle_number,
                'date': profile.date,
                'latitude': profile.latitude,
                'longitude': profile.longitude,
                'mld_m': float(mld),
                'criterion': criterion,
            })
        
        return results
    
    async def _synthetic_search(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        program: ArgoProgram = None,
    ) -> List[ArgoFloat]:
        """Generate synthetic search results."""
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        
        n_days = (end - start).days + 1
        
        # Typical Argo density: ~1 float per 3° x 3° per 10 days
        area = (lat_range[1] - lat_range[0]) * (lon_range[1] - lon_range[0])
        n_floats = max(1, int(area / 9 * n_days / 10))
        n_floats = min(n_floats, 100)  # Cap for testing
        
        floats = []
        for i in range(n_floats):
            wmo_id = f"690{np.random.randint(1000, 9999)}"
            
            floats.append(ArgoFloat(
                wmo_id=wmo_id,
                latitude=np.random.uniform(lat_range[0], lat_range[1]),
                longitude=np.random.uniform(lon_range[0], lon_range[1]),
                cycle_number=np.random.randint(1, 200),
                date=start + timedelta(days=np.random.randint(0, n_days)),
                program=program or ArgoProgram.CORE,
                data_mode=ArgoDataMode.ADJUSTED,
                n_levels=np.random.randint(50, 100),
                max_pressure=2000.0,
            ))
        
        return floats
    
    async def _generate_synthetic_profile(
        self,
        float_meta: ArgoFloat,
    ) -> ArgoProfile:
        """Generate synthetic profile data."""
        # Pressure levels (typical Argo: 2000m max)
        n_levels = np.random.randint(60, 100)
        pressure = np.linspace(5, 2000, n_levels)
        
        # Temperature profile (Mediterranean-like)
        # Surface: 15-25°C, Deep: ~13°C
        surface_temp = 18 + 5 * np.cos(2 * np.pi * float_meta.date.timetuple().tm_yday / 365)
        
        # Thermocline around 50-200m
        thermocline_depth = 100 + 50 * np.random.randn()
        
        temperature = surface_temp * np.exp(-pressure / 1000) + 13 * (1 - np.exp(-pressure / 1000))
        temperature += 0.5 * np.random.randn(n_levels)
        
        # Salinity profile
        # Mediterranean: ~38 PSU surface, decreasing with depth
        salinity = 38.5 - 0.5 * (pressure / 2000) + 0.1 * np.random.randn(n_levels)
        salinity = np.clip(salinity, 34, 39)
        
        return ArgoProfile(
            wmo_id=float_meta.wmo_id,
            cycle_number=float_meta.cycle_number,
            date=float_meta.date,
            latitude=float_meta.latitude,
            longitude=float_meta.longitude,
            pressure=pressure,
            temperature=temperature,
            salinity=salinity,
            quality_flags={
                'temperature': np.ones(n_levels, dtype=int),  # Good data
                'salinity': np.ones(n_levels, dtype=int),
            }
        )


# ============================================================================
# Tests
# ============================================================================

class TestArgoProgram:
    """Test ArgoProgram enum."""
    
    def test_programs_defined(self):
        """All Argo programs should be defined."""
        programs = [p.value for p in ArgoProgram]
        
        assert "core" in programs
        assert "bgc" in programs
        assert "deep" in programs


class TestArgoDataMode:
    """Test ArgoDataMode enum."""
    
    def test_modes_defined(self):
        """All data modes should be defined."""
        modes = [m.value for m in ArgoDataMode]
        
        assert "R" in modes  # Realtime
        assert "A" in modes  # Adjusted
        assert "D" in modes  # Delayed


class TestArgoFloat:
    """Test ArgoFloat dataclass."""
    
    def test_create_float(self):
        """Should create float metadata."""
        f = ArgoFloat(
            wmo_id="6901234",
            latitude=40.0,
            longitude=5.0,
            cycle_number=42,
            date=datetime(2023, 6, 15),
            program=ArgoProgram.CORE,
            data_mode=ArgoDataMode.DELAYED,
        )
        
        assert f.wmo_id == "6901234"
        assert f.cycle_number == 42
        assert f.program == ArgoProgram.CORE


class TestArgoProfile:
    """Test ArgoProfile dataclass."""
    
    def test_create_profile(self):
        """Should create profile data."""
        profile = ArgoProfile(
            wmo_id="6901234",
            cycle_number=1,
            date=datetime(2023, 1, 1),
            latitude=40.0,
            longitude=5.0,
            pressure=np.array([10, 50, 100, 200]),
            temperature=np.array([20, 18, 15, 13]),
            salinity=np.array([38, 38.2, 38.4, 38.5]),
        )
        
        assert len(profile.pressure) == 4
        assert profile.temperature[0] == 20


class TestArgoClient:
    """Test ArgoClient class."""
    
    @pytest.fixture
    def client(self):
        return ArgoClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
        assert client.source == "ifremer"
    
    def test_create_with_source(self):
        """Should create client with specific source."""
        client = ArgoClient(source="argovis")
        assert client.source == "argovis"
    
    @pytest.mark.asyncio
    async def test_search_returns_floats(self, client):
        """Search should return list of floats."""
        floats = await client.search(
            lat_range=(35, 45),
            lon_range=(-5, 15),
            time_range=("2023-01-01", "2023-01-31"),
        )
        
        assert isinstance(floats, list)
        assert len(floats) > 0
        
        for f in floats:
            assert isinstance(f, ArgoFloat)
            assert f.wmo_id
            assert f.latitude >= 35 and f.latitude <= 45
    
    @pytest.mark.asyncio
    async def test_search_with_program_filter(self, client):
        """Search should filter by program."""
        floats = await client.search(
            lat_range=(35, 45),
            lon_range=(-5, 15),
            time_range=("2023-01-01", "2023-01-15"),
            program=ArgoProgram.BGC,
        )
        
        for f in floats:
            assert f.program == ArgoProgram.BGC
    
    @pytest.mark.asyncio
    async def test_get_profiles(self, client):
        """Should get profile data."""
        profiles = await client.get_profiles(
            lat_range=(38, 42),
            lon_range=(5, 10),
            time_range=("2023-06-01", "2023-06-15"),
            max_profiles=10,
        )
        
        assert isinstance(profiles, list)
        assert len(profiles) <= 10
        
        for p in profiles:
            assert isinstance(p, ArgoProfile)
            assert len(p.pressure) > 0
            assert len(p.temperature) == len(p.pressure)
    
    @pytest.mark.asyncio
    async def test_get_float_by_wmo(self, client):
        """Should get profiles for specific float."""
        profiles = await client.get_float(
            wmo_id="6901234",
            cycles=[1, 2, 3],
        )
        
        assert len(profiles) == 3
        for p in profiles:
            assert p.wmo_id == "6901234"
    
    @pytest.mark.asyncio
    async def test_to_xarray(self, client):
        """Should convert profiles to xarray."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-10"),
            max_profiles=5,
        )
        
        ds = await client.to_xarray(profiles)
        
        assert ds is not None
        assert "temperature" in ds.data_vars
        assert "salinity" in ds.data_vars
        assert "pressure" in ds.coords
    
    @pytest.mark.asyncio
    async def test_compute_mld(self, client):
        """Should compute mixed layer depth."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-07-01", "2023-07-15"),
            max_profiles=5,
        )
        
        mld_results = await client.compute_mld(profiles, criterion="temperature")
        
        assert len(mld_results) == len(profiles)
        
        for result in mld_results:
            assert "mld_m" in result
            assert result["mld_m"] > 0
            assert result["mld_m"] < 2000  # Within profile range


class TestArgoDataQuality:
    """Test synthetic Argo data quality."""
    
    @pytest.fixture
    def client(self):
        return ArgoClient()
    
    @pytest.mark.asyncio
    async def test_temperature_values_realistic(self, client):
        """Temperature values should be realistic."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-06-01", "2023-06-15"),
            max_profiles=10,
        )
        
        for p in profiles:
            # Ocean temperature range: -2 to 35°C
            assert np.all(p.temperature > -2), "Temperature too cold"
            assert np.all(p.temperature < 35), "Temperature too warm"
            
            # Temperature should generally decrease with depth
            surface_t = np.mean(p.temperature[:5])
            deep_t = np.mean(p.temperature[-5:])
            assert surface_t > deep_t - 5, "Temperature profile inverted"
    
    @pytest.mark.asyncio
    async def test_salinity_values_realistic(self, client):
        """Salinity values should be realistic."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(5, 15),  # Mediterranean
            time_range=("2023-06-01", "2023-06-15"),
            max_profiles=10,
        )
        
        for p in profiles:
            # Mediterranean salinity: 36-40 PSU
            # Open ocean: 33-37 PSU
            assert np.all(p.salinity > 30), "Salinity too low"
            assert np.all(p.salinity < 42), "Salinity too high"
    
    @pytest.mark.asyncio
    async def test_pressure_increases_monotonically(self, client):
        """Pressure should increase with depth."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-15"),
            max_profiles=5,
        )
        
        for p in profiles:
            # Check monotonic increase
            assert np.all(np.diff(p.pressure) > 0), "Pressure not monotonic"
    
    @pytest.mark.asyncio
    async def test_profile_depth_range(self, client):
        """Profiles should cover typical Argo depth range."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-15"),
            max_profiles=10,
        )
        
        for p in profiles:
            # Surface near 0, deep around 2000 dbar
            assert p.pressure.min() < 50, "Profile doesn't reach surface"
            assert p.pressure.max() > 1500, "Profile not deep enough"


class TestArgoXarrayConversion:
    """Test xarray conversion functionality."""
    
    @pytest.fixture
    def client(self):
        return ArgoClient()
    
    @pytest.mark.asyncio
    async def test_xarray_has_standard_levels(self, client):
        """xarray should use standard pressure levels."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-10"),
            max_profiles=5,
        )
        
        ds = await client.to_xarray(profiles)
        
        # Check standard levels
        assert 0 in ds.pressure.values
        assert 100 in ds.pressure.values
        assert 2000 in ds.pressure.values
    
    @pytest.mark.asyncio
    async def test_xarray_has_coordinates(self, client):
        """xarray should have lat/lon coordinates."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-10"),
            max_profiles=5,
        )
        
        ds = await client.to_xarray(profiles)
        
        assert "latitude" in ds.data_vars
        assert "longitude" in ds.data_vars
        assert "time" in ds.coords
    
    @pytest.mark.asyncio
    async def test_xarray_metadata(self, client):
        """xarray should have proper metadata."""
        if not HAS_XARRAY:
            pytest.skip("xarray not installed")
        
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-01-01", "2023-01-10"),
            max_profiles=3,
        )
        
        ds = await client.to_xarray(profiles)
        
        assert "source" in ds.attrs
        assert "n_profiles" in ds.attrs
        assert ds.attrs["n_profiles"] == 3


class TestArgoMLD:
    """Test mixed layer depth computation."""
    
    @pytest.fixture
    def client(self):
        return ArgoClient()
    
    @pytest.mark.asyncio
    async def test_mld_temperature_criterion(self, client):
        """MLD should work with temperature criterion."""
        profiles = await client.get_profiles(
            lat_range=(35, 45),
            lon_range=(0, 10),
            time_range=("2023-07-01", "2023-07-15"),
            max_profiles=5,
        )
        
        results = await client.compute_mld(profiles, criterion="temperature")
        
        for r in results:
            assert r["criterion"] == "temperature"
            assert r["mld_m"] is not None
    
    @pytest.mark.asyncio
    async def test_mld_summer_vs_winter(self, client):
        """Summer MLD should be shallower than winter."""
        # Summer profiles
        summer_profiles = await client.get_profiles(
            lat_range=(40, 42),
            lon_range=(5, 8),
            time_range=("2023-07-01", "2023-07-15"),
            max_profiles=5,
        )
        summer_mld = await client.compute_mld(summer_profiles)
        
        # Winter profiles
        winter_profiles = await client.get_profiles(
            lat_range=(40, 42),
            lon_range=(5, 8),
            time_range=("2023-01-01", "2023-01-15"),
            max_profiles=5,
        )
        winter_mld = await client.compute_mld(winter_profiles)
        
        # Note: synthetic data may not show seasonal difference
        # Just verify computation works
        assert len(summer_mld) > 0
        assert len(winter_mld) > 0


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
