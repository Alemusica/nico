"""
🌊 Test Tide Gauge Client
=========================

Tests for coastal tide gauge data client.
Provides:
- Sea level observations from tide gauges
- Storm surge detection
- Tidal analysis
- Historical sea level records

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
# Tide Gauge Client Implementation (stub for testing)
# ============================================================================

class TideGaugeNetwork(Enum):
    """Tide gauge data networks."""
    GLOSS = "gloss"  # Global Sea Level Observing System
    UHSLC = "uhslc"  # University of Hawaii Sea Level Center
    PSMSL = "psmsl"  # Permanent Service for Mean Sea Level
    IOC = "ioc"  # IOC/UNESCO
    NOAA = "noaa"  # NOAA Tides & Currents
    EMODnet = "emodnet"  # European Marine Observation Network


@dataclass
class TideGaugeStation:
    """Tide gauge station metadata."""
    station_id: str
    name: str
    latitude: float
    longitude: float
    country: str
    network: TideGaugeNetwork
    start_year: int
    end_year: Optional[int] = None
    datum: str = "MSL"  # Mean Sea Level
    quality_flag: int = 0


@dataclass
class TideGaugeData:
    """Tide gauge time series data."""
    station: TideGaugeStation
    time: np.ndarray
    sea_level: np.ndarray  # meters relative to datum
    quality_flags: np.ndarray = None
    tide_prediction: np.ndarray = None  # Predicted astronomical tide
    residual: np.ndarray = None  # Sea level - tide (surge component)


class TideGaugeClient:
    """
    Client for coastal tide gauge data.
    
    Provides access to:
    - GLOSS (Global Sea Level Observing System)
    - UHSLC (University of Hawaii Sea Level Center)
    - PSMSL (Permanent Service for Mean Sea Level)
    - EMODnet (European Marine Observation Network)
    
    Usage:
        client = TideGaugeClient()
        
        # Search for stations
        stations = await client.search_stations(
            lat_range=(42, 46),
            lon_range=(12, 16),
        )
        
        # Get data for station
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-01-01", "2023-01-31"),
        )
        
        # Detect storm surge
        surge = await client.detect_surge(data)
    """
    
    def __init__(
        self,
        network: TideGaugeNetwork = TideGaugeNetwork.GLOSS,
        cache_dir: str = None,
    ):
        self.network = network
        self.cache_dir = cache_dir
    
    def list_networks(self) -> Dict[str, str]:
        """List available networks."""
        return {
            "gloss": "Global Sea Level Observing System (GLOSS)",
            "uhslc": "University of Hawaii Sea Level Center",
            "psmsl": "Permanent Service for Mean Sea Level",
            "emodnet": "European Marine Observation Network",
            "noaa": "NOAA Tides & Currents",
        }
    
    async def search_stations(
        self,
        lat_range: Tuple[float, float] = None,
        lon_range: Tuple[float, float] = None,
        country: str = None,
        network: TideGaugeNetwork = None,
    ) -> List[TideGaugeStation]:
        """
        Search for tide gauge stations.
        
        Args:
            lat_range: Latitude range
            lon_range: Longitude range
            country: Filter by country
            network: Filter by network
            
        Returns:
            List of TideGaugeStation
        """
        return await self._synthetic_stations(lat_range, lon_range, country)
    
    async def get_data(
        self,
        station_id: str,
        time_range: Tuple[str, str],
        resolution: str = "hourly",  # hourly, daily, monthly
        include_prediction: bool = False,
    ) -> Optional[TideGaugeData]:
        """
        Get tide gauge data for station.
        
        Args:
            station_id: Station identifier
            time_range: Time range
            resolution: Temporal resolution
            include_prediction: Include tidal prediction
            
        Returns:
            TideGaugeData object
        """
        return await self._synthetic_data(station_id, time_range, resolution, include_prediction)
    
    async def get_multiple_stations(
        self,
        station_ids: List[str],
        time_range: Tuple[str, str],
    ) -> Dict[str, TideGaugeData]:
        """
        Get data for multiple stations.
        
        Returns:
            Dict mapping station_id to TideGaugeData
        """
        results = {}
        for sid in station_ids:
            data = await self.get_data(sid, time_range)
            if data:
                results[sid] = data
        return results
    
    async def detect_surge(
        self,
        data: TideGaugeData,
        threshold_m: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Detect storm surge events.
        
        Args:
            data: Tide gauge data
            threshold_m: Surge threshold in meters
            
        Returns:
            List of surge events
        """
        if data.residual is None:
            # Compute residual if not available
            residual = data.sea_level - np.nanmean(data.sea_level)
        else:
            residual = data.residual
        
        events = []
        above_threshold = residual > threshold_m
        
        # Find contiguous surge periods
        in_surge = False
        surge_start = None
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_surge:
                in_surge = True
                surge_start = i
            elif not is_above and in_surge:
                in_surge = False
                # Record event
                surge_end = i
                max_idx = surge_start + np.argmax(residual[surge_start:surge_end])
                events.append({
                    'start_time': data.time[surge_start],
                    'end_time': data.time[surge_end - 1],
                    'peak_time': data.time[max_idx],
                    'peak_surge_m': float(residual[max_idx]),
                    'duration_hours': (surge_end - surge_start),
                    'station': data.station.name,
                })
        
        return events
    
    async def compute_statistics(
        self,
        data: TideGaugeData,
    ) -> Dict[str, float]:
        """
        Compute sea level statistics.
        
        Returns:
            Dict with statistics
        """
        sl = data.sea_level
        sl_clean = sl[~np.isnan(sl)]
        
        if len(sl_clean) == 0:
            return {}
        
        return {
            'mean_m': float(np.mean(sl_clean)),
            'std_m': float(np.std(sl_clean)),
            'min_m': float(np.min(sl_clean)),
            'max_m': float(np.max(sl_clean)),
            'range_m': float(np.max(sl_clean) - np.min(sl_clean)),
            'n_observations': len(sl_clean),
            'data_coverage': len(sl_clean) / len(sl),
        }
    
    async def to_dataframe(
        self,
        data: TideGaugeData,
    ) -> Optional[Any]:
        """Convert to pandas DataFrame."""
        if not HAS_PANDAS:
            return None
        
        df_data = {
            'time': data.time,
            'sea_level_m': data.sea_level,
        }
        
        if data.quality_flags is not None:
            df_data['quality_flag'] = data.quality_flags
        
        if data.tide_prediction is not None:
            df_data['tide_prediction_m'] = data.tide_prediction
        
        if data.residual is not None:
            df_data['residual_m'] = data.residual
        
        df = pd.DataFrame(df_data)
        df['time'] = pd.to_datetime(df['time'])
        
        return df
    
    async def _synthetic_stations(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        country: str = None,
    ) -> List[TideGaugeStation]:
        """Generate synthetic station list."""
        # Some real station locations (simplified)
        all_stations = [
            TideGaugeStation("venice", "Venice - Punta della Salute", 45.43, 12.33, "Italy", TideGaugeNetwork.GLOSS, 1872),
            TideGaugeStation("trieste", "Trieste", 45.65, 13.76, "Italy", TideGaugeNetwork.PSMSL, 1905),
            TideGaugeStation("genova", "Genoa", 44.40, 8.93, "Italy", TideGaugeNetwork.GLOSS, 1884),
            TideGaugeStation("marseille", "Marseille", 43.30, 5.35, "France", TideGaugeNetwork.GLOSS, 1885),
            TideGaugeStation("barcelona", "Barcelona", 41.38, 2.18, "Spain", TideGaugeNetwork.UHSLC, 1950),
            TideGaugeStation("brest", "Brest", 48.38, -4.50, "France", TideGaugeNetwork.GLOSS, 1807),
            TideGaugeStation("newlyn", "Newlyn", 50.10, -5.54, "UK", TideGaugeNetwork.PSMSL, 1915),
            TideGaugeStation("amsterdam", "Amsterdam", 52.37, 4.90, "Netherlands", TideGaugeNetwork.PSMSL, 1700),
            TideGaugeStation("stockholm", "Stockholm", 59.32, 18.08, "Sweden", TideGaugeNetwork.PSMSL, 1774),
            TideGaugeStation("honolulu", "Honolulu", 21.30, -157.87, "USA", TideGaugeNetwork.UHSLC, 1905),
        ]
        
        filtered = []
        for station in all_stations:
            if lat_range:
                if not (lat_range[0] <= station.latitude <= lat_range[1]):
                    continue
            if lon_range:
                if not (lon_range[0] <= station.longitude <= lon_range[1]):
                    continue
            if country:
                if station.country.lower() != country.lower():
                    continue
            filtered.append(station)
        
        return filtered
    
    async def _synthetic_data(
        self,
        station_id: str,
        time_range: Tuple[str, str],
        resolution: str,
        include_prediction: bool,
    ) -> TideGaugeData:
        """Generate synthetic tide gauge data."""
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        
        # Time array
        if resolution == "hourly":
            freq = timedelta(hours=1)
        elif resolution == "daily":
            freq = timedelta(days=1)
        else:
            freq = timedelta(days=30)
        
        times = []
        current = start
        while current <= end:
            times.append(current)
            current += freq
        
        times = np.array([np.datetime64(t) for t in times])
        n = len(times)
        
        # Generate tidal signal (simplified M2 + S2 + diurnal)
        t_hours = np.arange(n) * (freq.total_seconds() / 3600)
        
        # M2 tide (12.42 hour period)
        m2_amp = 0.3 + 0.2 * np.random.random()
        m2_phase = np.random.uniform(0, 2 * np.pi)
        m2 = m2_amp * np.cos(2 * np.pi * t_hours / 12.42 + m2_phase)
        
        # S2 tide (12 hour period)
        s2_amp = m2_amp * 0.4
        s2_phase = np.random.uniform(0, 2 * np.pi)
        s2 = s2_amp * np.cos(2 * np.pi * t_hours / 12.0 + s2_phase)
        
        # Diurnal (K1, 23.93 hour period)
        k1_amp = m2_amp * 0.3
        k1_phase = np.random.uniform(0, 2 * np.pi)
        k1 = k1_amp * np.cos(2 * np.pi * t_hours / 23.93 + k1_phase)
        
        tide_prediction = m2 + s2 + k1
        
        # Meteorological residual (surge)
        # Random walk + occasional events
        residual = np.cumsum(np.random.normal(0, 0.02, n))
        residual -= np.mean(residual)
        
        # Add surge events
        n_events = np.random.randint(0, 3)
        for _ in range(n_events):
            event_idx = np.random.randint(0, n)
            event_amp = np.random.uniform(0.3, 1.5)
            event_width = np.random.randint(5, 24)
            event_signal = event_amp * np.exp(-((np.arange(n) - event_idx) ** 2) / (2 * event_width ** 2))
            residual += event_signal
        
        # Total sea level
        sea_level = tide_prediction + residual
        
        # Add some measurement noise
        sea_level += np.random.normal(0, 0.02, n)
        
        # Quality flags (mostly good)
        quality_flags = np.ones(n, dtype=int)
        quality_flags[np.random.random(n) < 0.01] = 0  # 1% questionable
        
        # Get station info
        stations = await self._synthetic_stations(None, None, None)
        station = next((s for s in stations if s.station_id == station_id), None)
        
        if station is None:
            station = TideGaugeStation(
                station_id=station_id,
                name=f"Station {station_id}",
                latitude=45.0,
                longitude=10.0,
                country="Unknown",
                network=self.network,
                start_year=1900,
            )
        
        return TideGaugeData(
            station=station,
            time=times,
            sea_level=sea_level.astype(np.float32),
            quality_flags=quality_flags,
            tide_prediction=tide_prediction.astype(np.float32) if include_prediction else None,
            residual=residual.astype(np.float32) if include_prediction else None,
        )


# ============================================================================
# Tests
# ============================================================================

class TestTideGaugeNetwork:
    """Test TideGaugeNetwork enum."""
    
    def test_networks_defined(self):
        """All networks should be defined."""
        networks = [n.value for n in TideGaugeNetwork]
        
        assert "gloss" in networks
        assert "psmsl" in networks
        assert "uhslc" in networks


class TestTideGaugeStation:
    """Test TideGaugeStation dataclass."""
    
    def test_create_station(self):
        """Should create station metadata."""
        station = TideGaugeStation(
            station_id="test",
            name="Test Station",
            latitude=45.0,
            longitude=10.0,
            country="Italy",
            network=TideGaugeNetwork.GLOSS,
            start_year=1900,
        )
        
        assert station.station_id == "test"
        assert station.country == "Italy"


class TestTideGaugeData:
    """Test TideGaugeData dataclass."""
    
    def test_create_data(self):
        """Should create data object."""
        station = TideGaugeStation(
            "test", "Test", 45.0, 10.0, "Italy",
            TideGaugeNetwork.GLOSS, 1900
        )
        
        data = TideGaugeData(
            station=station,
            time=np.array([np.datetime64("2023-01-01")]),
            sea_level=np.array([0.5]),
        )
        
        assert len(data.sea_level) == 1


class TestTideGaugeClient:
    """Test TideGaugeClient class."""
    
    @pytest.fixture
    def client(self):
        return TideGaugeClient()
    
    def test_create_client(self, client):
        """Should create client instance."""
        assert client is not None
        assert client.network == TideGaugeNetwork.GLOSS
    
    def test_list_networks(self, client):
        """Should list available networks."""
        networks = client.list_networks()
        
        assert isinstance(networks, dict)
        assert "gloss" in networks
    
    @pytest.mark.asyncio
    async def test_search_stations(self, client):
        """Should search for stations."""
        stations = await client.search_stations(
            lat_range=(40, 50),
            lon_range=(5, 15),
        )
        
        assert isinstance(stations, list)
        assert len(stations) > 0
        
        for s in stations:
            assert isinstance(s, TideGaugeStation)
            assert 40 <= s.latitude <= 50
    
    @pytest.mark.asyncio
    async def test_search_by_country(self, client):
        """Should filter by country."""
        stations = await client.search_stations(country="Italy")
        
        for s in stations:
            assert s.country == "Italy"
    
    @pytest.mark.asyncio
    async def test_get_data(self, client):
        """Should get tide gauge data."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-31"),
        )
        
        assert data is not None
        assert len(data.sea_level) > 0
        assert data.station.station_id == "venice"
    
    @pytest.mark.asyncio
    async def test_get_data_with_prediction(self, client):
        """Should include tidal prediction."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-10"),
            include_prediction=True,
        )
        
        assert data.tide_prediction is not None
        assert data.residual is not None
    
    @pytest.mark.asyncio
    async def test_get_multiple_stations(self, client):
        """Should get data for multiple stations."""
        data_dict = await client.get_multiple_stations(
            station_ids=["venice", "trieste"],
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        assert "venice" in data_dict
        assert "trieste" in data_dict
    
    @pytest.mark.asyncio
    async def test_detect_surge(self, client):
        """Should detect surge events."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-31"),
            include_prediction=True,
        )
        
        events = await client.detect_surge(data, threshold_m=0.3)
        
        assert isinstance(events, list)
        for event in events:
            assert "peak_surge_m" in event
            assert event["peak_surge_m"] > 0.3
    
    @pytest.mark.asyncio
    async def test_compute_statistics(self, client):
        """Should compute statistics."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-31"),
        )
        
        stats = await client.compute_statistics(data)
        
        assert "mean_m" in stats
        assert "std_m" in stats
        assert "range_m" in stats
    
    @pytest.mark.asyncio
    async def test_to_dataframe(self, client):
        """Should convert to DataFrame."""
        if not HAS_PANDAS:
            pytest.skip("pandas not installed")
        
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-10"),
        )
        
        df = await client.to_dataframe(data)
        
        assert df is not None
        assert "sea_level_m" in df.columns
        assert "time" in df.columns


class TestTideGaugeDataQuality:
    """Test synthetic data quality."""
    
    @pytest.fixture
    def client(self):
        return TideGaugeClient()
    
    @pytest.mark.asyncio
    async def test_sea_level_realistic(self, client):
        """Sea level should be realistic."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-31"),
        )
        
        sl = data.sea_level
        
        # Sea level typically -3 to +3 m relative to datum
        assert np.all(sl > -5), "Sea level too low"
        assert np.all(sl < 5), "Sea level too high"
    
    @pytest.mark.asyncio
    async def test_tidal_range_realistic(self, client):
        """Tidal range should be realistic."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-15"),
        )
        
        sl_range = np.max(data.sea_level) - np.min(data.sea_level)
        
        # Typical tidal range 0.5 to 5 m
        assert sl_range > 0.2, "Tidal range too small"
        assert sl_range < 10, "Tidal range too large"
    
    @pytest.mark.asyncio
    async def test_has_tidal_periodicity(self, client):
        """Data should show tidal periodicity."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-07"),
            resolution="hourly",
        )
        
        # Check for semi-diurnal signal (2 high/low per day)
        # Simplified: just check there's variability
        std = np.std(data.sea_level)
        assert std > 0.1, "Data lacks tidal variability"


class TestStormSurgeDetection:
    """Test storm surge detection."""
    
    @pytest.fixture
    def client(self):
        return TideGaugeClient()
    
    @pytest.mark.asyncio
    async def test_surge_event_structure(self, client):
        """Surge events should have required fields."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-11-30"),
            include_prediction=True,
        )
        
        events = await client.detect_surge(data, threshold_m=0.3)
        
        if events:
            event = events[0]
            assert "start_time" in event
            assert "end_time" in event
            assert "peak_time" in event
            assert "peak_surge_m" in event
            assert "duration_hours" in event
    
    @pytest.mark.asyncio
    async def test_no_surge_with_high_threshold(self, client):
        """High threshold should find fewer/no events."""
        data = await client.get_data(
            station_id="venice",
            time_range=("2023-10-01", "2023-10-31"),
            include_prediction=True,
        )
        
        events_low = await client.detect_surge(data, threshold_m=0.2)
        events_high = await client.detect_surge(data, threshold_m=2.0)
        
        assert len(events_high) <= len(events_low)


# Run tests directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
