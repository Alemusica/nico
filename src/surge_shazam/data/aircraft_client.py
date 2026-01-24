"""
✈️ Aircraft Meteorological Data Client
=======================================

Aircraft as flying weather sensors - thousands of vertical profiles daily.

Implements the DataClient interface ("parking spot" contract).

Data Sources:
- AMDAR (Aircraft Meteorological Data Relay): WMO standard
- Mode-S EHS: Derived from ADS-B radar transponders
- TAMDAR: US regional airlines (temperature, humidity, icing)

Why Aircraft Data for Early Warning:
- Real-time vertical profiles of atmosphere
- High temporal frequency (continuous)
- Ocean coverage during transatlantic flights
- Wind shear detection = severe weather precursor

Variables:
- Temperature (at altitude)
- Wind speed and direction
- Humidity (dewpoint)
- Turbulence (EDR)
- Altitude/pressure level
"""

import os
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
import json

# Import base client interface
from .base_client import (
    DataClient,
    DataFormat,
    ClientStatus,
    BoundingBox,
    TimeRange,
    HealthCheckResult,
    DataClientError,
    PointDataClientMixin,
)

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from opensky_api import OpenSkyApi
    HAS_OPENSKY = True
except ImportError:
    HAS_OPENSKY = False


@dataclass
class AircraftObservation:
    """Single aircraft meteorological observation."""
    timestamp: datetime
    latitude: float
    longitude: float
    altitude_m: float
    pressure_hpa: float = None
    
    # Meteorological
    temperature_k: float = None
    wind_speed_ms: float = None
    wind_direction_deg: float = None
    humidity_percent: float = None
    dewpoint_k: float = None
    
    # Aircraft info
    aircraft_id: str = ""
    flight_id: str = ""
    source: str = ""  # "amdar", "mode_s", "tamdar"
    
    # Quality
    quality_flag: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "lat": self.latitude,
            "lon": self.longitude,
            "alt_m": self.altitude_m,
            "temp_k": self.temperature_k,
            "wind_speed": self.wind_speed_ms,
            "wind_dir": self.wind_direction_deg,
            "humidity": self.humidity_percent,
            "source": self.source,
            "aircraft": self.aircraft_id,
        }


@dataclass
class VerticalProfile:
    """Vertical profile from aircraft ascent/descent."""
    flight_id: str
    timestamp_start: datetime
    timestamp_end: datetime
    latitude: float
    longitude: float
    profile_type: str  # "ascent", "descent", "cruise"
    observations: List[AircraftObservation] = field(default_factory=list)
    
    @property
    def altitude_range(self) -> Tuple[float, float]:
        if not self.observations:
            return (0, 0)
        alts = [o.altitude_m for o in self.observations]
        return (min(alts), max(alts))
    
    def to_dataframe(self) -> Any:
        """Convert to pandas DataFrame."""
        if not HAS_PANDAS:
            return None
        return pd.DataFrame([o.to_dict() for o in self.observations])


class OpenSkyModeS:
    """
    Mode-S EHS derived meteorological data from OpenSky Network.
    
    Mode-S Enhanced Surveillance includes:
    - Selected Altitude (barometric)
    - True Airspeed
    - Mach number
    - Magnetic heading
    
    From these, we can derive:
    - Temperature (from TAS and Mach)
    - Wind (from ground speed, TAS, heading)
    
    API: https://opensky-network.org/apidoc/rest.html
    
    Rate limits:
    - Anonymous: 100 requests/day
    - Registered: 1000 requests/day
    - Full access: No limit
    """
    
    BASE_URL = "https://opensky-network.org/api"
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        cache_dir: Path = None,
    ):
        self.username = username or os.getenv("OPENSKY_USERNAME")
        self.password = password or os.getenv("OPENSKY_PASSWORD")
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "opensky"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._session = None
    
    async def _get_session(self):
        if self._session is None:
            auth = None
            if self.username and self.password:
                auth = aiohttp.BasicAuth(self.username, self.password)
            self._session = aiohttp.ClientSession(auth=auth)
        return self._session
    
    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
    
    async def get_states(
        self,
        bbox: Tuple[float, float, float, float] = None,
        time: int = None,
    ) -> List[Dict]:
        """
        Get current aircraft state vectors.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time: Unix timestamp (default: now)
            
        Returns:
            List of state vectors with position, velocity, etc.
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required: pip install aiohttp")
            return []
        
        session = await self._get_session()
        
        params = {}
        if bbox:
            params['lamin'] = bbox[1]
            params['lamax'] = bbox[3]
            params['lomin'] = bbox[0]
            params['lomax'] = bbox[2]
        if time:
            params['time'] = time
        
        try:
            async with session.get(f"{self.BASE_URL}/states/all", params=params) as resp:
                if resp.status != 200:
                    logger.warning(f"OpenSky API error: {resp.status}")
                    return []
                
                data = await resp.json()
                
                if not data or 'states' not in data:
                    return []
                
                # Parse state vectors
                states = []
                for sv in data['states']:
                    if sv[5] is None or sv[6] is None:  # lat, lon
                        continue
                    
                    states.append({
                        'icao24': sv[0],
                        'callsign': (sv[1] or '').strip(),
                        'origin_country': sv[2],
                        'time_position': sv[3],
                        'last_contact': sv[4],
                        'longitude': sv[5],
                        'latitude': sv[6],
                        'baro_altitude': sv[7],  # meters
                        'on_ground': sv[8],
                        'velocity': sv[9],  # m/s ground speed
                        'true_track': sv[10],  # degrees from north
                        'vertical_rate': sv[11],  # m/s
                        'geo_altitude': sv[13],  # meters
                        'squawk': sv[14],
                        'spi': sv[15],
                        'position_source': sv[16],
                    })
                
                return states
                
        except Exception as e:
            logger.error(f"OpenSky API error: {e}")
            return []
    
    async def get_flights(
        self,
        begin: datetime,
        end: datetime,
        airport: str = None,
    ) -> List[Dict]:
        """
        Get flight data for time range.
        
        Args:
            begin: Start time
            end: End time
            airport: ICAO airport code (optional filter)
        """
        if not HAS_AIOHTTP:
            return []
        
        session = await self._get_session()
        
        begin_ts = int(begin.timestamp())
        end_ts = int(end.timestamp())
        
        if airport:
            url = f"{self.BASE_URL}/flights/departure"
            params = {'airport': airport, 'begin': begin_ts, 'end': end_ts}
        else:
            url = f"{self.BASE_URL}/flights/all"
            params = {'begin': begin_ts, 'end': end_ts}
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                return await resp.json()
        except Exception as e:
            logger.error(f"OpenSky flights error: {e}")
            return []
    
    def derive_meteorology(self, state: Dict) -> Optional[AircraftObservation]:
        """
        Derive meteorological parameters from Mode-S state vector.
        
        Temperature from Mach and TAS:
            T = TAS² / (γ·R·M²)
            where γ=1.4, R=287 J/(kg·K)
            
        Wind from ground speed, TAS, and heading:
            Wind = Ground_velocity - Air_velocity
        """
        if state['on_ground']:
            return None
        
        altitude = state.get('baro_altitude') or state.get('geo_altitude')
        if altitude is None:
            return None
        
        # For now, return with available data
        # Full Mode-S EHS decoding requires additional data
        obs = AircraftObservation(
            timestamp=datetime.fromtimestamp(state['time_position'] or state['last_contact']),
            latitude=state['latitude'],
            longitude=state['longitude'],
            altitude_m=altitude,
            wind_speed_ms=None,  # Would need TAS + heading
            wind_direction_deg=None,
            temperature_k=None,  # Would need Mach + TAS
            aircraft_id=state['icao24'],
            flight_id=state.get('callsign', ''),
            source="mode_s",
        )
        
        return obs


class MADISClient:
    """
    NOAA MADIS AMDAR Client.
    
    MADIS (Meteorological Assimilation Data Ingest System) provides
    quality-controlled aircraft data from AMDAR.
    
    Variables:
    - Temperature
    - Wind speed/direction
    - Moisture (some aircraft)
    - Turbulence (EDR)
    
    Data access: https://madis.ncep.noaa.gov/
    Requires registration.
    """
    
    BASE_URL = "https://madis-data.cprk.ncep.noaa.gov/madisPublic1/data/point/acars/netcdf"
    
    def __init__(
        self,
        username: str = None,
        password: str = None,
        cache_dir: Path = None,
    ):
        self.username = username or os.getenv("MADIS_USER")
        self.password = password or os.getenv("MADIS_PASSWORD")
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "madis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def download_hourly(
        self,
        date: datetime,
        hour: int,
    ) -> Optional[Any]:
        """
        Download hourly AMDAR data file.
        
        Files are in NetCDF format, ~hourly updates.
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required")
            return None
        
        # MADIS file naming: YYYYMMDDHH00.gz
        filename = f"{date.strftime('%Y%m%d')}{hour:02d}00.gz"
        url = f"{self.BASE_URL}/{filename}"
        
        # Would need proper MADIS auth and file parsing
        # Placeholder for structure
        logger.warning(f"MADIS download not yet implemented: {url}")
        return None
    
    async def get_observations(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[datetime, datetime],
    ) -> List[AircraftObservation]:
        """
        Get AMDAR observations for region and time.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time_range: (start, end) datetime
            
        Returns:
            List of AircraftObservation
        """
        # Placeholder - would download and parse MADIS NetCDF
        return []


class AircraftClient(PointDataClientMixin, DataClient):
    """
    Unified aircraft meteorological data client.

    Implements the DataClient interface for unified data access.

    Aggregates from multiple sources:
    - OpenSky Mode-S (free, real-time)
    - MADIS AMDAR (requires registration)
    - TAMDAR (US regional, if available)

    Usage (new interface):
        client = AircraftClient()

        df = await client.download(
            variables=["temperature", "wind_speed"],
            bbox=BoundingBox(lon_min=7.0, lat_min=44.0, lon_max=12.0, lat_max=47.0),
            time_range=TimeRange.from_strings("2024-01-01", "2024-01-02"),
        )

    Usage (legacy):
        # Get current observations in region
        obs = await client.get_observations(
            bbox=(7.0, 44.0, 11.0, 47.0),
            time_range=(now - timedelta(hours=6), now),
        )
    """

    # =========================================================================
    # DataClient REQUIRED PROPERTIES
    # =========================================================================

    @property
    def source_id(self) -> str:
        """Unique identifier matching api_registry.py."""
        return "mode_s_ehs"

    # output_format is provided by PointDataClientMixin

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):
        self.opensky = OpenSkyModeS()
        self.madis = MADISClient()

    # =========================================================================
    # DataClient REQUIRED METHODS
    # =========================================================================

    def list_products(self) -> Dict[str, str]:
        """List available products."""
        return {
            "temperature": "Temperature at altitude (K)",
            "wind_speed": "Wind speed (m/s)",
            "wind_direction": "Wind direction (degrees)",
            "altitude": "Barometric altitude (m)",
            "humidity": "Relative humidity (%)",
        }

    async def health_check(self) -> HealthCheckResult:
        """Check if OpenSky/aircraft data APIs are available."""
        start_time = time.time()

        if not HAS_AIOHTTP:
            return HealthCheckResult(
                status=ClientStatus.DEGRADED,
                message="aiohttp not installed - synthetic data only",
                details={"install": "pip install aiohttp"}
            )

        # Try OpenSky API
        try:
            states = await self.opensky.get_states(bbox=(-10, 35, 30, 60))  # Europe
            latency_ms = (time.time() - start_time) * 1000

            if states:
                return HealthCheckResult(
                    status=ClientStatus.HEALTHY,
                    message="OpenSky API ready",
                    latency_ms=latency_ms,
                    details={"current_aircraft": len(states)}
                )
        except Exception as e:
            logger.debug(f"OpenSky health check failed: {e}")

        return HealthCheckResult(
            status=ClientStatus.DEGRADED,
            message="OpenSky API unavailable - synthetic data available",
            latency_ms=(time.time() - start_time) * 1000,
        )

    async def download(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> Union[pd.DataFrame, Any]:
        """
        Download aircraft meteorological data (DataClient interface).

        Args:
            variables: List of variable names (e.g., ["temperature", "wind_speed"])
            bbox: Geographic bounding box
            time_range: Start and end time

        Returns:
            pd.DataFrame with aircraft observations

        Raises:
            DataClientError: If download fails
        """
        try:
            bbox_tuple = bbox.to_tuple()
            time_tuple = (time_range.start, time_range.end)

            observations = await self.get_observations(bbox_tuple, time_tuple)

            if not observations:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    message="No aircraft observations found",
                    fallback_available=True
                )

            df = self.observations_to_dataframe(observations)
            return df

        except DataClientError:
            raise
        except Exception as e:
            logger.warning(f"[{self.source_id}] Download failed, trying synthetic: {e}")
            try:
                return await self.generate_synthetic_data(variables, bbox, time_range)
            except Exception:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    original_error=e,
                    message="Both real and synthetic download failed",
                    fallback_available=False
                )

    async def generate_synthetic_data(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> pd.DataFrame:
        """Generate synthetic aircraft data (DataClient interface)."""
        logger.info(f"Generating synthetic aircraft data for {bbox}")

        bbox_tuple = bbox.to_tuple()
        time_tuple = (time_range.start, time_range.end)

        observations = await self.generate_synthetic(bbox_tuple, time_tuple, n_flights=50)
        df = self.observations_to_dataframe(observations)
        df.attrs["synthetic"] = True
        df.attrs["source"] = self.source_id
        return df

    # =========================================================================
    # LEGACY METHODS
    # =========================================================================
    
    async def close(self):
        await self.opensky.close()
    
    async def get_current_observations(
        self,
        bbox: Tuple[float, float, float, float],
    ) -> List[AircraftObservation]:
        """
        Get current aircraft observations in bounding box.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            
        Returns:
            List of current aircraft observations
        """
        observations = []
        
        # OpenSky current states
        states = await self.opensky.get_states(bbox=bbox)
        
        for state in states:
            obs = self.opensky.derive_meteorology(state)
            if obs:
                observations.append(obs)
        
        logger.info(f"Got {len(observations)} aircraft observations")
        return observations
    
    async def get_observations(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[datetime, datetime],
        sources: List[str] = None,
    ) -> List[AircraftObservation]:
        """
        Get aircraft observations for region and time.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time_range: (start, end)
            sources: ["opensky", "madis"] - default both
            
        Returns:
            List of AircraftObservation
        """
        sources = sources or ["opensky", "madis"]
        observations = []
        
        # For historical data, would need to query archives
        # For now, if recent, get current state
        if time_range[1] >= datetime.utcnow() - timedelta(minutes=5):
            if "opensky" in sources:
                current = await self.get_current_observations(bbox)
                observations.extend(current)
        
        # MADIS historical
        if "madis" in sources:
            madis_obs = await self.madis.get_observations(bbox, time_range)
            observations.extend(madis_obs)
        
        return observations
    
    async def get_profiles(
        self,
        airport: str,
        date: datetime,
        profile_type: str = "all",
    ) -> List[VerticalProfile]:
        """
        Get vertical profiles near an airport.
        
        Args:
            airport: ICAO code (e.g., "LIMC")
            date: Date
            profile_type: "ascent", "descent", or "all"
            
        Returns:
            List of VerticalProfile
        """
        # Get flights from/to airport
        start = datetime.combine(date.date(), datetime.min.time())
        end = start + timedelta(days=1)
        
        flights = await self.opensky.get_flights(start, end, airport=airport)
        
        # Would need to track individual flights to build profiles
        # Placeholder
        profiles = []
        
        return profiles
    
    def observations_to_dataframe(
        self,
        observations: List[AircraftObservation],
    ) -> Optional[Any]:
        """Convert observations to pandas DataFrame."""
        if not HAS_PANDAS or not observations:
            return None
        
        return pd.DataFrame([o.to_dict() for o in observations])
    
    async def generate_synthetic(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[datetime, datetime],
        n_flights: int = 50,
    ) -> List[AircraftObservation]:
        """
        Generate synthetic aircraft data for testing.
        
        Simulates flight paths with realistic meteorological profiles.
        """
        if not HAS_PANDAS:
            return []
        
        logger.warning("🔧 Generating synthetic aircraft data...")
        
        observations = []
        
        lon_min, lat_min, lon_max, lat_max = bbox
        start, end = time_range
        
        # Standard atmosphere
        def isa_temperature(altitude_m: float) -> float:
            """ISA temperature at altitude."""
            if altitude_m < 11000:
                return 288.15 - 0.0065 * altitude_m
            else:
                return 216.65  # Tropopause
        
        # Generate flights
        for flight_idx in range(n_flights):
            # Random departure/arrival within bbox
            dep_lon = np.random.uniform(lon_min, lon_max)
            dep_lat = np.random.uniform(lat_min, lat_max)
            arr_lon = np.random.uniform(lon_min, lon_max)
            arr_lat = np.random.uniform(lat_min, lat_max)
            
            # Random departure time
            dep_time = start + timedelta(
                seconds=np.random.randint(0, int((end - start).total_seconds()))
            )
            
            # Flight duration based on distance
            dist_deg = np.sqrt((arr_lon - dep_lon)**2 + (arr_lat - dep_lat)**2)
            duration_h = max(0.5, dist_deg / 8)  # ~8 deg/hour cruise
            
            # Cruise altitude
            cruise_alt = np.random.uniform(9000, 12000)  # meters
            
            # Generate observations along flight
            n_obs = int(duration_h * 60)  # One per minute
            
            for i in range(n_obs):
                t = i / n_obs
                
                # Position interpolation
                lat = dep_lat + t * (arr_lat - dep_lat)
                lon = dep_lon + t * (arr_lon - dep_lon)
                
                # Altitude profile (climb, cruise, descent)
                if t < 0.2:  # Climb
                    alt = cruise_alt * (t / 0.2)
                elif t > 0.8:  # Descent
                    alt = cruise_alt * (1 - (t - 0.8) / 0.2)
                else:  # Cruise
                    alt = cruise_alt
                
                # Temperature (ISA + random deviation)
                temp = isa_temperature(alt) + np.random.normal(0, 3)
                
                # Wind (increases with altitude, adds variability)
                wind_speed = 5 + (alt / 1000) * 3 + np.random.normal(0, 5)
                wind_dir = np.random.uniform(0, 360)
                
                obs = AircraftObservation(
                    timestamp=dep_time + timedelta(hours=duration_h * t),
                    latitude=lat,
                    longitude=lon,
                    altitude_m=alt,
                    temperature_k=temp,
                    wind_speed_ms=max(0, wind_speed),
                    wind_direction_deg=wind_dir,
                    aircraft_id=f"SYNTH{flight_idx:03d}",
                    flight_id=f"SYN{flight_idx:04d}",
                    source="synthetic",
                )
                observations.append(obs)
        
        logger.info(f"Generated {len(observations)} synthetic aircraft observations")
        return observations


# Module-level client
Client = AircraftClient


async def get_aircraft_data(
    bbox: Tuple[float, float, float, float],
    time_range: Tuple[datetime, datetime] = None,
) -> List[AircraftObservation]:
    """Quick aircraft data fetch."""
    client = AircraftClient()
    try:
        if time_range:
            return await client.get_observations(bbox, time_range)
        else:
            return await client.get_current_observations(bbox)
    finally:
        await client.close()


# CLI test
if __name__ == "__main__":
    async def test():
        print("=== Aircraft Data Client Test ===\n")
        
        client = AircraftClient()
        
        # Test bbox: Northern Italy / Alps
        bbox = (7.0, 44.0, 12.0, 47.0)
        
        print(f"📍 Test region: lon={bbox[0]}-{bbox[2]}, lat={bbox[1]}-{bbox[3]}")
        
        # Get current observations
        print("\n1. Current aircraft observations...")
        current = await client.get_current_observations(bbox)
        print(f"   Found {len(current)} aircraft")
        
        if current:
            for obs in current[:5]:
                print(f"   ✈️ {obs.flight_id}: {obs.latitude:.2f}°N, {obs.longitude:.2f}°E, {obs.altitude_m:.0f}m")
        
        # Generate synthetic for testing
        print("\n2. Generating synthetic data...")
        time_range = (
            datetime.utcnow() - timedelta(hours=6),
            datetime.utcnow()
        )
        synthetic = await client.generate_synthetic(bbox, time_range, n_flights=10)
        print(f"   Generated {len(synthetic)} observations")
        
        # Convert to DataFrame
        df = client.observations_to_dataframe(synthetic)
        if df is not None:
            print(f"\n3. DataFrame summary:")
            print(f"   Shape: {df.shape}")
            print(f"   Altitude range: {df['alt_m'].min():.0f} - {df['alt_m'].max():.0f} m")
            if df['temp_k'].notna().any():
                print(f"   Temperature range: {df['temp_k'].min():.1f} - {df['temp_k'].max():.1f} K")
        
        await client.close()
        print("\n✅ Test complete")
    
    asyncio.run(test())
