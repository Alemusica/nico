"""
🌊 ARGO Floats Client
=====================

ARGO: Global array of ~4000 autonomous profiling floats.
Measures temperature and salinity from surface to 2000m depth.

Implements the DataClient interface ("parking spot" contract).

For Early Warning:
- Steric height calculation (thermal expansion → sea level)
- Ocean heat content (storm intensification)
- Subsurface anomalies (precursors to surface events)

Data:
- ~10 day cycle per float
- Global coverage (sparse in time/space)
- Free, no auth required

Sources:
- Argo GDAC: ftp://ftp.ifremer.fr/ifremer/argo/
- Argo API: https://argovis.colorado.edu/
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


@dataclass
class ArgoProfile:
    """Single Argo float profile."""
    float_id: str
    cycle: int
    timestamp: datetime
    latitude: float
    longitude: float
    
    # Profile data (arrays)
    pressure: List[float] = field(default_factory=list)  # dbar
    temperature: List[float] = field(default_factory=list)  # °C
    salinity: List[float] = field(default_factory=list)  # PSU
    
    # Quality flags
    position_qc: int = 0
    
    def max_depth(self) -> float:
        """Maximum depth in meters (approx pressure in dbar)."""
        return max(self.pressure) if self.pressure else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "float_id": self.float_id,
            "cycle": self.cycle,
            "timestamp": self.timestamp.isoformat(),
            "lat": self.latitude,
            "lon": self.longitude,
            "max_depth": self.max_depth(),
            "n_levels": len(self.pressure),
        }


class ArgovisClient:
    """
    Argovis API Client.
    
    Argovis is a modern REST API for Argo data.
    No authentication required.
    
    API Docs: https://argovis.colorado.edu/api/docs
    
    Usage:
        client = ArgovisClient()
        
        # Get profiles in region
        profiles = await client.get_profiles(
            bbox=(7.0, 44.0, 12.0, 47.0),
            time_range=("2024-01-01", "2024-01-31"),
        )
        
        # Get specific float
        float_profiles = await client.get_float_history("6902756")
    """
    
    BASE_URL = "https://argovis.colorado.edu/api"
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent / "data" / "cache" / "argo"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get_profiles(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
        max_profiles: int = 1000,
    ) -> List[ArgoProfile]:
        """
        Get Argo profiles in region and time range.
        
        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            time_range: (start, end) as "YYYY-MM-DD"
            max_profiles: Maximum profiles to return
            
        Returns:
            List of ArgoProfile
        """
        if not HAS_AIOHTTP:
            logger.error("aiohttp required")
            return await self._generate_synthetic(bbox, time_range)
        
        # Argovis uses polygon format
        polygon = f"[[{bbox[0]},{bbox[1]}],[{bbox[2]},{bbox[1]}],[{bbox[2]},{bbox[3]}],[{bbox[0]},{bbox[3]}],[{bbox[0]},{bbox[1]}]]"
        
        params = {
            "startDate": time_range[0],
            "endDate": time_range[1],
            "polygon": polygon,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/argo",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Argovis API error: {resp.status}")
                        return await self._generate_synthetic(bbox, time_range)
                    
                    data = await resp.json()
                    
                    profiles = []
                    for item in data[:max_profiles]:
                        try:
                            profile = ArgoProfile(
                                float_id=str(item.get('_id', '')).split('_')[0],
                                cycle=item.get('cycle_number', 0),
                                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                                latitude=item['geolocation']['coordinates'][1],
                                longitude=item['geolocation']['coordinates'][0],
                                pressure=item.get('pres', []),
                                temperature=item.get('temp', []),
                                salinity=item.get('psal', []),
                            )
                            profiles.append(profile)
                        except (KeyError, ValueError) as e:
                            continue
                    
                    logger.info(f"Got {len(profiles)} Argo profiles")
                    return profiles
                    
        except Exception as e:
            logger.error(f"Argovis error: {e}")
            return await self._generate_synthetic(bbox, time_range)
    
    async def get_float_history(
        self,
        float_id: str,
        time_range: Tuple[str, str] = None,
    ) -> List[ArgoProfile]:
        """
        Get all profiles from a specific float.
        
        Args:
            float_id: WMO float ID (e.g., "6902756")
            time_range: Optional time filter
            
        Returns:
            List of ArgoProfile sorted by time
        """
        if not HAS_AIOHTTP:
            return []
        
        params = {"platform": float_id}
        if time_range:
            params["startDate"] = time_range[0]
            params["endDate"] = time_range[1]
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/argo",
                    params=params,
                    timeout=30,
                ) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    
                    profiles = []
                    for item in data:
                        try:
                            profile = ArgoProfile(
                                float_id=float_id,
                                cycle=item.get('cycle_number', 0),
                                timestamp=datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00')),
                                latitude=item['geolocation']['coordinates'][1],
                                longitude=item['geolocation']['coordinates'][0],
                                pressure=item.get('pres', []),
                                temperature=item.get('temp', []),
                                salinity=item.get('psal', []),
                            )
                            profiles.append(profile)
                        except (KeyError, ValueError):
                            continue
                    
                    return sorted(profiles, key=lambda p: p.timestamp)
                    
        except Exception as e:
            logger.error(f"Float history error: {e}")
            return []
    
    async def _generate_synthetic(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> List[ArgoProfile]:
        """Generate synthetic Argo profiles."""
        if not HAS_PANDAS:
            return []
        
        logger.warning("🔧 Generating synthetic Argo data")
        
        profiles = []
        
        # Generate ~50 profiles
        n_floats = 5
        n_cycles = 10
        
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        days = (end - start).days
        
        for f in range(n_floats):
            float_id = f"SYNTH{f:04d}"
            
            # Float drift
            lon = np.random.uniform(bbox[0], bbox[2])
            lat = np.random.uniform(bbox[1], bbox[3])
            
            for c in range(n_cycles):
                # Time
                t = start + timedelta(days=int(c * days / n_cycles))
                
                # Drift
                lon += np.random.normal(0, 0.1)
                lat += np.random.normal(0, 0.05)
                
                # Keep in bbox
                lon = np.clip(lon, bbox[0], bbox[2])
                lat = np.clip(lat, bbox[1], bbox[3])
                
                # Profile levels
                n_levels = 50
                pressure = np.linspace(5, 2000, n_levels).tolist()
                
                # Temperature profile (typical Mediterranean)
                # Surface ~20°C, thermocline ~100-300m, deep ~13°C
                temp_surface = 18 + np.random.normal(0, 2)
                temp_deep = 13 + np.random.normal(0, 0.5)
                
                temperature = []
                for p in pressure:
                    if p < 100:
                        t_val = temp_surface - (temp_surface - 15) * (p / 100)
                    elif p < 500:
                        t_val = 15 - (15 - temp_deep) * ((p - 100) / 400)
                    else:
                        t_val = temp_deep + np.random.normal(0, 0.1)
                    temperature.append(t_val)
                
                # Salinity profile
                salinity = [38.5 + np.random.normal(0, 0.1) for _ in pressure]
                
                profile = ArgoProfile(
                    float_id=float_id,
                    cycle=c + 1,
                    timestamp=t,
                    latitude=lat,
                    longitude=lon,
                    pressure=pressure,
                    temperature=temperature,
                    salinity=salinity,
                )
                profiles.append(profile)
        
        return profiles


class ArgoClient(PointDataClientMixin, DataClient):
    """
    Unified Argo data client.

    Implements the DataClient interface for unified data access.

    Usage (new interface):
        client = ArgoClient()

        df = await client.download(
            variables=["temperature", "salinity"],
            bbox=BoundingBox(lon_min=5.0, lat_min=40.0, lon_max=15.0, lat_max=45.0),
            time_range=TimeRange.from_strings("2024-01-01", "2024-01-31"),
        )

    Usage (legacy):
        # Get profiles in region
        profiles = await client.get_profiles(
            bbox=(7.0, 44.0, 12.0, 47.0),
            time_range=("2024-01-01", "2024-01-31"),
        )
    """

    # =========================================================================
    # DataClient REQUIRED PROPERTIES
    # =========================================================================

    @property
    def source_id(self) -> str:
        """Unique identifier matching api_registry.py."""
        return "argo_floats"

    # output_format is provided by PointDataClientMixin

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self):
        self.argovis = ArgovisClient()

    # =========================================================================
    # DataClient REQUIRED METHODS
    # =========================================================================

    def list_products(self) -> Dict[str, str]:
        """List available products."""
        return {
            "temperature": "In-situ temperature profiles (°C)",
            "salinity": "In-situ salinity profiles (PSU)",
            "pressure": "Pressure levels (dbar)",
            "steric_height": "Computed steric height anomaly",
        }

    async def health_check(self) -> HealthCheckResult:
        """Check if Argovis API is available."""
        start_time = time.time()

        if not HAS_AIOHTTP:
            return HealthCheckResult(
                status=ClientStatus.DEGRADED,
                message="aiohttp not installed - synthetic data only",
                details={"install": "pip install aiohttp"}
            )

        # Try a minimal API call
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.argovis.BASE_URL}/ping",
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    latency_ms = (time.time() - start_time) * 1000
                    if resp.status == 200:
                        return HealthCheckResult(
                            status=ClientStatus.HEALTHY,
                            message="Argovis API ready",
                            latency_ms=latency_ms,
                        )
        except Exception as e:
            logger.debug(f"Argovis ping failed: {e}")

        return HealthCheckResult(
            status=ClientStatus.DEGRADED,
            message="Argovis API unavailable - synthetic data available",
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
        Download Argo profile data (DataClient interface).

        Args:
            variables: List of variable names (e.g., ["temperature", "salinity"])
            bbox: Geographic bounding box
            time_range: Start and end time

        Returns:
            pd.DataFrame with Argo profiles

        Raises:
            DataClientError: If download fails
        """
        try:
            bbox_tuple = bbox.to_tuple()
            time_tuple = (
                time_range.start.strftime("%Y-%m-%d"),
                time_range.end.strftime("%Y-%m-%d")
            )

            profiles = await self.argovis.get_profiles(bbox_tuple, time_tuple)

            if not profiles:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    message="No Argo profiles found",
                    fallback_available=True
                )

            df = self.profiles_to_dataframe(profiles)
            return df

        except DataClientError:
            raise
        except Exception as e:
            logger.warning(f"[{self.source_id}] Download failed, trying synthetic: {e}")
            try:
                return await self.generate_synthetic(variables, bbox, time_range)
            except Exception:
                raise DataClientError(
                    source_id=self.source_id,
                    operation="download",
                    original_error=e,
                    message="Both real and synthetic download failed",
                    fallback_available=False
                )

    async def generate_synthetic(
        self,
        variables: List[str],
        bbox: BoundingBox,
        time_range: TimeRange,
        **kwargs
    ) -> pd.DataFrame:
        """Generate synthetic Argo data."""
        logger.info(f"Generating synthetic Argo data for {bbox}")

        bbox_tuple = bbox.to_tuple()
        time_tuple = (
            time_range.start.strftime("%Y-%m-%d"),
            time_range.end.strftime("%Y-%m-%d")
        )

        profiles = await self.argovis._generate_synthetic(bbox_tuple, time_tuple)
        df = self.profiles_to_dataframe(profiles)
        df.attrs["synthetic"] = True
        df.attrs["source"] = self.source_id
        return df

    # =========================================================================
    # LEGACY METHODS
    # =========================================================================
    
    async def get_profiles(
        self,
        bbox: Tuple[float, float, float, float],
        time_range: Tuple[str, str],
    ) -> List[ArgoProfile]:
        """Get Argo profiles in region."""
        return await self.argovis.get_profiles(bbox, time_range)
    
    async def compute_steric_height(
        self,
        profiles: List[ArgoProfile],
        ref_depth: float = 1000,
    ) -> Dict[str, float]:
        """
        Compute steric height from T/S profiles.
        
        Steric height = integral of specific volume anomaly.
        Indicates thermal expansion contribution to sea level.
        
        Args:
            profiles: List of ArgoProfile
            ref_depth: Reference depth (dbar)
            
        Returns:
            Dict with mean steric height and std
        """
        if not HAS_PANDAS:
            return {"steric_height_m": 0, "std_m": 0}
        
        steric_heights = []
        
        for profile in profiles:
            if not profile.temperature or not profile.salinity:
                continue
            
            # Simple steric height calculation
            # Full calculation needs seawater equation of state
            
            # Approximate: thermal expansion coefficient ~2e-4 /°C
            alpha = 2e-4
            
            # Integrate temperature anomaly (from 15°C reference)
            sh = 0
            for i, (p, t) in enumerate(zip(profile.pressure, profile.temperature)):
                if p > ref_depth:
                    break
                if i > 0:
                    dp = profile.pressure[i] - profile.pressure[i-1]
                    t_anom = t - 15.0  # Reference temp
                    sh += alpha * t_anom * dp  # meters
            
            steric_heights.append(sh)
        
        if steric_heights:
            return {
                "steric_height_m": np.mean(steric_heights),
                "std_m": np.std(steric_heights),
                "n_profiles": len(steric_heights),
            }
        
        return {"steric_height_m": 0, "std_m": 0, "n_profiles": 0}
    
    def profiles_to_dataframe(
        self,
        profiles: List[ArgoProfile],
    ) -> Optional[Any]:
        """Convert profiles to DataFrame."""
        if not HAS_PANDAS or not profiles:
            return None
        
        rows = []
        for p in profiles:
            for i in range(len(p.pressure)):
                rows.append({
                    "float_id": p.float_id,
                    "cycle": p.cycle,
                    "timestamp": p.timestamp,
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "pressure": p.pressure[i],
                    "temperature": p.temperature[i] if i < len(p.temperature) else np.nan,
                    "salinity": p.salinity[i] if i < len(p.salinity) else np.nan,
                })
        
        return pd.DataFrame(rows)


# Module interface
Client = ArgoClient


async def get_argo_profiles(
    bbox: Tuple[float, float, float, float],
    time_range: Tuple[str, str],
) -> List[ArgoProfile]:
    """Quick Argo data access."""
    client = ArgoClient()
    return await client.get_profiles(bbox, time_range)


# CLI test
if __name__ == "__main__":
    async def test():
        print("=== Argo Client Test ===\n")
        
        client = ArgoClient()
        
        # Mediterranean
        bbox = (5.0, 40.0, 15.0, 45.0)
        time_range = ("2024-01-01", "2024-01-31")
        
        print(f"1. Getting profiles in Mediterranean...")
        profiles = await client.get_profiles(bbox, time_range)
        print(f"   Got {len(profiles)} profiles")
        
        if profiles:
            print(f"\n   Sample profile:")
            p = profiles[0]
            print(f"   Float: {p.float_id}, Cycle: {p.cycle}")
            print(f"   Location: {p.latitude:.2f}°N, {p.longitude:.2f}°E")
            print(f"   Max depth: {p.max_depth():.0f} dbar")
            print(f"   Surface temp: {p.temperature[0]:.1f}°C")
        
        print(f"\n2. Computing steric height...")
        steric = await client.compute_steric_height(profiles)
        print(f"   Steric height: {steric['steric_height_m']*100:.1f} ± {steric['std_m']*100:.1f} cm")
        
        print(f"\n3. Converting to DataFrame...")
        df = client.profiles_to_dataframe(profiles)
        if df is not None:
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
        
        print("\n✅ Test complete")
    
    asyncio.run(test())
