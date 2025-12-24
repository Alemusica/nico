"""
🌤️ ERA5 Reanalysis Client
==========================

Download meteorological data from ECMWF's ERA5 via CDS API.

ERA5 provides hourly data from 1940 to present:
- Surface: temperature, precipitation, wind, pressure
- Pressure levels: geopotential, temperature, humidity
- Sea: SST, wave height

Authentication:
    1. Register at: https://cds.climate.copernicus.eu
    2. Go to your profile to get your Personal Access Token
    3. Create ~/.cdsapirc with:
    
       url: https://cds.climate.copernicus.eu/api
       key: <PERSONAL-ACCESS-TOKEN>
    
    4. Install cdsapi: pip install "cdsapi>=0.7.7"
    
    For advanced users, consider:
        pip install ecmwf-datastores-client
        
    Documentation: https://cds.climate.copernicus.eu/how-to-api
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
import json

try:
    import cdsapi
    HAS_CDS = True
except ImportError:
    HAS_CDS = False

try:
    import xarray as xr
    import numpy as np
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


@dataclass
class ERA5Variable:
    """ERA5 variable definition."""
    cds_name: str
    short_name: str
    units: str
    description: str
    level_type: str = "single"  # single, pressure


# Available ERA5 variables for hydro-meteorological analysis
ERA5_VARIABLES = {
    # Surface variables
    "precipitation": ERA5Variable(
        cds_name="total_precipitation",
        short_name="tp",
        units="m",
        description="Total precipitation (accumulated)",
    ),
    "temperature_2m": ERA5Variable(
        cds_name="2m_temperature",
        short_name="t2m",
        units="K",
        description="2 metre temperature",
    ),
    "pressure_msl": ERA5Variable(
        cds_name="mean_sea_level_pressure",
        short_name="msl",
        units="Pa",
        description="Mean sea level pressure",
    ),
    "u_wind_10m": ERA5Variable(
        cds_name="10m_u_component_of_wind",
        short_name="u10",
        units="m/s",
        description="10 metre U wind component",
    ),
    "v_wind_10m": ERA5Variable(
        cds_name="10m_v_component_of_wind",
        short_name="v10",
        units="m/s",
        description="10 metre V wind component",
    ),
    "evaporation": ERA5Variable(
        cds_name="evaporation",
        short_name="e",
        units="m",
        description="Evaporation (accumulated)",
    ),
    "runoff": ERA5Variable(
        cds_name="runoff",
        short_name="ro",
        units="m",
        description="Total runoff (accumulated)",
    ),
    "soil_moisture": ERA5Variable(
        cds_name="volumetric_soil_water_layer_1",
        short_name="swvl1",
        units="m3/m3",
        description="Soil moisture (0-7 cm)",
    ),
    "snow_depth": ERA5Variable(
        cds_name="snow_depth",
        short_name="sd",
        units="m",
        description="Snow depth (water equivalent)",
    ),
    
    # Sea surface
    "sst": ERA5Variable(
        cds_name="sea_surface_temperature",
        short_name="sst",
        units="K",
        description="Sea surface temperature",
    ),
    "wave_height": ERA5Variable(
        cds_name="significant_height_of_combined_wind_waves_and_swell",
        short_name="swh",
        units="m",
        description="Significant wave height",
    ),
    
    # Pressure level variables
    "geopotential_500": ERA5Variable(
        cds_name="geopotential",
        short_name="z",
        units="m2/s2",
        description="Geopotential at 500 hPa",
        level_type="pressure",
    ),
    
    # Humidity variables (Issue #7 - GitHub)
    "dewpoint_2m": ERA5Variable(
        cds_name="2m_dewpoint_temperature",
        short_name="d2m",
        units="K",
        description="2 metre dewpoint temperature (proxy for humidity)",
    ),
    "specific_humidity_850": ERA5Variable(
        cds_name="specific_humidity",
        short_name="q",
        units="kg/kg",
        description="Specific humidity at 850 hPa",
        level_type="pressure",
    ),
    "relative_humidity_850": ERA5Variable(
        cds_name="relative_humidity",
        short_name="r",
        units="%",
        description="Relative humidity at 850 hPa",
        level_type="pressure",
    ),
}


# Common variable sets for analysis
VARIABLE_SETS = {
    "flood_analysis": [
        "precipitation", "temperature_2m", "pressure_msl",
        "u_wind_10m", "v_wind_10m", "soil_moisture", "runoff"
    ],
    "storm_surge": [
        "pressure_msl", "u_wind_10m", "v_wind_10m", "wave_height"
    ],
    "drought": [
        "precipitation", "temperature_2m", "evaporation", "soil_moisture"
    ],
    "basic": [
        "precipitation", "temperature_2m", "pressure_msl"
    ],
}


class ERA5Client:
    """
    Client for downloading ERA5 reanalysis data.
    
    Usage:
        client = ERA5Client()
        
        # Download precipitation for Lago Maggiore floods
        ds = await client.download(
            variables=["precipitation", "temperature_2m"],
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
        
        # Use preset variable set
        ds = await client.download_for_flood(
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
    """
    
    def __init__(
        self,
        cache_dir: Path = None,
    ):
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent.parent.parent / "data" / "cache" / "era5"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = None
        self._api_configured = False
        if HAS_CDS:
            try:
                self.client = cdsapi.Client()
                self._api_configured = True
                print("✅ CDS API client initialized")
            except Exception as e:
                print(f"⚠️ CDS API not configured: {e}")
                print("   To configure, create ~/.cdsapirc with:")
                print("   url: https://cds.climate.copernicus.eu/api")
                print("   key: <YOUR-PERSONAL-ACCESS-TOKEN>")
    
    @property
    def is_configured(self) -> bool:
        """Check if CDS API is properly configured."""
        return self._api_configured and self.client is not None
    
    def list_variables(self) -> Dict[str, str]:
        """List available variables."""
        return {k: v.description for k, v in ERA5_VARIABLES.items()}
    
    def list_variable_sets(self) -> Dict[str, List[str]]:
        """List variable sets."""
        return VARIABLE_SETS.copy()
    
    async def download(
        self,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        output_file: Path = None,
        force_download: bool = False,
        hours: List[int] = None,  # [0, 6, 12, 18] for 6-hourly
    ) -> Optional[Any]:
        """
        Download ERA5 data.
        
        Args:
            variables: List of variable keys
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude
            time_range: (start, end) dates as "YYYY-MM-DD"
            output_file: Output NetCDF path
            force_download: Re-download even if cached
            hours: Hours to download (default: all 24)
            
        Returns:
            xarray.Dataset with requested data
        """
        # Default to daily data (00:00)
        hours = hours or list(range(24))
        
        # Generate cache key
        cache_key = self._cache_key(variables, lat_range, lon_range, time_range)
        cache_file = self.cache_dir / f"{cache_key}.nc"
        
        if cache_file.exists() and not force_download:
            print(f"📁 Loading from cache: {cache_file}")
            return xr.open_dataset(cache_file)
        
        # Try CDS API
        if self.client:
            return await self._download_cds(
                variables, lat_range, lon_range, time_range, hours, cache_file
            )
        else:
            print("⚠️ CDS API not available, using fallback")
            return await self._download_fallback(
                variables, lat_range, lon_range, time_range
            )
    
    async def _download_cds(
        self,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        hours: List[int],
        output_file: Path,
    ) -> Optional[Any]:
        """Download using CDS API."""
        # Parse time range
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        
        # Get years, months, days as lists (new API format)
        years = sorted(list(set([str(y) for y in range(start.year, end.year + 1)])))
        
        # For the months/days, be more precise based on the actual date range
        if start.year == end.year:
            months = [f"{m:02d}" for m in range(start.month, end.month + 1)]
        else:
            months = [f"{m:02d}" for m in range(1, 13)]
            
        days = [f"{d:02d}" for d in range(1, 32)]
        
        # Map variable names
        cds_variables = []
        for var in variables:
            if var in ERA5_VARIABLES:
                cds_variables.append(ERA5_VARIABLES[var].cds_name)
            else:
                print(f"⚠️ Unknown variable: {var}")
        
        if not cds_variables:
            return None
        
        # Build request (new CDS API format - uses lists for all multi-value params)
        request = {
            "product_type": ["reanalysis"],
            "data_format": "netcdf",
            "variable": cds_variables,
            "year": years,
            "month": months,
            "day": days,
            "time": [f"{h:02d}:00" for h in hours],
            "area": [lat_range[1], lon_range[0], lat_range[0], lon_range[1]],  # N, W, S, E
        }
        
        print(f"⬇️ Downloading ERA5 data...")
        print(f"   Variables: {cds_variables}")
        print(f"   Area: lat={lat_range}, lon={lon_range}")
        print(f"   Time: {time_range[0]} to {time_range[1]}")
        print(f"   Years: {years}, Months: {months[:3]}...")
        
        try:
            # Download (this blocks, hence in executor)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.retrieve(
                    "reanalysis-era5-single-levels",
                    request,
                    str(output_file),
                )
            )
            
            if output_file.exists():
                print(f"✅ Downloaded: {output_file}")
                return xr.open_dataset(output_file)
            
        except Exception as e:
            print(f"❌ CDS download error: {e}")
            print("   Falling back to synthetic data...")
            return await self._download_fallback(variables, lat_range, lon_range, time_range)
        
        return None
    
    async def _download_fallback(
        self,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Generate synthetic ERA5-like data for testing."""
        if not HAS_XARRAY:
            return None
        
        print("🔧 Generating synthetic ERA5 data for testing...")
        
        # Create coordinates
        resolution = 0.25  # ERA5 resolution
        lats = np.arange(lat_range[0], lat_range[1], resolution)
        lons = np.arange(lon_range[0], lon_range[1], resolution)
        times = np.arange(
            np.datetime64(time_range[0]),
            np.datetime64(time_range[1]) + np.timedelta64(1, 'D'),
            np.timedelta64(1, 'D')
        )
        
        # Create data arrays
        data_vars = {}
        
        for var in variables:
            var_info = ERA5_VARIABLES.get(var)
            if not var_info:
                continue
            
            shape = (len(times), len(lats), len(lons))
            
            if var == "precipitation":
                # Precipitation: exponential distribution, mm/day
                # Add rainy events
                data = np.random.exponential(2, shape)
                # Add random intense precipitation events
                mask = np.random.random(shape) > 0.9
                data[mask] += np.random.exponential(20, mask.sum())
                data = np.clip(data, 0, 200) / 1000  # Convert to m
                
            elif var == "temperature_2m":
                # Temperature: depends on latitude and season
                base = 288 - 0.5 * (lats - lat_range[0])[:, np.newaxis]  # ~15°C
                seasonal = 5 * np.sin(2 * np.pi * np.arange(len(times)) / 365)[:, np.newaxis, np.newaxis]
                noise = np.random.normal(0, 2, shape)
                data = base + seasonal + noise
                
            elif var == "pressure_msl":
                # Pressure: typical range 980-1040 hPa
                data = 101325 + np.random.normal(0, 1500, shape)
                # Add pressure drops for storm events
                storm_mask = np.random.random(shape) > 0.95
                data[storm_mask] -= np.random.uniform(2000, 5000, storm_mask.sum())
                
            elif var in ["u_wind_10m", "v_wind_10m"]:
                # Wind: typical range -20 to 20 m/s
                data = np.random.normal(0, 5, shape)
                # Strong wind events
                mask = np.random.random(shape) > 0.95
                data[mask] = np.random.uniform(-20, 20, mask.sum())
                
            elif var == "soil_moisture":
                # Soil moisture: 0-0.5 m3/m3
                data = np.random.uniform(0.1, 0.4, shape)
                
            elif var == "runoff":
                # Runoff: correlate with precipitation
                if "precipitation" in data_vars:
                    precip = data_vars["precipitation"][1]
                    data = precip * np.random.uniform(0.2, 0.6, shape)
                else:
                    data = np.random.exponential(1, shape) / 1000
                    
            else:
                data = np.random.normal(0, 1, shape)
            
            data_vars[var_info.short_name] = (
                ['time', 'latitude', 'longitude'],
                data.astype(np.float32)
            )
        
        # Create dataset
        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons,
            },
            attrs={
                'source': 'synthetic_era5_fallback',
                'description': 'Synthetic ERA5-like data for testing',
                'warning': 'This is synthetic data, not real ERA5 data',
                'conventions': 'CF-1.6',
            }
        )
        
        return ds
    
    def _cache_key(
        self,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> str:
        """Generate cache key."""
        import hashlib
        
        key_parts = [
            "_".join(sorted(variables)),
            f"{lat_range}",
            f"{lon_range}",
            f"{time_range}",
        ]
        
        key_str = "_".join(key_parts)
        h = hashlib.md5(key_str.encode()).hexdigest()[:12]
        return f"era5_{h}"
    
    # Convenience methods for specific analysis types
    async def download_for_flood(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Download variables for flood analysis."""
        return await self.download(
            variables=VARIABLE_SETS["flood_analysis"],
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        )
    
    async def download_for_storm_surge(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Download variables for storm surge analysis."""
        return await self.download(
            variables=VARIABLE_SETS["storm_surge"],
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        )
    
    async def download_for_drought(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
    ) -> Optional[Any]:
        """Download variables for drought analysis."""
        return await self.download(
            variables=VARIABLE_SETS["drought"],
            lat_range=lat_range,
            lon_range=lon_range,
            time_range=time_range,
        )


# Convenience function
async def download_era5(
    variables: List[str],
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_range: Tuple[str, str],
) -> Optional[Any]:
    """Quick ERA5 download."""
    client = ERA5Client()
    return await client.download(
        variables=variables,
        lat_range=lat_range,
        lon_range=lon_range,
        time_range=time_range,
    )


# CLI test
if __name__ == "__main__":
    async def test():
        client = ERA5Client()
        
        print("=== Available Variables ===")
        for key, desc in client.list_variables().items():
            print(f"  {key}: {desc}")
        
        print("\n=== Variable Sets ===")
        for name, vars in client.list_variable_sets().items():
            print(f"  {name}: {vars}")
        
        print("\n=== Downloading for Flood Analysis (Lago Maggiore 2000) ===")
        ds = await client.download_for_flood(
            lat_range=(44.0, 47.0),
            lon_range=(7.0, 11.0),
            time_range=("2000-10-01", "2000-10-31"),
        )
        
        if ds is not None:
            print(f"✅ Got dataset:")
            print(ds)
    
    asyncio.run(test())
