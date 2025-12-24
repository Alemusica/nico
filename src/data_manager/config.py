"""
🔧 Data Source Configuration
============================

User-configurable settings for data sources and resolutions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum
import json
from pathlib import Path


class TemporalResolution(str, Enum):
    """Available temporal resolutions."""
    HOURLY = "hourly"           # Every hour
    THREE_HOURLY = "3-hourly"   # Every 3 hours
    SIX_HOURLY = "6-hourly"     # Every 6 hours (00, 06, 12, 18)
    DAILY = "daily"             # Once per day
    MONTHLY = "monthly"         # Monthly mean
    

class SpatialResolution(str, Enum):
    """Available spatial resolutions."""
    HIGH = "0.1"       # 0.1° (~11 km)
    MEDIUM = "0.25"    # 0.25° (~28 km) - ERA5 native
    LOW = "0.5"        # 0.5° (~55 km)
    COARSE = "1.0"     # 1.0° (~111 km)


@dataclass
class ResolutionConfig:
    """Resolution configuration for a data request."""
    temporal: TemporalResolution = TemporalResolution.DAILY
    spatial: SpatialResolution = SpatialResolution.MEDIUM
    
    @property
    def hours(self) -> List[int]:
        """Get hours to request based on temporal resolution."""
        if self.temporal == TemporalResolution.HOURLY:
            return list(range(24))
        elif self.temporal == TemporalResolution.THREE_HOURLY:
            return [0, 3, 6, 9, 12, 15, 18, 21]
        elif self.temporal == TemporalResolution.SIX_HOURLY:
            return [0, 6, 12, 18]
        elif self.temporal == TemporalResolution.DAILY:
            return [12]  # Noon
        else:  # Monthly
            return [12]
    
    @property
    def spatial_deg(self) -> float:
        """Get spatial resolution in degrees."""
        return float(self.spatial.value)
    
    def to_dict(self) -> Dict:
        return {
            "temporal": self.temporal.value,
            "spatial": self.spatial.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ResolutionConfig":
        return cls(
            temporal=TemporalResolution(data.get("temporal", "daily")),
            spatial=SpatialResolution(data.get("spatial", "0.25")),
        )


@dataclass  
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    enabled: bool = True
    api_url: str = ""
    credentials: Dict[str, str] = field(default_factory=dict)
    default_resolution: ResolutionConfig = field(default_factory=ResolutionConfig)
    variables: List[str] = field(default_factory=list)
    description: str = ""
    
    # API limits
    max_items_per_request: int = 100000
    max_area_deg2: float = 360.0  # Max area in square degrees
    min_start_date: str = "1940-01-01"  # Earliest available data
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "api_url": self.api_url,
            "credentials": {k: "***" for k in self.credentials},  # Hide credentials
            "default_resolution": self.default_resolution.to_dict(),
            "variables": self.variables,
            "description": self.description,
            "max_items_per_request": self.max_items_per_request,
            "max_area_deg2": self.max_area_deg2,
            "min_start_date": self.min_start_date,
        }


# Default configurations for known data sources
DEFAULT_SOURCES: Dict[str, DataSourceConfig] = {
    "era5": DataSourceConfig(
        name="ERA5 Reanalysis",
        api_url="https://cds.climate.copernicus.eu/api",
        description="ECMWF ERA5 atmospheric reanalysis (1940-present)",
        default_resolution=ResolutionConfig(
            temporal=TemporalResolution.SIX_HOURLY,
            spatial=SpatialResolution.MEDIUM,
        ),
        variables=[
            "total_precipitation",
            "2m_temperature", 
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "volumetric_soil_water_layer_1",
            "runoff",
        ],
        max_items_per_request=120000,
        min_start_date="1940-01-01",
    ),
    "cmems_sla": DataSourceConfig(
        name="CMEMS Sea Level",
        api_url="https://data.marine.copernicus.eu/api",
        description="Copernicus Marine sea level anomaly",
        default_resolution=ResolutionConfig(
            temporal=TemporalResolution.DAILY,
            spatial=SpatialResolution.MEDIUM,
        ),
        variables=["sla", "adt"],
        max_items_per_request=50000,
        min_start_date="1993-01-01",
    ),
    "cmems_sst": DataSourceConfig(
        name="CMEMS SST",
        api_url="https://data.marine.copernicus.eu/api",
        description="Copernicus Marine sea surface temperature",
        default_resolution=ResolutionConfig(
            temporal=TemporalResolution.DAILY,
            spatial=SpatialResolution.HIGH,
        ),
        variables=["analysed_sst"],
        max_items_per_request=50000,
        min_start_date="1982-01-01",
    ),
    "climate_indices": DataSourceConfig(
        name="Climate Indices",
        api_url="https://psl.noaa.gov/data/",
        description="NAO, AO, ONI, AMO, PDO, EA, SCAND",
        default_resolution=ResolutionConfig(
            temporal=TemporalResolution.MONTHLY,
            spatial=SpatialResolution.COARSE,  # Global indices, no spatial
        ),
        variables=["nao", "ao", "oni", "amo", "pdo", "ea", "scand"],
        max_items_per_request=1000000,  # No real limit
        min_start_date="1950-01-01",
    ),
}


@dataclass
class SystemConfig:
    """Global system configuration."""
    cache_dir: Path = field(default_factory=lambda: Path("data/cache"))
    sources: Dict[str, DataSourceConfig] = field(default_factory=lambda: DEFAULT_SOURCES.copy())
    
    # Default resolution for investigations
    investigation_resolution: ResolutionConfig = field(
        default_factory=lambda: ResolutionConfig(
            temporal=TemporalResolution.DAILY,
            spatial=SpatialResolution.MEDIUM,
        )
    )
    
    def save(self, path: Path = None):
        """Save configuration to JSON."""
        path = path or (self.cache_dir.parent / "config" / "data_sources.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "cache_dir": str(self.cache_dir),
            "investigation_resolution": self.investigation_resolution.to_dict(),
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path = None) -> "SystemConfig":
        """Load configuration from JSON."""
        path = path or Path("data/config/data_sources.json")
        
        if not path.exists():
            return cls()
        
        with open(path) as f:
            data = json.load(f)
        
        config = cls(
            cache_dir=Path(data.get("cache_dir", "data/cache")),
            investigation_resolution=ResolutionConfig.from_dict(
                data.get("investigation_resolution", {})
            ),
        )
        
        # Load sources (keeping defaults for missing)
        for name, source_data in data.get("sources", {}).items():
            if name in config.sources:
                config.sources[name].enabled = source_data.get("enabled", True)
                if "default_resolution" in source_data:
                    config.sources[name].default_resolution = ResolutionConfig.from_dict(
                        source_data["default_resolution"]
                    )
        
        return config
