"""
Configuration management for Surge-Shazam-DK.

Handles variable mapping for heterogeneous data sources (ERA5, DMI, CMEMS).
Inspired by nico/src/core/config.py pattern.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VariableMapping:
    """Maps source-specific variable names to canonical names."""
    
    # Canonical name used internally
    canonical: str
    
    # Source-specific aliases
    era5: str | None = None
    dmi: str | None = None
    cmems: str | None = None
    
    # Units and description
    units: str = ""
    description: str = ""
    
    def get_name(self, source: str) -> str | None:
        """Get variable name for a specific source."""
        return getattr(self, source.lower(), None)


@dataclass
class DatasetConfig:
    """Configuration for a specific dataset/source."""
    
    name: str
    source: str  # "era5", "dmi", "cmems", "noaa"
    
    # API/access config
    api_endpoint: str = ""
    api_key_env: str = ""  # Environment variable name for API key
    
    # Spatial bounds
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    
    # Temporal config
    time_resolution_hours: float = 1.0
    
    # Variables available
    variables: list[str] = field(default_factory=list)
    
    # Extra metadata
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Standard Variable Mappings
# =============================================================================

VARIABLE_MAPPINGS = {
    "wind_u": VariableMapping(
        canonical="wind_u",
        era5="u10",
        dmi="wind_speed_u",
        units="m/s",
        description="Eastward wind component at 10m"
    ),
    "wind_v": VariableMapping(
        canonical="wind_v",
        era5="v10",
        dmi="wind_speed_v",
        units="m/s",
        description="Northward wind component at 10m"
    ),
    "pressure": VariableMapping(
        canonical="pressure",
        era5="msl",
        dmi="mean_sea_level_pressure",
        units="Pa",
        description="Mean sea level pressure"
    ),
    "sea_surface_height": VariableMapping(
        canonical="sea_surface_height",
        era5=None,
        dmi="water_level",
        cmems="ssh",
        units="m",
        description="Sea surface height / water level"
    ),
    "sst": VariableMapping(
        canonical="sst",
        era5="sst",
        cmems="thetao",
        units="K",
        description="Sea surface temperature"
    ),
}


# =============================================================================
# Pre-configured Dataset Configs
# =============================================================================

ERA5_CONFIG = DatasetConfig(
    name="ERA5 Reanalysis",
    source="era5",
    api_endpoint="https://cds.climate.copernicus.eu/api/v2",
    api_key_env="CDSAPI_KEY",
    lat_min=35.0,
    lat_max=65.0,
    lon_min=-30.0,
    lon_max=15.0,
    time_resolution_hours=1.0,
    variables=["wind_u", "wind_v", "pressure", "sst"],
)

DMI_CONFIG = DatasetConfig(
    name="DMI Open Data",
    source="dmi",
    api_endpoint="https://dmigw.govcloud.dk/v2/oceanObs",
    api_key_env="DMI_API_KEY",
    lat_min=54.5,
    lat_max=58.0,
    lon_min=7.5,
    lon_max=15.5,
    time_resolution_hours=0.5,
    variables=["sea_surface_height", "wind_u", "wind_v"],
)

CMEMS_CONFIG = DatasetConfig(
    name="Copernicus Marine",
    source="cmems",
    api_endpoint="https://my.cmems-du.eu/motu-web/Motu",
    api_key_env="CMEMS_USER",
    lat_min=50.0,
    lat_max=62.0,
    lon_min=-5.0,
    lon_max=15.0,
    time_resolution_hours=1.0,
    variables=["sea_surface_height", "sst"],
)


def get_variable_name(canonical: str, source: str) -> str | None:
    """Get source-specific variable name from canonical name."""
    mapping = VARIABLE_MAPPINGS.get(canonical)
    if mapping:
        return mapping.get_name(source)
    return None


def resolve_variables(source: str, variables: list[str]) -> dict[str, str]:
    """
    Resolve list of canonical variable names to source-specific names.
    
    Returns dict: {canonical_name: source_name}
    """
    resolved = {}
    for var in variables:
        source_name = get_variable_name(var, source)
        if source_name:
            resolved[var] = source_name
    return resolved
