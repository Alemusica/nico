"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DATA CLIENTS MODULE                                        ║
║                                                                              ║
║   Unified data access layer for all data sources in the Early Warning System.║
║                                                                              ║
║   Categories:                                                                ║
║   - Satellite: CMEMS, ERA5, CYGNSS, GPM, GRACE, Sentinel                     ║
║   - Aircraft: AMDAR, Mode-S, OpenSky                                         ║
║   - In-situ: Tide Gauges, ARGO, Weather Stations                             ║
║   - Indices: Climate teleconnections (NAO, ENSO, AMO...)                     ║
║                                                                              ║
║   Usage:                                                                     ║
║       from src.surge_shazam.data import API_REGISTRY, CMEMSClient            ║
║       from src.surge_shazam.data import get_precipitation, get_tide_data     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# API Registry - Central catalog
from .api_registry import (
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

# Satellite clients
from .cmems_client import CMEMSClient, CMEMS_DATASETS, download_cmems
from .era5_client import ERA5Client, ERA5_VARIABLES, VARIABLE_SETS, download_era5
from .cygnss_client import CYGNSSClient

# Precipitation
from .gpm_client import GPMClient, GPM_PRODUCTS, get_precipitation

# Climate indices
from .climate_indices import ClimateIndicesClient, CLIMATE_INDICES, get_climate_indices_for_event

# Aircraft data
from .aircraft_client import (
    AircraftClient,
    AircraftObservation,
    VerticalProfile,
    OpenSkyModeS,
    MADISClient,
    get_aircraft_data,
)

# Tide gauges
from .tide_gauge_client import (
    TideGaugeClient,
    TideGaugeStation,
    TideGaugeObservation,
    IOCSeaLevelClient,
    EUROPEAN_STATIONS,
    get_tide_data,
)

__all__ = [
    # Registry
    "API_REGISTRY",
    "DataSource",
    "DataCategory",
    "Latency",
    "Status",
    "get_sources_by_category",
    "get_sources_by_status",
    "get_sources_by_latency",
    "get_high_priority_sources",
    "get_physics_variables_map",
    
    # CMEMS
    "CMEMSClient",
    "CMEMS_DATASETS",
    "download_cmems",
    
    # ERA5
    "ERA5Client",
    "ERA5_VARIABLES",
    "VARIABLE_SETS",
    "download_era5",
    
    # CYGNSS
    "CYGNSSClient",
    
    # GPM
    "GPMClient",
    "GPM_PRODUCTS",
    "get_precipitation",
    
    # Climate Indices
    "ClimateIndicesClient",
    "CLIMATE_INDICES",
    "get_climate_indices_for_event",
    
    # Aircraft
    "AircraftClient",
    "AircraftObservation",
    "VerticalProfile",
    "OpenSkyModeS",
    "MADISClient",
    "get_aircraft_data",
    
    # Tide Gauges
    "TideGaugeClient",
    "TideGaugeStation",
    "TideGaugeObservation",
    "IOCSeaLevelClient",
    "EUROPEAN_STATIONS",
    "get_tide_data",
]
