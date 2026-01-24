"""
Data Clients Module - "The Garage"
==================================

Unified data access layer with standard interface.

All clients inherit from DataClient base class for consistent:
- Method signatures (download, list_products, health_check)
- Return types (xarray.Dataset or pandas.DataFrame)
- Error handling (DataClientError)
- Caching behavior

See docs/DEVELOPMENT_APPROACH.md for the "Garage First" philosophy.
"""

# Base client interface (the "parking spot" contract)
from .base_client import (
    DataClient,
    DataFormat,
    ClientStatus,
    BoundingBox,
    TimeRange,
    HealthCheckResult,
    DataClientError,
    XArrayClientMixin,
    DataFrameClientMixin,
    PointDataClientMixin,
)

# Concrete clients (the "cars")
from .cmems_client import CMEMSClient, CMEMS_DATASETS, download_cmems
from .era5_client import ERA5Client, ERA5_VARIABLES, VARIABLE_SETS, download_era5
from .climate_indices import ClimateIndicesClient, CLIMATE_INDICES, get_climate_indices_for_event

__all__ = [
    # Base interface
    "DataClient",
    "DataFormat",
    "ClientStatus",
    "BoundingBox",
    "TimeRange",
    "HealthCheckResult",
    "DataClientError",
    "XArrayClientMixin",
    "DataFrameClientMixin",
    "PointDataClientMixin",
    # CMEMS
    "CMEMSClient",
    "CMEMS_DATASETS",
    "download_cmems",
    # ERA5
    "ERA5Client",
    "ERA5_VARIABLES",
    "VARIABLE_SETS",
    "download_era5",
    # Climate Indices
    "ClimateIndicesClient",
    "CLIMATE_INDICES",
    "get_climate_indices_for_event",
]
