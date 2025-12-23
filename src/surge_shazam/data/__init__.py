"""
Data Clients Module
==================

CMEMS, ERA5, and Climate Indices clients for data download.
"""

from .cmems_client import CMEMSClient, CMEMS_DATASETS, download_cmems
from .era5_client import ERA5Client, ERA5_VARIABLES, VARIABLE_SETS, download_era5
from .climate_indices import ClimateIndicesClient, CLIMATE_INDICES, get_climate_indices_for_event

__all__ = [
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
