"""
📊 Data Manager
===============

Centralized data management system for:
- API connections (CMEMS, ERA5, Climate Indices)
- Cache management
- Resolution configuration
- Data inventory
- Dataset catalog (Copernicus Marine)
"""

from .manager import DataManager
from .cache import DataCache
from .config import DataSourceConfig, ResolutionConfig
from .catalog import (
    CopernicusCatalog,
    DataCategory,
    DataProduct,
    Variable,
    SpatialCoverage,
    TemporalCoverage,
    get_catalog,
    search_products,
    check_availability,
)

__all__ = [
    "DataManager", 
    "DataCache", 
    "DataSourceConfig", 
    "ResolutionConfig",
    # Catalog
    "CopernicusCatalog",
    "DataCategory", 
    "DataProduct",
    "Variable",
    "SpatialCoverage",
    "TemporalCoverage",
    "get_catalog",
    "search_products",
    "check_availability",
]
