"""
📊 Data Manager
===============

Centralized data management system for:
- API connections (CMEMS, ERA5, Climate Indices)
- Cache management
- Resolution configuration
- Data inventory
"""

from .manager import DataManager
from .cache import DataCache
from .config import DataSourceConfig, ResolutionConfig

__all__ = ["DataManager", "DataCache", "DataSourceConfig", "ResolutionConfig"]
