"""
Data Loaders Module - Separated from sidebar UI.
Handles loading data from various sources.
"""

from .base import DataLoaderResult, BaseDataLoader
from .slcci_loader import load_slcci_data
from .dtu_loader import load_dtu_data
from .cmems_l4_loader import load_cmems_l4_data

__all__ = [
    "DataLoaderResult",
    "BaseDataLoader",
    "load_slcci_data",
    "load_dtu_data",
    "load_cmems_l4_data",
]
