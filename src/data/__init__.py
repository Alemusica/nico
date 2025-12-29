"""
Data Loading Module
===================
Functions for loading and preprocessing NetCDF satellite data.
Includes multi-satellite data fusion capabilities.
"""
from .loaders import load_cycle, load_multiple_cycles, load_from_upload
from .geoid import interpolate_geoid, add_geoid_to_dataset
from .filters import apply_quality_filter, filter_by_pass
from .satellite_fusion import (
    SatelliteStatus,
    InstrumentType,
    SatelliteConfig,
    SatelliteObservation,
    SatelliteFusionEngine,
    DynamicIndexCalculator,
    SATELLITE_CONFIGS,
)

# New unified loader
try:
    from .unified_loader import UnifiedLoader, CacheManager, load_data
    UNIFIED_LOADER_AVAILABLE = True
except ImportError:
    UNIFIED_LOADER_AVAILABLE = False

__all__ = [
    # Loaders
    "load_cycle",
    "load_multiple_cycles",
    "load_from_upload",
    # Geoid
    "interpolate_geoid",
    "add_geoid_to_dataset",
    # Filters
    "apply_quality_filter",
    "filter_by_pass",
    # Satellite fusion
    "SatelliteStatus",
    "InstrumentType",
    "SatelliteConfig",
    "SatelliteObservation",
    "SatelliteFusionEngine",
    "DynamicIndexCalculator",
    "SATELLITE_CONFIGS",
    # Unified loader (if available)
    "UnifiedLoader",
    "CacheManager",
    "load_data",
]
