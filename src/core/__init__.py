"""
Core Module - Low-level utilities and base functions
====================================================
"""
from .satellite import detect_satellite_type, get_cycle_info
from .coordinates import wrap_longitudes, lon_in_bounds, get_lon_lat_arrays
from .helpers import extract_cycle_number, extract_strait_info
from .config import (
    DatasetConfig,
    DatasetFormat,
    VariableMapping,
    CoordinateMapping,
    get_dataset_config,
    register_dataset_config,
    list_supported_formats,
    DATASET_CONFIGS,
)
from .resolver import VariableResolver, auto_load, compare_formats

# New unified models
try:
    from .models import (
        BoundingBox,
        TimeRange,
        GateModel,
        DataRequest,
        TemporalResolution,
        SpatialResolution,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

__all__ = [
    # Satellite
    "detect_satellite_type",
    "get_cycle_info", 
    # Coordinates
    "wrap_longitudes",
    "lon_in_bounds",
    "get_lon_lat_arrays",
    # Helpers
    "extract_cycle_number",
    "extract_strait_info",
    # Config
    "DatasetConfig",
    "DatasetFormat",
    "VariableMapping",
    "CoordinateMapping",
    "get_dataset_config",
    "register_dataset_config",
    "list_supported_formats",
    "DATASET_CONFIGS",
    # Resolver
    "VariableResolver",
    "auto_load",
    "compare_formats",
    # Models (if available)
    "BoundingBox",
    "TimeRange",
    "GateModel",
    "DataRequest",
    "TemporalResolution",
    "SpatialResolution",
]
