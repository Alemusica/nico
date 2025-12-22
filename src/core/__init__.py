"""
Core Module - Low-level utilities and base functions
====================================================
"""
from .satellite import detect_satellite_type, get_cycle_info
from .coordinates import wrap_longitudes, lon_in_bounds, get_lon_lat_arrays
from .helpers import extract_cycle_number, extract_strait_info

__all__ = [
    "detect_satellite_type",
    "get_cycle_info", 
    "wrap_longitudes",
    "lon_in_bounds",
    "get_lon_lat_arrays",
    "extract_cycle_number",
    "extract_strait_info",
]
