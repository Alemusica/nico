"""
Services Layer
==============
Business logic services shared by API and Streamlit.

This layer provides:
- GateService: Gate selection and spatial operations
- DataService: Data loading and filtering
- AnalysisService: Analysis operations
- SLCCIService: ESA Sea Level CCI data loading and processing (along-track)
- CMEMSService: CMEMS L3 along-track data loading (track variable)
- CMEMSL4Service: CMEMS L4 gridded data loading via API
- DTUService: DTUSpace v4 gridded DOT data loading and processing
- DataCache: Persistent cache for processed PassData objects
- BathymetryService: GEBCO bathymetry extraction along gates
- VolumeTransport: Volume transport calculations from geostrophic velocity

Dataset Comparison:
    | Dataset      | Type        | Filter Variable | Source    | DOI                                    |
    |--------------|-------------|-----------------|-----------|----------------------------------------|
    | SLCCI        | Along-track | pass            | Local     | ESA CCI                                |
    | CMEMS L3     | Along-track | track           | Local     | https://doi.org/10.48670/moi-00149     |
    | CMEMS L4     | Gridded     | (none)          | API       | https://doi.org/10.48670/moi-00148     |
    | DTUSpace     | Gridded     | (none)          | Local     | DTU Space                              |

Usage:
    from src.services import GateService, DataService, SLCCIService, DTUService
    from src.services import CMEMSService, CMEMSL4Service
    from src.services import DataCache, get_cache, BathymetryService
    
    gate_service = GateService()
    bbox = gate_service.get_bbox("fram_strait")
    
    # Along-track datasets
    slcci_service = SLCCIService()
    pass_data = slcci_service.load_pass_data(gate_path, pass_number)
    
    cmems_l3_service = CMEMSService()  # Along-track with track selection
    track_data = cmems_l3_service.load_pass_data(config)
    
    # Gridded datasets
    dtu_service = DTUService()
    dtu_data = dtu_service.load_gate_data(nc_path, gate_path)
    
    cmems_l4_service = CMEMSL4Service()  # Gridded via API
    l4_data = cmems_l4_service.load_gate_data(config)
    
    # Cache and utilities
    cache = get_cache()
    if cache.exists("slcci", "fram_strait", pass_number=248):
        pass_data = cache.load("slcci", "fram_strait", pass_number=248)
"""

from src.services.gate_service import GateService
from src.services.data_service import DataService
from src.services.analysis_service import AnalysisService
from src.services.slcci_service import SLCCIService, SLCCIConfig, PassData, CacheConfig, SLCCICache
from src.services.cmems_service import CMEMSService, CMEMSConfig
from src.services.cmems_service import PassData as CMEMSPassData
from src.services.cmems_l4_service import CMEMSL4Service, CMEMSL4Config, CMEMSL4PassData
from src.services.dtu_service import DTUService, DTUConfig, DTUPassData

# Intelligent cache (shared across services)
from src.services.intelligent_cache import (
    IntelligentCache,
    CacheConfig as IntelligentCacheConfig,
    get_intelligent_cache,
    clear_global_cache,
    save_global_cache,
)

# Bathymetry and Transport services
from src.services.bathymetry_service import BathymetryService, BathymetryProfile
from src.services.transport_service import (
    VolumeTransportResult,
    compute_perpendicular_velocity,
    compute_segment_widths,
    calculate_volume_transport,
)

# Legacy cache - DEPRECATED, use IntelligentCache instead
# Keeping imports for backward compatibility but will be removed
try:
    from src.services.cache_service import DataCache, get_cache, cache_pass_data, load_cached_pass_data, is_cached
    _LEGACY_CACHE_AVAILABLE = True
except ImportError:
    _LEGACY_CACHE_AVAILABLE = False
    # Create stubs for backward compatibility
    DataCache = IntelligentCache  # Alias to new cache
    def get_cache(): return get_intelligent_cache()
    def cache_pass_data(*args, **kwargs): pass
    def load_cached_pass_data(*args, **kwargs): return None
    def is_cached(*args, **kwargs): return False

__all__ = [
    # Core services
    "GateService",
    "DataService",
    "AnalysisService",
    # Along-track datasets
    "SLCCIService",
    "SLCCIConfig",
    "PassData",
    "CacheConfig",  # SLCCI intelligent cache configuration
    "SLCCICache",   # SLCCI intelligent cache
    "CMEMSService",  # L3 along-track
    "CMEMSConfig",
    "CMEMSPassData",
    # Gridded datasets
    "CMEMSL4Service",  # L4 gridded via API
    "CMEMSL4Config",
    "CMEMSL4PassData",
    "DTUService",
    "DTUConfig",
    "DTUPassData",
    # Intelligent cache (shared) - PRIMARY CACHE SYSTEM
    "IntelligentCache",
    "IntelligentCacheConfig",
    "get_intelligent_cache",
    "clear_global_cache",
    "save_global_cache",
    # Legacy cache - DEPRECATED (kept for backward compatibility)
    "DataCache",
    "get_cache",
    "cache_pass_data",
    "load_cached_pass_data",
    "is_cached",
    # Bathymetry service
    "BathymetryService",
    "BathymetryProfile",
    # Transport service
    "VolumeTransportResult",
    "compute_perpendicular_velocity",
    "compute_segment_widths",
    "calculate_volume_transport",
]
