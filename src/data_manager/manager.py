"""
📊 Data Manager
===============

Central hub for managing data sources, downloads, and cache.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import os

from .config import (
    SystemConfig, DataSourceConfig, ResolutionConfig,
    TemporalResolution, SpatialResolution, DEFAULT_SOURCES,
)
from .cache import DataCache, CacheEntry

# Import data clients
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from surge_shazam.data.era5_client import ERA5Client
    HAS_ERA5 = True
except ImportError:
    HAS_ERA5 = False
    ERA5Client = None

try:
    from surge_shazam.data.cmems_client import CMEMSClient
    HAS_CMEMS = True
except ImportError:
    HAS_CMEMS = False
    CMEMSClient = None

try:
    from surge_shazam.data.climate_indices import ClimateIndicesClient
    HAS_INDICES = True
except ImportError:
    HAS_INDICES = False
    ClimateIndicesClient = None


@dataclass
class DataRequest:
    """A data request specification."""
    source: str
    variables: List[str]
    lat_range: Tuple[float, float]
    lon_range: Tuple[float, float]
    time_range: Tuple[str, str]
    resolution: ResolutionConfig
    description: str = ""
    estimated_size_mb: float = 0.0
    estimated_time_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "variables": self.variables,
            "lat_range": list(self.lat_range),
            "lon_range": list(self.lon_range),
            "time_range": list(self.time_range),
            "resolution": self.resolution.to_dict(),
            "description": self.description,
            "estimated_size_mb": self.estimated_size_mb,
            "estimated_time_sec": self.estimated_time_sec,
        }


@dataclass
class InvestigationBriefing:
    """
    Data briefing for user confirmation before download.
    
    LLM generates this, user confirms, then download proceeds.
    """
    query: str
    location_name: str
    location_bbox: Tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    event_type: str
    time_period: Tuple[str, str]  # Event period
    precursor_period: Tuple[str, str]  # Extended period for precursors
    
    data_requests: List[DataRequest] = field(default_factory=list)
    
    total_estimated_size_mb: float = 0.0
    total_estimated_time_sec: float = 0.0
    
    # What's already cached
    cached_sources: List[str] = field(default_factory=list)
    needs_download: List[str] = field(default_factory=list)
    
    # User decisions
    confirmed: bool = False
    modified_requests: List[DataRequest] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "location_name": self.location_name,
            "location_bbox": list(self.location_bbox),
            "event_type": self.event_type,
            "time_period": list(self.time_period),
            "precursor_period": list(self.precursor_period),
            "data_requests": [r.to_dict() for r in self.data_requests],
            "total_estimated_size_mb": self.total_estimated_size_mb,
            "total_estimated_time_sec": self.total_estimated_time_sec,
            "cached_sources": self.cached_sources,
            "needs_download": self.needs_download,
            "confirmed": self.confirmed,
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"📍 **Location**: {self.location_name}",
            f"   Bbox: {self.location_bbox[0]:.2f}°-{self.location_bbox[1]:.2f}°N, {self.location_bbox[2]:.2f}°-{self.location_bbox[3]:.2f}°E",
            f"📅 **Event Period**: {self.time_period[0]} to {self.time_period[1]}",
            f"📅 **Analysis Period**: {self.precursor_period[0]} to {self.precursor_period[1]}",
            f"🎯 **Event Type**: {self.event_type}",
            "",
            "📦 **Data Requests**:",
        ]
        
        for req in self.data_requests:
            status = "✅ cached" if req.source in self.cached_sources else "⬇️ download"
            lines.append(f"   • {req.source}: {', '.join(req.variables[:3])}{'...' if len(req.variables) > 3 else ''}")
            lines.append(f"     Resolution: {req.resolution.temporal.value}, {req.resolution.spatial.value}°")
            lines.append(f"     Status: {status}")
            if req.source not in self.cached_sources:
                lines.append(f"     Est. size: {req.estimated_size_mb:.1f} MB, time: {req.estimated_time_sec:.0f}s")
        
        if self.needs_download:
            lines.append("")
            lines.append(f"⏱️ **Total download**: {self.total_estimated_size_mb:.1f} MB, ~{self.total_estimated_time_sec:.0f}s")
        else:
            lines.append("")
            lines.append("✅ **All data cached** - instant analysis!")
        
        return "\n".join(lines)


class DataManager:
    """
    Central data management hub.
    
    Features:
    - Manage data source configurations
    - Check cache before downloading
    - Generate briefings for user confirmation
    - Download with progress callbacks
    - Store downloaded data in cache
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig.load()
        self.cache = DataCache(self.config.cache_dir)
        
        # Initialize clients
        self._clients = {}
        self._init_clients()
    
    def _init_clients(self):
        """Initialize data clients."""
        if HAS_ERA5:
            self._clients["era5"] = ERA5Client(cache_dir=self.config.cache_dir / "era5")
        
        if HAS_CMEMS:
            username = os.environ.get("CMEMS_USERNAME")
            password = os.environ.get("CMEMS_PASSWORD")
            if username and password:
                self._clients["cmems"] = CMEMSClient(
                    username=username,
                    password=password,
                    cache_dir=self.config.cache_dir / "cmems",
                )
        
        if HAS_INDICES:
            self._clients["indices"] = ClimateIndicesClient(
                cache_dir=self.config.cache_dir / "indices"
            )
    
    def get_available_sources(self) -> Dict[str, Dict]:
        """Get available data sources with status."""
        sources = {}
        
        for name, config in self.config.sources.items():
            sources[name] = {
                "name": config.name,
                "description": config.description,
                "enabled": config.enabled,
                "connected": name.split("_")[0] in self._clients or name == "climate_indices",
                "variables": config.variables,
                "default_resolution": config.default_resolution.to_dict(),
                "min_start_date": config.min_start_date,
            }
        
        return sources
    
    def estimate_request(
        self,
        source: str,
        variables: List[str],
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[str, str],
        resolution: ResolutionConfig,
    ) -> Tuple[float, float]:
        """
        Estimate download size and time.
        
        Returns:
            (estimated_size_mb, estimated_time_sec)
        """
        # Calculate dimensions
        lat_span = abs(lat_range[1] - lat_range[0])
        lon_span = abs(lon_range[1] - lon_range[0])
        
        start = datetime.strptime(time_range[0], "%Y-%m-%d")
        end = datetime.strptime(time_range[1], "%Y-%m-%d")
        days = (end - start).days + 1
        
        # Grid points
        lat_points = max(1, int(lat_span / resolution.spatial_deg))
        lon_points = max(1, int(lon_span / resolution.spatial_deg))
        
        # Time points
        time_points = days * len(resolution.hours)
        
        # Total data points
        total_points = lat_points * lon_points * time_points * len(variables)
        
        # Estimate size (4 bytes per float, compressed ~50%)
        size_mb = (total_points * 4 * 0.5) / (1024 * 1024)
        
        # Estimate time based on source
        if source == "era5":
            # CDS queue time + download
            time_sec = 60 + (size_mb * 2)  # 1 min queue + 2 sec/MB
        elif source.startswith("cmems"):
            time_sec = 30 + (size_mb * 3)  # 30s setup + 3 sec/MB
        else:
            time_sec = 5  # Local/fast sources
        
        return round(size_mb, 2), round(time_sec, 0)
    
    def create_briefing(
        self,
        query: str,
        location_name: str,
        location_bbox: Tuple[float, float, float, float],
        event_type: str,
        event_time_range: Tuple[str, str],
        precursor_days: int = 30,
        resolution: ResolutionConfig = None,
    ) -> InvestigationBriefing:
        """
        Create a data briefing for user confirmation.
        
        Args:
            query: Original user query
            location_name: Human-readable location name
            location_bbox: (lat_min, lat_max, lon_min, lon_max)
            event_type: flood, drought, storm_surge, etc.
            event_time_range: (start, end) of event
            precursor_days: Days before event to analyze
            resolution: Desired resolution (uses default if None)
            
        Returns:
            InvestigationBriefing for user confirmation
        """
        resolution = resolution or self.config.investigation_resolution
        
        # Calculate extended time range for precursors
        event_start = datetime.strptime(event_time_range[0], "%Y-%m-%d")
        event_end = datetime.strptime(event_time_range[1], "%Y-%m-%d")
        precursor_start = event_start - timedelta(days=precursor_days)
        
        precursor_range = (
            precursor_start.strftime("%Y-%m-%d"),
            event_end.strftime("%Y-%m-%d"),
        )
        
        lat_range = (location_bbox[0], location_bbox[1])
        lon_range = (location_bbox[2], location_bbox[3])
        
        # Build data requests based on event type
        requests = []
        
        # ERA5 - always include for meteorological context
        if "era5" in self.config.sources and self.config.sources["era5"].enabled:
            era5_vars = self._get_variables_for_event("era5", event_type)
            size, time = self.estimate_request(
                "era5", era5_vars, lat_range, lon_range, precursor_range, resolution
            )
            requests.append(DataRequest(
                source="era5",
                variables=era5_vars,
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=precursor_range,
                resolution=resolution,
                description="Meteorological reanalysis",
                estimated_size_mb=size,
                estimated_time_sec=time,
            ))
        
        # Climate indices - always useful
        if "climate_indices" in self.config.sources:
            requests.append(DataRequest(
                source="climate_indices",
                variables=["nao", "ao", "oni"],
                lat_range=lat_range,  # Not used for indices
                lon_range=lon_range,
                time_range=precursor_range,
                resolution=ResolutionConfig(
                    temporal=TemporalResolution.MONTHLY,
                    spatial=SpatialResolution.COARSE,
                ),
                description="Climate oscillation indices",
                estimated_size_mb=0.01,
                estimated_time_sec=5,
            ))
        
        # Check cache
        cached = []
        needs_download = []
        total_size = 0.0
        total_time = 0.0
        
        for req in requests:
            cache_entry = self.cache.find(
                req.source, req.variables, req.lat_range, req.lon_range,
                req.time_range, req.resolution.temporal.value, req.resolution.spatial.value
            )
            
            if cache_entry:
                cached.append(req.source)
            else:
                needs_download.append(req.source)
                total_size += req.estimated_size_mb
                total_time += req.estimated_time_sec
        
        return InvestigationBriefing(
            query=query,
            location_name=location_name,
            location_bbox=location_bbox,
            event_type=event_type,
            time_period=event_time_range,
            precursor_period=precursor_range,
            data_requests=requests,
            total_estimated_size_mb=total_size,
            total_estimated_time_sec=total_time,
            cached_sources=cached,
            needs_download=needs_download,
        )
    
    def _get_variables_for_event(self, source: str, event_type: str) -> List[str]:
        """Get relevant variables for an event type."""
        if source == "era5":
            base = ["total_precipitation", "2m_temperature", "mean_sea_level_pressure"]
            
            if event_type in ["flood", "flash_flood"]:
                return base + ["runoff", "volumetric_soil_water_layer_1"]
            elif event_type in ["storm_surge", "coastal_flood"]:
                return base + ["10m_u_component_of_wind", "10m_v_component_of_wind"]
            elif event_type == "drought":
                return base + ["evaporation", "volumetric_soil_water_layer_1"]
            else:
                return base + ["10m_u_component_of_wind", "10m_v_component_of_wind"]
        
        elif source.startswith("cmems"):
            if "sla" in source:
                return ["sla", "adt"]
            elif "sst" in source:
                return ["analysed_sst"]
        
        return self.config.sources.get(source, DataSourceConfig(name="")).variables
    
    async def download_briefing(
        self,
        briefing: InvestigationBriefing,
        progress_callback: Callable[[str, float, str], None] = None,
    ) -> Dict[str, Any]:
        """
        Download all data from a confirmed briefing.
        
        Args:
            briefing: Confirmed InvestigationBriefing
            progress_callback: Called with (source, progress_pct, message)
            
        Returns:
            Dict mapping source name to downloaded data
        """
        results = {}
        
        for i, req in enumerate(briefing.data_requests):
            source = req.source
            
            # Check cache first
            if source in briefing.cached_sources:
                if progress_callback:
                    progress_callback(source, 100.0, f"✅ Loading from cache")
                
                cache_entry = self.cache.find(
                    source, req.variables, req.lat_range, req.lon_range,
                    req.time_range
                )
                if cache_entry:
                    results[source] = self.cache.load(cache_entry)
                continue
            
            # Download
            if progress_callback:
                progress_callback(source, 0.0, f"⬇️ Starting download...")
            
            try:
                data = await self._download_source(req, progress_callback)
                
                if data is not None:
                    # Cache the data
                    self.cache.add(
                        source=source,
                        variables=req.variables,
                        lat_range=req.lat_range,
                        lon_range=req.lon_range,
                        time_range=req.time_range,
                        resolution_temporal=req.resolution.temporal.value,
                        resolution_spatial=req.resolution.spatial.value,
                        data=data,
                    )
                    results[source] = data
                    
                    if progress_callback:
                        progress_callback(source, 100.0, f"✅ Downloaded and cached")
                else:
                    if progress_callback:
                        progress_callback(source, 100.0, f"⚠️ No data available")
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(source, 100.0, f"❌ Error: {str(e)}")
                results[source] = None
        
        return results
    
    async def _download_source(
        self,
        req: DataRequest,
        progress_callback: Callable = None,
    ) -> Any:
        """Download data from a specific source."""
        source_base = req.source.split("_")[0]
        
        if source_base == "era5" and "era5" in self._clients:
            client = self._clients["era5"]
            
            # Map variable names
            var_mapping = {
                "total_precipitation": "precipitation",
                "2m_temperature": "temperature_2m",
                "mean_sea_level_pressure": "pressure_msl",
                "10m_u_component_of_wind": "u_wind_10m",
                "10m_v_component_of_wind": "v_wind_10m",
                "volumetric_soil_water_layer_1": "soil_moisture",
                "runoff": "runoff",
                "evaporation": "evaporation",
            }
            
            variables = [var_mapping.get(v, v) for v in req.variables]
            
            return await client.download(
                variables=variables,
                lat_range=req.lat_range,
                lon_range=req.lon_range,
                time_range=req.time_range,
                hours=req.resolution.hours,
            )
        
        elif source_base == "cmems" and "cmems" in self._clients:
            client = self._clients["cmems"]
            # Determine dataset based on variables
            dataset = "sea_level_global" if "sla" in req.variables else "sst_global"
            
            return await client.download(
                dataset_id=dataset,
                variables=req.variables,
                lat_range=req.lat_range,
                lon_range=req.lon_range,
                time_range=req.time_range,
            )
        
        elif req.source == "climate_indices" and "indices" in self._clients:
            client = self._clients["indices"]
            return await client.get_indices(
                indices=req.variables,
                time_range=req.time_range,
            )
        
        return None
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def list_cached_data(self, source: str = None) -> List[Dict]:
        """List cached data entries."""
        entries = self.cache.list_entries(source)
        return [e.to_dict() for e in entries]
    
    def clear_cache(self, source: str = None, older_than_days: int = None):
        """Clear cache entries."""
        self.cache.clear(source, older_than_days)
    
    def update_resolution(
        self,
        temporal: str = None,
        spatial: str = None,
    ):
        """Update default investigation resolution."""
        if temporal:
            self.config.investigation_resolution.temporal = TemporalResolution(temporal)
        if spatial:
            self.config.investigation_resolution.spatial = SpatialResolution(spatial)
        self.config.save()
    
    def enable_source(self, source: str, enabled: bool = True):
        """Enable or disable a data source."""
        if source in self.config.sources:
            self.config.sources[source].enabled = enabled
            self.config.save()
