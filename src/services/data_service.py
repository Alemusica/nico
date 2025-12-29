"""
Data Service
============
Business logic for data loading and filtering.
Unified interface for Copernicus, ERA5, and other datasets.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

import yaml

from src.core.models import (
    BoundingBox,
    TimeRange,
    DataRequest,
    TemporalResolution,
    SpatialResolution,
    GateModel
)

logger = logging.getLogger(__name__)


class DataService:
    """
    Service for data loading and filtering.
    
    Provides a unified interface for:
    - Building data requests
    - Loading from Copernicus/ERA5/CMEMS
    - Filtering by time/space
    - Caching results
    
    Example:
        >>> service = DataService()
        >>> request = service.build_request(
        ...     variables=["sla", "adt"],
        ...     bbox=bbox,
        ...     time_range=time_range
        ... )
        >>> data = service.load(request)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize data service.
        
        Args:
            config_path: Path to datasets.yaml configuration
        """
        self.config_path = Path(config_path) if config_path else Path("config/datasets.yaml")
        self._datasets: Dict[str, Dict] = {}
        self._defaults: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load datasets configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self._datasets = config.get("datasets", {})
                self._defaults = config.get("defaults", {})
                logger.info(f"Loaded {len(self._datasets)} datasets from config")
        else:
            logger.warning(f"Config not found: {self.config_path}")
    
    def list_datasets(self) -> List[str]:
        """List available datasets."""
        return list(self._datasets.keys())
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset configuration."""
        return self._datasets.get(dataset_id)
    
    def get_variables(self, dataset_id: str) -> List[str]:
        """Get variables for a dataset."""
        ds = self._datasets.get(dataset_id, {})
        return ds.get("variables", [])
    
    def build_request(
        self,
        gate: Optional[GateModel] = None,
        bbox: Optional[BoundingBox] = None,
        time_range: Optional[TimeRange] = None,
        variables: Optional[List[str]] = None,
        dataset_id: str = "cmems_sla",
        temporal_resolution: TemporalResolution = TemporalResolution.DAILY,
        spatial_resolution: SpatialResolution = SpatialResolution.MEDIUM
    ) -> DataRequest:
        """
        Build a data request.
        
        Args:
            gate: Gate to use for bbox (optional)
            bbox: Bounding box (optional if gate provided)
            time_range: Time range for data
            variables: Variables to load
            dataset_id: Dataset identifier
            temporal_resolution: Time resolution
            spatial_resolution: Spatial resolution
            
        Returns:
            DataRequest ready for loading
        """
        # Use gate bbox if no explicit bbox
        if bbox is None and gate:
            bbox = BoundingBox(
                lat_min=gate.lat_min,
                lat_max=gate.lat_max,
                lon_min=gate.lon_min,
                lon_max=gate.lon_max
            )
        
        # Default bbox for Arctic
        if bbox is None:
            bbox = BoundingBox(
                lat_min=self._defaults.get("lat_min", 60.0),
                lat_max=self._defaults.get("lat_max", 85.0),
                lon_min=self._defaults.get("lon_min", -180.0),
                lon_max=self._defaults.get("lon_max", 180.0)
            )
        
        # Default time range: last 30 days
        if time_range is None:
            end = datetime.now()
            start = end - timedelta(days=30)
            time_range = TimeRange(start=start, end=end)
        
        # Get variables from dataset config if not specified
        if variables is None:
            variables = self.get_variables(dataset_id) or ["sla"]
        
        return DataRequest(
            bbox=bbox,
            time_range=time_range,
            variables=variables,
            dataset_id=dataset_id,
            temporal_resolution=temporal_resolution,
            spatial_resolution=spatial_resolution
        )
    
    def load(self, request: DataRequest) -> Optional[Any]:
        """
        Load data based on request.
        
        This method routes to the appropriate data loader based on dataset_id.
        
        Args:
            request: DataRequest with parameters
            
        Returns:
            xarray Dataset or None if loading fails
        """
        dataset_id = request.dataset_id
        
        # Route to appropriate loader
        if "cmems" in dataset_id or "copernicus" in dataset_id:
            return self._load_cmems(request)
        elif "era5" in dataset_id:
            return self._load_era5(request)
        else:
            logger.warning(f"Unknown dataset type: {dataset_id}")
            return None
    
    def _load_cmems(self, request: DataRequest) -> Optional[Any]:
        """
        Load data from CMEMS/Copernicus.
        
        Uses existing cmems_client if available.
        """
        try:
            # Import existing client
            from src.surge_shazam.data.cmems_client import CMEMSClient
            
            client = CMEMSClient()
            return client.load(
                dataset_id=request.dataset_id,
                variables=request.variables,
                bbox=(
                    request.bbox.lon_min,
                    request.bbox.lat_min,
                    request.bbox.lon_max,
                    request.bbox.lat_max
                ),
                time_range=(request.time_range.start, request.time_range.end)
            )
        except ImportError:
            logger.warning("CMEMSClient not available")
            return self._load_mock_data(request)
        except Exception as e:
            logger.error(f"CMEMS load error: {e}")
            return None
    
    def _load_era5(self, request: DataRequest) -> Optional[Any]:
        """
        Load data from ERA5.
        
        Uses existing era5_client if available.
        """
        try:
            # Import existing client
            from src.surge_shazam.data.era5_client import ERA5Client
            
            client = ERA5Client()
            return client.load(
                variables=request.variables,
                bbox=(
                    request.bbox.lon_min,
                    request.bbox.lat_min,
                    request.bbox.lon_max,
                    request.bbox.lat_max
                ),
                time_range=(request.time_range.start, request.time_range.end)
            )
        except ImportError:
            logger.warning("ERA5Client not available")
            return self._load_mock_data(request)
        except Exception as e:
            logger.error(f"ERA5 load error: {e}")
            return None
    
    def _load_mock_data(self, request: DataRequest) -> Optional[Any]:
        """
        Generate mock data for testing.
        
        Returns an xarray-like structure with random data.
        """
        try:
            import numpy as np
            import xarray as xr
            
            # Generate coordinate arrays
            lats = np.linspace(
                request.bbox.lat_min,
                request.bbox.lat_max,
                50
            )
            lons = np.linspace(
                request.bbox.lon_min,
                request.bbox.lon_max,
                50
            )
            
            # Generate time array
            times = np.arange(
                np.datetime64(request.time_range.start.isoformat()[:10]),
                np.datetime64(request.time_range.end.isoformat()[:10]),
                np.timedelta64(1, 'D')
            )
            
            # Create data variables
            data_vars = {}
            for var in request.variables:
                data = np.random.randn(len(times), len(lats), len(lons)) * 0.1
                data_vars[var] = (["time", "lat", "lon"], data)
            
            ds = xr.Dataset(
                data_vars,
                coords={
                    "time": times,
                    "lat": lats,
                    "lon": lons
                },
                attrs={
                    "dataset_id": request.dataset_id,
                    "source": "mock_data",
                    "generated": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Generated mock data: {ds.dims}")
            return ds
            
        except ImportError:
            logger.error("numpy/xarray not available for mock data")
            return None
    
    def filter_by_passes(
        self,
        data: Any,
        passes: List[int],
        orbit_cycle_days: int = 10
    ) -> Optional[Any]:
        """
        Filter data by satellite passes.
        
        This is a simplified filter - real implementation would use
        along-track filtering based on pass geometry.
        
        Args:
            data: xarray Dataset
            passes: List of pass numbers
            orbit_cycle_days: Days in orbit cycle
            
        Returns:
            Filtered dataset
        """
        if data is None:
            return None
        
        # Placeholder - real implementation would filter
        # based on satellite ground tracks
        logger.info(f"Pass filtering requested for passes: {passes}")
        return data
    
    def get_data_latency(self, dataset_id: str) -> Optional[str]:
        """Get data latency for dataset."""
        ds = self._datasets.get(dataset_id, {})
        return ds.get("latency")
    
    def is_nrt_available(self, dataset_id: str) -> bool:
        """Check if near-real-time data is available."""
        latency = self.get_data_latency(dataset_id)
        if latency:
            # Parse latency string
            if "hour" in latency:
                return True
            elif "day" in latency:
                # NRT if less than 3 days
                days = int(latency.split()[0])
                return days < 3
        return False
