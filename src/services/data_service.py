"""
Data Service
============
Business logic for data loading and filtering.
Unified interface for Copernicus, ERA5, local files, and other datasets.

This service follows the NICO Unified Architecture:
- Called by UI (Streamlit/React) and API
- Uses GateService for spatial bounds
- Routes to appropriate loader based on dataset type
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import os
import glob

import yaml
import numpy as np
import xarray as xr

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
        
        Routes to the appropriate data loader based on:
        1. Dataset configuration in config/datasets.yaml
        2. Dataset ID prefix (cmems_, era5_, local_, etc.)
        
        Args:
            request: DataRequest with parameters
            
        Returns:
            xarray Dataset or None if loading fails
        """
        dataset_id = request.dataset_id
        logger.info(f"Loading dataset: {dataset_id}")
        
        # Check if dataset is configured
        dataset_config = self._datasets.get(dataset_id)
        
        if dataset_config:
            provider = dataset_config.get("provider", "")
            logger.info(f"Dataset {dataset_id} uses provider: {provider}")
            
            # Route based on provider from config
            if "copernicus" in provider or "cmems" in provider:
                return self._load_cmems(request, dataset_config)
            elif "ecmwf" in provider or "era5" in provider.lower():
                return self._load_era5(request, dataset_config)
            elif "noaa" in provider:
                return self._load_noaa(request, dataset_config)
            elif "nasa" in provider or "podaac" in provider:
                return self._load_nasa(request, dataset_config)
        
        # Fallback: route based on dataset_id prefix
        dataset_lower = dataset_id.lower()
        if "cmems" in dataset_lower or "copernicus" in dataset_lower:
            return self._load_cmems(request)
        elif "era5" in dataset_lower:
            return self._load_era5(request)
        elif "local" in dataset_lower or "slcci" in dataset_lower:
            return self._load_local_files(request)
        elif "demo" in dataset_lower or "mock" in dataset_lower:
            return self._load_mock_data(request)
        else:
            logger.warning(f"Unknown dataset type: {dataset_id}, trying mock data")
            return self._load_mock_data(request)
    
    # =========================================================================
    # LOCAL FILE LOADING (from legacy/j2_utils.py patterns)
    # =========================================================================
    
    def load_local_netcdf(
        self,
        data_dir: str,
        bbox: Optional[BoundingBox] = None,
        gate_id: Optional[str] = None,
        cycles: Optional[List[int]] = None,
        pass_number: Optional[int] = None,
        use_quality_flag: bool = True,
        buffer_km: float = 50.0,
        verbose: bool = True
    ) -> tuple[List[Any], List[Dict]]:
        """
        Load local NetCDF files with optional spatial filtering.
        
        This is the main entry point for loading local SLCCI data.
        Based on legacy/j2_utils.py patterns.
        
        Args:
            data_dir: Directory containing NetCDF files
            bbox: Bounding box for spatial filtering
            gate_id: Gate ID to use for filtering (alternative to bbox)
            cycles: Specific cycle numbers to load (None = all)
            pass_number: Filter by specific pass number
            use_quality_flag: Apply validation_flag filter
            buffer_km: Buffer around gate in km
            verbose: Print loading progress
            
        Returns:
            Tuple of (datasets, cycle_info) where:
            - datasets: List of xarray.Dataset
            - cycle_info: List of dicts with cycle metadata
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return [], []
        
        # Get bbox from gate if provided
        if gate_id and bbox is None:
            try:
                from src.services import GateService
                gs = GateService()
                gate = gs.get_gate(gate_id)
                if gate and gate.bbox:
                    bbox = gate.bbox
                    logger.info(f"Using bbox from gate '{gate_id}'")
            except Exception as e:
                logger.warning(f"Could not get gate bbox: {e}")
        
        # Find NetCDF files
        patterns = [
            str(data_path / "SLCCI_ALTDB_*.nc"),
            str(data_path / "*.nc"),
        ]
        
        files = []
        for pattern in patterns:
            files = sorted(glob.glob(pattern))
            if files:
                break
        
        if not files:
            logger.warning(f"No NetCDF files found in {data_dir}")
            return [], []
        
        logger.info(f"Found {len(files)} NetCDF files")
        
        # Load and filter each file
        datasets = []
        cycle_info = []
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Extract cycle number from filename
            cycle_num = self._extract_cycle_number(filename)
            
            # Skip if not in requested cycles
            if cycles and cycle_num not in cycles:
                continue
            
            try:
                ds = self._load_and_filter_netcdf(
                    filepath=filepath,
                    bbox=bbox,
                    buffer_km=buffer_km,
                    pass_number=pass_number,
                    use_quality_flag=use_quality_flag,
                    verbose=verbose
                )
                
                if ds is not None and ds.sizes.get("time", 0) > 0:
                    datasets.append(ds)
                    cycle_info.append({
                        "filename": filename,
                        "cycle": cycle_num,
                        "path": filepath,
                        "n_points": ds.sizes.get("time", 0)
                    })
                    if verbose:
                        logger.info(f"✅ Loaded {filename}: {ds.sizes.get('time', 0)} points")
                        
            except Exception as e:
                logger.warning(f"⚠️ Error loading {filename}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(datasets)} cycles")
        return datasets, cycle_info
    
    def _load_and_filter_netcdf(
        self,
        filepath: str,
        bbox: Optional[BoundingBox] = None,
        buffer_km: float = 50.0,
        pass_number: Optional[int] = None,
        use_quality_flag: bool = True,
        verbose: bool = False
    ) -> Optional[Any]:
        """
        Load a single NetCDF file with spatial and quality filtering.
        
        Based on legacy/j2_utils.py load_filtered_cycles_serial_J2.
        """
        try:
            ds = xr.open_dataset(filepath, decode_times=False)
            
            # Get lon/lat
            if "longitude" in ds and "latitude" in ds:
                lon = ds["longitude"].values.flatten()
                lat = ds["latitude"].values.flatten()
            elif "lon" in ds and "lat" in ds:
                lon = ds["lon"].values.flatten()
                lat = ds["lat"].values.flatten()
            else:
                logger.warning(f"No coordinates found in {filepath}")
                return None
            
            # Wrap longitudes to [-180, 180]
            lon_wrapped = ((lon + 180) % 360) - 180
            
            # Apply spatial filter
            if bbox:
                # Add buffer (approx 1 deg = 111 km)
                buffer_deg = buffer_km / 111.0
                lat_min = bbox.lat_min - buffer_deg
                lat_max = bbox.lat_max + buffer_deg
                lon_min = bbox.lon_min - buffer_deg
                lon_max = bbox.lon_max + buffer_deg
                
                # Wrap lon bounds
                lon_min_w = ((lon_min + 180) % 360) - 180
                lon_max_w = ((lon_max + 180) % 360) - 180
                
                # Dateline-aware mask
                if lon_min_w <= lon_max_w:
                    lon_mask = (lon_wrapped >= lon_min_w) & (lon_wrapped <= lon_max_w)
                else:
                    # Crosses dateline
                    lon_mask = (lon_wrapped >= lon_min_w) | (lon_wrapped <= lon_max_w)
                
                mask_spatial = (
                    (lat >= lat_min) &
                    (lat <= lat_max) &
                    lon_mask
                )
                
                if mask_spatial.sum() == 0:
                    if verbose:
                        logger.debug(f"No points in spatial window for {filepath}")
                    return None
                
                ds = ds.isel(time=mask_spatial)
            
            # Quality flag filter
            if use_quality_flag and "validation_flag" in ds:
                valid_mask = ds["validation_flag"] == 0
                ds = ds.isel(time=valid_mask)
            
            # Pass filter
            if pass_number is not None:
                pass_var = None
                for var in ["pass", "track", "pass_number", "track_number"]:
                    if var in ds.variables:
                        pass_var = var
                        break
                
                if pass_var:
                    pass_vals = np.round(ds[pass_var].values).astype(int)
                    pass_mask = pass_vals == int(pass_number)
                    if pass_mask.sum() > 0:
                        ds = ds.isel(time=pass_mask)
                    else:
                        return None
            
            return ds
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    def _extract_cycle_number(self, filename: str) -> int:
        """Extract cycle number from filename."""
        import re
        # Try pattern SLCCI_ALTDB_J2_CycleXXX_V2.nc
        match = re.search(r'Cycle(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Try pattern with cycle_ prefix
        match = re.search(r'cycle[_-]?(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Default: return hash of filename
        return hash(filename) % 1000
    
    def _load_local_files(self, request: DataRequest) -> Optional[Any]:
        """Load local files based on DataRequest."""
        # Default data directory
        data_dir = self._defaults.get("local_data_dir", "data/slcci")
        
        datasets, _ = self.load_local_netcdf(
            data_dir=data_dir,
            bbox=request.bbox,
            use_quality_flag=True,
            verbose=True
        )
        
        if datasets:
            # Merge into single dataset
            try:
                return xr.concat(datasets, dim="time")
            except Exception:
                return datasets[0] if datasets else None
        return None
    
    def _load_cmems(self, request: DataRequest, config: Optional[Dict] = None) -> Optional[Any]:
        """
        Load data from CMEMS/Copernicus.
        
        Uses existing cmems_client if available.
        Routes based on config/datasets.yaml product_id.
        """
        try:
            # Import existing client
            from src.surge_shazam.data.cmems_client import CMEMSClient
            
            client = CMEMSClient()
            
            # Get product_id from config if available
            product_id = config.get("product_id", request.dataset_id) if config else request.dataset_id
            variables = config.get("variables", request.variables) if config else request.variables
            
            return client.load(
                dataset_id=product_id,
                variables=variables if variables else None,
                bbox=(
                    request.bbox.lon_min,
                    request.bbox.lat_min,
                    request.bbox.lon_max,
                    request.bbox.lat_max
                ),
                time_range=(request.time_range.start, request.time_range.end)
            )
        except ImportError:
            logger.warning("CMEMSClient not available, using mock data")
            return self._load_mock_data(request)
        except Exception as e:
            logger.error(f"CMEMS load error: {e}")
            return self._load_mock_data(request)
    
    def _load_era5(self, request: DataRequest, config: Optional[Dict] = None) -> Optional[Any]:
        """
        Load data from ERA5.
        
        Uses existing era5_client if available.
        Routes based on config/datasets.yaml product_id.
        """
        try:
            # Import existing client
            from src.surge_shazam.data.era5_client import ERA5Client
            
            client = ERA5Client()
            
            # Get variables from config if available
            variables = config.get("variables", request.variables) if config else request.variables
            
            return client.load(
                variables=variables if variables else None,
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
    
    def _load_noaa(self, request: DataRequest, config: Optional[Dict] = None) -> Optional[Any]:
        """
        Load data from NOAA.
        
        Uses climate indices client if available.
        """
        try:
            from src.surge_shazam.data.climate_indices import ClimateIndicesClient
            
            client = ClimateIndicesClient()
            variables = config.get("variables", request.variables) if config else request.variables
            
            return client.load(
                variables=variables,
                time_range=(request.time_range.start, request.time_range.end)
            )
        except ImportError:
            logger.warning("ClimateIndicesClient not available")
            return self._load_mock_data(request)
        except Exception as e:
            logger.error(f"NOAA load error: {e}")
            return None
    
    def _load_nasa(self, request: DataRequest, config: Optional[Dict] = None) -> Optional[Any]:
        """
        Load data from NASA PO.DAAC (e.g., CYGNSS).
        
        Uses CYGNSS client if available.
        """
        try:
            from src.surge_shazam.data.cygnss_client import CYGNSSClient
            
            client = CYGNSSClient()
            
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
            logger.warning("CYGNSSClient not available")
            return self._load_mock_data(request)
        except Exception as e:
            logger.error(f"NASA/CYGNSS load error: {e}")
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
