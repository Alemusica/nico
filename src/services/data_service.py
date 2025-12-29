"""
Data Service
============
Business logic for data loading and filtering.
Unified interface for ALL data sources via IntakeCatalogBridge.

This service follows the NICO Unified Architecture:
- Called by UI (Streamlit/React) and API
- Uses GateService for spatial bounds
- Routes to IntakeCatalogBridge which reads from catalog.yaml

CRITICAL: This is the ONLY entry point for data loading.
See docs/UNIFIED_DATA_PIPELINE.md for architecture.
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

# Try to import IntakeCatalogBridge (the unified catalog)
INTAKE_BRIDGE_AVAILABLE = False
try:
    from src.data_manager.intake_bridge import IntakeCatalogBridge, get_catalog
    INTAKE_BRIDGE_AVAILABLE = True
    logger.info("IntakeCatalogBridge available - using unified catalog")
except ImportError as e:
    logger.warning(f"IntakeCatalogBridge not available: {e}")


class DataService:
    """
    Service for data loading and filtering.
    
    Routes ALL data requests through IntakeCatalogBridge which:
    - Reads dataset definitions from catalog.yaml
    - Instantiates appropriate client (CMEMS, ERA5, etc.)
    - Returns xarray Dataset
    
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
        
        Uses IntakeCatalogBridge as primary source, falls back to
        config/datasets.yaml if intake not available.
        """
        self.config_path = Path(config_path) if config_path else Path("config/datasets.yaml")
        self._datasets: Dict[str, Dict] = {}
        self._defaults: Dict[str, Any] = {}
        
        # Initialize catalog bridge if available
        self._catalog_bridge = None
        if INTAKE_BRIDGE_AVAILABLE:
            try:
                self._catalog_bridge = IntakeCatalogBridge()
                logger.info(f"Loaded {len(self._catalog_bridge.list_datasets())} datasets from catalog.yaml")
            except Exception as e:
                logger.warning(f"Could not initialize IntakeCatalogBridge: {e}")
        
        # Load fallback config
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
        
        ROUTING PRIORITY:
        1. IntakeCatalogBridge (reads catalog.yaml) - PREFERRED
        2. Fallback to config/datasets.yaml routing
        3. Fallback to dataset_id prefix matching
        
        Args:
            request: DataRequest with parameters
            
        Returns:
            xarray Dataset or None if loading fails
        """
        dataset_id = request.dataset_id
        logger.info(f"Loading dataset: {dataset_id}")
        
        # === PRIORITY 1: Use IntakeCatalogBridge ===
        if self._catalog_bridge is not None:
            try:
                # Check if dataset exists in catalog
                if dataset_id in self._catalog_bridge.list_datasets():
                    logger.info(f"Loading {dataset_id} via IntakeCatalogBridge")
                    
                    # Build bbox tuple (lat_min, lat_max, lon_min, lon_max) as per intake_bridge
                    bbox_tuple = (
                        request.bbox.lat_min,
                        request.bbox.lat_max,
                        request.bbox.lon_min, 
                        request.bbox.lon_max
                    )
                    
                    # Time range tuple
                    time_tuple = (request.time_range.start, request.time_range.end)
                    
                    # Load via bridge - note: load() is async in intake_bridge
                    # For now, use synchronous fallback via intake directly
                    try:
                        source = self._catalog_bridge.catalog[dataset_id]
                        # Try to read (may fail if urlpath doesn't exist)
                        data = source.read()
                        if data is not None:
                            logger.info(f"✅ Loaded {dataset_id} via catalog source")
                            return data
                    except Exception as e:
                        logger.debug(f"Direct catalog read failed: {e}")
                        
                    # Try via client if available
                    client = self._catalog_bridge.get_client(dataset_id)
                    if client is not None:
                        logger.info(f"Using client for {dataset_id}")
                        # Client-based loading handled below
                        
            except Exception as e:
                logger.warning(f"Catalog bridge load failed for {dataset_id}: {e}")
                # Fall through to other methods
        
        # === PRIORITY 2: Use config/datasets.yaml ===
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
        
        # === PRIORITY 3: Fallback to prefix matching ===
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
        Generate mock data for testing - MULTIPLE CYCLES with variable slope.
        
        Returns a LIST of xarray Datasets in ALTIMETRY format (along-track):
        - Dimension: time
        - Variables: corssh, mean_sea_surface, latitude, longitude, pass, cycle
        
        This matches the SLCCI format expected by visualization tabs.
        """
        try:
            import numpy as np
            import xarray as xr
            import pandas as pd
            
            # Parse time range
            start_str = request.time_range.start[:10] if isinstance(request.time_range.start, str) else request.time_range.start.isoformat()[:10]
            
            # Configuration
            n_cycles = 10  # Generate 10 cycles for timeline variation
            n_points_per_cycle = 500  # Points per pass
            pass_number = 481  # Example pass number (Fram Strait S3)
            
            # Gate geometry
            lat_center = (request.bbox.lat_min + request.bbox.lat_max) / 2
            lon_center = (request.bbox.lon_min + request.bbox.lon_max) / 2
            lat_range = request.bbox.lat_max - request.bbox.lat_min
            lon_range = request.bbox.lon_max - request.bbox.lon_min
            
            datasets = []
            
            for cycle in range(1, n_cycles + 1):
                # Simulate satellite track (along-track, roughly N-S)
                t = np.linspace(0, 1, n_points_per_cycle)
                lats = request.bbox.lat_min + lat_range * t + np.random.randn(n_points_per_cycle) * 0.005
                lons = lon_center + np.sin(t * 2 * np.pi) * lon_range * 0.2 + np.random.randn(n_points_per_cycle) * 0.005
                
                # Sort by longitude for profile plots
                sort_idx = np.argsort(lons)
                lats = lats[sort_idx]
                lons = lons[sort_idx]
                
                # Generate DOT with VARIABLE SLOPE per cycle - REALISTIC VALUES
                # Seasonal and interannual variability
                seasonal = 1 + 0.5 * np.sin(2 * np.pi * cycle / 12)
                interannual = 0.3 * np.sin(2 * np.pi * cycle / 24)
                random_var = np.random.randn() * 0.2
                
                # Base slope: 0.0002 to 0.0008 mm/m (realistic)
                base_slope_mm_m = (0.0003 + 0.0002 * interannual) * seasonal * (1 + random_var)
                
                lon_diff = lons - lon_center
                distance_m = lon_diff * 111000 * np.cos(np.radians(lat_center))
                base_dot = base_slope_mm_m * distance_m / 1000
                
                # Add mesoscale eddy variability
                mesoscale = 0.03 * np.sin(2 * np.pi * lons / 1.5 + cycle * 0.5) * np.cos(2 * np.pi * lats / 0.8)
                
                # Add realistic noise
                noise = np.random.randn(n_points_per_cycle) * 0.025
                
                # Total DOT
                dot = base_dot + mesoscale + noise
                
                # SSH = DOT + MSS
                mss = -30.0 + 0.5 * np.sin(2 * np.pi * lons / 10)  # Variable MSS
                corssh = dot + mss
                
                # Generate times (each cycle is ~10 days apart)
                base_time = pd.Timestamp(start_str) + pd.Timedelta(days=10 * (cycle - 1))
                times = [base_time + pd.Timedelta(seconds=i * 0.5) for i in range(n_points_per_cycle)]
                
                # Create Dataset with pass and cycle info
                ds = xr.Dataset(
                    {
                        "corssh": (["time"], corssh.astype(np.float32)),
                        "mean_sea_surface": (["time"], mss.astype(np.float32)),
                        "latitude": (["time"], lats.astype(np.float32)),
                        "longitude": (["time"], lons.astype(np.float32)),
                        "dot": (["time"], dot.astype(np.float32)),
                        "pass": (["time"], np.full(n_points_per_cycle, pass_number, dtype=np.int32)),
                        "cycle": (["time"], np.full(n_points_per_cycle, cycle, dtype=np.int32)),
                    },
                    coords={"time": times},
                    attrs={
                        "dataset_id": request.dataset_id,
                        "source": "mock_altimetry_data",
                        "generated": datetime.now().isoformat(),
                        "cycle": cycle,
                        "pass": pass_number,
                        "gate": getattr(request, 'gate_id', 'unknown'),
                        "description": f"Demo altimetry cycle {cycle}, pass {pass_number}"
                    }
                )
                
                datasets.append(ds)
            
            # Return first dataset for backward compatibility
            # The sidebar should call load_multi_cycle for multiple datasets
            logger.info(f"Generated {n_cycles} mock cycles with variable slope")
            return datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="time")
            
        except Exception as e:
            logger.error(f"Mock data generation error: {e}")
            return None
    
    def load_multi_cycle_demo(self, request: DataRequest, n_cycles: int = 10) -> tuple[List[Any], List[Dict]]:
        """
        Load multiple demo cycles for timeline analysis.
        
        Returns:
            Tuple of (datasets, cycle_info) matching the format expected by visualization.
        """
        try:
            import numpy as np
            import xarray as xr
            import pandas as pd
            
            start_str = request.time_range.start[:10] if isinstance(request.time_range.start, str) else request.time_range.start.isoformat()[:10]
            
            n_points = 500
            pass_number = 481
            
            lat_center = (request.bbox.lat_min + request.bbox.lat_max) / 2
            lon_center = (request.bbox.lon_min + request.bbox.lon_max) / 2
            lat_range = request.bbox.lat_max - request.bbox.lat_min
            lon_range = request.bbox.lon_max - request.bbox.lon_min
            
            datasets = []
            cycle_info = []
            
            for cycle in range(1, n_cycles + 1):
                t = np.linspace(0, 1, n_points)
                lats = request.bbox.lat_min + lat_range * t + np.random.randn(n_points) * 0.005
                lons = lon_center + np.sin(t * 2 * np.pi) * lon_range * 0.2 + np.random.randn(n_points) * 0.005
                
                sort_idx = np.argsort(lons)
                lats = lats[sort_idx]
                lons = lons[sort_idx]
                
                # Variable slope per cycle - REALISTIC VALUES
                # Real DOT slope across Fram Strait: ~0.0001 to 0.001 mm/m
                # Seasonal and interannual variability
                seasonal = 1 + 0.5 * np.sin(2 * np.pi * cycle / 12)  # ±50% seasonal
                interannual = 0.3 * np.sin(2 * np.pi * cycle / 24)   # Slower variation
                random_var = np.random.randn() * 0.2  # Some randomness
                
                # Base slope: 0.0002 to 0.0008 mm/m (realistic range)
                base_slope_mm_m = (0.0003 + 0.0002 * interannual) * seasonal * (1 + random_var)
                
                # Convert to DOT: slope in m per degree longitude
                # DOT change across strait ≈ slope_mm_m * distance_m / 1000
                lon_diff = lons - lon_center  # degrees
                distance_m = lon_diff * 111000 * np.cos(np.radians(lat_center))  # meters
                base_dot = base_slope_mm_m * distance_m / 1000  # Convert mm to m
                
                # Add mesoscale variability (eddies, waves) - ±2-5 cm
                mesoscale = 0.03 * np.sin(2 * np.pi * lons / 1.5 + cycle * 0.5) * np.cos(2 * np.pi * lats / 0.5)
                
                # Add realistic noise (altimeter precision ~2-3 cm)
                noise = np.random.randn(n_points) * 0.025
                
                dot = base_dot + mesoscale + noise
                
                mss = -30.0 + 0.5 * np.sin(2 * np.pi * lons / 10)
                corssh = dot + mss
                
                base_time = pd.Timestamp(start_str) + pd.Timedelta(days=10 * (cycle - 1))
                times = [base_time + pd.Timedelta(seconds=i * 0.5) for i in range(n_points)]
                
                ds = xr.Dataset(
                    {
                        "corssh": (["time"], corssh.astype(np.float32)),
                        "mean_sea_surface": (["time"], mss.astype(np.float32)),
                        "latitude": (["time"], lats.astype(np.float32)),
                        "longitude": (["time"], lons.astype(np.float32)),
                        "pass": (["time"], np.full(n_points, pass_number, dtype=np.int32)),
                    },
                    coords={"time": times},
                    attrs={"cycle": cycle, "pass": pass_number}
                )
                
                datasets.append(ds)
                cycle_info.append({
                    "filename": f"demo_cycle_{cycle}_pass_{pass_number}.nc",
                    "cycle": cycle,
                    "pass": pass_number,
                    "path": "demo",
                    "n_points": n_points,
                    "date": base_time.strftime("%Y-%m-%d")
                })
            
            return datasets, cycle_info
            
        except Exception as e:
            logger.error(f"Multi-cycle demo error: {e}")
            return [], []
    
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
