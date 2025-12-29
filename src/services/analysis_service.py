"""
Analysis Service
================
Business logic for ocean analysis computations.
DOT, binning, statistics, and causal analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np

from src.core.models import BoundingBox, TimeRange, GateModel

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for analysis computations.
    
    Provides a unified interface for:
    - DOT (Dynamic Ocean Topography) calculations
    - Spatial binning
    - Time series statistics
    - Causal analysis triggers
    
    Example:
        >>> service = AnalysisService()
        >>> dot = service.compute_dot(data)
        >>> binned = service.spatial_bin(data, resolution=0.25)
        >>> stats = service.compute_statistics(data)
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize analysis service.
        
        Args:
            config_path: Path to defaults.yaml configuration
        """
        self.config_path = Path(config_path) if config_path else Path("config/defaults.yaml")
        self._defaults: Dict[str, Any] = {}
        self._load_defaults()
    
    def _load_defaults(self) -> None:
        """Load default analysis parameters."""
        try:
            import yaml
            if self.config_path.exists():
                with open(self.config_path) as f:
                    self._defaults = yaml.safe_load(f) or {}
                    logger.info(f"Loaded defaults from {self.config_path}")
        except ImportError:
            logger.warning("PyYAML not available, using hardcoded defaults")
        except Exception as e:
            logger.warning(f"Could not load defaults: {e}")
    
    def compute_dot(
        self,
        data: Any,
        mdt_reference: str = "CNES-CLS18"
    ) -> Optional[Any]:
        """
        Compute Dynamic Ocean Topography.
        
        DOT = SSH - MDT (Mean Dynamic Topography)
        
        Args:
            data: xarray Dataset with SSH
            mdt_reference: MDT reference to use
            
        Returns:
            xarray Dataset with DOT
        """
        if data is None:
            return None
        
        try:
            # Check for SSH variable
            ssh_vars = ["ssh", "sla", "adt", "sea_surface_height"]
            ssh_var = None
            for var in ssh_vars:
                if var in data:
                    ssh_var = var
                    break
            
            if ssh_var is None:
                logger.warning("No SSH variable found in data")
                return data
            
            # For SLA, DOT ≈ SLA (simplified)
            # Real implementation would subtract MDT
            if "sla" in ssh_var:
                data["dot"] = data[ssh_var]
                logger.info("DOT computed from SLA")
            else:
                # Placeholder: would load MDT and subtract
                data["dot"] = data[ssh_var]
                logger.info(f"DOT computed from {ssh_var} (MDT subtraction pending)")
            
            return data
            
        except Exception as e:
            logger.error(f"DOT computation error: {e}")
            return None
    
    def spatial_bin(
        self,
        data: Any,
        resolution: float = 0.25,
        method: str = "mean",
        lat_resolution: Optional[float] = None,
        lon_resolution: Optional[float] = None
    ) -> Optional[Any]:
        """
        Bin data to specified spatial resolution.
        
        Supports independent lat/lon resolution for anisotropic binning.
        
        Args:
            data: xarray Dataset
            resolution: Target resolution in degrees (used for both lat/lon if not specified separately)
            method: Aggregation method (mean, median, std)
            lat_resolution: Optional separate latitude resolution in degrees
            lon_resolution: Optional separate longitude resolution in degrees
            
        Returns:
            Binned xarray Dataset
            
        Example:
            >>> from src.core.models import SpatialResolution
            >>> # Use enum
            >>> binned = service.spatial_bin(data, resolution=SpatialResolution.MEDIUM.value)
            >>> # Use custom float
            >>> binned = service.spatial_bin(data, resolution=0.15)
            >>> # Different lat/lon resolution
            >>> binned = service.spatial_bin(data, lat_resolution=0.25, lon_resolution=0.5)
        
        Note:
            - If lat_resolution or lon_resolution is provided, they override `resolution`
            - If data is already at target resolution, returns unchanged
            - Uses xarray.coarsen() for efficient binning
        """
        if data is None:
            return None
        
        # Handle SpatialResolution enum if passed
        if hasattr(resolution, 'value'):
            resolution = resolution.value
        
        # Use separate resolutions if provided
        target_lat_res = lat_resolution if lat_resolution is not None else resolution
        target_lon_res = lon_resolution if lon_resolution is not None else resolution
        
        try:
            import xarray as xr
            
            # Check current resolution
            if "lat" in data.coords and "lon" in data.coords:
                lat_res = abs(float(data.lat[1] - data.lat[0])) if len(data.lat) > 1 else 1.0
                lon_res = abs(float(data.lon[1] - data.lon[0])) if len(data.lon) > 1 else 1.0
                
                if lat_res >= target_lat_res and lon_res >= target_lon_res:
                    logger.info(f"Data already at {lat_res}°x{lon_res}° - no binning needed")
                    return data
                
                # Compute binning factors
                lat_factor = max(1, int(target_lat_res / lat_res))
                lon_factor = max(1, int(target_lon_res / lon_res))
                
                if lat_factor > 1 or lon_factor > 1:
                    # Use coarsen for binning
                    coarsened = data.coarsen(
                        lat=lat_factor,
                        lon=lon_factor,
                        boundary="trim"
                    )
                    
                    if method == "mean":
                        data = coarsened.mean()
                    elif method == "median":
                        data = coarsened.median()
                    elif method == "std":
                        data = coarsened.std()
                    else:
                        data = coarsened.mean()
                    
                    logger.info(f"Binned to {target_lat_res}°(lat) x {target_lon_res}°(lon) using {method}")
            
            return data
            
        except Exception as e:
            logger.error(f"Spatial binning error: {e}")
            return data
    
    def temporal_bin(
        self,
        data: Any,
        resolution: str = "daily",
        method: str = "mean"
    ) -> Optional[Any]:
        """
        Bin data to specified temporal resolution.
        
        Args:
            data: xarray Dataset
            resolution: Target resolution (hourly, daily, weekly, monthly)
            method: Aggregation method
            
        Returns:
            Resampled xarray Dataset
        """
        if data is None:
            return None
        
        try:
            freq_map = {
                "hourly": "1H",
                "daily": "1D",
                "weekly": "1W",
                "monthly": "1M",
                "yearly": "1Y"
            }
            
            freq = freq_map.get(resolution, "1D")
            
            if "time" in data.dims:
                resampled = data.resample(time=freq)
                
                if method == "mean":
                    data = resampled.mean()
                elif method == "median":
                    data = resampled.median()
                elif method == "std":
                    data = resampled.std()
                elif method == "sum":
                    data = resampled.sum()
                else:
                    data = resampled.mean()
                
                logger.info(f"Resampled to {resolution} using {method}")
            
            return data
            
        except Exception as e:
            logger.error(f"Temporal binning error: {e}")
            return data
    
    def compute_statistics(
        self,
        data: Any,
        variables: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute basic statistics for variables.
        
        Args:
            data: xarray Dataset
            variables: Variables to compute stats for
            
        Returns:
            Dict of statistics per variable
        """
        stats = {}
        
        if data is None:
            return stats
        
        try:
            if variables is None:
                variables = list(data.data_vars)
            
            for var in variables:
                if var in data:
                    arr = data[var].values
                    stats[var] = {
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "median": float(np.nanmedian(arr)),
                        "count": int(np.sum(~np.isnan(arr)))
                    }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics computation error: {e}")
            return stats
    
    def compute_cross_gate_transport(
        self,
        data: Any,
        gate: GateModel,
        method: str = "geostrophic"
    ) -> Optional[Dict[str, Any]]:
        """
        Compute transport across a gate.
        
        Args:
            data: xarray Dataset with velocity or height
            gate: Gate definition
            method: Transport computation method
            
        Returns:
            Dict with transport estimates
        """
        if data is None or gate is None:
            return None
        
        try:
            # Placeholder for transport computation
            # Real implementation would:
            # 1. Extract cross-gate section
            # 2. Compute geostrophic velocity from SSH gradient
            # 3. Integrate across section
            
            result = {
                "gate_id": gate.id,
                "method": method,
                "transport_sv": None,  # Sverdrup (10^6 m³/s)
                "uncertainty_sv": None,
                "computed_at": datetime.now().isoformat(),
                "status": "placeholder"
            }
            
            logger.info(f"Transport computation for {gate.id} (placeholder)")
            return result
            
        except Exception as e:
            logger.error(f"Transport computation error: {e}")
            return None
    
    def detect_anomalies(
        self,
        data: Any,
        variable: str,
        threshold_std: float = 2.0
    ) -> Optional[Any]:
        """
        Detect anomalies in time series.
        
        Args:
            data: xarray Dataset
            variable: Variable to check
            threshold_std: Number of standard deviations
            
        Returns:
            Boolean mask of anomalies
        """
        if data is None:
            return None
        
        try:
            if variable in data:
                values = data[variable].values
                mean = np.nanmean(values)
                std = np.nanstd(values)
                
                # Z-score anomaly detection
                z_scores = (values - mean) / std
                anomalies = np.abs(z_scores) > threshold_std
                
                logger.info(f"Found {np.sum(anomalies)} anomalies in {variable}")
                return anomalies
            
            return None
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return None
    
    def prepare_causal_input(
        self,
        data: Any,
        variables: List[str],
        detrend: bool = True,
        standardize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Prepare data for causal analysis (PCMCI).
        
        Args:
            data: xarray Dataset
            variables: Variables to include
            detrend: Whether to remove trend
            standardize: Whether to standardize (z-score)
            
        Returns:
            Array shaped (T, N) for PCMCI
        """
        if data is None:
            return None
        
        try:
            # Extract time series for each variable
            series = []
            for var in variables:
                if var in data:
                    # Spatial mean to get time series
                    ts = data[var].mean(dim=["lat", "lon"]).values
                    series.append(ts)
            
            if not series:
                logger.warning("No variables found for causal analysis")
                return None
            
            # Stack into (T, N) array
            arr = np.column_stack(series)
            
            if detrend:
                from scipy import signal
                arr = signal.detrend(arr, axis=0)
            
            if standardize:
                arr = (arr - np.nanmean(arr, axis=0)) / np.nanstd(arr, axis=0)
            
            logger.info(f"Prepared causal input: {arr.shape}")
            return arr
            
        except ImportError:
            logger.warning("scipy not available for detrending")
            return np.column_stack(series) if series else None
        except Exception as e:
            logger.error(f"Causal prep error: {e}")
            return None
