"""
🛰️ Multi-Satellite Data Fusion Engine
=======================================

Integrates data from multiple satellites with different:
- Orbits (inclination, repeat period)
- Instruments (Ku, Ka, C-band altimeters)
- Measurements (SSH, SWH, Wind, SST)
- Quality levels and availability

Inspired by:
- AVISO/DUACS L4 gridded products
- ESA CCI Sea Level project
- Copernicus Marine Service

Key Features:
1. Query multiple satellites simultaneously
2. Handle offline/unavailable sensors gracefully
3. Weight by quality and proximity
4. Interpolate to common grid
5. Provide uncertainty estimates
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import cKDTree


# =============================================================================
# SATELLITE DEFINITIONS
# =============================================================================

class SatelliteStatus(Enum):
    """Current operational status of satellite."""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"  # Partial capability
    OFFLINE = "offline"  # Temporary outage
    DECOMMISSIONED = "decommissioned"
    UNKNOWN = "unknown"


class InstrumentType(Enum):
    """Type of measurement instrument."""
    RADAR_ALTIMETER = "radar_altimeter"  # SSH, SWH, sigma0
    RADIOMETER = "radiometer"  # Wet troposphere
    DORIS = "doris"  # Precise orbit
    GPS = "gps"  # Orbit
    SAR = "sar"  # High resolution mode
    PASSIVE_MW = "passive_mw"  # SST


@dataclass
class SatelliteConfig:
    """
    Configuration for a satellite in the constellation.
    """
    name: str
    short_name: str
    
    # Orbital parameters
    inclination_deg: float
    repeat_days: int
    altitude_km: float
    ground_track_spacing_km: float
    
    # Instruments
    instruments: List[InstrumentType] = field(default_factory=list)
    frequency_ghz: float = 13.6  # Ku-band default
    
    # Coverage
    lat_min: float = -66.0
    lat_max: float = 66.0
    
    # Data characteristics
    along_track_resolution_km: float = 7.0
    measurement_precision_cm: float = 2.0
    
    # Status
    status: SatelliteStatus = SatelliteStatus.OPERATIONAL
    launch_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Data quality weight (0-1)
    quality_weight: float = 1.0
    
    def is_available(self, date: datetime = None) -> bool:
        """Check if satellite is available for given date."""
        if self.status in [SatelliteStatus.OFFLINE, SatelliteStatus.DECOMMISSIONED]:
            return False
        
        if date and self.launch_date:
            launch = datetime.fromisoformat(self.launch_date)
            if date < launch:
                return False
        
        if date and self.end_date:
            end = datetime.fromisoformat(self.end_date)
            if date > end:
                return False
        
        return True


# Pre-defined satellite configurations
SATELLITE_CONFIGS = {
    "sentinel-3a": SatelliteConfig(
        name="Sentinel-3A",
        short_name="S3A",
        inclination_deg=98.65,
        repeat_days=27,
        altitude_km=814,
        ground_track_spacing_km=104,
        instruments=[InstrumentType.RADAR_ALTIMETER, InstrumentType.RADIOMETER],
        frequency_ghz=13.575,
        lat_min=-81.5,
        lat_max=81.5,
        along_track_resolution_km=0.3,  # SAR mode
        measurement_precision_cm=1.5,
        launch_date="2016-02-16",
        quality_weight=0.95,
    ),
    "sentinel-3b": SatelliteConfig(
        name="Sentinel-3B",
        short_name="S3B",
        inclination_deg=98.65,
        repeat_days=27,
        altitude_km=814,
        ground_track_spacing_km=104,
        instruments=[InstrumentType.RADAR_ALTIMETER, InstrumentType.RADIOMETER],
        frequency_ghz=13.575,
        lat_min=-81.5,
        lat_max=81.5,
        along_track_resolution_km=0.3,
        measurement_precision_cm=1.5,
        launch_date="2018-04-25",
        quality_weight=0.95,
    ),
    "jason-3": SatelliteConfig(
        name="Jason-3",
        short_name="J3",
        inclination_deg=66.0,
        repeat_days=10,
        altitude_km=1336,
        ground_track_spacing_km=315,
        instruments=[InstrumentType.RADAR_ALTIMETER, InstrumentType.RADIOMETER, InstrumentType.DORIS],
        frequency_ghz=13.6,
        lat_min=-66.0,
        lat_max=66.0,
        along_track_resolution_km=7.0,
        measurement_precision_cm=2.5,
        launch_date="2016-01-17",
        quality_weight=0.90,
    ),
    "sentinel-6a": SatelliteConfig(
        name="Sentinel-6A Michael Freilich",
        short_name="S6A",
        inclination_deg=66.0,
        repeat_days=10,
        altitude_km=1336,
        ground_track_spacing_km=315,
        instruments=[InstrumentType.RADAR_ALTIMETER, InstrumentType.RADIOMETER, InstrumentType.GPS],
        frequency_ghz=13.6,
        lat_min=-66.0,
        lat_max=66.0,
        along_track_resolution_km=0.3,  # SAR
        measurement_precision_cm=1.0,
        launch_date="2020-11-21",
        quality_weight=1.0,  # Reference mission
    ),
    "cryosat-2": SatelliteConfig(
        name="CryoSat-2",
        short_name="C2",
        inclination_deg=92.0,
        repeat_days=369,  # Geodetic orbit
        altitude_km=717,
        ground_track_spacing_km=8,  # Very dense
        instruments=[InstrumentType.RADAR_ALTIMETER, InstrumentType.SAR],
        frequency_ghz=13.575,
        lat_min=-88.0,
        lat_max=88.0,
        along_track_resolution_km=0.25,  # SAR/SARIn
        measurement_precision_cm=2.0,
        launch_date="2010-04-08",
        quality_weight=0.85,
    ),
    "icesat-2": SatelliteConfig(
        name="ICESat-2",
        short_name="IS2",
        inclination_deg=92.0,
        repeat_days=91,
        altitude_km=500,
        ground_track_spacing_km=3.3,
        instruments=[],  # Laser, not radar
        frequency_ghz=0.0,  # Laser
        lat_min=-88.0,
        lat_max=88.0,
        along_track_resolution_km=0.07,  # Very high res
        measurement_precision_cm=3.0,  # Over ocean
        launch_date="2018-09-15",
        quality_weight=0.70,  # Lower over ocean
    ),
    "swot": SatelliteConfig(
        name="SWOT",
        short_name="SWOT",
        inclination_deg=77.6,
        repeat_days=21,
        altitude_km=891,
        ground_track_spacing_km=120,
        instruments=[InstrumentType.SAR, InstrumentType.RADAR_ALTIMETER],
        frequency_ghz=35.75,  # Ka-band
        lat_min=-78.0,
        lat_max=78.0,
        along_track_resolution_km=2.0,  # 2D imaging
        measurement_precision_cm=5.0,  # Over ocean
        launch_date="2022-12-16",
        quality_weight=0.98,
    ),
}


@dataclass
class SatelliteObservation:
    """Single observation from a satellite."""
    satellite: str
    time: datetime
    lat: float
    lon: float
    
    # Measurements
    ssh: Optional[float] = None  # Sea Surface Height [m]
    sla: Optional[float] = None  # Sea Level Anomaly [m]
    swh: Optional[float] = None  # Significant Wave Height [m]
    wind_speed: Optional[float] = None  # Wind speed [m/s]
    sigma0: Optional[float] = None  # Backscatter [dB]
    
    # Quality
    quality_flag: int = 0
    quality_weight: float = 1.0
    
    # Uncertainty
    ssh_error: float = 0.0
    
    def is_valid(self) -> bool:
        return self.quality_flag == 0 and self.ssh is not None


# =============================================================================
# MULTI-SATELLITE FUSION ENGINE
# =============================================================================

class SatelliteFusionEngine:
    """
    Query and fuse data from multiple satellites.
    
    Features:
    - Handles offline satellites gracefully
    - Weights by quality and proximity
    - Provides uncertainty estimates
    - Supports both along-track and gridded output
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent.parent / "data"
        self.satellites = SATELLITE_CONFIGS.copy()
        self._loaded_data: Dict[str, xr.Dataset] = {}
        
    def get_available_satellites(self, date: datetime = None) -> List[str]:
        """List satellites available for given date."""
        date = date or datetime.now()
        return [
            name for name, config in self.satellites.items()
            if config.is_available(date)
        ]
    
    def get_satellite_status(self) -> Dict[str, Dict]:
        """Get status of all satellites."""
        return {
            name: {
                "status": config.status.value,
                "quality_weight": config.quality_weight,
                "lat_range": (config.lat_min, config.lat_max),
                "instruments": [i.value for i in config.instruments],
            }
            for name, config in self.satellites.items()
        }
    
    def set_satellite_status(self, name: str, status: SatelliteStatus):
        """Update satellite status (e.g., mark as offline)."""
        if name in self.satellites:
            self.satellites[name].status = status
    
    def query_observations(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[datetime, datetime],
        satellites: List[str] = None,
        variable: str = "ssh",
    ) -> List[SatelliteObservation]:
        """
        Query observations from multiple satellites.
        
        Args:
            lat_range: (min, max) latitude
            lon_range: (min, max) longitude  
            time_range: (start, end) datetime
            satellites: List of satellite names (None = all available)
            variable: Variable to query
            
        Returns:
            List of observations from all queried satellites
        """
        if satellites is None:
            satellites = self.get_available_satellites(time_range[0])
        
        all_obs = []
        
        for sat_name in satellites:
            if sat_name not in self.satellites:
                continue
            
            config = self.satellites[sat_name]
            
            # Check if satellite covers this region
            if lat_range[0] > config.lat_max or lat_range[1] < config.lat_min:
                continue
            
            # Try to load data
            obs = self._load_satellite_data(
                sat_name, config, lat_range, lon_range, time_range, variable
            )
            all_obs.extend(obs)
        
        return all_obs
    
    def _load_satellite_data(
        self,
        sat_name: str,
        config: SatelliteConfig,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_range: Tuple[datetime, datetime],
        variable: str,
    ) -> List[SatelliteObservation]:
        """
        Load data for a single satellite.
        
        This is a placeholder - in production would query actual data files.
        """
        # Check for cached data
        cache_key = f"{sat_name}_{time_range[0].date()}"
        
        if cache_key in self._loaded_data:
            ds = self._loaded_data[cache_key]
        else:
            # Try to load from file
            # In production: search for matching files
            return []
        
        # Filter and return observations
        # Placeholder implementation
        return []
    
    def fuse_to_grid(
        self,
        observations: List[SatelliteObservation],
        grid_lat: np.ndarray,
        grid_lon: np.ndarray,
        method: str = "weighted_average",
        search_radius_km: float = 50.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse observations to regular grid.
        
        Args:
            observations: List of satellite observations
            grid_lat: 1D array of grid latitudes
            grid_lon: 1D array of grid longitudes
            method: Interpolation method ("weighted_average", "rbf", "kriging")
            search_radius_km: Search radius for nearest neighbor
            
        Returns:
            (gridded_values, gridded_uncertainty)
        """
        if not observations:
            shape = (len(grid_lat), len(grid_lon))
            return np.full(shape, np.nan), np.full(shape, np.nan)
        
        # Extract observation points
        obs_lats = np.array([o.lat for o in observations])
        obs_lons = np.array([o.lon for o in observations])
        obs_vals = np.array([o.ssh for o in observations if o.ssh is not None])
        obs_weights = np.array([o.quality_weight for o in observations if o.ssh is not None])
        
        if len(obs_vals) == 0:
            shape = (len(grid_lat), len(grid_lon))
            return np.full(shape, np.nan), np.full(shape, np.nan)
        
        # Create output grid
        lon_grid, lat_grid = np.meshgrid(grid_lon, grid_lat)
        
        if method == "weighted_average":
            return self._weighted_average_interp(
                obs_lats, obs_lons, obs_vals, obs_weights,
                lat_grid, lon_grid, search_radius_km
            )
        elif method == "rbf":
            return self._rbf_interp(
                obs_lats, obs_lons, obs_vals,
                lat_grid, lon_grid
            )
        else:
            # Simple griddata
            points = np.column_stack([obs_lats, obs_lons])
            values = griddata(points, obs_vals, (lat_grid, lon_grid), method='linear')
            uncertainty = np.full_like(values, 0.05)  # Placeholder
            return values, uncertainty
    
    def _weighted_average_interp(
        self,
        obs_lats: np.ndarray,
        obs_lons: np.ndarray,
        obs_vals: np.ndarray,
        obs_weights: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        search_radius_km: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted average interpolation using inverse distance and quality weights.
        
        Formula: value = Σ(wi * vi) / Σ(wi)
        where wi = quality_weight / distance²
        """
        # Build KDTree for fast search
        obs_points = np.column_stack([obs_lats, obs_lons])
        tree = cKDTree(obs_points)
        
        # Convert search radius to degrees (approximate)
        search_deg = search_radius_km / 111.0
        
        # Output arrays
        gridded = np.full(lat_grid.shape, np.nan)
        uncertainty = np.full(lat_grid.shape, np.nan)
        
        # Query each grid point
        for i in range(lat_grid.shape[0]):
            for j in range(lat_grid.shape[1]):
                point = [lat_grid[i, j], lon_grid[i, j]]
                
                # Find nearby observations
                indices = tree.query_ball_point(point, search_deg)
                
                if len(indices) == 0:
                    continue
                
                # Calculate weights
                distances = np.array([
                    self._haversine_distance(point[0], point[1], obs_lats[k], obs_lons[k])
                    for k in indices
                ])
                
                # Avoid division by zero
                distances = np.maximum(distances, 0.1)
                
                # Combined weight: quality / distance²
                weights = obs_weights[indices] / (distances ** 2)
                values = obs_vals[indices]
                
                # Weighted average
                gridded[i, j] = np.sum(weights * values) / np.sum(weights)
                
                # Uncertainty: weighted std
                if len(indices) > 1:
                    variance = np.sum(weights * (values - gridded[i, j])**2) / np.sum(weights)
                    uncertainty[i, j] = np.sqrt(variance)
                else:
                    uncertainty[i, j] = 0.05  # Default
        
        return gridded, uncertainty
    
    def _rbf_interp(
        self,
        obs_lats: np.ndarray,
        obs_lons: np.ndarray,
        obs_vals: np.ndarray,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Radial Basis Function interpolation."""
        points = np.column_stack([obs_lats, obs_lons])
        
        try:
            rbf = RBFInterpolator(points, obs_vals, kernel='thin_plate_spline')
            grid_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
            values = rbf(grid_points).reshape(lat_grid.shape)
            uncertainty = np.full_like(values, 0.03)
            return values, uncertainty
        except Exception:
            # Fallback to simple griddata
            values = griddata(points, obs_vals, (lat_grid, lon_grid), method='linear')
            uncertainty = np.full_like(values, 0.05)
            return values, uncertainty
    
    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in km."""
        R = 6371.0  # Earth radius in km
        
        lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_coverage_map(
        self,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        grid_resolution: float = 0.5,
    ) -> Dict[str, np.ndarray]:
        """
        Get map of which satellites cover which grid cells.
        
        Useful for understanding data availability before querying.
        """
        lats = np.arange(lat_range[0], lat_range[1], grid_resolution)
        lons = np.arange(lon_range[0], lon_range[1], grid_resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        coverage = {}
        
        for name, config in self.satellites.items():
            if config.status == SatelliteStatus.DECOMMISSIONED:
                continue
            
            # Simple coverage mask based on latitude limits
            mask = (lat_grid >= config.lat_min) & (lat_grid <= config.lat_max)
            coverage[name] = mask.astype(float) * config.quality_weight
        
        # Combined coverage (max weight at each point)
        coverage["combined"] = np.maximum.reduce(list(coverage.values()))
        
        return coverage


# =============================================================================
# DYNAMIC INDEX CALCULATOR
# =============================================================================

class DynamicIndexCalculator:
    """
    Calculate dynamic indices from fused satellite data.
    
    Replaces static indices with real-time values:
    - thermodynamics: From SST anomalies
    - oceanography: From SSH/SLA patterns
    - cryosphere: From sea ice concentration
    - anemometry: From altimeter wind
    """
    
    def __init__(self, fusion_engine: SatelliteFusionEngine = None):
        self.fusion = fusion_engine or SatelliteFusionEngine()
    
    def calculate_thermodynamic_index(
        self,
        sst_data: np.ndarray,
        climatology: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Calculate thermodynamic index from SST.
        
        High values = warm anomaly = more energy available
        """
        if climatology is not None:
            anomaly = sst_data - climatology
        else:
            anomaly = sst_data - np.nanmean(sst_data)
        
        # Index: normalized to 0-10 scale
        # Positive anomaly = higher score
        mean_anomaly = np.nanmean(anomaly)
        max_anomaly = np.nanmax(np.abs(anomaly))
        
        # Scale: ±5°C anomaly maps to 0-10
        score = 5.0 + (mean_anomaly / 5.0) * 5.0
        score = np.clip(score, 0, 10)
        
        return {
            "index": float(score),
            "mean_anomaly_C": float(mean_anomaly),
            "max_anomaly_C": float(max_anomaly),
            "data_coverage": float(np.sum(~np.isnan(sst_data)) / sst_data.size),
        }
    
    def calculate_oceanography_index(
        self,
        ssh_data: np.ndarray,
        sla_data: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Calculate oceanographic index from SSH/SLA.
        
        High values = strong sea level gradients = active dynamics
        """
        if sla_data is not None:
            data = sla_data
        else:
            data = ssh_data - np.nanmean(ssh_data)
        
        # Calculate gradient magnitude
        grad_y, grad_x = np.gradient(data)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        mean_gradient = np.nanmean(gradient_mag)
        std_data = np.nanstd(data)
        
        # Scale: 10cm std maps to score 5
        score = (std_data / 0.10) * 5.0
        score = np.clip(score, 0, 10)
        
        return {
            "index": float(score),
            "ssh_std_m": float(std_data),
            "mean_gradient": float(mean_gradient),
            "data_coverage": float(np.sum(~np.isnan(ssh_data)) / ssh_data.size),
        }
    
    def calculate_cryosphere_index(
        self,
        ice_concentration: np.ndarray,
        ice_thickness: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Calculate cryosphere index from sea ice data.
        
        High values = significant ice presence/change
        """
        mean_conc = np.nanmean(ice_concentration)
        
        if ice_thickness is not None:
            mean_thick = np.nanmean(ice_thickness)
            volume_proxy = mean_conc * mean_thick
        else:
            volume_proxy = mean_conc
        
        # Scale: 50% concentration = score 5
        score = (mean_conc / 50.0) * 5.0
        score = np.clip(score, 0, 10)
        
        return {
            "index": float(score),
            "mean_concentration_pct": float(mean_conc),
            "volume_proxy": float(volume_proxy),
        }
    
    def calculate_anemometry_index(
        self,
        wind_speed: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate wind/anemometry index.
        
        High values = strong winds = atmospheric forcing
        """
        mean_wind = np.nanmean(wind_speed)
        max_wind = np.nanmax(wind_speed)
        
        # Scale: 20 m/s mean = score 10
        score = (mean_wind / 20.0) * 10.0
        score = np.clip(score, 0, 10)
        
        return {
            "index": float(score),
            "mean_wind_ms": float(mean_wind),
            "max_wind_ms": float(max_wind),
            "data_coverage": float(np.sum(~np.isnan(wind_speed)) / wind_speed.size),
        }
    
    def calculate_all_indices(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, Dict]:
        """
        Calculate all available indices from data dict.
        
        Args:
            data: Dict with keys like 'sst', 'ssh', 'ice', 'wind'
        """
        indices = {}
        
        if 'sst' in data:
            indices['thermodynamics'] = self.calculate_thermodynamic_index(data['sst'])
        else:
            indices['thermodynamics'] = {"index": 0.0, "reason": "no SST data"}
        
        if 'ssh' in data:
            indices['oceanography'] = self.calculate_oceanography_index(
                data['ssh'], 
                data.get('sla')
            )
        else:
            indices['oceanography'] = {"index": 0.0, "reason": "no SSH data"}
        
        if 'ice' in data:
            indices['cryosphere'] = self.calculate_cryosphere_index(
                data['ice'],
                data.get('ice_thickness')
            )
        else:
            indices['cryosphere'] = {"index": 0.0, "reason": "no ice data"}
        
        if 'wind' in data:
            indices['anemometry'] = self.calculate_anemometry_index(data['wind'])
        else:
            indices['anemometry'] = {"index": 0.0, "reason": "no wind data"}
        
        # Precipitation placeholder (typically from reanalysis, not satellites)
        indices['precipitation'] = {"index": 5.0, "reason": "requires ERA5/reanalysis"}
        
        return indices
