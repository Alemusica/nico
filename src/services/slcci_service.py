"""
SLCCI Service
=============
Service for loading and processing ESA Sea Level CCI (SLCCI) data.

This service wraps the functions from legacy/j2_utils.py into a clean service layer
following the NICO Unified Architecture pattern.

Data Flow:
    UI → SLCCIService → load_filtered_cycles_serial_J2 → NetCDF files
                      → interpolate_geoid → TUM_ogmoc.nc
                      → DOT calculation → DataFrame
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SLCCIConfig:
    """Configuration for SLCCI data loading."""
    base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/J2"
    geoid_path: str = "/Users/nicolocaron/Desktop/ARCFRESH/TUM_ogmoc.nc"
    cycles: List[int] = field(default_factory=lambda: list(range(1, 282)))
    use_flag: bool = True
    lat_buffer_deg: float = 2.0
    lon_buffer_deg: float = 5.0


@dataclass
class PassData:
    """Container for pass data analysis results."""
    pass_number: int
    strait_name: str
    satellite: str
    df: pd.DataFrame  # Main data
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    x_km: np.ndarray
    time_periods: List
    slope_series: np.ndarray
    profile_mean: np.ndarray
    dot_matrix: np.ndarray  # (n_gate_pts, ntime)
    time_array: np.ndarray


# ==============================================================================
# SLCCI SERVICE CLASS
# ==============================================================================

class SLCCIService:
    """
    Service for loading and processing ESA Sea Level CCI data.
    
    Example usage:
        service = SLCCIService()
        
        # Load data for a gate
        pass_data = service.load_pass_data(
            gate_path="/path/to/gate.shp",
            pass_number=248,
        )
        
        # Or find closest pass automatically
        closest_pass = service.find_closest_pass(gate_path="/path/to/gate.shp")
        pass_data = service.load_pass_data(gate_path, pass_number=closest_pass)
    """
    
    def __init__(self, config: Optional[SLCCIConfig] = None):
        """Initialize SLCCI service with configuration."""
        self.config = config or SLCCIConfig()
        self._geoid_interp: Optional[RegularGridInterpolator] = None
        self._gate_points_cache: Dict = {}
        
        # Validate paths exist
        if not os.path.exists(self.config.base_dir):
            logger.warning(f"SLCCI base_dir not found: {self.config.base_dir}")
        if not os.path.exists(self.config.geoid_path):
            logger.warning(f"Geoid file not found: {self.config.geoid_path}")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    @log_call(logger)
    def load_pass_data(
        self,
        gate_path: str,
        pass_number: int,
        cycles: Optional[List[int]] = None,
    ) -> Optional[PassData]:
        """
        Load satellite data for a specific pass and compute DOT analysis.
        
        Parameters
        ----------
        gate_path : str
            Path to gate shapefile
        pass_number : int
            Pass number to load
        cycles : List[int], optional
            Cycles to load (defaults to config.cycles)
            
        Returns
        -------
        PassData
            Container with all analysis results, or None if no data
        """
        import os
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Fix missing .shx files
        
        cycles = cycles or self.config.cycles
        
        logger.info(f"Loading pass {pass_number} for gate: {Path(gate_path).name}")
        
        # 1. Load gate geometry
        gate_gdf = gpd.read_file(gate_path).to_crs("EPSG:4326")
        strait_name = self._extract_strait_name(gate_path)
        
        # 2. Load satellite data
        ds = self._load_filtered_cycles(
            cycles=cycles,
            gate_path=gate_path,
            pass_number=pass_number,
        )
        
        if ds is None or ds.sizes.get("time", 0) == 0:
            logger.warning(f"No data found for pass {pass_number}")
            return None
        
        # 3. Interpolate and add geoid
        geoid_values = self._interpolate_geoid(
            ds["latitude"].values,
            ds["longitude"].values,
        )
        
        # 4. Build DataFrame with DOT
        df = self._build_pass_dataframe(ds, geoid_values, pass_number)
        
        if df is None or len(df) == 0:
            logger.warning(f"Empty DataFrame for pass {pass_number}")
            return None
        
        # 5. Build gate profile points
        gate_lon_pts, gate_lat_pts, x_km = self._get_gate_profile_points(gate_gdf)
        
        # 6. Compute DOT matrix and slope series
        dot_matrix, time_periods = self._build_dot_matrix(df, gate_lon_pts, gate_lat_pts)
        slope_series = self._compute_slope_series(dot_matrix, x_km)
        
        # 7. Compute profile mean
        profile_mean = np.nanmean(dot_matrix, axis=1)
        time_array = np.array([pd.Timestamp(str(p)) for p in time_periods])
        
        satellite = ds.attrs.get("satellite_type", "J2")
        
        logger.info(f"Loaded {len(df)} observations for pass {pass_number}")
        
        return PassData(
            pass_number=pass_number,
            strait_name=strait_name,
            satellite=satellite,
            df=df,
            gate_lon_pts=gate_lon_pts,
            gate_lat_pts=gate_lat_pts,
            x_km=x_km,
            time_periods=time_periods,
            slope_series=slope_series,
            profile_mean=profile_mean,
            dot_matrix=dot_matrix,
            time_array=time_array,
        )
    
    @log_call(logger)
    def find_closest_pass(
        self,
        gate_path: str,
        cycles: Optional[List[int]] = None,
        n_passes: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Find the N closest satellite passes to a gate.
        
        Parameters
        ----------
        gate_path : str
            Path to gate shapefile
        cycles : List[int], optional
            Cycles to search (defaults to subset for speed)
        n_passes : int
            Number of closest passes to return
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (pass_number, distance_km) sorted by distance
        """
        import os
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Fix missing .shx files
        
        cycles = cycles or list(range(1, 100))  # Use subset for speed
        
        gate_gdf = gpd.read_file(gate_path).to_crs("EPSG:4326")
        gate_centroid = gate_gdf.geometry.centroid.iloc[0]
        gate_lon, gate_lat = gate_centroid.x, gate_centroid.y
        
        logger.info(f"Finding closest passes to gate centroid: ({gate_lat:.4f}, {gate_lon:.4f})")
        
        # Load data with expanded bounds
        ds = self._load_filtered_cycles(
            cycles=cycles,
            gate_path=gate_path,
            pass_number=None,  # Load all passes
            lat_buffer_deg=10.0,
            lon_buffer_deg=15.0,
        )
        
        if ds is None or ds.sizes.get("time", 0) == 0:
            logger.warning("No data found for closest pass search")
            return [(1, float('inf'))]
        
        # Get unique passes
        if "pass" not in ds:
            logger.warning("No 'pass' variable in dataset")
            return [(1, float('inf'))]
        
        all_passes = set(int(p) for p in np.unique(ds["pass"].values) if not np.isnan(p))
        
        # Calculate min distance for each pass
        pass_distances = {}
        
        for pass_num in all_passes:
            mask = ds["pass"].values == pass_num
            lons = ds["longitude"].values[mask]
            lats = ds["latitude"].values[mask]
            
            if len(lons) == 0:
                continue
            
            # Calculate distances using Haversine-like approximation
            dlat = (lats - gate_lat) * 111.0  # km per degree
            dlon = (lons - gate_lon) * 111.0 * np.cos(np.radians(gate_lat))
            distances = np.sqrt(dlat**2 + dlon**2)
            
            pass_distances[pass_num] = np.min(distances)
        
        # Sort by distance
        sorted_passes = sorted(pass_distances.items(), key=lambda x: x[1])
        
        return sorted_passes[:n_passes]
    
    def get_available_passes_for_gate(self, gate_path: str) -> List[int]:
        """
        Get list of available pass numbers for a gate.
        
        Extracts from filename first, then searches data.
        """
        # Check filename first
        _, pass_from_filename = self._extract_strait_info(gate_path)
        
        if pass_from_filename:
            return [pass_from_filename]
        
        # Search in data
        closest = self.find_closest_pass(gate_path, n_passes=10)
        return [p[0] for p in closest]
    
    # ==========================================================================
    # PRIVATE METHODS - DATA LOADING
    # ==========================================================================
    
    def _load_filtered_cycles(
        self,
        cycles: List[int],
        gate_path: str,
        pass_number: Optional[int] = None,
        lat_buffer_deg: Optional[float] = None,
        lon_buffer_deg: Optional[float] = None,
    ) -> Optional[xr.Dataset]:
        """
        Load and filter satellite altimetry cycles for a specific region.
        
        Adapted from legacy/j2_utils.py::load_filtered_cycles_serial_J2
        """
        import os
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Fix missing .shx files
        
        lat_buffer = lat_buffer_deg or self.config.lat_buffer_deg
        lon_buffer = lon_buffer_deg or self.config.lon_buffer_deg
        
        # Load gate bounds
        gate = gpd.read_file(gate_path).to_crs("EPSG:4326")
        lon_min_g, lat_min_g, lon_max_g, lat_max_g = gate.total_bounds
        
        lat_min = lat_min_g - lat_buffer
        lat_max = lat_max_g + lat_buffer
        lon_min = self._wrap_longitude(lon_min_g - lon_buffer)
        lon_max = self._wrap_longitude(lon_max_g + lon_buffer)
        
        satellite_type = self._detect_satellite_type()
        
        cycle_datasets = []
        
        for cycle in cycles:
            cycle_str = str(cycle).zfill(3)
            filename = f"SLCCI_ALTDB_{satellite_type}_Cycle{cycle_str}_V2.nc"
            filepath = os.path.join(self.config.base_dir, filename)
            
            if not os.path.exists(filepath):
                continue
            
            try:
                with xr.open_dataset(filepath, decode_times=False) as ds:
                    # Spatial filtering
                    lon = ds["longitude"].values
                    lat = ds["latitude"].values
                    lon_wrapped = self._wrap_longitude(lon)
                    
                    mask_spatial = (
                        (lat >= lat_min) & (lat <= lat_max) &
                        self._lon_in_bounds(lon_wrapped, lon_min, lon_max)
                    )
                    
                    if mask_spatial.sum() == 0:
                        continue
                    
                    ds_filtered = ds.isel(time=mask_spatial)
                    
                    # Update longitude to wrapped values
                    ds_filtered = ds_filtered.assign_coords(
                        longitude=(("time",), lon_wrapped[mask_spatial])
                    )
                    
                    # Decode time
                    time_vals = pd.to_datetime(
                        ds_filtered["time"].values, origin="1950-01-01", unit="D"
                    )
                    ds_filtered = ds_filtered.assign_coords(time=time_vals)
                    
                    # Quality filtering
                    if self.config.use_flag and "validation_flag" in ds_filtered:
                        valid_mask = ds_filtered["validation_flag"] == 0
                        ds_filtered = ds_filtered.isel(time=valid_mask)
                    
                    # Standardize pass variable name
                    for var in ["pass", "track", "pass_number", "track_number"]:
                        if var in ds_filtered.variables:
                            if var != "pass":
                                ds_filtered = ds_filtered.rename({var: "pass"})
                            break
                    
                    # Filter by pass number if specified
                    if pass_number is not None and "pass" in ds_filtered:
                        pass_vals = np.round(ds_filtered["pass"].values).astype(int)
                        mask_pass = pass_vals == int(pass_number)
                        if mask_pass.sum() == 0:
                            continue
                        ds_filtered = ds_filtered.isel(time=mask_pass)
                    
                    if ds_filtered.sizes.get("time", 0) == 0:
                        continue
                    
                    # Add cycle coordinate
                    ds_filtered = ds_filtered.assign_coords(
                        cycle=("time", np.full(ds_filtered.sizes["time"], cycle))
                    )
                    
                    cycle_datasets.append(ds_filtered)
                    
            except Exception as e:
                logger.debug(f"Error loading cycle {cycle}: {e}")
                continue
        
        if not cycle_datasets:
            return None
        
        combined = xr.concat(cycle_datasets, dim="time")
        combined.attrs["satellite_type"] = satellite_type
        
        return combined
    
    # ==========================================================================
    # PRIVATE METHODS - GEOID INTERPOLATION
    # ==========================================================================
    
    def _get_geoid_interpolator(self) -> RegularGridInterpolator:
        """Get or create the geoid interpolator (cached)."""
        if self._geoid_interp is not None:
            return self._geoid_interp
        
        logger.info("Building geoid interpolator...")
        
        ds_geoid = xr.open_dataset(self.config.geoid_path)
        lat_geoid = ds_geoid["lat"].values
        lon_geoid = ds_geoid["lon"].values
        geoid_values = ds_geoid["value"].values
        
        # Wrap and sort longitudes
        lon_wrapped = self._wrap_longitude(lon_geoid)
        sort_idx = np.argsort(lon_wrapped)
        lon_sorted = lon_wrapped[sort_idx]
        
        # Remove duplicates
        unique_idx = np.concatenate(([True], np.diff(lon_sorted) != 0))
        lon_sorted = lon_sorted[unique_idx]
        geoid_sorted = geoid_values[:, sort_idx][:, unique_idx]
        
        self._geoid_interp = RegularGridInterpolator(
            (lat_geoid, lon_sorted),
            geoid_sorted,
            method="nearest",
            bounds_error=False,
            fill_value=np.nan,
        )
        
        return self._geoid_interp
    
    def _interpolate_geoid(
        self,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
    ) -> np.ndarray:
        """Interpolate geoid values at target positions."""
        interp = self._get_geoid_interpolator()
        
        target_lons_wrapped = self._wrap_longitude(target_lons)
        points = np.column_stack([target_lats, target_lons_wrapped])
        
        return interp(points)
    
    # ==========================================================================
    # PRIVATE METHODS - DOT COMPUTATION
    # ==========================================================================
    
    def _build_pass_dataframe(
        self,
        ds: xr.Dataset,
        geoid_values: np.ndarray,
        pass_number: int,
    ) -> Optional[pd.DataFrame]:
        """Build DataFrame with DOT computed from corssh - geoid."""
        if "corssh" not in ds.data_vars:
            logger.warning("No 'corssh' variable in dataset")
            return None
        
        dot = ds["corssh"].values - geoid_values
        
        df = pd.DataFrame({
            "cycle": ds["cycle"].values,
            "pass": pass_number,
            "lat": ds["latitude"].values,
            "lon": ds["longitude"].values,
            "corssh": ds["corssh"].values,
            "geoid": geoid_values,
            "dot": dot,
            "time": pd.to_datetime(ds["time"].values),
        })
        
        df["month"] = df["time"].dt.month
        df["year"] = df["time"].dt.year
        df["year_month"] = df["time"].dt.to_period("M")
        
        return df
    
    def _build_dot_matrix(
        self,
        df: pd.DataFrame,
        gate_lon_pts: np.ndarray,
        gate_lat_pts: np.ndarray,
        search_radius: float = 0.5,
    ) -> Tuple[np.ndarray, List]:
        """
        Build DOT matrix along gate profile for each time period.
        
        Returns
        -------
        dot_matrix : np.ndarray
            Shape (n_gate_pts, n_time_periods)
        time_periods : List
            List of time periods
        """
        time_periods = sorted(df["year_month"].unique())
        n_gate_pts = len(gate_lon_pts)
        n_time = len(time_periods)
        
        dot_matrix = np.full((n_gate_pts, n_time), np.nan, dtype=float)
        gate_coords = np.column_stack([gate_lon_pts, gate_lat_pts])
        
        for it, period in enumerate(time_periods):
            month_data = df[df["year_month"] == period]
            if month_data.empty:
                continue
            
            month_coords = month_data[["lon", "lat"]].to_numpy()
            month_dots = month_data["dot"].to_numpy()
            
            # Use KDTree for spatial matching
            tree = cKDTree(month_coords)
            idx_lists = tree.query_ball_point(gate_coords, r=search_radius)
            
            for ig, idx_near in enumerate(idx_lists):
                if idx_near:
                    dot_matrix[ig, it] = np.nanmean(month_dots[idx_near])
        
        return dot_matrix, time_periods
    
    def _compute_slope_series(
        self,
        dot_matrix: np.ndarray,
        x_km: np.ndarray,
    ) -> np.ndarray:
        """Compute slope time series (m / 100 km) from DOT matrix."""
        n_time = dot_matrix.shape[1]
        slope_series = np.full(n_time, np.nan, dtype=float)
        
        for it in range(n_time):
            y = dot_matrix[:, it]
            mask = np.isfinite(x_km) & np.isfinite(y)
            
            if np.sum(mask) < 2:
                continue
            
            # Linear regression: slope in m/km
            a, _ = np.polyfit(x_km[mask], y[mask], 1)
            slope_series[it] = a * 100.0  # Convert to m / 100 km
        
        return slope_series
    
    # ==========================================================================
    # PRIVATE METHODS - GATE GEOMETRY
    # ==========================================================================
    
    def _get_gate_profile_points(
        self,
        gate_gdf: gpd.GeoDataFrame,
        n_pts: int = 500,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample points along the gate line for profile analysis.
        
        Returns
        -------
        gate_lon_pts : np.ndarray
        gate_lat_pts : np.ndarray
        x_km : np.ndarray (distance along gate in km)
        """
        from shapely.ops import linemerge
        from shapely.geometry import MultiLineString, LineString
        
        gate_bounds = tuple(gate_gdf.total_bounds)
        
        if gate_bounds in self._gate_points_cache:
            return self._gate_points_cache[gate_bounds]
        
        geom = gate_gdf.geometry.unary_union
        if isinstance(geom, MultiLineString):
            geom = linemerge(geom)
        
        if isinstance(geom, LineString):
            total_length = geom.length
            distances = np.linspace(0, total_length, n_pts)
            points = [geom.interpolate(d) for d in distances]
            gate_lon_pts = np.array([p.x for p in points])
            gate_lat_pts = np.array([p.y for p in points])
        else:
            # Fallback for non-linestring geometries
            bounds = gate_gdf.total_bounds
            gate_lon_pts = np.linspace(bounds[0], bounds[2], n_pts)
            gate_lat_pts = np.linspace(bounds[1], bounds[3], n_pts)
        
        # Calculate distance in km
        R_earth = 6371.0
        lat0_rad = np.deg2rad(np.mean(gate_lat_pts))
        lon_rad = np.deg2rad(gate_lon_pts)
        lat_rad = np.deg2rad(gate_lat_pts)
        
        dlon = lon_rad - lon_rad[0]
        dlat = lat_rad - lat_rad[0]
        x_km = R_earth * np.sqrt((dlon * np.cos(lat0_rad))**2 + dlat**2)
        
        result = (gate_lon_pts, gate_lat_pts, x_km)
        self._gate_points_cache[gate_bounds] = result
        
        return result
    
    # ==========================================================================
    # PRIVATE METHODS - UTILITIES
    # ==========================================================================
    
    def _detect_satellite_type(self) -> str:
        """Detect satellite type from base_dir name."""
        dir_name = os.path.basename(self.config.base_dir.rstrip("/"))
        if "J2" in dir_name.upper():
            return "J2"
        return "J1"
    
    @staticmethod
    def _wrap_longitude(lon) -> np.ndarray:
        """Wrap longitude to [-180, 180]."""
        arr = np.asarray(lon, dtype=float)
        return ((arr + 180) % 360) - 180
    
    @staticmethod
    def _lon_in_bounds(lon_wrapped, lon_min, lon_max) -> np.ndarray:
        """Dateline-aware longitude check."""
        if lon_min <= lon_max:
            return (lon_wrapped >= lon_min) & (lon_wrapped <= lon_max)
        # Crosses dateline
        return (lon_wrapped >= lon_min) | (lon_wrapped <= lon_max)
    
    def _extract_strait_name(self, path: str) -> str:
        """Extract strait name from file path."""
        filename = Path(path).stem
        return filename.replace("_", " ").replace("-", " ").title()
    
    def _extract_strait_info(self, path: str) -> Tuple[str, Optional[int]]:
        """Extract strait name and pass number from gate shapefile path."""
        import re
        filename = Path(path).stem
        strait_name = filename.replace("_", " ").replace("-", " ").title()
        match = re.search(r'pass[_\s]*(\d+)', filename, re.IGNORECASE)
        pass_from_filename = int(match.group(1)) if match else None
        return strait_name, pass_from_filename
