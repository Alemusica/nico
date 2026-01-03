"""
CMEMS Service
=============
Service for loading and processing Copernicus Marine (CMEMS) L3 1Hz data.

This service processes Jason-1/2/3 along-track altimetry data from CMEMS
and produces PassData objects compatible with the NICO visualization tabs.

Data Flow:
    UI → CMEMSService → load_all_jason_files → NetCDF files (local J1/J2/J3)
                      → filter_by_gate → DataFrame
                      → compute_monthly_slopes → slope_series, v_geostrophic
                      → build_pass_data → PassData object
                      
Key Differences from SLCCI:
    - DOT = sla_filtered + mdt (MDT included, no external geoid)
    - Merges J1, J2, J3 satellites automatically
    - No pass selection (uses gate as "synthetic pass")
    - Lower resolution binning (0.1° default vs 0.01° for SLCCI)
    - Jason coverage limited to ±66° latitude
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, date
import re

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy import stats

from src.core.logging_config import get_logger, log_call

logger = get_logger(__name__)

# Constants for geostrophic calculations
G = 9.81  # m/s² - gravitational acceleration
OMEGA = 7.2921e-5  # rad/s - Earth's angular velocity
R_EARTH = 6371.0  # km - Earth's radius


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CMEMSConfig:
    """Configuration for CMEMS data loading."""
    
    # Data paths
    base_dir: str = "/Users/nicolocaron/Desktop/ARCFRESH/COPERNICUS DATA"
    
    # Data source mode: "local" or "api"
    source_mode: str = "local"
    
    # Processing parameters
    lon_bin_size: float = 0.1  # Binning resolution (degrees)
    buffer_deg: float = 5.0  # Geographic buffer for filtering (degrees) - from Copernicus notebook
    min_points_per_month: int = 10  # Minimum points for valid month
    
    # Jason coverage limit
    max_latitude: float = 66.0  # Jason satellite coverage limit
    
    # Satellites to include (default: all)
    satellites: List[str] = field(default_factory=lambda: ["J1", "J2", "J3"])


@dataclass
class PassData:
    """
    Standard interface for satellite pass data.
    Compatible with tabs.py visualization.
    """
    # Metadata
    strait_name: str
    pass_number: int  # Synthetic pass from gate name
    
    # Tab 1: Slope Timeline
    slope_series: np.ndarray  # Shape: (n_periods,) - m/100km
    time_array: np.ndarray  # Shape: (n_periods,) - datetime objects
    time_periods: List[str]  # ["2002-01", "2002-02", ...]
    
    # Tab 2: DOT Profile
    profile_mean: np.ndarray  # Shape: (n_lon_bins,)
    x_km: np.ndarray  # Shape: (n_lon_bins,) - distance in km
    dot_matrix: np.ndarray  # Shape: (n_lon_bins, n_periods)
    
    # Tab 3 & 4: Spatial & Monthly
    df: pd.DataFrame  # Columns: lat, lon, dot, month, time, satellite, etc.
    gate_lon_pts: np.ndarray
    gate_lat_pts: np.ndarray
    
    # Tab 5: Geostrophic Velocity (NEW)
    v_geostrophic_series: np.ndarray  # Shape: (n_periods,) - m/s
    mean_latitude: float  # For display
    coriolis_f: float  # Coriolis parameter


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _load_gate_gdf(gate_path: str) -> gpd.GeoDataFrame:
    """Load gate shapefile and ensure it's in EPSG:4326."""
    os.environ['SHAPE_RESTORE_SHX'] = 'YES'
    gate_gdf = gpd.read_file(gate_path)
    
    if gate_gdf.crs is None:
        logger.warning(f"Gate {gate_path} has no CRS, assuming EPSG:3413")
        gate_gdf = gate_gdf.set_crs("EPSG:3413")
    
    return gate_gdf.to_crs("EPSG:4326")


def _extract_pass_from_gate_name(gate_path: str) -> Tuple[str, Optional[int]]:
    """
    Extract strait name and pass number from gate filename.
    
    Examples:
        - "barents_sea_opening_S3_pass_481.shp" -> ("Barents Sea Opening", 481)
        - "denmark_strait_TPJ_pass_248.shp" -> ("Denmark Strait", 248)
        - "fram_strait.shp" -> ("Fram Strait", None)
    
    Returns:
        Tuple of (strait_name, pass_number or None if not found)
    """
    filename = Path(gate_path).stem
    pass_number = None
    
    # Pattern 1: _pass_XXX at end (most common)
    match = re.search(r'_pass[_]?(\d+)$', filename, re.IGNORECASE)
    if match:
        pass_number = int(match.group(1))
        logger.info(f"Extracted pass {pass_number} from gate filename (pattern: _pass_XXX)")
    else:
        # Pattern 2: Trailing number after underscore (e.g., gate_name_481)
        match = re.search(r'_(\d{2,4})$', filename)
        if match:
            pass_number = int(match.group(1))
            logger.info(f"Extracted pass {pass_number} from gate filename (pattern: trailing number)")
        else:
            # Pattern 3: pass_XXX anywhere in name
            match = re.search(r'pass[_]?(\d+)', filename, re.IGNORECASE)
            if match:
                pass_number = int(match.group(1))
                logger.info(f"Extracted pass {pass_number} from gate filename (pattern: pass_XXX)")
            else:
                logger.info(f"No pass number found in gate filename: {filename}")
    
    # Clean up strait name
    name_part = re.sub(r'_pass[_]?\d+$', '', filename, flags=re.IGNORECASE)  # Remove _pass_XXX
    name_part = re.sub(r'_\d{2,4}$', '', name_part)  # Remove trailing numbers
    name_part = re.sub(r'_TPJ$', '', name_part, flags=re.IGNORECASE)  # Remove satellite markers
    name_part = re.sub(r'_S\d$', '', name_part)  # Remove _S3, _S2, etc.
    name_part = name_part.replace('_', ' ').replace('-', ' ').title()
    
    return name_part, pass_number


def _deg_to_meters(deg_lon: float, lat_deg: float) -> float:
    """Convert longitude degrees to meters at given latitude."""
    lat_rad = np.deg2rad(lat_deg)
    meters = deg_lon * (np.pi * R_EARTH * 1000 * np.cos(lat_rad) / 180.0)
    return meters


def _coriolis_parameter(lat_deg: float) -> float:
    """Calculate Coriolis parameter f = 2*Omega*sin(lat)."""
    lat_rad = np.deg2rad(lat_deg)
    return 2 * OMEGA * np.sin(lat_rad)


def _lon_to_km(lon_array: np.ndarray, reference_lat: float) -> np.ndarray:
    """Convert longitude array to km distance from minimum."""
    lat_rad = np.deg2rad(reference_lat)
    dlon_rad = np.deg2rad(lon_array) - np.deg2rad(lon_array.min())
    return R_EARTH * dlon_rad * np.cos(lat_rad)


# ==============================================================================
# CMEMS SERVICE CLASS
# ==============================================================================

class CMEMSService:
    """
    Service for loading and processing CMEMS L3 1Hz altimetry data.
    
    Supports both LOCAL files and API access.
    Merges Jason-1, Jason-2, Jason-3 data automatically.
    Produces PassData objects compatible with the standard visualization tabs.
    
    Local file structure expected:
        base_dir/
        ├── J1_netcdf/YYYY/MM/dt_global_j1_phy_l3_1hz_*.nc
        ├── J2_netcdf/YYYY/MM/dt_global_j2_phy_l3_1hz_*.nc
        └── J3_netcdf/YYYY/MM/dt_global_j3_phy_l3_1hz_*.nc
    
    Example usage:
        config = CMEMSConfig(base_dir="/path/to/COPERNICUS DATA", source_mode="local")
        service = CMEMSService(config)
        pass_data = service.load_pass_data(gate_path="/path/to/gate.shp")
    """
    
    def __init__(self, config: Optional[CMEMSConfig] = None):
        self.config = config or CMEMSConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration."""
        base_path = Path(self.config.base_dir)
        if not base_path.exists():
            logger.warning(f"CMEMS base directory not found: {base_path}")
    
    # ==========================================================================
    # PUBLIC METHODS
    # ==========================================================================
    
    def count_files(self) -> Dict[str, int]:
        """
        Count NetCDF files available for each satellite.
        Useful for sidebar display.
        
        Returns:
            Dict with satellite names as keys and file counts as values.
        """
        base_dir = Path(self.config.base_dir)
        counts = {}
        total = 0
        
        for sat in self.config.satellites:
            folder = base_dir / f"{sat}_netcdf"
            if folder.exists():
                count = len(list(folder.rglob("*.nc")))
                counts[sat] = count
                total += count
            else:
                counts[sat] = 0
        
        counts["total"] = total
        return counts
    
    def get_date_range(self) -> Dict[str, Any]:
        """
        Get date range of available data by checking file names.
        Pattern: dt_global_jX_phy_l3_1hz_YYYYMMDD_*.nc
        
        Returns:
            Dict with min_date, max_date, and years list.
        """
        base_dir = Path(self.config.base_dir)
        years = set()
        
        for sat in self.config.satellites:
            folder = base_dir / f"{sat}_netcdf"
            if folder.exists():
                # Check year folders
                for year_folder in folder.iterdir():
                    if year_folder.is_dir() and year_folder.name.isdigit():
                        years.add(int(year_folder.name))
        
        if not years:
            return {"min_date": None, "max_date": None, "years": []}
        
        sorted_years = sorted(years)
        return {
            "min_date": f"{sorted_years[0]}-01-01",
            "max_date": f"{sorted_years[-1]}-12-31",
            "years": sorted_years
        }
    
    @log_call(logger)
    def load_pass_data(
        self, 
        gate_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[PassData]:
        """
        Load CMEMS data for a gate and return PassData object.
        
        Args:
            gate_path: Path to gate shapefile
            progress_callback: Optional callback(processed, total) for progress updates
            
        Returns:
            PassData object compatible with tabs.py visualization
        """
        # Load gate
        gate_gdf = _load_gate_gdf(gate_path)
        strait_name, pass_number = _extract_pass_from_gate_name(gate_path)
        
        logger.info(f"Loading CMEMS data for {strait_name} (pass {pass_number})")
        
        # Check latitude coverage warning
        bounds = gate_gdf.total_bounds  # [minx, miny, maxx, maxy]
        if bounds[3] > self.config.max_latitude:
            logger.warning(
                f"Gate {strait_name} extends beyond Jason coverage ({self.config.max_latitude}°N). "
                f"Data will be limited to ±{self.config.max_latitude}°."
            )
        
        # Get gate geometry for profile
        gate_lon_pts, gate_lat_pts = self._get_gate_points(gate_gdf)
        
        # Calculate gate bounds with buffer
        gate_bounds = {
            "lon_min": bounds[0] - self.config.buffer_deg,
            "lon_max": bounds[2] + self.config.buffer_deg,
            "lat_min": max(bounds[1] - self.config.buffer_deg, -self.config.max_latitude),
            "lat_max": min(bounds[3] + self.config.buffer_deg, self.config.max_latitude),
        }
        
        # Load all Jason data with progress callback
        df = self._load_all_jason_files(gate_bounds, progress_callback=progress_callback)
        
        if df is None or df.empty:
            logger.error(f"No data found for gate {strait_name}")
            return None
        
        logger.info(f"Loaded {len(df):,} observations from {df['time'].min()} to {df['time'].max()}")
        
        # Add month column
        df['month'] = df['time'].dt.month
        
        # Compute monthly slopes and geostrophic velocity
        monthly_results = self._compute_monthly_slopes(df)
        
        if monthly_results.empty:
            logger.error(f"No valid monthly results for {strait_name}")
            return None
        
        # Build DOT matrix
        dot_matrix, time_periods, x_km, lon_bins = self._build_dot_matrix(
            df, gate_lon_pts, gate_lat_pts
        )
        
        # Compute profile mean
        profile_mean = np.nanmean(dot_matrix, axis=1)
        
        # Extract time series
        slope_series = monthly_results['slope_m_100km'].values
        v_geostrophic_series = monthly_results['v_geostrophic_m_s'].values
        time_array = monthly_results['date'].values
        time_period_labels = [str(p) for p in monthly_results['period'].values]
        
        # Mean latitude for geostrophic
        mean_lat = df['lat'].mean()
        f_coriolis = _coriolis_parameter(mean_lat)
        
        return PassData(
            strait_name=strait_name,
            pass_number=pass_number,
            slope_series=slope_series,
            time_array=time_array,
            time_periods=time_period_labels,
            profile_mean=profile_mean,
            x_km=x_km,
            dot_matrix=dot_matrix,
            df=df,
            gate_lon_pts=gate_lon_pts,
            gate_lat_pts=gate_lat_pts,
            v_geostrophic_series=v_geostrophic_series,
            mean_latitude=mean_lat,
            coriolis_f=f_coriolis,
        )
    
    def check_gate_coverage(self, gate_path: str) -> Dict[str, Any]:
        """
        Check if gate is within Jason coverage.
        
        Returns dict with coverage info and warnings.
        """
        gate_gdf = _load_gate_gdf(gate_path)
        bounds = gate_gdf.total_bounds
        
        is_within = bounds[3] <= self.config.max_latitude and bounds[1] >= -self.config.max_latitude
        
        return {
            "gate_lat_min": bounds[1],
            "gate_lat_max": bounds[3],
            "jason_limit": self.config.max_latitude,
            "is_fully_covered": is_within,
            "warning": None if is_within else (
                f"Gate extends beyond Jason coverage (±{self.config.max_latitude}°). "
                f"Data will be limited."
            )
        }
    
    # ==========================================================================
    # PRIVATE METHODS - DATA LOADING
    # ==========================================================================
    
    def _load_all_jason_files(
        self, 
        gate_bounds: Dict[str, float],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load and merge all Jason-1/2/3 files within gate bounds.
        
        Parameters
        ----------
        gate_bounds : dict
            Geographic bounds (lon_min, lon_max, lat_min, lat_max)
        progress_callback : callable, optional
            Function(processed: int, total: int) called during processing
            
        File structure expected:
            base_dir/J1_netcdf/YYYY/MM/*.nc
            base_dir/J2_netcdf/YYYY/MM/*.nc
            base_dir/J3_netcdf/YYYY/MM/*.nc
        """
        
        base_dir = Path(self.config.base_dir)
        
        # Map satellite config names to folder names
        jason_folders = {}
        for sat in self.config.satellites:
            folder = base_dir / f"{sat}_netcdf"
            if folder.exists():
                jason_folders[sat] = folder
            else:
                logger.warning(f"Folder not found: {folder}")
        
        if not jason_folders:
            logger.error(f"No satellite folders found in {base_dir}")
            return None
        
        # Find all NetCDF files (recursive search in YYYY/MM structure)
        all_files = []
        for jason, folder in jason_folders.items():
            # Use rglob to find all .nc files recursively
            nc_files = list(folder.rglob("*.nc"))
            all_files.extend([(jason, f) for f in nc_files])
            logger.info(f"{jason}: {len(nc_files)} files found in {folder}")
        
        if not all_files:
            logger.error("No NetCDF files found")
            return None
        
        total_files = len(all_files)
        logger.info(f"Processing {total_files} total files (this may take a while)...")
        
        # Process files with progress
        data_chunks = []
        processed = 0
        skipped = 0
        with_data = 0
        
        for jason, nc_file in all_files:
            processed += 1
            
            # Call progress callback if provided
            if progress_callback is not None and processed % 100 == 0:
                try:
                    progress_callback(processed, total_files)
                except Exception:
                    pass  # Don't fail if callback fails
            
            if processed % 500 == 0:
                logger.info(f"Progress: {processed}/{total_files} files, {with_data} with data in gate")
            
            try:
                chunk = self._process_single_file(jason, nc_file, gate_bounds)
                if chunk is not None and len(chunk) > 0:
                    data_chunks.append(chunk)
                    with_data += 1
            except Exception as e:
                skipped += 1
                if skipped % 50 == 0:
                    logger.debug(f"Skipped {skipped} files (last error: {e})")
        
        # Final callback
        if progress_callback is not None:
            try:
                progress_callback(total_files, total_files)
            except Exception:
                pass
        
        logger.info(f"Processed: {processed}, Skipped: {skipped}, With data: {with_data}")
        
        if not data_chunks:
            return None
        
        # Combine all chunks
        df = pd.concat(data_chunks, ignore_index=True)
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        df = df.dropna(subset=['time'])
        df = df.sort_values('time')
        
        # NO date range filter - use ALL data in the folder
        logger.info(f"Total observations: {len(df):,} from {df['time'].min()} to {df['time'].max()}")
        
        return df
    
    def _process_single_file(
        self, 
        satellite: str, 
        nc_file: Path, 
        gate_bounds: Dict[str, float]
    ) -> Optional[pd.DataFrame]:
        """Process a single NetCDF file and return filtered DataFrame."""
        
        ds = xr.open_dataset(nc_file)
        
        # Check required variables
        if 'sla_filtered' not in ds.variables or 'mdt' not in ds.variables:
            ds.close()
            return None
        
        # Extract arrays
        lats = ds['latitude'].values
        lons = ds['longitude'].values
        
        # Handle time
        if 'time' in ds.variables:
            time_var = ds['time']
            if np.issubdtype(time_var.dtype, np.datetime64):
                times = time_var.values
            else:
                times = pd.to_datetime(time_var.values, unit='s', errors='coerce')
        else:
            ds.close()
            return None
        
        # Check dimensions
        if not (len(lats) == len(lons) == len(times)):
            ds.close()
            return None
        
        # Calculate DOT
        dot = (ds['sla_filtered'] + ds['mdt']).values
        
        if len(dot) != len(lats):
            ds.close()
            return None
        
        # Geographic filter
        mask = (
            (lons >= gate_bounds['lon_min']) & (lons <= gate_bounds['lon_max']) &
            (lats >= gate_bounds['lat_min']) & (lats <= gate_bounds['lat_max']) &
            np.isfinite(dot)
        )
        
        if np.sum(mask) == 0:
            ds.close()
            return None
        
        # Build DataFrame
        df = pd.DataFrame({
            'satellite': satellite,
            'time': times[mask],
            'lat': lats[mask],
            'lon': lons[mask],
            'sla_filtered': ds['sla_filtered'].values[mask],
            'mdt': ds['mdt'].values[mask],
            'dot': dot[mask],
            'cycle': ds['cycle'].values[mask] if 'cycle' in ds.variables else -1,
            'track': ds['track'].values[mask] if 'track' in ds.variables else -1,
        })
        
        ds.close()
        
        # Convert time
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        df = df.dropna(subset=['time'])
        
        return df if len(df) > 0 else None
    
    # ==========================================================================
    # PRIVATE METHODS - SLOPE & GEOSTROPHIC COMPUTATION
    # ==========================================================================
    
    def _compute_monthly_slopes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute monthly DOT slopes and geostrophic velocities.
        
        Returns DataFrame with columns:
            period, date, n_obs, slope_m_deg, slope_m_100km, v_geostrophic_m_s, r_squared
        """
        df['year_month'] = df['time'].dt.to_period('M')
        
        mean_lat = df['lat'].mean()
        f_coriolis = _coriolis_parameter(mean_lat)
        meters_per_deg = _deg_to_meters(1.0, mean_lat)
        
        monthly_groups = df.groupby('year_month')
        results = []
        
        for period, group in monthly_groups:
            if len(group) < self.config.min_points_per_month:
                continue
            
            # Sort by longitude
            group_sorted = group.sort_values('lon')
            
            # Binning
            lon_bins = np.arange(
                group_sorted['lon'].min(),
                group_sorted['lon'].max() + self.config.lon_bin_size,
                self.config.lon_bin_size
            )
            
            if len(lon_bins) < 3:
                continue
            
            # Compute bin means
            lon_centers = []
            dot_means = []
            
            for i in range(len(lon_bins) - 1):
                mask = (
                    (group_sorted['lon'] >= lon_bins[i]) & 
                    (group_sorted['lon'] < lon_bins[i+1])
                )
                dots = group_sorted.loc[mask, 'dot']
                
                if len(dots) > 0:
                    lon_centers.append((lon_bins[i] + lon_bins[i+1]) / 2)
                    dot_means.append(dots.mean())
            
            if len(lon_centers) < 3:
                continue
            
            lon_arr = np.array(lon_centers)
            dot_arr = np.array(dot_means)
            
            # Linear regression
            slope_deg, intercept, r_val, p_val, std_err = stats.linregress(lon_arr, dot_arr)
            
            # Convert slope
            slope_m_per_m = slope_deg / meters_per_deg
            slope_m_100km = slope_m_per_m * 100000  # m per 100km
            
            # Geostrophic velocity: v = -g/f * (dη/dx)
            v_geostrophic = -(G / f_coriolis) * slope_m_per_m
            
            results.append({
                'period': period,
                'date': period.to_timestamp(),
                'n_obs': len(group),
                'slope_m_deg': slope_deg,
                'slope_m_100km': slope_m_100km,
                'v_geostrophic_m_s': v_geostrophic,
                'r_squared': r_val**2,
                'p_value': p_val,
            })
        
        return pd.DataFrame(results)
    
    # ==========================================================================
    # PRIVATE METHODS - DOT MATRIX
    # ==========================================================================
    
    def _build_dot_matrix(
        self,
        df: pd.DataFrame,
        gate_lon_pts: np.ndarray,
        gate_lat_pts: np.ndarray,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Build DOT matrix [n_lon_bins, n_time_periods] for profile visualization.
        """
        # Use gate longitude extent
        lon_min, lon_max = gate_lon_pts.min(), gate_lon_pts.max()
        lon_bins = np.arange(lon_min, lon_max + self.config.lon_bin_size, self.config.lon_bin_size)
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        
        # Time periods (monthly)
        df['year_month'] = df['time'].dt.to_period('M')
        periods = sorted(df['year_month'].unique())
        
        # Initialize matrix
        n_lon = len(lon_centers)
        n_time = len(periods)
        dot_matrix = np.full((n_lon, n_time), np.nan)
        
        # Fill matrix
        for t_idx, period in enumerate(periods):
            period_df = df[df['year_month'] == period]
            
            for i in range(len(lon_bins) - 1):
                mask = (
                    (period_df['lon'] >= lon_bins[i]) & 
                    (period_df['lon'] < lon_bins[i+1])
                )
                dots = period_df.loc[mask, 'dot']
                
                if len(dots) > 0:
                    dot_matrix[i, t_idx] = dots.mean()
        
        # Compute x_km (distance along longitude)
        mean_lat = gate_lat_pts.mean()
        x_km = _lon_to_km(lon_centers, mean_lat)
        
        time_period_labels = [str(p) for p in periods]
        
        return dot_matrix, time_period_labels, x_km, lon_centers
    
    # ==========================================================================
    # PRIVATE METHODS - GATE GEOMETRY
    # ==========================================================================
    
    def _get_gate_points(
        self, 
        gate_gdf: gpd.GeoDataFrame, 
        n_pts: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract longitude and latitude points along gate line."""
        
        geom = gate_gdf.geometry.iloc[0]
        
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
        elif geom.geom_type == 'MultiLineString':
            coords = []
            for line in geom.geoms:
                coords.extend(list(line.coords))
        else:
            # Point or Polygon - use centroid
            centroid = geom.centroid
            return np.array([centroid.x]), np.array([centroid.y])
        
        lons = np.array([c[0] for c in coords])
        lats = np.array([c[1] for c in coords])
        
        # Interpolate to n_pts
        if len(lons) > 1:
            t_orig = np.linspace(0, 1, len(lons))
            t_new = np.linspace(0, 1, n_pts)
            lons = np.interp(t_new, t_orig, lons)
            lats = np.interp(t_new, t_orig, lats)
        
        return lons, lats
