"""
Gate Buffer Calculations
========================
Calculate buffer zones around gate geometries.
"""

from pathlib import Path
from typing import Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import geopandas as gpd
    from shapely.ops import unary_union
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    gpd = None

from src.core.models import BoundingBox


# Approximate km per degree at different latitudes
# At equator: 1 degree ≈ 111 km
# At 60°N: 1 degree lat ≈ 111 km, 1 degree lon ≈ 55.5 km
# At 75°N: 1 degree lat ≈ 111 km, 1 degree lon ≈ 28.7 km
KM_PER_DEGREE_LAT = 111.0


def km_to_degrees(km: float, latitude: float = 0.0) -> Tuple[float, float]:
    """
    Convert kilometers to degrees at a given latitude.
    
    Args:
        km: Distance in kilometers
        latitude: Reference latitude for longitude conversion
        
    Returns:
        (lat_degrees, lon_degrees) tuple
    """
    import math
    lat_deg = km / KM_PER_DEGREE_LAT
    lon_deg = km / (KM_PER_DEGREE_LAT * math.cos(math.radians(latitude)))
    return (lat_deg, lon_deg)


class GateBuffer:
    """
    Calculate buffer zones around gate geometries.
    
    Provides methods to create buffered areas around gates for
    data selection and spatial filtering.
    
    Example:
        >>> buffer = GateBuffer()
        >>> buffered_gdf = buffer.apply_buffer(gdf, buffer_km=50)
        >>> bbox = buffer.get_buffered_bbox(gdf, buffer_km=50)
    """
    
    def __init__(self, default_buffer_km: float = 50.0):
        """
        Initialize buffer calculator.
        
        Args:
            default_buffer_km: Default buffer distance in kilometers
        """
        self.default_buffer_km = default_buffer_km
    
    @property
    def available(self) -> bool:
        """Check if shapely is available."""
        return HAS_SHAPELY
    
    def apply_buffer(
        self,
        geometry: Any,
        buffer_km: Optional[float] = None,
        reference_lat: Optional[float] = None
    ) -> Any:
        """
        Apply buffer to geometry.
        
        Args:
            geometry: GeoDataFrame or Shapely geometry
            buffer_km: Buffer distance in km (default: self.default_buffer_km)
            reference_lat: Latitude for degree conversion (auto-detected if None)
            
        Returns:
            Buffered geometry
        """
        if not HAS_SHAPELY:
            logger.warning("shapely not installed. Cannot apply buffer.")
            return geometry
        
        buffer_km = buffer_km or self.default_buffer_km
        
        # Get reference latitude for conversion
        if reference_lat is None:
            if hasattr(geometry, 'total_bounds'):
                bounds = geometry.total_bounds
                reference_lat = (bounds[1] + bounds[3]) / 2
            else:
                reference_lat = 0.0
        
        # Convert km to degrees (simplified - uses average)
        buffer_deg = buffer_km / KM_PER_DEGREE_LAT
        
        # Apply buffer
        if hasattr(geometry, 'buffer'):
            # GeoDataFrame or GeoSeries
            return geometry.buffer(buffer_deg)
        else:
            logger.warning(f"Cannot buffer geometry of type {type(geometry)}")
            return geometry
    
    def get_buffered_bbox(
        self,
        geometry: Any,
        buffer_km: Optional[float] = None
    ) -> Optional[BoundingBox]:
        """
        Get bounding box of buffered geometry.
        
        Args:
            geometry: GeoDataFrame or Shapely geometry
            buffer_km: Buffer distance in km
            
        Returns:
            BoundingBox of buffered area
        """
        if not HAS_SHAPELY:
            return None
        
        buffered = self.apply_buffer(geometry, buffer_km)
        
        if hasattr(buffered, 'total_bounds'):
            bounds = buffered.total_bounds
        elif hasattr(buffered, 'bounds'):
            bounds = buffered.bounds
        else:
            # Try unary union
            try:
                unified = unary_union(buffered.geometry)
                bounds = unified.bounds
            except Exception:
                return None
        
        return BoundingBox(
            lon_min=bounds[0],
            lat_min=bounds[1],
            lon_max=bounds[2],
            lat_max=bounds[3]
        )
    
    def get_buffered_area(self, geometry: Any, buffer_km: Optional[float] = None) -> Any:
        """
        Get unified buffered area as single geometry.
        
        Args:
            geometry: GeoDataFrame or Shapely geometry
            buffer_km: Buffer distance in km
            
        Returns:
            Single Shapely geometry representing buffered area
        """
        if not HAS_SHAPELY:
            return None
        
        buffered = self.apply_buffer(geometry, buffer_km)
        
        try:
            if hasattr(buffered, 'geometry'):
                return unary_union(buffered.geometry)
            return unary_union(buffered)
        except Exception as e:
            logger.error(f"Error creating buffered area: {e}")
            return None
    
    def expand_bbox(self, bbox: BoundingBox, buffer_km: float) -> BoundingBox:
        """
        Expand a bounding box by a buffer distance.
        
        Args:
            bbox: Original bounding box
            buffer_km: Buffer distance in km
            
        Returns:
            Expanded bounding box
        """
        # Get center latitude for conversion
        center_lat = (bbox.lat_min + bbox.lat_max) / 2
        lat_buffer, lon_buffer = km_to_degrees(buffer_km, center_lat)
        
        return BoundingBox(
            lat_min=max(-90, bbox.lat_min - lat_buffer),
            lat_max=min(90, bbox.lat_max + lat_buffer),
            lon_min=max(-180, bbox.lon_min - lon_buffer),
            lon_max=min(180, bbox.lon_max + lon_buffer)
        )
    
    def point_in_buffered_area(
        self,
        lon: float,
        lat: float,
        geometry: Any,
        buffer_km: Optional[float] = None
    ) -> bool:
        """
        Check if a point is within buffered gate area.
        
        Args:
            lon: Longitude of point
            lat: Latitude of point
            geometry: Gate geometry
            buffer_km: Buffer distance
            
        Returns:
            True if point is within buffered area
        """
        if not HAS_SHAPELY:
            return False
        
        from shapely.geometry import Point
        
        buffered_area = self.get_buffered_area(geometry, buffer_km)
        if buffered_area is None:
            return False
        
        point = Point(lon, lat)
        return buffered_area.contains(point) or buffered_area.intersects(point)
