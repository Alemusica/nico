"""
Coordinate utilities for Danish coast and North Sea regions.

Defines vulnerable coastal segments, tide gauge locations, and grid helpers.
"""

from dataclasses import dataclass
from typing import NamedTuple
import numpy as np


class LatLon(NamedTuple):
    """Simple lat/lon coordinate."""
    lat: float
    lon: float


@dataclass
class BoundingBox:
    """Rectangular bounding box in lat/lon."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is inside box."""
        return (self.lat_min <= lat <= self.lat_max and
                self.lon_min <= lon <= self.lon_max)
    
    def to_slice(self, lat_arr: np.ndarray, lon_arr: np.ndarray) -> tuple:
        """Get array slices for this box."""
        lat_mask = (lat_arr >= self.lat_min) & (lat_arr <= self.lat_max)
        lon_mask = (lon_arr >= self.lon_min) & (lon_arr <= self.lon_max)
        return lat_mask, lon_mask


# =============================================================================
# Danish Coastal Regions (Vulnerable to Storm Surge)
# =============================================================================

COASTAL_REGIONS = {
    "jutland_west": BoundingBox(
        lat_min=54.8, lat_max=57.5,
        lon_min=7.5, lon_max=9.0
    ),
    "wadden_sea": BoundingBox(
        lat_min=54.8, lat_max=55.5,
        lon_min=8.0, lon_max=9.0
    ),
    "limfjord": BoundingBox(
        lat_min=56.5, lat_max=57.2,
        lon_min=8.5, lon_max=10.5
    ),
    "kattegat": BoundingBox(
        lat_min=55.5, lat_max=57.5,
        lon_min=10.0, lon_max=12.5
    ),
    "copenhagen": BoundingBox(
        lat_min=55.5, lat_max=55.9,
        lon_min=12.3, lon_max=12.8
    ),
    "south_zealand": BoundingBox(
        lat_min=54.5, lat_max=55.5,
        lon_min=11.0, lon_max=12.5
    ),
    "bornholm": BoundingBox(
        lat_min=54.9, lat_max=55.4,
        lon_min=14.5, lon_max=15.2
    ),
}


# =============================================================================
# DMI Tide Gauge Stations
# =============================================================================

TIDE_GAUGES = {
    "esbjerg": LatLon(55.4667, 8.4500),
    "hvide_sande": LatLon(56.0000, 8.1333),
    "thyboron": LatLon(56.7000, 8.2167),
    "hirtshals": LatLon(57.5833, 9.9500),
    "frederikshavn": LatLon(57.4333, 10.5333),
    "aarhus": LatLon(56.1500, 10.2167),
    "copenhagen": LatLon(55.6761, 12.5683),
    "korsor": LatLon(55.3333, 11.1333),
    "gedser": LatLon(54.5667, 11.9333),
    "rodby": LatLon(54.6500, 11.3500),
}


# =============================================================================
# Remote Forcing Regions (Teleconnections)
# =============================================================================

REMOTE_REGIONS = {
    "iberian_atlantic": BoundingBox(
        lat_min=35.0, lat_max=45.0,
        lon_min=-15.0, lon_max=-5.0
    ),
    "biscay": BoundingBox(
        lat_min=43.0, lat_max=48.0,
        lon_min=-10.0, lon_max=0.0
    ),
    "uk_atlantic": BoundingBox(
        lat_min=48.0, lat_max=60.0,
        lon_min=-15.0, lon_max=-5.0
    ),
    "norwegian_sea": BoundingBox(
        lat_min=60.0, lat_max=70.0,
        lon_min=-5.0, lon_max=15.0
    ),
    "central_north_sea": BoundingBox(
        lat_min=54.0, lat_max=58.0,
        lon_min=0.0, lon_max=8.0
    ),
}


# =============================================================================
# Grid Utilities
# =============================================================================

def create_grid(bbox: BoundingBox, resolution: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """
    Create regular lat/lon grid for a bounding box.
    
    Args:
        bbox: Bounding box
        resolution: Grid spacing in degrees
        
    Returns:
        lat_grid, lon_grid: 2D arrays
    """
    lats = np.arange(bbox.lat_min, bbox.lat_max + resolution, resolution)
    lons = np.arange(bbox.lon_min, bbox.lon_max + resolution, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid, lon_grid


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points [km].
    
    Simple formula, good enough for North Sea scales.
    """
    R = 6371.0  # Earth radius [km]
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def find_nearest_gauge(lat: float, lon: float) -> tuple[str, float]:
    """
    Find nearest tide gauge to a given point.
    
    Returns:
        (gauge_name, distance_km)
    """
    min_dist = float('inf')
    nearest = None
    
    for name, coords in TIDE_GAUGES.items():
        dist = haversine_distance(lat, lon, coords.lat, coords.lon)
        if dist < min_dist:
            min_dist = dist
            nearest = name
    
    return nearest, min_dist
