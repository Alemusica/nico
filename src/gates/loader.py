"""
Gate Shapefile Loader
=====================
Load gate geometries from shapefiles using geopandas.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Optional geopandas import
try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    gpd = None

from src.core.models import BoundingBox


class GateLoader:
    """
    Load gate geometries from shapefiles.
    
    Uses geopandas to load ESRI shapefiles and extract geometry information.
    Caches loaded geometries for performance.
    
    Attributes:
        gates_dir: Directory containing gate shapefiles
        crs: Coordinate reference system (default: EPSG:4326)
    
    Example:
        >>> loader = GateLoader()
        >>> gdf = loader.load("fram_strait")
        >>> print(gdf.total_bounds)
        [-20.0, 78.0, 10.0, 80.0]
    """
    
    def __init__(
        self,
        gates_dir: Optional[Path] = None,
        crs: str = "EPSG:4326"
    ):
        """
        Initialize the gate loader.
        
        Args:
            gates_dir: Directory containing shapefiles
            crs: Coordinate reference system
        """
        self.gates_dir = Path(gates_dir) if gates_dir else Path("gates/")
        self.crs = crs
        self._cache: Dict[str, Any] = {}
    
    @property
    def available(self) -> bool:
        """Check if geopandas is available."""
        return HAS_GEOPANDAS
    
    def load(self, gate_id: str, filename: Optional[str] = None) -> Optional[Any]:
        """
        Load gate geometry from shapefile.
        
        Args:
            gate_id: Gate identifier (used for caching)
            filename: Shapefile filename. If None, uses {gate_id}.shp
            
        Returns:
            GeoDataFrame with gate geometry, or None if loading fails
        """
        if not HAS_GEOPANDAS:
            logger.warning("geopandas not installed. Cannot load shapefiles.")
            return None
        
        # Check cache
        if gate_id in self._cache:
            return self._cache[gate_id]
        
        # Determine filepath
        if filename:
            filepath = self.gates_dir / filename
        else:
            # Try common patterns
            patterns = [
                f"{gate_id}.shp",
                f"{gate_id}_*.shp",
            ]
            filepath = None
            for pattern in patterns:
                matches = list(self.gates_dir.glob(pattern))
                if matches:
                    filepath = matches[0]
                    break
            
            if not filepath:
                logger.warning(f"No shapefile found for gate: {gate_id}")
                return None
        
        if not filepath.exists():
            logger.warning(f"Shapefile not found: {filepath}")
            return None
        
        try:
            gdf = gpd.read_file(filepath)
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf = gdf.set_crs(self.crs)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Cache result
            self._cache[gate_id] = gdf
            logger.info(f"Loaded gate geometry: {gate_id} ({len(gdf)} features)")
            
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading shapefile {filepath}: {e}")
            return None
    
    def load_from_catalog(self, gate_id: str, catalog: Any) -> Optional[Any]:
        """
        Load gate geometry using catalog information.
        
        Args:
            gate_id: Gate identifier
            catalog: GateCatalog instance
            
        Returns:
            GeoDataFrame with gate geometry
        """
        gate = catalog.get(gate_id)
        if not gate:
            return None
        return self.load(gate_id, gate.file)
    
    def get_bounds(self, gate_id: str, filename: Optional[str] = None) -> Optional[BoundingBox]:
        """
        Get bounding box from gate geometry.
        
        Args:
            gate_id: Gate identifier
            filename: Optional shapefile filename
            
        Returns:
            BoundingBox from geometry bounds
        """
        gdf = self.load(gate_id, filename)
        if gdf is None:
            return None
        
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        return BoundingBox(
            lon_min=bounds[0],
            lat_min=bounds[1],
            lon_max=bounds[2],
            lat_max=bounds[3]
        )
    
    def get_centroid(self, gate_id: str) -> Optional[tuple]:
        """
        Get centroid of gate geometry.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            (lon, lat) tuple of centroid
        """
        gdf = self.load(gate_id)
        if gdf is None:
            return None
        
        centroid = gdf.geometry.unary_union.centroid
        return (centroid.x, centroid.y)
    
    def clear_cache(self, gate_id: Optional[str] = None) -> None:
        """
        Clear geometry cache.
        
        Args:
            gate_id: Specific gate to clear, or None for all
        """
        if gate_id:
            self._cache.pop(gate_id, None)
        else:
            self._cache.clear()
    
    def is_loaded(self, gate_id: str) -> bool:
        """Check if gate is in cache."""
        return gate_id in self._cache
    
    def list_available_shapefiles(self) -> list:
        """List all shapefile names in gates directory."""
        if not self.gates_dir.exists():
            return []
        return [f.name for f in self.gates_dir.glob("*.shp")]
