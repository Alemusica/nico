"""
Gate Service
============
Business logic for gate operations.
Used by both Streamlit and FastAPI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from src.core.models import GateModel, BoundingBox
from src.gates.catalog import GateCatalog
from src.gates.loader import GateLoader
from src.gates.buffer import GateBuffer
from src.gates.passes import PassFilter

logger = logging.getLogger(__name__)


class GateService:
    """
    Service for gate-related operations.
    
    Provides a unified interface for:
    - Selecting gates
    - Loading geometries
    - Getting bounding boxes with buffer
    - Getting satellite passes
    
    This service is used by both Streamlit sidebar and API endpoints.
    
    Example:
        >>> service = GateService()
        >>> gate = service.select_gate("fram_strait")
        >>> bbox = service.get_bbox("fram_strait", buffer_km=50)
        >>> passes = service.get_closest_passes("fram_strait")
    """
    
    def __init__(
        self,
        gates_dir: Optional[Path] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize gate service.
        
        Args:
            gates_dir: Directory containing gate shapefiles
            config_path: Path to gates.yaml configuration
        """
        self.gates_dir = Path(gates_dir) if gates_dir else Path("gates/")
        self.config_path = Path(config_path) if config_path else Path("config/gates.yaml")
        
        # Initialize components
        self._catalog = GateCatalog(config_path=self.config_path, gates_dir=self.gates_dir)
        self._loader = GateLoader(gates_dir=self.gates_dir)
        self._buffer = GateBuffer()
        self._pass_filter = PassFilter()
        
        # Cache for loaded geometries
        self._geometry_cache: Dict[str, Any] = {}
    
    @property
    def catalog(self) -> GateCatalog:
        """Get the gate catalog."""
        return self._catalog
    
    def select_gate(self, gate_id: str) -> Optional[GateModel]:
        """
        Select a gate by ID.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            GateModel if found, None otherwise
        """
        return self._catalog.get(gate_id)
    
    def list_gates(self) -> List[GateModel]:
        """
        List all available gates.
        
        Returns:
            List of all gates
        """
        return self._catalog.list_all()
    
    def list_gates_by_region(self, region: str) -> List[GateModel]:
        """
        List gates in a specific region.
        
        Args:
            region: Region name
            
        Returns:
            List of gates in region
        """
        return self._catalog.by_region(region)
    
    def get_regions(self) -> List[str]:
        """
        Get list of unique regions.
        
        Returns:
            List of region names
        """
        return self._catalog.get_regions()
    
    def load_geometry(self, gate_id: str) -> Optional[Any]:
        """
        Load gate shapefile geometry.
        
        Uses caching to avoid reloading.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            GeoDataFrame with gate geometry
        """
        if gate_id in self._geometry_cache:
            return self._geometry_cache[gate_id]
        
        gate = self._catalog.get(gate_id)
        if not gate:
            logger.warning(f"Gate not found: {gate_id}")
            return None
        
        gdf = self._loader.load(gate_id, gate.file)
        if gdf is not None:
            self._geometry_cache[gate_id] = gdf
        
        return gdf
    
    def get_bbox(
        self,
        gate_id: str,
        buffer_km: float = 50.0,
        use_geometry: bool = True
    ) -> Optional[BoundingBox]:
        """
        Get bounding box for a gate with buffer.
        
        Args:
            gate_id: Gate identifier
            buffer_km: Buffer distance in kilometers
            use_geometry: If True, load shapefile for accurate bbox.
                         If False, use pre-defined bbox from config.
            
        Returns:
            BoundingBox with buffer applied
        """
        if use_geometry:
            gdf = self.load_geometry(gate_id)
            if gdf is not None:
                return self._buffer.get_buffered_bbox(gdf, buffer_km)
        
        # Fallback to config bbox
        bbox = self._catalog.get_default_bbox(gate_id)
        if bbox:
            return self._buffer.expand_bbox(bbox, buffer_km)
        
        logger.warning(f"Could not get bbox for gate: {gate_id}")
        return None
    
    def get_closest_passes(self, gate_id: str, n: int = 5) -> List[int]:
        """
        Get pre-computed closest satellite passes.
        
        Args:
            gate_id: Gate identifier
            n: Number of passes to return
            
        Returns:
            List of pass numbers
        """
        # First check catalog
        gate = self._catalog.get(gate_id)
        if gate and gate.closest_passes:
            return gate.closest_passes[:n]
        
        # Fallback to pass filter
        return self._pass_filter.get_closest_passes(gate_id, n)
    
    def get_primary_pass(self, gate_id: str) -> Optional[int]:
        """
        Get the primary (closest) satellite pass.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            Primary pass number
        """
        passes = self.get_closest_passes(gate_id, n=1)
        return passes[0] if passes else None
    
    def get_gate_info(self, gate_id: str, buffer_km: float = 50.0) -> Optional[Dict]:
        """
        Get complete gate information including bbox and passes.
        
        Args:
            gate_id: Gate identifier
            buffer_km: Buffer for bbox
            
        Returns:
            Dict with gate info, bbox, and passes
        """
        gate = self.select_gate(gate_id)
        if not gate:
            return None
        
        bbox = self.get_bbox(gate_id, buffer_km)
        passes = self.get_closest_passes(gate_id)
        
        return {
            "gate": gate.model_dump(),
            "bbox": bbox.model_dump() if bbox else None,
            "closest_passes": passes,
            "geometry_available": self._loader.available and gate_id in self._geometry_cache
        }
    
    def point_in_gate_area(
        self,
        gate_id: str,
        lon: float,
        lat: float,
        buffer_km: float = 50.0
    ) -> bool:
        """
        Check if a point is within buffered gate area.
        
        Args:
            gate_id: Gate identifier
            lon: Longitude
            lat: Latitude
            buffer_km: Buffer distance
            
        Returns:
            True if point is in gate area
        """
        gdf = self.load_geometry(gate_id)
        if gdf is None:
            # Fallback to bbox check
            bbox = self.get_bbox(gate_id, buffer_km)
            if bbox:
                return (bbox.lat_min <= lat <= bbox.lat_max and
                        bbox.lon_min <= lon <= bbox.lon_max)
            return False
        
        return self._buffer.point_in_buffered_area(lon, lat, gdf, buffer_km)
    
    def clear_cache(self, gate_id: Optional[str] = None) -> None:
        """
        Clear geometry cache.
        
        Args:
            gate_id: Specific gate to clear, or None for all
        """
        if gate_id:
            self._geometry_cache.pop(gate_id, None)
            self._loader.clear_cache(gate_id)
        else:
            self._geometry_cache.clear()
            self._loader.clear_cache()
    
    def gate_exists(self, gate_id: str) -> bool:
        """Check if gate exists."""
        return self._catalog.exists(gate_id)
