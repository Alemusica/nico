"""
Gates Catalog
=============
Central registry for ocean gate definitions.

Loads gate metadata from config/gates.yaml and provides
access to gate information.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from src.core.models import GateModel, BoundingBox


class GateCatalog:
    """
    Central registry for ocean gates.
    
    Loads gate definitions from YAML configuration file and provides
    methods to query and retrieve gate information.
    
    Attributes:
        config_path: Path to gates.yaml configuration file
        gates_dir: Directory containing gate shapefiles
    
    Example:
        >>> catalog = GateCatalog()
        >>> gate = catalog.get("fram_strait")
        >>> print(gate.name)
        🧊 Fram Strait
        
        >>> atlantic_gates = catalog.by_region("Atlantic Sector")
        >>> print(len(atlantic_gates))
        5
    """
    
    def __init__(
        self, 
        config_path: Optional[Path] = None,
        gates_dir: Optional[Path] = None
    ):
        """
        Initialize the gate catalog.
        
        Args:
            config_path: Path to gates.yaml. Defaults to config/gates.yaml
            gates_dir: Directory containing shapefiles. Defaults to gates/
        """
        self.config_path = config_path or Path("config/gates.yaml")
        self.gates_dir = gates_dir or Path("gates/")
        self._gates: Dict[str, GateModel] = {}
        self._metadata: Dict = {}
        self._regions: Dict = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load gate configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Gates config not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self._metadata = data.get("metadata", {})
        self._regions = data.get("regions", {})
        
        # Get default buffer from metadata
        default_buffer = self._metadata.get("default_buffer_km", 50.0)
        
        # Parse gates
        for gate_id, info in data.get("gates", {}).items():
            self._gates[gate_id] = GateModel(
                id=gate_id,
                name=info.get("name", gate_id),
                file=info.get("file", f"{gate_id}.shp"),
                description=info.get("description", ""),
                region=info.get("region", "Unknown"),
                closest_passes=info.get("closest_passes"),
                datasets=info.get("datasets"),
                default_buffer_km=info.get("default_buffer_km", default_buffer),
                latitude_range=info.get("latitude_range"),
                longitude_range=info.get("longitude_range"),
            )
    
    def get(self, gate_id: str) -> Optional[GateModel]:
        """
        Get a gate by ID.
        
        Args:
            gate_id: Unique gate identifier
            
        Returns:
            GateModel if found, None otherwise
        """
        return self._gates.get(gate_id)
    
    def list_all(self) -> List[GateModel]:
        """
        List all available gates.
        
        Returns:
            List of all GateModel instances
        """
        return list(self._gates.values())
    
    def list_ids(self) -> List[str]:
        """
        List all gate IDs.
        
        Returns:
            List of gate ID strings
        """
        return list(self._gates.keys())
    
    def by_region(self, region: str) -> List[GateModel]:
        """
        Get gates in a specific region.
        
        Args:
            region: Region name (e.g., "Atlantic Sector")
            
        Returns:
            List of gates in the specified region
        """
        return [g for g in self._gates.values() if g.region == region]
    
    def get_regions(self) -> List[str]:
        """
        Get list of unique regions.
        
        Returns:
            List of region names
        """
        return list(set(g.region for g in self._gates.values()))
    
    def get_shapefile_path(self, gate_id: str) -> Optional[Path]:
        """
        Get the full path to a gate's shapefile.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            Path to shapefile, or None if gate not found
        """
        gate = self.get(gate_id)
        if not gate:
            return None
        return self.gates_dir / gate.file
    
    def exists(self, gate_id: str) -> bool:
        """
        Check if a gate exists.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            True if gate exists
        """
        return gate_id in self._gates
    
    def get_default_bbox(self, gate_id: str) -> Optional[BoundingBox]:
        """
        Get default bounding box for a gate from config.
        
        This returns the pre-defined bbox without loading the shapefile.
        For accurate bbox from geometry, use GateLoader.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            BoundingBox if defined, None otherwise
        """
        if not self.config_path.exists():
            return None
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        gate_data = data.get("gates", {}).get(gate_id)
        if not gate_data:
            return None
            
        lat_range = gate_data.get("latitude_range")
        lon_range = gate_data.get("longitude_range")
        
        if lat_range and lon_range:
            return BoundingBox(
                lat_min=lat_range[0],
                lat_max=lat_range[1],
                lon_min=lon_range[0],
                lon_max=lon_range[1]
            )
        return None
    
    @property
    def metadata(self) -> Dict:
        """Get catalog metadata."""
        return self._metadata
    
    def __len__(self) -> int:
        """Return number of gates."""
        return len(self._gates)
    
    def __contains__(self, gate_id: str) -> bool:
        """Check if gate exists."""
        return gate_id in self._gates
    
    def __iter__(self):
        """Iterate over gates."""
        return iter(self._gates.values())
