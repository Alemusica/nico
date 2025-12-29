"""
Satellite Pass Filtering
========================
Filter satellite data by pass number relative to ocean gates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import logging

logger = logging.getLogger(__name__)

# Optional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from shapely.geometry import Point
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# Pre-computed closest passes for each gate
# These are the satellite ground tracks that pass closest to each gate
GATE_CLOSEST_PASSES: Dict[str, List[int]] = {
    "fram_strait": [481, 254, 127, 308, 55],
    "bering_strait": [76, 152, 228, 304, 380],
    "denmark_strait": [246, 172, 320, 98, 394],
    "davis_strait": [180, 106, 254, 32, 328],
    "barents_opening": [481, 254, 127, 308, 55],
    "norwegian_boundary": [220, 146, 294, 72, 368],
    "nares_strait": [167, 93, 241, 19, 315],
    "lancaster_sound": [195, 121, 269, 47, 343],
}


class PassFilter:
    """
    Filter satellite data by pass number.
    
    Provides methods to:
    - Get closest passes to a gate
    - Filter datasets by pass number
    - Find passes that intersect gate areas
    
    Example:
        >>> pf = PassFilter()
        >>> passes = pf.get_closest_passes("fram_strait")
        >>> print(passes)
        [481, 254, 127, 308, 55]
        
        >>> filtered = pf.filter_by_pass(dataset, pass_number=481)
    """
    
    def __init__(self, closest_passes: Optional[Dict[str, List[int]]] = None):
        """
        Initialize pass filter.
        
        Args:
            closest_passes: Custom closest passes dict (default: GATE_CLOSEST_PASSES)
        """
        self._closest_passes = closest_passes or GATE_CLOSEST_PASSES.copy()
    
    def get_closest_passes(self, gate_id: str, n: int = 5) -> List[int]:
        """
        Get closest satellite passes for a gate.
        
        Args:
            gate_id: Gate identifier
            n: Number of passes to return
            
        Returns:
            List of pass numbers, sorted by proximity
        """
        passes = self._closest_passes.get(gate_id, [])
        return passes[:n]
    
    def get_primary_pass(self, gate_id: str) -> Optional[int]:
        """
        Get the primary (closest) pass for a gate.
        
        Args:
            gate_id: Gate identifier
            
        Returns:
            Primary pass number, or None
        """
        passes = self.get_closest_passes(gate_id, n=1)
        return passes[0] if passes else None
    
    def has_passes(self, gate_id: str) -> bool:
        """Check if gate has pre-computed passes."""
        return gate_id in self._closest_passes and len(self._closest_passes[gate_id]) > 0
    
    def add_gate_passes(self, gate_id: str, passes: List[int]) -> None:
        """
        Add or update closest passes for a gate.
        
        Args:
            gate_id: Gate identifier
            passes: List of pass numbers
        """
        self._closest_passes[gate_id] = passes
    
    def filter_by_pass(
        self,
        dataset: Any,
        pass_number: int,
        pass_var: str = "pass"
    ) -> Any:
        """
        Filter xarray dataset by pass number.
        
        Args:
            dataset: xarray Dataset
            pass_number: Pass number to filter
            pass_var: Name of pass variable in dataset
            
        Returns:
            Filtered dataset
        """
        if not HAS_NUMPY:
            logger.warning("numpy not installed. Cannot filter by pass.")
            return dataset
        
        # Try different pass variable names
        pass_vars = [pass_var, "pass_number", "pass_num", "track"]
        
        for var in pass_vars:
            if var in dataset:
                pass_values = dataset[var].values
                if pass_values.ndim > 1:
                    pass_values = pass_values.flatten()
                
                # Round to integers for comparison
                pass_int = np.round(pass_values).astype(int)
                mask = pass_int == int(pass_number)
                
                if mask.sum() == 0:
                    logger.warning(f"No data found for pass {pass_number}")
                    return dataset.isel(time=slice(0, 0))  # Empty dataset
                
                return dataset.isel(time=mask)
        
        logger.warning(f"Pass variable not found. Tried: {pass_vars}")
        return dataset
    
    def filter_by_multiple_passes(
        self,
        dataset: Any,
        passes: List[int],
        pass_var: str = "pass"
    ) -> Any:
        """
        Filter xarray dataset by multiple pass numbers.
        
        Args:
            dataset: xarray Dataset
            passes: List of pass numbers
            pass_var: Name of pass variable
            
        Returns:
            Filtered dataset
        """
        if not HAS_NUMPY:
            return dataset
        
        pass_vars = [pass_var, "pass_number", "pass_num", "track"]
        
        for var in pass_vars:
            if var in dataset:
                pass_values = np.round(dataset[var].values).astype(int)
                if pass_values.ndim > 1:
                    pass_values = pass_values.flatten()
                
                mask = np.isin(pass_values, passes)
                
                if mask.sum() == 0:
                    logger.warning(f"No data found for passes {passes}")
                    return dataset.isel(time=slice(0, 0))
                
                return dataset.isel(time=mask)
        
        return dataset
    
    def find_passes_in_area(
        self,
        dataset: Any,
        gate_geometry: Any,
        buffer_km: float = 50.0,
        sample_size: int = 10
    ) -> List[int]:
        """
        Find satellite passes that intersect a gate area.
        
        Args:
            dataset: xarray Dataset with lat/lon/pass
            gate_geometry: GeoDataFrame of gate
            buffer_km: Buffer around gate in km
            sample_size: Number of datasets to sample
            
        Returns:
            List of unique pass numbers in area
        """
        if not HAS_NUMPY or not HAS_SHAPELY:
            return []
        
        try:
            from shapely.ops import unary_union
            
            # Buffer gate
            buffer_deg = buffer_km / 111.0
            buffered = gate_geometry.buffer(buffer_deg)
            gate_area = unary_union(buffered.geometry)
            
            # Get coordinates from dataset
            lat_var = "latitude" if "latitude" in dataset else "lat"
            lon_var = "longitude" if "longitude" in dataset else "lon"
            pass_var = "pass_number" if "pass_number" in dataset else "pass"
            
            if lat_var not in dataset or lon_var not in dataset:
                return []
            
            lats = dataset[lat_var].values.flatten()
            lons = dataset[lon_var].values.flatten()
            
            if pass_var not in dataset:
                return []
            
            passes = dataset[pass_var].values.flatten()
            
            # Filter valid coordinates
            valid = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(passes))
            lats = lats[valid]
            lons = lons[valid]
            passes = passes[valid]
            
            # Find passes in area
            found_passes: Set[int] = set()
            
            for i in range(len(lats)):
                point = Point(lons[i], lats[i])
                if gate_area.contains(point) or gate_area.intersects(point):
                    found_passes.add(int(passes[i]))
            
            return sorted(list(found_passes))
            
        except Exception as e:
            logger.error(f"Error finding passes in area: {e}")
            return []
    
    def list_all_gates(self) -> List[str]:
        """List all gates with pre-computed passes."""
        return list(self._closest_passes.keys())
