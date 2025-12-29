"""
Gates Router
============
REST API endpoints for ocean gates.

Provides endpoints for:
- Listing gates
- Getting gate info
- Getting bounding boxes
- Getting satellite passes
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.core.models import GateModel, BoundingBox
from src.services.gate_service import GateService

router = APIRouter(
    prefix="/gates",
    tags=["gates"],
    responses={404: {"description": "Gate not found"}}
)

# Initialize service (singleton pattern)
_gate_service: Optional[GateService] = None


def get_gate_service() -> GateService:
    """Get or create gate service instance."""
    global _gate_service
    if _gate_service is None:
        _gate_service = GateService()
    return _gate_service


@router.get("/", response_model=List[dict])
async def list_gates(
    region: Optional[str] = Query(None, description="Filter by region")
) -> List[dict]:
    """
    List all available ocean gates.
    
    Optionally filter by region (e.g., "Atlantic Sector").
    
    Returns:
        List of gate definitions
    """
    service = get_gate_service()
    
    if region:
        gates = service.list_gates_by_region(region)
    else:
        gates = service.list_gates()
    
    return [g.model_dump() for g in gates]


@router.get("/regions", response_model=List[str])
async def list_regions() -> List[str]:
    """
    List available regions.
    
    Returns:
        List of region names
    """
    service = get_gate_service()
    return service.get_regions()


@router.get("/{gate_id}")
async def get_gate(gate_id: str) -> dict:
    """
    Get gate by ID.
    
    Args:
        gate_id: Gate identifier (e.g., "fram_strait")
        
    Returns:
        Gate definition
        
    Raises:
        404: Gate not found
    """
    service = get_gate_service()
    gate = service.select_gate(gate_id)
    
    if not gate:
        raise HTTPException(status_code=404, detail=f"Gate not found: {gate_id}")
    
    return gate.model_dump()


@router.get("/{gate_id}/bbox")
async def get_gate_bbox(
    gate_id: str,
    buffer_km: float = Query(50.0, ge=0, le=500, description="Buffer in km")
) -> dict:
    """
    Get bounding box for a gate with buffer.
    
    Args:
        gate_id: Gate identifier
        buffer_km: Buffer distance in kilometers (default: 50)
        
    Returns:
        Bounding box with lat/lon min/max
        
    Raises:
        404: Gate not found
    """
    service = get_gate_service()
    
    if not service.gate_exists(gate_id):
        raise HTTPException(status_code=404, detail=f"Gate not found: {gate_id}")
    
    bbox = service.get_bbox(gate_id, buffer_km=buffer_km)
    
    if not bbox:
        raise HTTPException(status_code=500, detail="Could not compute bbox")
    
    return bbox.model_dump()


@router.get("/{gate_id}/passes", response_model=List[int])
async def get_gate_passes(
    gate_id: str,
    n: int = Query(5, ge=1, le=20, description="Number of passes to return")
) -> List[int]:
    """
    Get closest satellite passes for a gate.
    
    Returns pre-computed passes ordered by proximity.
    
    Args:
        gate_id: Gate identifier
        n: Number of passes to return (default: 5)
        
    Returns:
        List of pass numbers
        
    Raises:
        404: Gate not found
    """
    service = get_gate_service()
    
    if not service.gate_exists(gate_id):
        raise HTTPException(status_code=404, detail=f"Gate not found: {gate_id}")
    
    return service.get_closest_passes(gate_id, n=n)


@router.get("/{gate_id}/info")
async def get_gate_info(
    gate_id: str,
    buffer_km: float = Query(50.0, ge=0, le=500, description="Buffer for bbox")
) -> dict:
    """
    Get complete gate information.
    
    Includes gate definition, bbox, and closest passes.
    
    Args:
        gate_id: Gate identifier
        buffer_km: Buffer distance for bbox
        
    Returns:
        Complete gate info
        
    Raises:
        404: Gate not found
    """
    service = get_gate_service()
    
    info = service.get_gate_info(gate_id, buffer_km=buffer_km)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"Gate not found: {gate_id}")
    
    return info


@router.get("/{gate_id}/contains")
async def check_point_in_gate(
    gate_id: str,
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    buffer_km: float = Query(50.0, ge=0, le=500, description="Buffer in km")
) -> dict:
    """
    Check if a point is within a gate's buffered area.
    
    Args:
        gate_id: Gate identifier
        lon: Longitude
        lat: Latitude
        buffer_km: Buffer distance
        
    Returns:
        Boolean result with gate info
        
    Raises:
        404: Gate not found
    """
    service = get_gate_service()
    
    if not service.gate_exists(gate_id):
        raise HTTPException(status_code=404, detail=f"Gate not found: {gate_id}")
    
    is_inside = service.point_in_gate_area(gate_id, lon, lat, buffer_km)
    
    return {
        "gate_id": gate_id,
        "point": {"lon": lon, "lat": lat},
        "buffer_km": buffer_km,
        "is_inside": is_inside
    }
